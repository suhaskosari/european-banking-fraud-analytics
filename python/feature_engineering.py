"""
Behavioural and risk-indicator feature engineering.

Builds per-transaction features used by both the statistical tests and the
fraud classifier: spending-velocity, amount-vs-customer-baseline deviation,
geographic distance from home country, new-device flags, and merchant risk.
"""
import numpy as np
import pandas as pd

PROC_DIR = "data/processed"

CENTROIDS = {
    "DE": (51.16, 10.45), "FR": (46.60, 2.35), "ES": (40.30, -3.70),
    "IT": (42.50, 12.50), "NL": (52.30, 5.75), "SE": (60.10, 18.65),
    "PL": (52.00, 19.15), "IE": (53.35, -8.25),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def geo_distance_km(home, txn):
    h = CENTROIDS.get(home)
    t = CENTROIDS.get(txn)
    if h is None or t is None:
        return np.nan
    return haversine_km(h[0], h[1], t[0], t[1])


def build_features():
    txns = pd.read_csv(f"{PROC_DIR}/transactions.csv", parse_dates=["timestamp"])
    merchants = pd.read_csv(f"{PROC_DIR}/merchants.csv")
    txns = txns.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    # --- Customer baseline (mean/std amount) computed leakage-safely with an
    # expanding window shifted by one, i.e. only using each customer's history
    # strictly *before* the current transaction. ---
    grp = txns.groupby("customer_id")["amount"]
    txns["cust_hist_mean"] = grp.transform(lambda s: s.shift(1).expanding().mean())
    txns["cust_hist_std"] = grp.transform(lambda s: s.shift(1).expanding().std())
    txns["cust_hist_count"] = grp.transform(lambda s: s.shift(1).expanding().count())

    # Cold-start (first transaction ever, or too few prior txns): fall back to
    # the customer's home-country segment average via a global default.
    global_mean, global_std = txns["amount"].mean(), txns["amount"].std()
    txns["cust_hist_mean"] = txns["cust_hist_mean"].fillna(global_mean)
    txns["cust_hist_std"] = txns["cust_hist_std"].fillna(global_std).replace(0, global_std)

    txns["amount_zscore"] = (txns["amount"] - txns["cust_hist_mean"]) / txns["cust_hist_std"]
    txns["amount_zscore"] = txns["amount_zscore"].clip(-10, 10)

    # --- Velocity features: count of transactions by the same customer in the
    # trailing 1h / 24h windows (rolling time-window count, leakage-safe). ---
    def rolling_count(s, window):
        return s.rolling(window, closed="left").count()

    txns = txns.set_index("timestamp")
    counts_1h, counts_24h = [], []
    for cust_id, g in txns.groupby("customer_id"):
        idx = pd.Series(1, index=g.index)
        counts_1h.append(idx.rolling("1h", closed="left").sum())
        counts_24h.append(idx.rolling("24h", closed="left").sum())
    txns["velocity_1h"] = pd.concat(counts_1h).reindex(txns.index).fillna(0)
    txns["velocity_24h"] = pd.concat(counts_24h).reindex(txns.index).fillna(0)
    txns = txns.reset_index()

    # --- Geographic risk: distance between home country and transaction country ---
    txns["geo_distance_km"] = txns.apply(
        lambda r: geo_distance_km(r["home_country"], r["txn_country"]), axis=1
    )
    txns["is_cross_border"] = (txns["home_country"] != txns["txn_country"]).astype(int)

    # --- Device / channel risk ---
    txns["is_new_device"] = (txns["device_id"] == "UNKNOWN").astype(int)
    txns["is_high_risk_channel"] = txns["channel"].isin(["card_not_present", "wire_transfer", "atm_withdrawal"]).astype(int)

    # --- Temporal risk: unusual hour-of-day for this customer ---
    txns["txn_hour"] = txns["timestamp"].dt.hour
    txns["is_night_txn"] = txns["txn_hour"].between(0, 5).astype(int)

    # --- Merchant risk ---
    txns = txns.merge(
        merchants[["merchant_id", "mcc_category", "base_risk_multiplier"]],
        on="merchant_id", how="left"
    )
    txns["base_risk_multiplier"] = txns["base_risk_multiplier"].fillna(1.5)  # non-merchant (wire/atm) default
    txns["mcc_category"] = txns["mcc_category"].fillna("N/A (wire/ATM)")

    # --- Round-amount flag (common in collusion / structuring) ---
    txns["is_round_amount"] = (txns["amount"] % 50 == 0).astype(int)

    txns.to_csv(f"{PROC_DIR}/transactions_features.csv", index=False)
    print(f"Feature table written: {PROC_DIR}/transactions_features.csv "
          f"({len(txns):,} rows, {txns.shape[1]} columns)")
    return txns


if __name__ == "__main__":
    build_features()
