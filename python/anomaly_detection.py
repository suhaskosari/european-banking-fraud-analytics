"""
Unsupervised anomaly detection as a fraud-analyst triage layer.

Two complementary approaches:
  1. Statistical: per-customer z-score on transaction amount (already computed
     in feature_engineering.py) flags any transaction far from that
     customer's own spending baseline.
  2. Model-based: Isolation Forest over the full behavioural feature set,
     which can catch multivariate anomalies a single z-score threshold misses
     (e.g. a normal amount at an unusual hour, from a new device, far from home).

This is run *without* the fraud label (unsupervised) and then compared
against it purely for evaluation -- exactly how it would be used in
production to surface transactions for manual review before enough labeled
fraud exists to train a supervised model.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROC_DIR = "data/processed"
OUT_DIR = "outputs"

Z_THRESHOLD = 3.0
CONTAMINATION = 0.01  # assumed prior fraction of anomalies for Isolation Forest


def zscore_flagging(df):
    df["flag_zscore_anomaly"] = (df["amount_zscore"].abs() > Z_THRESHOLD).astype(int)
    return df


def isolation_forest_flagging(df):
    features = [
        "amount", "amount_zscore", "velocity_1h", "velocity_24h",
        "geo_distance_km", "is_cross_border", "is_new_device",
        "is_high_risk_channel", "is_night_txn", "base_risk_multiplier", "is_round_amount",
    ]
    X = df[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=CONTAMINATION, random_state=42, n_jobs=-1
    )
    predictions = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    df["iso_forest_score"] = (predictions == -1).astype(int)
    df["iso_forest_raw_score"] = model.decision_function(X_scaled)
    return df, features


def evaluate_against_labels(df, method_col, label_col="is_fraud"):
    tp = ((df[method_col] == 1) & (df[label_col])).sum()
    fp = ((df[method_col] == 1) & (~df[label_col])).sum()
    fn = ((df[method_col] == 0) & (df[label_col])).sum()
    tn = ((df[method_col] == 0) & (~df[label_col])).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"flagged": int(df[method_col].sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def main():
    df = pd.read_csv(f"{PROC_DIR}/transactions_features.csv", parse_dates=["timestamp"])
    df["is_fraud"] = df["is_fraud"].astype(bool)

    df = zscore_flagging(df)
    df, feats = isolation_forest_flagging(df)

    z_eval = evaluate_against_labels(df, "flag_zscore_anomaly")
    iso_eval = evaluate_against_labels(df, "iso_forest_score")

    print("Per-customer amount z-score anomaly flagging (unsupervised, univariate):")
    print(f"  {z_eval}")
    print("\nIsolation Forest anomaly flagging (unsupervised, multivariate):")
    print(f"  {iso_eval}")

    flagged = df[(df["flag_zscore_anomaly"] == 1) | (df["iso_forest_score"] == 1)]
    flagged[[
        "transaction_id", "customer_id", "merchant_id", "timestamp", "amount",
        "channel", "txn_country", "home_country", "amount_zscore", "velocity_1h",
        "flag_zscore_anomaly", "iso_forest_score", "is_fraud", "fraud_type",
    ]].to_csv(f"{OUT_DIR}/anomaly_flagged_transactions.csv", index=False)

    summary = pd.DataFrame([
        {"method": "zscore_amount", **z_eval},
        {"method": "isolation_forest", **iso_eval},
    ])
    summary.to_csv(f"{OUT_DIR}/anomaly_detection_summary.csv", index=False)
    print(f"\n{len(flagged):,} transactions flagged by at least one method -> "
          f"{OUT_DIR}/anomaly_flagged_transactions.csv")
    print(f"Summary -> {OUT_DIR}/anomaly_detection_summary.csv")


if __name__ == "__main__":
    main()
