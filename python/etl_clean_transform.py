"""
Pandas cleaning / transform layer.

Takes the messy raw source-system exports in data/raw/ and produces
analysis-ready tables in data/processed/, logging every fix applied so the
cleaning is auditable (outputs/data_quality_report.md).
"""
import os

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
REPORT_PATH = "outputs/data_quality_report.md"

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def parse_mixed_timestamp(series):
    """Two known formats in the export: 'YYYY-MM-DD HH:MM:SS' and 'DD/MM/YYYY HH:MM'."""
    iso = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dmy = pd.to_datetime(series, format="%d/%m/%Y %H:%M", errors="coerce")
    parsed = iso.fillna(dmy)
    still_missing = parsed.isna().sum()
    if still_missing:
        parsed = parsed.fillna(pd.to_datetime(series, errors="coerce"))
    return parsed


def clean_customers():
    df = pd.read_csv(f"{RAW_DIR}/customers_raw.csv")
    n0 = len(df)

    df["full_name"] = df["full_name"].str.strip().str.title()
    df["email"] = df["email"].str.strip().str.lower()
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    dupes = df.duplicated(subset="customer_id").sum()
    df = df.drop_duplicates(subset="customer_id")

    log(f"customers: {n0:,} rows -> {len(df):,} after dedup ({dupes} duplicate customer_id removed); "
        f"names trimmed/title-cased, emails lower-cased.")
    return df


def clean_accounts():
    df = pd.read_csv(f"{RAW_DIR}/accounts_raw.csv")
    df["opened_date"] = pd.to_datetime(df["opened_date"], errors="coerce")
    df["credit_limit"] = df["credit_limit"].fillna(0.0)
    log(f"accounts: {len(df):,} rows; credit_limit nulls (non credit-card accounts) filled with 0.")
    return df


def clean_merchants():
    df = pd.read_csv(f"{RAW_DIR}/merchants_raw.csv")
    df["onboarded_date"] = pd.to_datetime(df["onboarded_date"], errors="coerce")
    log(f"merchants: {len(df):,} rows loaded, no cleaning required.")
    return df


def clean_transactions(customers, accounts):
    df = pd.read_csv(f"{RAW_DIR}/transactions_raw.csv")
    n0 = len(df)

    # 1. De-duplicate double-posted transactions (same id, or same account+timestamp+amount+merchant)
    exact_dupes = df.duplicated(subset="transaction_id").sum()
    df = df.drop_duplicates(subset="transaction_id")
    fuzzy_dupes = df.duplicated(subset=["account_id", "timestamp", "amount", "merchant_id"]).sum()
    df = df.drop_duplicates(subset=["account_id", "timestamp", "amount", "merchant_id"])

    # 2. Mixed timestamp formats -> single datetime dtype
    df["timestamp"] = parse_mixed_timestamp(df["timestamp"])
    bad_ts = df["timestamp"].isna().sum()
    df = df.dropna(subset=["timestamp"])

    # 3. Sign-error correction: a transaction amount should never be negative
    #    (all outflow/inflow direction is carried by `channel`, not sign).
    neg_amounts = (df["amount"] < 0).sum()
    df["amount"] = df["amount"].abs()

    # 4. Currency code casing normalization
    lower_currency = df["currency"].str.islower().sum()
    df["currency"] = df["currency"].str.upper()

    # 5. Missing device_id: legitimate for ATM/wire, imputed as "UNKNOWN" elsewhere
    #    so it can still be used as a categorical feature (missingness itself is
    #    informative for card-not-present fraud).
    missing_device = df["device_id"].isna().sum()
    df["device_id"] = df["device_id"].fillna("UNKNOWN")

    # 6. Referential integrity: drop orphaned rows (should be none, but verify)
    valid_customers = set(customers["customer_id"])
    valid_accounts = set(accounts["account_id"])
    orphan_cust = (~df["customer_id"].isin(valid_customers)).sum()
    orphan_acct = (~df["account_id"].isin(valid_accounts)).sum()
    df = df[df["customer_id"].isin(valid_customers) & df["account_id"].isin(valid_accounts)]

    # 7. Type normalization
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["merchant_id"] = df["merchant_id"].replace({np.nan: None})

    log(f"transactions: {n0:,} rows -> {len(df):,} after cleaning")
    log(f"  - {exact_dupes} exact-duplicate transaction_id removed")
    log(f"  - {fuzzy_dupes} fuzzy duplicate (same account/timestamp/amount/merchant) removed")
    log(f"  - {bad_ts} rows with unparseable timestamps dropped")
    log(f"  - {neg_amounts} negative amounts corrected (sign-flip export bug) via abs()")
    log(f"  - {lower_currency} lower-cased currency codes normalized to upper")
    log(f"  - {missing_device} missing device_id imputed as 'UNKNOWN'")
    log(f"  - {orphan_cust} rows with unknown customer_id dropped, {orphan_acct} with unknown account_id dropped")
    log(f"  - fraud rate after cleaning: {df['is_fraud'].mean()*100:.3f}% "
        f"({int(df['is_fraud'].sum()):,} of {len(df):,})")
    return df


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    customers = clean_customers()
    accounts = clean_accounts()
    merchants = clean_merchants()
    transactions = clean_transactions(customers, accounts)

    customers.to_csv(f"{PROC_DIR}/customers.csv", index=False)
    accounts.to_csv(f"{PROC_DIR}/accounts.csv", index=False)
    merchants.to_csv(f"{PROC_DIR}/merchants.csv", index=False)
    transactions.to_csv(f"{PROC_DIR}/transactions.csv", index=False)

    with open(REPORT_PATH, "w") as f:
        f.write("# Data Quality Report\n\n")
        f.write("Generated by `python/etl_clean_transform.py`.\n\n")
        for line in log_lines:
            f.write(f"- {line}\n")

    log(f"\nClean tables written to {PROC_DIR}/, report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
