"""
GDPR-aware data handling layer.

Produces a pseudonymized, minimized copy of the processed tables suitable
for analytics/BI use and for sharing with a wider audience (e.g. a Power BI
workspace) without exposing direct identifiers. This is a deliberate,
documented transformation -- see docs/gdpr_governance.md for the full policy
this implements (lawful basis, retention, access tiers, DPIA summary).

Design choices:
  - Direct identifiers (name, email) are dropped entirely from the
    analytics copy (data minimization, Art. 5(1)(c)).
  - customer_id / account_id are replaced with a keyed pseudonym (HMAC-SHA256)
    so the same real ID always maps to the same pseudonym *within this
    dataset*, preserving joinability for analysis, without being reversible
    without the secret key (pseudonymization, Art. 4(5)).
  - The mapping key is generated at runtime and never written to disk in the
    analytics output -- only a salted, one-way pseudonym is exported. The key
    itself would live in a separate, access-controlled secrets store in
    production (kept here in outputs/.pseudonymization_key.txt, gitignored,
    purely so the pipeline is re-runnable locally).
  - Precise birthdate/age is bucketed into age bands; exact device_id is
    dropped in favour of a boolean new-device flag (already computed
    upstream) -- both reduce re-identification risk while preserving
    analytical value.
"""
import hashlib
import hmac
import os
import secrets

import pandas as pd

PROC_DIR = "data/processed"
OUT_DIR = "outputs"
KEY_PATH = f"{OUT_DIR}/.pseudonymization_key.txt"


def get_or_create_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "r") as f:
            return bytes.fromhex(f.read().strip())
    key = secrets.token_bytes(32)
    with open(KEY_PATH, "w") as f:
        f.write(key.hex())
    return key


def pseudonymize(value, key):
    if pd.isna(value):
        return None
    return hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()[:16]


def age_band(age):
    if pd.isna(age):
        return "unknown"
    bins = [17, 25, 35, 45, 55, 65, 120]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    for hi, label in zip(bins[1:], labels):
        if age <= hi:
            return label
    return "66+"


def pseudonymize_customers(key):
    df = pd.read_csv(f"{PROC_DIR}/customers.csv")
    out = pd.DataFrame({
        "customer_pseudo_id": df["customer_id"].apply(lambda v: pseudonymize(v, key)),
        "country": df["country"],
        "age_band": df["age"].apply(age_band),
        "segment": df["segment"],
        "signup_year_month": pd.to_datetime(df["signup_date"]).dt.to_period("M").astype(str),
        "kyc_verified": df["kyc_verified"],
    })
    return out


def pseudonymize_transactions(key):
    df = pd.read_csv(f"{PROC_DIR}/transactions_features.csv", parse_dates=["timestamp"])
    out = df.copy()
    out["customer_pseudo_id"] = out["customer_id"].apply(lambda v: pseudonymize(v, key))
    out["account_pseudo_id"] = out["account_id"].apply(lambda v: pseudonymize(v, key))
    out = out.drop(columns=["customer_id", "account_id", "device_id"])
    # Precise timestamp retained (needed for velocity/time-series analysis);
    # this is a documented, risk-accepted exception -- see docs/gdpr_governance.md.
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    key = get_or_create_key()

    customers_pseudo = pseudonymize_customers(key)
    transactions_pseudo = pseudonymize_transactions(key)

    customers_pseudo.to_csv(f"{OUT_DIR}/customers_pseudonymized.csv", index=False)
    transactions_pseudo.to_csv(f"{OUT_DIR}/transactions_pseudonymized.csv", index=False)

    direct_identifiers_dropped = ["full_name", "email", "device_id", "customer_id (raw)", "account_id (raw)"]
    print("GDPR pseudonymization complete.")
    print(f"  - Direct identifiers dropped from analytics copy: {direct_identifiers_dropped}")
    print(f"  - {customers_pseudo['customer_pseudo_id'].nunique():,} unique customer pseudonyms "
          f"(HMAC-SHA256, key held out-of-band)")
    print(f"  - Age generalized to 6 bands; signup date generalized to year-month")
    print(f"  - Outputs: {OUT_DIR}/customers_pseudonymized.csv, {OUT_DIR}/transactions_pseudonymized.csv")
    print(f"  - Key stored at {KEY_PATH} (gitignored -- would be a secrets manager in production)")


if __name__ == "__main__":
    main()
