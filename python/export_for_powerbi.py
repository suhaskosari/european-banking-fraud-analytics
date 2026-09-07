"""
Builds Power BI-ready CSV exports from the pseudonymized analytics layer:
a fact table with risk scores/flags, and supporting dimension/summary
tables for the dashboards described in powerbi/data_model.md.
"""
import os

import pandas as pd

OUT_DIR = "outputs"
PBI_DIR = "powerbi/exports"


def main():
    os.makedirs(PBI_DIR, exist_ok=True)

    txns = pd.read_csv(f"{OUT_DIR}/transactions_pseudonymized.csv", parse_dates=["timestamp"])
    customers = pd.read_csv(f"{OUT_DIR}/customers_pseudonymized.csv")
    merchants = pd.read_csv("data/processed/merchants.csv")
    scored = pd.read_csv(f"{OUT_DIR}/scored_transactions.csv", parse_dates=["timestamp"])
    anomalies = pd.read_csv(f"{OUT_DIR}/anomaly_flagged_transactions.csv")

    # --- fct_transactions: the core fact table for the dashboard ---
    fct = txns[[
        "transaction_id", "customer_pseudo_id", "account_pseudo_id", "merchant_id",
        "timestamp", "amount", "currency", "channel", "txn_country", "home_country",
        "mcc_category", "is_cross_border", "is_new_device", "is_night_txn",
        "velocity_1h", "velocity_24h", "amount_zscore", "is_fraud",
    ]].copy()
    fct["date"] = fct["timestamp"].dt.date
    fct["year_month"] = fct["timestamp"].dt.to_period("M").astype(str)
    fct = fct.merge(
        scored[["transaction_id", "fraud_risk_score", "predicted_fraud"]],
        on="transaction_id", how="left",
    )
    fct = fct.merge(
        anomalies[["transaction_id", "flag_zscore_anomaly", "iso_forest_score"]],
        on="transaction_id", how="left",
    ).fillna({"flag_zscore_anomaly": 0, "iso_forest_score": 0})
    fct.to_csv(f"{PBI_DIR}/fct_transactions.csv", index=False)

    # --- dim_customer ---
    customers.to_csv(f"{PBI_DIR}/dim_customer.csv", index=False)

    # --- dim_merchant ---
    merchants[["merchant_id", "merchant_name", "mcc", "mcc_category", "country", "base_risk_multiplier"]].to_csv(
        f"{PBI_DIR}/dim_merchant.csv", index=False
    )

    # --- dim_country: fraud-rate-by-country summary for map visuals ---
    country_summary = (
        txns.groupby("txn_country")
        .agg(total_transactions=("transaction_id", "count"),
             total_amount=("amount", "sum"),
             fraud_transactions=("is_fraud", "sum"))
        .reset_index()
    )
    country_summary["fraud_rate_pct"] = (
        country_summary["fraud_transactions"] / country_summary["total_transactions"] * 100
    ).round(4)
    country_summary.to_csv(f"{PBI_DIR}/summary_by_country.csv", index=False)

    # --- monthly KPI summary ---
    monthly = (
        fct.groupby("year_month")
        .agg(total_transactions=("transaction_id", "count"),
             total_amount=("amount", "sum"),
             fraud_transactions=("is_fraud", "sum"),
             avg_risk_score=("fraud_risk_score", "mean"))
        .reset_index()
    )
    monthly["fraud_rate_pct"] = (monthly["fraud_transactions"] / monthly["total_transactions"] * 100).round(4)
    monthly.to_csv(f"{PBI_DIR}/summary_monthly.csv", index=False)

    # --- model evaluation summary for a dashboard KPI card ---
    import json
    with open(f"{OUT_DIR}/model_evaluation.json") as f:
        model_eval = pd.DataFrame(json.load(f))
    model_eval.drop(columns=["confusion_matrix"]).to_csv(f"{PBI_DIR}/model_evaluation_summary.csv", index=False)

    print(f"Power BI exports written to {PBI_DIR}/:")
    for fp in sorted(os.listdir(PBI_DIR)):
        print(f"  - {fp}")


if __name__ == "__main__":
    main()
