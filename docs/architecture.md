# Architecture

## Pipeline

```
generate_data.py                 (synthetic customers/accounts/merchants/transactions
       |                          + 4 injected fraud patterns, deliberate messiness)
       v
data/raw/*.csv
       |
etl_clean_transform.py           (dedup, mixed-date parsing, sign-error fix,
       |                          currency casing, referential integrity)
       v
data/processed/*.csv  ---------> outputs/data_quality_report.md
       |
feature_engineering.py           (leakage-safe rolling z-score, 1h/24h velocity,
       |                          geo-distance, device/channel/time risk flags)
       v
data/processed/transactions_features.csv
       |
       +----------------------------+----------------------------+
       v                            v                            v
stats_hypothesis_testing.py   anomaly_detection.py         fraud_model.py
(Mann-Whitney U, chi-square,  (per-customer z-score +      (Logistic Regression vs.
 two-proportion z-test)        Isolation Forest, evaluated  Gradient Boosting, time-based
       |                       against labels)               split, precision/recall/F1/
       v                            |                         ROC-AUC/PR-AUC)
outputs/hypothesis_test_          |                            |
  results.json                    v                            v
                          outputs/anomaly_*.csv        outputs/scored_transactions.csv,
                                                         model_evaluation.json,
                                                         fraud_model_curves.png,
                                                         fraud_model_feature_importance.png
       |
       v
gdpr_pseudonymize.py              (drops direct identifiers, HMAC-pseudonymizes
       |                            customer/account IDs, generalizes age/date)
       v
outputs/*_pseudonymized.csv
       |
export_for_powerbi.py
       |
       v
powerbi/exports/*.csv  --------> Power BI Desktop (see powerbi/data_model.md)
```

`sql/schema.sql` documents the same shapes as PostgreSQL DDL (a `raw` schema with direct identifiers, and a permission-separated `analytics` schema matching the pseudonymized exports) for how this would be productionized as a warehouse rather than a local CSV pipeline; `sql/analysis_queries.sql` shows the equivalent analytical queries in SQL for teams that compute metrics warehouse-side instead of in pandas.

## Why a time-based train/test split for the fraud model

`fraud_model.py` splits by transaction timestamp (train on the earliest ~75%, test on the most recent ~25%) rather than a random split. A random split lets the model implicitly "see the future" -- e.g. a card-testing burst split half into train and half into test leaks the exact pattern it's being tested on. A time-based split is the only split that matches how the model would actually be evaluated in production: trained on history, scored on transactions that haven't happened yet.

## Why Logistic Regression *and* Gradient Boosting, not just the better model

Gradient Boosting outperforms Logistic Regression on every metric here (ROC-AUC 0.9999 vs. 0.9891, precision 0.90 vs. 0.36 at the same 0.5 threshold) -- but the Logistic Regression model is kept as the production-facing explainable baseline. In a regulated banking context, a declined transaction or frozen account is a decision with real consequences for a customer, and "the gradient boosting model said so" is a much harder answer to defend to a regulator or an appealing customer than a small set of signed coefficients. See [docs/gdpr_governance.md](gdpr_governance.md#7-dpia-summary-art-35) for how this ties into the DPIA's explainability requirement.

## Data quality issues deliberately injected (and how they're caught)

| Issue | Injected in | Caught by |
|---|---|---|
| Duplicate transaction rows (double-posting) | `add_messiness()` | `etl_clean_transform.py` dedup on `transaction_id` and fuzzy dedup on (account, timestamp, amount, merchant) |
| Mixed timestamp formats (`YYYY-MM-DD HH:MM:SS` vs `DD/MM/YYYY HH:MM`) | `add_messiness()` | `parse_mixed_timestamp()` tries both formats before falling back to pandas' generic parser |
| Sign-flip bug (negative amounts) | `add_messiness()` | `.abs()` correction, logged with a before/after count |
| Currency code casing (`eur` vs `EUR`) | `add_messiness()` | `.str.upper()` normalization |
| Missing `device_id` on some legitimate online transactions | `add_messiness()` | Imputed as `"UNKNOWN"` (not dropped -- missingness itself is a fraud signal, reused as the `is_new_device` feature) |
| 4 realistic fraud typologies: card-testing, account takeover, merchant collusion, mule accounts | `inject_fraud_patterns()` | Detected via engineered velocity/geo/device features feeding both the unsupervised anomaly detectors and the supervised classifier |

## Fraud typologies simulated

1. **Card-testing**: 6-18 rapid, tiny (< EUR 4) card-not-present transactions against random merchants within minutes -- validates a new stolen card before a larger purchase. Drives the `velocity_1h` / `velocity_24h` features.
2. **Account takeover (ATO)**: a single large, geographically distant transaction from a new device shortly after the account's normal behaviour -- validates `geo_distance_km`, `is_new_device`, `is_cross_border`.
3. **Merchant collusion**: a small set of merchants systematically process inflated, round-number transactions for a rotating set of customers -- validates `is_round_amount` and merchant-level aggregation (see `sql/analysis_queries.sql` query 4).
4. **Mule accounts**: a newly opened account receives one large wire, then drains it via ATM withdrawals in a different country within hours -- validates the interaction between `cust_hist_count` (cold-start accounts) and cross-border ATM channel risk.
