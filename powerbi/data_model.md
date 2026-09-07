# Power BI Data Model

## Files to import

From `powerbi/exports/`:

| File | Role | Grain |
|---|---|---|
| `fct_transactions.csv` | Fact table | 1 row per transaction |
| `dim_customer.csv` | Dimension | 1 row per pseudonymized customer |
| `dim_merchant.csv` | Dimension | 1 row per merchant |
| `summary_by_country.csv` | Pre-aggregated | 1 row per transaction country |
| `summary_monthly.csv` | Pre-aggregated | 1 row per calendar month |
| `model_evaluation_summary.csv` | Pre-aggregated | 1 row per model (for a KPI card) |

All tables use the **GDPR-pseudonymized** identifiers (`customer_pseudo_id`, `account_pseudo_id`) produced by `python/gdpr_pseudonymize.py` -- there are no names, emails, or device IDs anywhere in this layer. See [docs/gdpr_governance.md](../docs/gdpr_governance.md).

## Star schema

```
                dim_customer
                      |
                      | 1:*
                      |
dim_merchant ----- fct_transactions ----- (date table, auto or dim_date)
   1:*                    
```

- `fct_transactions[customer_pseudo_id]` -> `dim_customer[customer_pseudo_id]` (many-to-one)
- `fct_transactions[merchant_id]` -> `dim_merchant[merchant_id]` (many-to-one, some transactions -- ATM/wire -- have no merchant, so this relationship must allow blanks)
- Mark `fct_transactions[date]` as a date column and either enable Power BI's auto date/time or build a `dim_date` table and relate on `date`, for period-over-period DAX (`SAMEPERIODLASTYEAR`, `DATEADD`, etc.)

## Power Query notes

- `fct_transactions[timestamp]` imports as datetime; `[date]` is already a separate date-only column for the relationship, so you don't need to truncate time in Power Query.
- `is_fraud`, `predicted_fraud`, `is_cross_border`, `is_new_device`, `is_night_txn` import as booleans (True/False) -- convert to whole numbers (0/1) in Power Query if you need to `SUM()` them directly instead of using DAX's implicit TRUE()/FALSE() handling.
- `fraud_risk_score` is null for transactions outside the model's time-based test window (the training period) -- this is expected; see [docs/architecture.md](../docs/architecture.md) for why the split is time-based, not random.

## Suggested pages

1. **Executive Overview** -- fraud rate trend (`summary_monthly`), total exposure (EUR), model precision/recall KPI cards, fraud rate by country map.
2. **Risk Investigation** -- `fct_transactions` table visual filtered to `fraud_risk_score` deciles, drillthrough to a single customer's transaction timeline.
3. **Channel & Merchant Risk** -- fraud rate by channel and MCC category, cross-border vs. domestic split.
4. **Model Performance** -- ROC/PR curve values (`model_evaluation_summary.csv`), confusion-matrix KPI cards, calibration by risk decile (mirrors `sql/analysis_queries.sql` query 6).
