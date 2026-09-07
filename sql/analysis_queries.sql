-- Example analytical queries against the analytics schema (sql/schema.sql).
-- Written for PostgreSQL; the repo's demo pipeline computes the same
-- metrics in pandas (see python/) so it runs without a live database.

-- 1. Fraud rate and volume by country, ranked (geographic risk pattern)
SELECT
    txn_country,
    COUNT(*)                                   AS total_transactions,
    SUM(amount)                                AS total_amount,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)  AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct
FROM analytics.fct_transactions
GROUP BY txn_country
ORDER BY fraud_rate_pct DESC;

-- 2. Fraud rate by channel and cross-border flag (risk segmentation)
SELECT
    channel,
    is_cross_border,
    COUNT(*)                                                              AS total_transactions,
    ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END), 4)           AS fraud_rate_pct,
    ROUND(AVG(fraud_risk_score)::numeric, 4)                              AS avg_model_risk_score
FROM analytics.fct_transactions
GROUP BY channel, is_cross_border
ORDER BY fraud_rate_pct DESC;

-- 3. Monthly fraud trend with month-over-month change
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', txn_timestamp)                          AS month,
        COUNT(*)                                                    AS total_transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                   AS fraud_transactions
    FROM analytics.fct_transactions
    GROUP BY 1
)
SELECT
    month,
    total_transactions,
    fraud_transactions,
    ROUND(100.0 * fraud_transactions / total_transactions, 4) AS fraud_rate_pct,
    ROUND(
        100.0 * fraud_transactions / total_transactions
        - LAG(100.0 * fraud_transactions / total_transactions) OVER (ORDER BY month),
    4) AS fraud_rate_pct_change_mom
FROM monthly
ORDER BY month;

-- 4. Top merchant categories by fraud exposure (amount at risk, not just count)
SELECT
    m.mcc_category,
    COUNT(*)                                                    AS total_transactions,
    SUM(f.amount) FILTER (WHERE f.is_fraud)                     AS fraud_amount_exposed,
    ROUND(100.0 * AVG(CASE WHEN f.is_fraud THEN 1 ELSE 0 END), 4) AS fraud_rate_pct
FROM analytics.fct_transactions f
JOIN analytics.dim_merchant m ON m.merchant_id = f.merchant_id
GROUP BY m.mcc_category
HAVING COUNT(*) > 100
ORDER BY fraud_amount_exposed DESC NULLS LAST;

-- 5. High-velocity customers in the trailing 24h (real-time-style watchlist)
SELECT
    customer_pseudo_id,
    COUNT(*)                       AS txns_last_24h_window,
    MAX(velocity_24h)              AS max_recorded_velocity_24h,
    SUM(amount)                    AS total_amount,
    BOOL_OR(is_fraud)              AS any_confirmed_fraud
FROM analytics.fct_transactions
WHERE txn_timestamp >= (SELECT MAX(txn_timestamp) FROM analytics.fct_transactions) - INTERVAL '1 day'
GROUP BY customer_pseudo_id
HAVING MAX(velocity_24h) >= 5
ORDER BY max_recorded_velocity_24h DESC;

-- 6. Model risk-score calibration check: does higher score bucket => higher actual fraud rate?
SELECT
    WIDTH_BUCKET(fraud_risk_score, 0, 1, 10)                    AS risk_decile,
    COUNT(*)                                                    AS n_transactions,
    ROUND(100.0 * AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END), 4) AS actual_fraud_rate_pct,
    ROUND(AVG(fraud_risk_score)::numeric, 4)                    AS avg_predicted_score
FROM analytics.fct_transactions
WHERE fraud_risk_score IS NOT NULL
GROUP BY risk_decile
ORDER BY risk_decile;

-- 7. Customers whose most recent transaction is a large outlier vs. their own history
--    (uses the same z-score logic as python/feature_engineering.py, expressed in SQL
--    via a window function, for teams that prefer to compute it warehouse-side)
WITH history AS (
    SELECT
        transaction_id,
        customer_pseudo_id,
        txn_timestamp,
        amount,
        AVG(amount) OVER (
            PARTITION BY customer_pseudo_id ORDER BY txn_timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS hist_mean,
        STDDEV(amount) OVER (
            PARTITION BY customer_pseudo_id ORDER BY txn_timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS hist_std
    FROM analytics.fct_transactions
)
SELECT
    customer_pseudo_id, transaction_id, txn_timestamp, amount,
    ROUND(((amount - hist_mean) / NULLIF(hist_std, 0))::numeric, 2) AS amount_zscore
FROM history
WHERE hist_std IS NOT NULL AND ABS((amount - hist_mean) / NULLIF(hist_std, 0)) > 3
ORDER BY amount_zscore DESC;
