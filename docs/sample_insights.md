# Sample Insights (generated output)

This is a real example of the numbers the pipeline produces end to end on the synthetic dataset (275,799 raw transactions across 6,000 customers, 8 European markets, 18 months).

## Data quality

From `outputs/data_quality_report.md`:

- 1,645 exact-duplicate transactions removed (double-posting simulation)
- 1,099 negative amounts corrected via a sign-flip fix
- 13,711 lower-cased currency codes normalized
- 1,995 missing `device_id` values imputed as `UNKNOWN` (kept as a feature, not dropped)
- Final clean dataset: 274,154 transactions, fraud rate 0.426%

## Hypothesis tests

From `outputs/hypothesis_test_results.json` (all significant at alpha = 0.05):

| Test | Finding |
|---|---|
| Mann-Whitney U, amount | Fraud and legitimate amounts differ significantly (p < 0.001); **median fraud amount (EUR 3.01) is actually lower** than legitimate (EUR 61.33) -- driven by high-volume, low-value card-testing fraud outnumbering the rarer high-value account-takeover cases |
| Chi-square, country | Fraud incidence depends on transaction country (p = 0.0045), though the effect is small (Cramer's V = 0.009) |
| Chi-square, channel | Fraud incidence strongly depends on channel (p < 0.001, Cramer's V = 0.080) |
| Two-proportion z-test, cross-border | Cross-border transactions have a significantly higher fraud rate than domestic (p < 0.001, Cohen's h = 0.087) |
| Mann-Whitney U, 1h velocity | Fraud transactions show significantly higher 1-hour transaction velocity (mean 3.89 vs. 0.0075 for legitimate) -- the clearest single behavioural signal in the dataset |

## Anomaly detection (unsupervised, no fraud label used)

| Method | Flagged | Precision | Recall | F1 |
|---|---|---|---|---|
| Per-customer amount z-score (univariate) | 10,735 | 0.022 | 0.199 | 0.039 |
| Isolation Forest (multivariate, 11 features) | 2,742 | 0.314 | 0.736 | 0.440 |

The univariate z-score is noisy on its own (too many legitimate but simply-large transactions get flagged) -- the multivariate Isolation Forest, using velocity/geo/device/channel signals together, is over an order of magnitude better on precision at similar recall. This is the practical argument for feature engineering before anomaly detection, not just after it.

## Supervised fraud classifier (time-based test split, most recent ~25% of transactions)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (interpretable) | 0.362 | 0.899 | 0.516 | 0.989 | 0.895 |
| Gradient Boosting | 0.904 | 0.965 | 0.934 | 0.9999 | 0.989 |

Gradient Boosting catches 633 of 656 fraud cases in the test window with only 67 false positives. Logistic Regression is kept as the explainable production-facing baseline (see [docs/architecture.md](architecture.md#why-logistic-regression-and-gradient-boosting-not-just-the-better-model)) despite the gap.

See `outputs/fraud_model_feature_importance.png` for which engineered features drive the Gradient Boosting model -- raw `amount` dominates (consistent with the amount-distribution finding above), followed by `is_round_amount` (the merchant-collusion signature) and `is_high_risk_channel`; velocity and geo-distance rank lower for this model despite being strong *univariate* signals, because gradient boosting can already separate most fraud on amount and channel alone once the training window contains enough of each fraud type -- a reminder that feature importance reflects what a specific model needed, not everything that's predictive in isolation (see the hypothesis tests above, where velocity's standalone effect is the largest of all).
