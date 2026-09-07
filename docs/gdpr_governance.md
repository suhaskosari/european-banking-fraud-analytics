# GDPR & Data Governance

This project handles simulated personal and financial data as if it were subject to the EU General Data Protection Regulation (GDPR), because that is the realistic constraint a fraud-analytics platform at a European bank operates under. No real personal data is used anywhere in this repository -- all customers, transactions, and identifiers are synthetically generated (`python/generate_data.py`).

## 1. Data inventory & classification

| Table | Fields | Classification |
|---|---|---|
| `raw.customers` | `full_name`, `email` | Direct identifier -- highest sensitivity |
| `raw.customers` | `country`, `age`, `segment`, `kyc_verified` | Personal data (indirect identifier / special context) |
| `raw.transactions` | `device_id` | Indirect identifier |
| `raw.transactions` | `amount`, `channel`, `timestamp`, `is_fraud` | Financial / behavioural data |
| `analytics.*` | pseudonymized IDs, generalized age/date fields | Pseudonymized personal data |

## 2. Lawful basis

For a real deployment, transaction monitoring for fraud prevention is processed under **Art. 6(1)(f) legitimate interest** (fraud prevention is explicitly named as a legitimate interest in Recital 47) and, for regulated entities, **Art. 6(1)(c) legal obligation** (AML/CTF and PSD2 fraud-monitoring duties). A Legitimate Interest Assessment (LIA) would be filed alongside this basis; it is out of scope for this demo but the pipeline is built as if one exists.

## 3. Data minimization (Art. 5(1)(c))

Implemented in `python/gdpr_pseudonymize.py`:

- `full_name` and `email` are **dropped entirely** from the analytics layer -- they are never needed for fraud modelling or dashboarding.
- Exact `device_id` is dropped in favour of the derived `is_new_device` boolean, which preserves the analytical signal without the identifier.
- Exact `age` is generalized into 6 age bands; exact `signup_date` is generalized to year-month.
- `data_model.md` / the Power BI layer only ever sees the minimized, pseudonymized tables -- never `raw.*`.

## 4. Pseudonymization (Art. 4(5), Art. 32)

- `customer_id` and `account_id` are replaced with a **keyed HMAC-SHA256 pseudonym**, truncated to 16 hex characters, so the same real ID always maps to the same pseudonym *within a given key* (preserving joinability for time-series and cohort analysis) while being computationally infeasible to reverse without the key.
- The key is generated at pipeline runtime and stored at `outputs/.pseudonymization_key.txt`, which is **git-ignored** -- it never leaves the local environment in this demo. In production this key would live in a secrets manager (e.g. AWS KMS / Azure Key Vault) with its own access-control policy, separate from the analytics data store, so that re-identification requires two independent approvals (the pseudonymized data plus the key).
- Precise `timestamp` is retained in the analytics layer because velocity-based fraud features require it; this is a **documented, risk-accepted exception** to full anonymization -- the data remains pseudonymized (re-identifiable in principle with the key), not anonymized, and is governed accordingly.

## 5. Access control matrix

Modeled in `sql/schema.sql` (commented `CREATE ROLE` / `GRANT` statements, since the demo has no live database):

| Role | `raw.*` | `analytics.*` | Typical user |
|---|---|---|---|
| `fraud_analyst_readonly` | No access | `SELECT` only | Analysts building dashboards, investigating flagged transactions by pseudonym |
| `compliance_officer` | `SELECT` | `SELECT` | Handling a Subject Access Request (SAR) or a confirmed-fraud case requiring re-identification |
| `ml_pipeline_service` | `SELECT` (own schema only, via service account) | `INSERT` on scored/flagged tables | The automated scoring pipeline (`fraud_model.py`, `anomaly_detection.py`) |

Every access by `compliance_officer` to `raw.*` in a production system would be logged (who, when, which `customer_id`, and the case/ticket justifying it) -- this is the audit trail a Data Protection Impact Assessment (DPIA) would require for a re-identification capability.

## 6. Data retention

- Transaction records: retained per the applicable AML statutory minimum (commonly 5 years post-relationship in EU member states), then purged or fully anonymized (irreversibly, unlike pseudonymization).
- The pseudonymization key: rotated periodically; rotating it without re-processing history effectively anonymizes all previously pseudonymized IDs against the old key.
- Model training data: capped to a rolling window (this demo uses 18 months) so stale behavioural patterns don't perpetually influence live risk scores.

## 7. DPIA summary (Art. 35)

A Data Protection Impact Assessment is required here because the processing involves: (a) systematic profiling used to produce legal/significant effects (a fraud flag can freeze an account or decline a payment), and (b) large-scale processing of financial data. Key mitigations already reflected in the pipeline design:

- **Human-in-the-loop**: `fraud_model.py` and `anomaly_detection.py` produce a *risk score and flag*, not an automated decision -- the intended production use is to route high-risk transactions to a human fraud analyst (Art. 22 safeguard against solely-automated decisions with legal effect).
- **Explainability**: Logistic Regression is trained alongside Gradient Boosting specifically to keep an interpretable model in the loop (coefficient signs and magnitudes are directly explainable to a customer or regulator), and Gradient Boosting's feature importances are exported (`outputs/fraud_model_feature_importance.png`) so any flag can be traced to the behavioural signals that drove it.
- **Fairness check (recommended next step, not yet implemented)**: fraud-rate and false-positive-rate should be monitored by `country` and `age_band` in `analytics.dim_customer` to catch a model that systematically over-flags a particular nationality or age group, which would itself be a compliance and fairness problem independent of accuracy.
- **Bias in training data**: the synthetic fraud injection in `generate_data.py` is pattern-based (card-testing, account takeover, merchant collusion, mule accounts) and not tied to protected characteristics, but any real deployment must re-run this fairness check against real, labeled fraud data before going live.

## 8. Right to erasure vs. legal retention conflict

Under Art. 17(3)(b), the right to erasure does not apply while data is needed for compliance with a legal obligation (AML retention). In practice: an erasure request during the statutory retention window is honored for **marketing/product** data but the transaction and KYC record is retained under the legal-obligation exception, with the customer informed of this exception per Art. 17(3) and the fields not required for the legal basis (e.g. marketing preferences, if any existed) still minimized/deleted immediately.
