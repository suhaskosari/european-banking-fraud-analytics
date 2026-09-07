# DAX Measures

Paste these into a dedicated `_Measures` table (Model view -> New Table -> `_Measures = {BLANK()}`, then add each measure to it) once the model in [data_model.md](data_model.md) is built.

## Volume & Exposure

```dax
Total Transactions =
COUNTROWS ( fct_transactions )

Total Amount (EUR) =
SUM ( fct_transactions[amount] )

Fraud Transactions =
CALCULATE ( COUNTROWS ( fct_transactions ), fct_transactions[is_fraud] = TRUE )

Fraud Amount Exposed (EUR) =
CALCULATE ( SUM ( fct_transactions[amount] ), fct_transactions[is_fraud] = TRUE )

Fraud Rate % =
DIVIDE ( [Fraud Transactions], [Total Transactions] )

Fraud Rate % LY =
CALCULATE ( [Fraud Rate %], SAMEPERIODLASTYEAR ( fct_transactions[date] ) )

Fraud Rate % MoM Change =
VAR PrevMonth = CALCULATE ( [Fraud Rate %], DATEADD ( fct_transactions[date], -1, MONTH ) )
RETURN
    [Fraud Rate %] - PrevMonth
```

## Risk Scoring & Model Performance

```dax
Avg Fraud Risk Score =
AVERAGE ( fct_transactions[fraud_risk_score] )

High Risk Transactions (Score > 0.5) =
CALCULATE ( COUNTROWS ( fct_transactions ), fct_transactions[fraud_risk_score] > 0.5 )

True Positives =
CALCULATE (
    COUNTROWS ( fct_transactions ),
    fct_transactions[predicted_fraud] = TRUE,
    fct_transactions[is_fraud] = TRUE
)

False Positives =
CALCULATE (
    COUNTROWS ( fct_transactions ),
    fct_transactions[predicted_fraud] = TRUE,
    fct_transactions[is_fraud] = FALSE
)

False Negatives =
CALCULATE (
    COUNTROWS ( fct_transactions ),
    fct_transactions[predicted_fraud] = FALSE,
    fct_transactions[is_fraud] = TRUE
)

Model Precision % =
DIVIDE ( [True Positives], [True Positives] + [False Positives] )

Model Recall % =
DIVIDE ( [True Positives], [True Positives] + [False Negatives] )

Model F1 Score =
VAR P = [Model Precision %]
VAR R = [Model Recall %]
RETURN
    DIVIDE ( 2 * P * R, P + R )
```

## Behavioural Risk Indicators

```dax
Cross-Border Fraud Rate % =
CALCULATE ( [Fraud Rate %], fct_transactions[is_cross_border] = TRUE )

Domestic Fraud Rate % =
CALCULATE ( [Fraud Rate %], fct_transactions[is_cross_border] = FALSE )

Avg Transaction Velocity (24h) =
AVERAGE ( fct_transactions[velocity_24h] )

New Device Transaction Share % =
DIVIDE (
    CALCULATE ( COUNTROWS ( fct_transactions ), fct_transactions[is_new_device] = TRUE ),
    [Total Transactions]
)

Night Transaction Fraud Rate % =
CALCULATE ( [Fraud Rate %], fct_transactions[is_night_txn] = TRUE )
```

## Customer & Merchant Risk

```dax
Customers with Confirmed Fraud =
CALCULATE (
    DISTINCTCOUNT ( fct_transactions[customer_pseudo_id] ),
    fct_transactions[is_fraud] = TRUE
)

Avg Amount by MCC Category =
AVERAGE ( fct_transactions[amount] )

Merchant Fraud Rate % =
VAR MerchantTxns = CALCULATE ( COUNTROWS ( fct_transactions ), ALLEXCEPT ( dim_merchant, dim_merchant[merchant_id] ) )
VAR MerchantFraud = CALCULATE (
    COUNTROWS ( fct_transactions ),
    fct_transactions[is_fraud] = TRUE,
    ALLEXCEPT ( dim_merchant, dim_merchant[merchant_id] )
)
RETURN
    DIVIDE ( MerchantFraud, MerchantTxns )
```
