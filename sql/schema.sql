-- PostgreSQL DDL for the European banking transaction & fraud analytics warehouse.
-- Mirrors the Python pipeline's cleaned tables (data/processed/) plus the
-- GDPR-pseudonymized analytics layer (outputs/*_pseudonymized.csv) as a
-- separate schema, so raw and analytics access can be permissioned separately.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ============================================================
-- RAW SCHEMA -- restricted access, contains direct identifiers
-- ============================================================

CREATE TABLE raw.customers (
    customer_id     VARCHAR(12) PRIMARY KEY,
    full_name       TEXT NOT NULL,
    email           TEXT NOT NULL,
    country         CHAR(2) NOT NULL,
    age             SMALLINT CHECK (age BETWEEN 16 AND 110),
    segment         VARCHAR(20) NOT NULL,
    signup_date     DATE NOT NULL,
    kyc_verified    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE raw.accounts (
    account_id      VARCHAR(12) PRIMARY KEY,
    customer_id     VARCHAR(12) NOT NULL REFERENCES raw.customers(customer_id),
    account_type    VARCHAR(20) NOT NULL CHECK (account_type IN ('checking','savings','credit_card')),
    opened_date     DATE NOT NULL,
    home_country    CHAR(2) NOT NULL,
    credit_limit    NUMERIC(12,2) DEFAULT 0
);

CREATE TABLE raw.merchants (
    merchant_id             VARCHAR(12) PRIMARY KEY,
    merchant_name           TEXT NOT NULL,
    mcc                     CHAR(4) NOT NULL,
    mcc_category            VARCHAR(60) NOT NULL,
    country                 CHAR(2) NOT NULL,
    base_risk_multiplier    NUMERIC(4,2) NOT NULL,
    onboarded_date          DATE NOT NULL
);

CREATE TABLE raw.transactions (
    transaction_id  UUID PRIMARY KEY,
    account_id      VARCHAR(12) NOT NULL REFERENCES raw.accounts(account_id),
    customer_id     VARCHAR(12) NOT NULL REFERENCES raw.customers(customer_id),
    merchant_id     VARCHAR(12) REFERENCES raw.merchants(merchant_id),
    txn_timestamp   TIMESTAMP NOT NULL,
    amount          NUMERIC(12,2) NOT NULL CHECK (amount >= 0),
    currency        CHAR(3) NOT NULL,
    channel         VARCHAR(20) NOT NULL CHECK (
        channel IN ('card_present','card_not_present','online','atm_withdrawal','wire_transfer')
    ),
    txn_country     CHAR(2) NOT NULL,
    home_country    CHAR(2) NOT NULL,
    device_id       VARCHAR(40),
    is_fraud        BOOLEAN NOT NULL DEFAULT FALSE,
    fraud_type      VARCHAR(30)
);

CREATE INDEX idx_transactions_customer_ts ON raw.transactions (customer_id, txn_timestamp);
CREATE INDEX idx_transactions_account_ts  ON raw.transactions (account_id, txn_timestamp);
CREATE INDEX idx_transactions_merchant    ON raw.transactions (merchant_id);
CREATE INDEX idx_transactions_fraud       ON raw.transactions (is_fraud) WHERE is_fraud;

-- ============================================================
-- ANALYTICS SCHEMA -- pseudonymized, minimized; safe for BI/reporting access
-- ============================================================

CREATE TABLE analytics.dim_customer (
    customer_pseudo_id     CHAR(16) PRIMARY KEY,
    country                CHAR(2) NOT NULL,
    age_band               VARCHAR(10) NOT NULL,
    segment                VARCHAR(20) NOT NULL,
    signup_year_month      CHAR(7) NOT NULL,
    kyc_verified            BOOLEAN NOT NULL
);

CREATE TABLE analytics.dim_merchant (
    merchant_id             VARCHAR(12) PRIMARY KEY,
    merchant_name           TEXT NOT NULL,
    mcc                     CHAR(4) NOT NULL,
    mcc_category            VARCHAR(60) NOT NULL,
    country                 CHAR(2) NOT NULL,
    base_risk_multiplier    NUMERIC(4,2) NOT NULL
);

CREATE TABLE analytics.fct_transactions (
    transaction_id      UUID PRIMARY KEY,
    customer_pseudo_id  CHAR(16) NOT NULL REFERENCES analytics.dim_customer(customer_pseudo_id),
    account_pseudo_id   CHAR(16) NOT NULL,
    merchant_id         VARCHAR(12) REFERENCES analytics.dim_merchant(merchant_id),
    txn_timestamp       TIMESTAMP NOT NULL,
    amount              NUMERIC(12,2) NOT NULL,
    currency            CHAR(3) NOT NULL,
    channel             VARCHAR(20) NOT NULL,
    txn_country         CHAR(2) NOT NULL,
    home_country        CHAR(2) NOT NULL,
    is_cross_border     BOOLEAN NOT NULL,
    is_new_device       BOOLEAN NOT NULL,
    is_night_txn        BOOLEAN NOT NULL,
    velocity_1h         SMALLINT NOT NULL,
    velocity_24h        SMALLINT NOT NULL,
    amount_zscore       NUMERIC(6,3),
    fraud_risk_score    NUMERIC(6,5),
    predicted_fraud     BOOLEAN,
    is_fraud            BOOLEAN NOT NULL
);

CREATE INDEX idx_fct_txn_customer ON analytics.fct_transactions (customer_pseudo_id, txn_timestamp);
CREATE INDEX idx_fct_txn_risk     ON analytics.fct_transactions (fraud_risk_score DESC);

-- ============================================================
-- ROLE-BASED ACCESS CONTROL (GDPR data-minimization boundary)
-- ============================================================
-- See docs/gdpr_governance.md for the full access-control matrix.

-- CREATE ROLE fraud_analyst_readonly;
-- GRANT USAGE ON SCHEMA analytics TO fraud_analyst_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO fraud_analyst_readonly;
-- -- fraud_analyst_readonly has NO grants on schema `raw` -- cannot see names/emails/device IDs.

-- CREATE ROLE compliance_officer;
-- GRANT USAGE ON SCHEMA raw, analytics TO compliance_officer;
-- GRANT SELECT ON ALL TABLES IN SCHEMA raw, analytics TO compliance_officer;
-- -- compliance_officer can re-identify a flagged transaction for a SAR/investigation,
-- -- with every such access expected to be logged at the application layer (see governance doc).
