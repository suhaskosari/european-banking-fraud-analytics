"""
Statistical hypothesis testing on fraud vs. legitimate transaction behaviour.

Each test states H0/H1 explicitly, picks a test appropriate to the data
shape (non-normal amounts -> Mann-Whitney U, categorical rates -> chi-square
test of independence), and reports the effect size alongside the p-value so
statistical significance isn't confused with practical significance.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

PROC_DIR = "data/processed"
OUT_DIR = "outputs"

ALPHA = 0.05
results = []


def record(test_name, h0, h1, statistic, p_value, effect_size=None, effect_label=None, extra=None):
    sig = p_value < ALPHA
    results.append({
        "test": test_name, "h0": h0, "h1": h1,
        "statistic": round(float(statistic), 4), "p_value": float(p_value),
        "significant_at_0.05": bool(sig),
        "effect_size": round(float(effect_size), 4) if effect_size is not None else None,
        "effect_label": effect_label,
        "extra": extra or {},
    })
    verdict = "REJECT H0" if sig else "FAIL TO REJECT H0"
    print(f"[{test_name}] stat={statistic:.4f}  p={p_value:.6g}  -> {verdict}"
          + (f"  ({effect_label}={effect_size:.4f})" if effect_size is not None else ""))


def mann_whitney_amount(df):
    """H0: fraud and legitimate transactions are drawn from the same amount
    distribution. H1: they differ. Amounts are heavily right-skewed (lognormal-
    like), so Mann-Whitney U (rank-based, distribution-free) is appropriate
    over a t-test, which assumes normality."""
    fraud = df.loc[df["is_fraud"], "amount"]
    legit = df.loc[~df["is_fraud"], "amount"]
    u_stat, p = stats.mannwhitneyu(fraud, legit, alternative="two-sided")
    # Rank-biserial correlation as effect size
    n1, n2 = len(fraud), len(legit)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    record(
        "Mann-Whitney U: transaction amount, fraud vs. legitimate",
        "Fraudulent and legitimate transactions have the same amount distribution.",
        "Fraudulent transactions have a different amount distribution.",
        u_stat, p, rank_biserial, "rank-biserial r",
        extra={"median_fraud": float(fraud.median()), "median_legit": float(legit.median()),
               "n_fraud": n1, "n_legit": n2},
    )


def chisq_country(df):
    """H0: fraud rate is independent of transaction country.
    H1: fraud rate depends on transaction country."""
    tbl = pd.crosstab(df["txn_country"], df["is_fraud"])
    chi2, p, dof, expected = stats.chi2_contingency(tbl)
    n = tbl.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tbl.shape) - 1)))
    rates = df.groupby("txn_country")["is_fraud"].mean().sort_values(ascending=False)
    record(
        "Chi-square test of independence: fraud vs. transaction country",
        "Fraud incidence is independent of transaction country.",
        "Fraud incidence depends on transaction country.",
        chi2, p, cramers_v, "Cramer's V",
        extra={"fraud_rate_by_country": {k: round(float(v), 5) for k, v in rates.items()}},
    )


def chisq_channel(df):
    """H0: fraud rate is independent of transaction channel.
    H1: fraud rate depends on channel."""
    tbl = pd.crosstab(df["channel"], df["is_fraud"])
    chi2, p, dof, expected = stats.chi2_contingency(tbl)
    n = tbl.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tbl.shape) - 1)))
    rates = df.groupby("channel")["is_fraud"].mean().sort_values(ascending=False)
    record(
        "Chi-square test of independence: fraud vs. channel",
        "Fraud incidence is independent of transaction channel.",
        "Fraud incidence depends on transaction channel.",
        chi2, p, cramers_v, "Cramer's V",
        extra={"fraud_rate_by_channel": {k: round(float(v), 5) for k, v in rates.items()}},
    )


def proportions_ztest_cross_border(df):
    """H0: cross-border and domestic transactions have equal fraud rates.
    H1: cross-border transactions have a higher fraud rate (one-sided)."""
    cb = df[df["is_cross_border"] == 1]["is_fraud"]
    dom = df[df["is_cross_border"] == 0]["is_fraud"]
    p1, p2 = cb.mean(), dom.mean()
    n1, n2 = len(cb), len(dom)
    p_pool = (cb.sum() + dom.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 1 - stats.norm.cdf(z)  # one-sided: cross-border rate > domestic rate
    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    record(
        "One-sided two-proportion z-test: cross-border vs. domestic fraud rate",
        "Cross-border and domestic transactions have equal fraud rates.",
        "Cross-border transactions have a higher fraud rate.",
        z, p_value, cohens_h, "Cohen's h",
        extra={"fraud_rate_cross_border": round(float(p1), 5), "fraud_rate_domestic": round(float(p2), 5)},
    )


def mann_whitney_velocity(df):
    """H0: fraud and legitimate transactions have the same 1-hour transaction
    velocity. H1: fraud transactions show higher velocity (card-testing /
    burst signature)."""
    fraud = df.loc[df["is_fraud"], "velocity_1h"]
    legit = df.loc[~df["is_fraud"], "velocity_1h"]
    u_stat, p = stats.mannwhitneyu(fraud, legit, alternative="greater")
    n1, n2 = len(fraud), len(legit)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    record(
        "Mann-Whitney U (one-sided): 1h transaction velocity, fraud vs. legitimate",
        "Fraud and legitimate transactions have equal 1-hour velocity.",
        "Fraud transactions have higher 1-hour velocity.",
        u_stat, p, rank_biserial, "rank-biserial r",
        extra={"mean_velocity_fraud": float(fraud.mean()), "mean_velocity_legit": float(legit.mean())},
    )


def main():
    df = pd.read_csv(f"{PROC_DIR}/transactions_features.csv", parse_dates=["timestamp"])
    df["is_fraud"] = df["is_fraud"].astype(bool)

    print(f"Running hypothesis tests on {len(df):,} transactions "
          f"({df['is_fraud'].sum():,} fraud, {df['is_fraud'].mean()*100:.3f}%)\n")

    mann_whitney_amount(df)
    chisq_country(df)
    chisq_channel(df)
    proportions_ztest_cross_border(df)
    mann_whitney_velocity(df)

    with open(f"{OUT_DIR}/hypothesis_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUT_DIR}/hypothesis_test_results.json")


if __name__ == "__main__":
    main()
