"""
Supervised fraud-risk classification.

Trains and compares Logistic Regression (interpretable baseline) and
Gradient Boosting (non-linear) classifiers on the engineered behavioural
features, with a time-based train/test split (no shuffling -- the test set
is strictly later in time than training, as it would be in production) and
class-weighting to handle the ~0.4% fraud prevalence. Evaluated on
precision, recall, F1, and ROC-AUC -- accuracy is intentionally not the
headline metric given the severe class imbalance.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

PROC_DIR = "data/processed"
OUT_DIR = "outputs"

NUMERIC_FEATURES = [
    "amount", "amount_zscore", "velocity_1h", "velocity_24h",
    "geo_distance_km", "is_cross_border", "is_new_device",
    "is_high_risk_channel", "is_night_txn", "base_risk_multiplier",
    "is_round_amount", "cust_hist_count",
]
CATEGORICAL_FEATURES = ["channel", "mcc_category"]


def load_features():
    df = pd.read_csv(f"{PROC_DIR}/transactions_features.csv", parse_dates=["timestamp"])
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["geo_distance_km"] = df["geo_distance_km"].fillna(0)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_design_matrix(df):
    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["is_fraud"]
    return X, y


def time_based_split(df, X, y, test_frac=0.25):
    cutoff = df["timestamp"].quantile(1 - test_frac)
    train_mask = df["timestamp"] < cutoff
    return (X[train_mask], X[~train_mask], y[train_mask], y[~train_mask],
            df.loc[~train_mask, "timestamp"].min())


def evaluate(name, y_true, y_pred, y_score):
    report = classification_report(y_true, y_pred, target_names=["legit", "fraud"], output_dict=True)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["legit", "fraud"]))
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    return {
        "model": name,
        "precision_fraud": round(report["fraud"]["precision"], 4),
        "recall_fraud": round(report["fraud"]["recall"], 4),
        "f1_fraud": round(report["fraud"]["f1-score"], 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
    }


def plot_curves(results_curves):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (fpr, tpr, roc_auc) in results_curves["roc"].items():
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right", fontsize=8)

    for name, (prec, rec, pr_auc) in results_curves["pr"].items():
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fraud_model_curves.png", dpi=140)
    plt.close()


def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    importances.plot(kind="barh", ax=ax, color="#2a6f97")
    ax.set_title("Gradient Boosting: Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fraud_model_feature_importance.png", dpi=140)
    plt.close()


def main():
    df = load_features()
    X, y = build_design_matrix(df)
    X_train, X_test, y_train, y_test, cutoff_date = time_based_split(df, X, y)

    print(f"Train: {len(X_train):,} rows ({y_train.mean()*100:.3f}% fraud)")
    print(f"Test:  {len(X_test):,} rows ({y_test.mean()*100:.3f}% fraud), from {cutoff_date.date()} onward")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_results = []
    curves = {"roc": {}, "pr": {}}

    # --- Logistic Regression (interpretable baseline) ---
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_scaled, y_train)
    logreg_scores = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_preds = (logreg_scores >= 0.5).astype(int)
    all_results.append(evaluate("Logistic Regression", y_test, logreg_preds, logreg_scores))
    fpr, tpr, _ = roc_curve(y_test, logreg_scores)
    curves["roc"]["Logistic Regression"] = (fpr, tpr, all_results[-1]["roc_auc"])
    prec, rec, _ = precision_recall_curve(y_test, logreg_scores)
    curves["pr"]["Logistic Regression"] = (prec, rec, all_results[-1]["pr_auc"])

    # --- Gradient Boosting (non-linear, handles interactions) ---
    # sample_weight substitutes for class_weight, which GradientBoostingClassifier lacks.
    sample_weight = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
    gbm = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42)
    gbm.fit(X_train, y_train, sample_weight=sample_weight)
    gbm_scores = gbm.predict_proba(X_test)[:, 1]
    gbm_preds = (gbm_scores >= 0.5).astype(int)
    all_results.append(evaluate("Gradient Boosting", y_test, gbm_preds, gbm_scores))
    fpr, tpr, _ = roc_curve(y_test, gbm_scores)
    curves["roc"]["Gradient Boosting"] = (fpr, tpr, all_results[-1]["roc_auc"])
    prec, rec, _ = precision_recall_curve(y_test, gbm_scores)
    curves["pr"]["Gradient Boosting"] = (prec, rec, all_results[-1]["pr_auc"])

    plot_curves(curves)
    plot_feature_importance(gbm, X.columns.tolist())

    # Score every transaction in the test window for the Power BI risk-score export
    df_test = df.loc[X_test.index].copy()
    df_test["fraud_risk_score"] = gbm_scores
    df_test["predicted_fraud"] = gbm_preds
    df_test[[
        "transaction_id", "customer_id", "merchant_id", "timestamp", "amount",
        "channel", "txn_country", "home_country", "mcc_category",
        "fraud_risk_score", "predicted_fraud", "is_fraud", "fraud_type",
    ]].to_csv(f"{OUT_DIR}/scored_transactions.csv", index=False)

    with open(f"{OUT_DIR}/model_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModel comparison written to {OUT_DIR}/model_evaluation.json")
    print(f"Scored test-set transactions written to {OUT_DIR}/scored_transactions.csv")
    print(f"Curves -> {OUT_DIR}/fraud_model_curves.png, importances -> {OUT_DIR}/fraud_model_feature_importance.png")


if __name__ == "__main__":
    main()
