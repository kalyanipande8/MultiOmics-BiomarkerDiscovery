"""
make_figures.py

Generates figures from the real run outputs in output/:
  - roc_curves.png: real ROC curves for RF and GB(XGBoost/fallback) on held-out test set
  - top_biomarkers.png: bar chart of top-18 combined importance scores, colored
    by whether each biomarker was truly injected or is a false-positive recovery
"""

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")


def main():
    with open("output/metrics.json") as f:
        metrics = json.load(f)

    test_df = pd.read_csv("output/held_out_test_set.csv")
    feature_cols = [c for c in test_df.columns if c != "diagnosis"]
    y_test = test_df["diagnosis"].values
    X_test = test_df[feature_cols]

    rf = joblib.load("output/rf_model.joblib")
    gb = joblib.load("output/gb_model.joblib")

    fig, ax = plt.subplots(figsize=(6, 5))
    for model, name, auc_key in [
        (rf, "Random Forest", "random_forest_auc"),
        (gb, metrics["gradient_boosting_model"].split(" ")[0], "gradient_boosting_auc"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics[auc_key]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Held-out ROC: Multi-Omics Diagnosis Classifiers")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("output/figures/roc_curves.png", dpi=150)
    plt.close(fig)

    ranking = pd.read_csv("output/feature_importance_ranking.csv", index_col=0)
    top18 = ranking.head(18)
    true_biomarkers = set(metrics["true_biomarkers"])
    colors = ["#2ca02c" if f in true_biomarkers else "#d62728" for f in top18.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top18.index[::-1], top18["combined_importance_score"][::-1], color=colors[::-1])
    ax.set_xlabel("Combined importance score (RF + GB + permutation, avg percentile rank)")
    ax.set_title(f"Top 18 Identified Biomarkers "
                 f"({len(metrics['recovered_biomarkers'])}/18 match truly-injected signal)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Truly-injected biomarker"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Not in injected set"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig("output/figures/top_biomarkers.png", dpi=150)
    plt.close(fig)

    print("Saved output/figures/roc_curves.png and output/figures/top_biomarkers.png")


if __name__ == "__main__":
    main()
