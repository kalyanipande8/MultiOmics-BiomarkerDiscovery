"""
train_models.py

Trains real RandomForestClassifier and (XGBoost, or a documented
GradientBoostingClassifier fallback -- see README) models on the integrated
multi-omics feature panel, evaluates held-out AUC, and identifies the top
~18 biomarkers via real feature-importance ranking. Verifies recovery
against the truly-injected predictive features simulated upstream.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier

    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

N_TOP_BIOMARKERS = 18
SEED = 42


def load_data():
    df = pd.read_csv("data/multiomics_cohort.csv")
    with open("data/true_biomarkers.txt") as f:
        true_biomarkers = [line.strip() for line in f if line.strip()]
    feature_cols = [c for c in df.columns if c not in ("patient_id", "diagnosis")]
    X = df[feature_cols]
    y = df["diagnosis"].values
    return df, X, y, feature_cols, true_biomarkers


def train_and_evaluate():
    df, X, y, feature_cols, true_biomarkers = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    if HAVE_XGBOOST:
        gb_model_name = "XGBoost"
        gb = XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        gb_model_name = "GradientBoostingClassifier (sklearn fallback for XGBoost)"
        gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=SEED,
        )
    gb.fit(X_train, y_train)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

    # --- Real biomarker identification: average feature importance across
    # both models (RF's native impurity importance + GB's native importance),
    # cross-checked with permutation importance on the held-out test set. ---
    rf_importance = pd.Series(rf.feature_importances_, index=feature_cols)
    gb_importance = pd.Series(gb.feature_importances_, index=feature_cols)

    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=15, random_state=SEED, scoring="roc_auc", n_jobs=-1
    )
    perm_importance = pd.Series(perm.importances_mean, index=feature_cols)

    def rank01(s):
        r = s.rank(pct=True)
        return r

    combined_score = (
        rank01(rf_importance) + rank01(gb_importance) + rank01(perm_importance)
    ) / 3.0
    ranked = combined_score.sort_values(ascending=False)
    top_biomarkers = list(ranked.head(N_TOP_BIOMARKERS).index)

    recovered = sorted(set(top_biomarkers) & set(true_biomarkers))
    recovery_rate = len(recovered) / len(true_biomarkers)

    # --- Sanity check: a genuinely predictive feature should outrank a pure
    # noise feature in combined importance. ---
    noise_features = [c for c in feature_cols if c not in true_biomarkers]
    example_true = true_biomarkers[0]
    example_noise = noise_features[0]
    sanity_true_beats_noise = combined_score[example_true] > combined_score[example_noise]

    results = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_cols),
        "n_true_biomarkers_injected": len(true_biomarkers),
        "random_forest_auc": float(rf_auc),
        "gradient_boosting_model": gb_model_name,
        "gradient_boosting_auc": float(gb_auc),
        "have_xgboost": HAVE_XGBOOST,
        "top_biomarkers": top_biomarkers,
        "true_biomarkers": true_biomarkers,
        "recovered_biomarkers": recovered,
        "recovery_rate": recovery_rate,
        "sanity_true_beats_noise_example": {
            "true_feature": example_true,
            "true_feature_score": float(combined_score[example_true]),
            "noise_feature": example_noise,
            "noise_feature_score": float(combined_score[example_noise]),
            "passes": bool(sanity_true_beats_noise),
        },
    }

    with open("output/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    ranked.to_csv("output/feature_importance_ranking.csv", header=["combined_importance_score"])

    # Persist trained models + split for the decision-support demo stage.
    import joblib

    joblib.dump(rf, "output/rf_model.joblib")
    joblib.dump(gb, "output/gb_model.joblib")
    X_test.assign(diagnosis=y_test).to_csv("output/held_out_test_set.csv", index=False)

    print(f"Train/test: {len(X_train)}/{len(X_test)} patients, {len(feature_cols)} features")
    print(f"Random Forest held-out AUC: {rf_auc:.4f}")
    print(f"{gb_model_name} held-out AUC: {gb_auc:.4f}")
    print(f"Top {N_TOP_BIOMARKERS} biomarkers (combined importance): {top_biomarkers}")
    print(f"Recovery vs {len(true_biomarkers)} truly-injected biomarkers: "
          f"{len(recovered)}/{len(true_biomarkers)} = {recovery_rate:.1%}")
    print(f"Sanity check (true feature '{example_true}' beats noise feature "
          f"'{example_noise}'): {sanity_true_beats_noise}")

    return results


if __name__ == "__main__":
    train_and_evaluate()
