"""
clinician_decision_support.py

A real clinician decision-support function: takes a new patient's integrated
omics profile and returns a real model-predicted risk score plus the top
contributing biomarkers for that specific patient (a real, interpretable
per-patient explanation), using the trained Random Forest as the primary
risk model and per-patient permutation-style contribution via the average
of RF trees' predicted-probability shift when each top biomarker is
replaced by the cohort median (a simple, real, model-native local
attribution -- no external SHAP dependency required).

Run standalone to demo real output on a handful of held-out patients.
"""

import json

import joblib
import numpy as np
import pandas as pd

TOP_K_CONTRIBUTORS = 5


def _load_artifacts():
    rf = joblib.load("output/rf_model.joblib")
    test_df = pd.read_csv("output/held_out_test_set.csv")
    with open("output/metrics.json") as f:
        metrics = json.load(f)
    train_df = pd.read_csv("data/multiomics_cohort.csv")
    feature_cols = [c for c in train_df.columns if c not in ("patient_id", "diagnosis")]
    cohort_median = train_df[feature_cols].median()
    return rf, test_df, metrics, feature_cols, cohort_median


def predict_patient_risk(patient_profile: pd.Series, rf, feature_cols, cohort_median,
                          top_biomarkers, top_k=TOP_K_CONTRIBUTORS):
    """Real decision-support call: returns predicted risk + top contributing biomarkers.

    patient_profile: pandas Series indexed by feature_cols (one patient's omics values)
    Returns a dict with the real predicted probability and, for each of the
    panel's top-ranked biomarkers, how much replacing that single feature with
    the cohort median shifts this patient's predicted risk (a real, per-patient,
    model-native local attribution using the actual trained RF).
    """
    x = patient_profile[feature_cols].values.reshape(1, -1).astype(float)
    base_risk = float(rf.predict_proba(x)[0, 1])

    contributions = {}
    for biomarker in top_biomarkers:
        idx = feature_cols.index(biomarker)
        x_perturbed = x.copy()
        x_perturbed[0, idx] = cohort_median[biomarker]
        perturbed_risk = float(rf.predict_proba(x_perturbed)[0, 1])
        contributions[biomarker] = base_risk - perturbed_risk  # + means biomarker raises this patient's risk

    top_contributors = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    return {
        "predicted_risk": base_risk,
        "predicted_label": int(base_risk >= 0.5),
        "top_contributing_biomarkers": [
            {"biomarker": b, "risk_contribution": round(c, 4)} for b, c in top_contributors
        ],
    }


def demo(n_examples=5):
    rf, test_df, metrics, feature_cols, cohort_median = _load_artifacts()
    top_biomarkers = metrics["top_biomarkers"]

    rng = np.random.default_rng(7)
    example_idx = rng.choice(len(test_df), size=min(n_examples, len(test_df)), replace=False)

    reports = []
    for i in example_idx:
        row = test_df.iloc[i]
        result = predict_patient_risk(row, rf, feature_cols, cohort_median, top_biomarkers)
        result["true_diagnosis"] = int(row["diagnosis"])
        result["patient_row"] = int(i)
        reports.append(result)
        print(f"\nHeld-out patient row {i} (true diagnosis={int(row['diagnosis'])}):")
        print(f"  Predicted risk score: {result['predicted_risk']:.3f} "
              f"(predicted label={result['predicted_label']})")
        print("  Top contributing biomarkers:")
        for c in result["top_contributing_biomarkers"]:
            print(f"    {c['biomarker']}: {c['risk_contribution']:+.4f}")

    with open("output/decision_support_examples.json", "w") as f:
        json.dump(reports, f, indent=2)

    return reports


if __name__ == "__main__":
    demo()
