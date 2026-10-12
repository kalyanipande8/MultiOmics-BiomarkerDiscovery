"""Disease-relevant expression signature discovery: differential
expression t-test per gene (disease vs. not), then a cross-validated
classifier using only the discovered significant genes -- validated
against the simulation's known true signature genes."""
from __future__ import annotations

import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .simulate_omics import EXPRESSION_GENES, SIGNATURE_GENES


def differential_expression(transcriptomic: pd.DataFrame, clinical: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    wide = transcriptomic.pivot(index="sample_id", columns="gene", values="expression_z")
    merged = wide.merge(clinical[["sample_id", "disease_status"]], on="sample_id")

    rows = []
    diseased = merged[merged["disease_status"] == 1]
    healthy = merged[merged["disease_status"] == 0]
    for gene in EXPRESSION_GENES:
        t_stat, p_value = stats.ttest_ind(diseased[gene], healthy[gene])
        rows.append({"gene": gene, "t_statistic": round(float(t_stat), 4),
                      "p_value": round(float(p_value), 6), "significant": p_value < alpha})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def signature_recovery(de_results: pd.DataFrame) -> dict:
    discovered = set(de_results.loc[de_results["significant"], "gene"])
    true_sig = set(SIGNATURE_GENES)
    return {
        "n_discovered": len(discovered), "discovered_genes": sorted(discovered),
        "n_true_signature_recovered": len(discovered & true_sig),
        "n_true_signature_total": len(true_sig),
    }


def cross_validated_classifier(transcriptomic: pd.DataFrame, clinical: pd.DataFrame,
                                signature_genes: list[str], cv: int = 5) -> dict:
    wide = transcriptomic.pivot(index="sample_id", columns="gene", values="expression_z")
    merged = wide.merge(clinical[["sample_id", "disease_status"]], on="sample_id")

    X = merged[signature_genes].values
    y = merged["disease_status"].values
    scores = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=cv, scoring="accuracy")
    return {"mean_cv_accuracy": round(float(scores.mean()), 4), "std_cv_accuracy": round(float(scores.std()), 4)}


if __name__ == "__main__":
    from .simulate_omics import simulate_all

    data = simulate_all()
    de = differential_expression(data["transcriptomic"], data["clinical"])
    print(de)
    recovery = signature_recovery(de)
    print(recovery)
    clf = cross_validated_classifier(data["transcriptomic"], data["clinical"], recovery["discovered_genes"])
    print(clf)
