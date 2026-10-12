"""Simulate 2,500 samples' genomic (mutation calls), transcriptomic (gene
expression), and clinical data -- 12,500+ total records across 3
modalities, with a disease status driven by a known subset of genuinely
disease-relevant expression signature genes."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SAMPLES = 2500
SEED = 131
EXPRESSION_GENES = [f"GENE_{i}" for i in range(20)]
# ground truth: only these 3 genes are genuinely part of the disease signature
SIGNATURE_GENES = ["GENE_2", "GENE_9", "GENE_15"]
MUTATION_GENES = [f"MUT_{i}" for i in range(15)]


def simulate_clinical(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sample_id": [f"S-{i:05d}" for i in range(n)],
        "age": rng.normal(56, 14, n).clip(18, 90), "sex": rng.choice(["M", "F"], n),
    })


def simulate_transcriptomic(clinical: pd.DataFrame, seed: int = SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(clinical)
    expr = {g: rng.normal(0, 1, n) for g in EXPRESSION_GENES}

    logit = sum(2.0 * expr[g] for g in SIGNATURE_GENES) + rng.normal(0, 1.0, n)
    prob = 1 / (1 + np.exp(-logit))
    disease = rng.binomial(1, prob)

    rows = []
    for i, sid in enumerate(clinical["sample_id"]):
        for gene in EXPRESSION_GENES:
            rows.append({"sample_id": sid, "gene": gene, "expression_z": round(float(expr[gene][i]), 3)})
    long_df = pd.DataFrame(rows)
    disease_df = pd.DataFrame({"sample_id": clinical["sample_id"], "disease_status": disease})
    return long_df, disease_df


def simulate_genomic(clinical: pd.DataFrame, mutations_per_sample: float = 1.8, seed: int = SEED + 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sid in clinical["sample_id"]:
        n_mut = rng.poisson(mutations_per_sample)
        for _ in range(n_mut):
            rows.append({"sample_id": sid, "mutated_gene": rng.choice(MUTATION_GENES)})
    return pd.DataFrame(rows)


def simulate_all(seed: int = SEED) -> dict:
    clinical = simulate_clinical(seed=seed)
    transcriptomic, disease = simulate_transcriptomic(clinical, seed=seed + 1)
    genomic = simulate_genomic(clinical, seed=seed + 2)
    clinical = clinical.merge(disease, on="sample_id")
    return {"clinical": clinical, "transcriptomic": transcriptomic, "genomic": genomic}


if __name__ == "__main__":
    data = simulate_all()
    total = sum(len(df) for df in data.values())
    for name, df in data.items():
        print(name, df.shape)
    print("Total records:", total)
