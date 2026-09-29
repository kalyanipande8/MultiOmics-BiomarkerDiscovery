"""
simulate_multiomics_data.py

Simulates an integrated multi-omics dataset for a rare-disease cohort:
  - genomics panel      (SNP-like dosage features, 0/1/2)
  - transcriptomics panel (log-normal expression-like features)
  - proteomics panel    (log-normal abundance-like features)

A binary diagnosis/phenotype label is generated from a genuine, hand-picked
subset of ~18 "truly predictive" features (spread across all three omics
layers, mimicking real integrative biology), combined with realistic noise,
so the task is learnable but not trivial. The remaining features are pure
noise (no relationship to the label) -- these act as decoys for the
biomarker-recovery check in the modeling stage.

Fixed random seed -> fully reproducible.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_PATIENTS = 420
N_GENOMICS = 60
N_TRANSCRIPTOMICS = 80
N_PROTEOMICS = 60
N_TRUE_BIOMARKERS = 18


def simulate(seed: int = RNG_SEED, steepness: float = 4.0, noise_std: float = 0.02,
             weight_lo: float = 0.7, weight_hi: float = 1.3):
    rng = np.random.default_rng(seed)
    n = N_PATIENTS

    # --- Genomics: SNP-like dosage (0, 1, 2 copies of minor allele) ---
    maf = rng.uniform(0.05, 0.45, size=N_GENOMICS)
    genomics = np.array([rng.binomial(2, p, size=n) for p in maf]).T
    genomics_cols = [f"GEN_{i:03d}" for i in range(N_GENOMICS)]

    # --- Transcriptomics: log-normal expression-like features ---
    transcript_mu = rng.uniform(2.0, 8.0, size=N_TRANSCRIPTOMICS)
    transcript_sigma = rng.uniform(0.3, 0.9, size=N_TRANSCRIPTOMICS)
    transcriptomics = np.array(
        [rng.lognormal(mu, sigma, size=n) for mu, sigma in zip(transcript_mu, transcript_sigma)]
    ).T
    transcript_cols = [f"TX_{i:03d}" for i in range(N_TRANSCRIPTOMICS)]

    # --- Proteomics: log-normal abundance-like features ---
    protein_mu = rng.uniform(1.0, 6.0, size=N_PROTEOMICS)
    protein_sigma = rng.uniform(0.2, 0.7, size=N_PROTEOMICS)
    proteomics = np.array(
        [rng.lognormal(mu, sigma, size=n) for mu, sigma in zip(protein_mu, protein_sigma)]
    ).T
    protein_cols = [f"PROT_{i:03d}" for i in range(N_PROTEOMICS)]

    all_cols = genomics_cols + transcript_cols + protein_cols
    X_raw = pd.DataFrame(
        np.hstack([genomics, transcriptomics, proteomics]), columns=all_cols
    )

    # --- Select ~18 truly predictive biomarkers spread across all 3 layers ---
    n_gen_true = 6
    n_tx_true = 7
    n_prot_true = N_TRUE_BIOMARKERS - n_gen_true - n_tx_true  # 5

    true_genomics = list(rng.choice(genomics_cols, size=n_gen_true, replace=False))
    true_transcript = list(rng.choice(transcript_cols, size=n_tx_true, replace=False))
    true_protein = list(rng.choice(protein_cols, size=n_prot_true, replace=False))
    true_biomarkers = true_genomics + true_transcript + true_protein
    assert len(true_biomarkers) == N_TRUE_BIOMARKERS

    # z-score the true-biomarker columns to build a clean linear signal, with
    # random signed weights (some protective, some risk-increasing) drawn
    # from a fairly narrow range -- each of the 18 truly-injected biomarkers
    # carries genuine, individually-detectable signal (rather than a few
    # markers dominating and the rest being negligible), which is what makes
    # the downstream biomarker-recovery check meaningful.
    signs = rng.choice([-1, 1], size=N_TRUE_BIOMARKERS, p=[0.35, 0.65])
    weights = rng.uniform(weight_lo, weight_hi, size=N_TRUE_BIOMARKERS) * signs

    z = (X_raw[true_biomarkers] - X_raw[true_biomarkers].mean()) / X_raw[true_biomarkers].std()
    logit = z.values @ weights

    # center and scale the signal, add irreducible per-patient noise, then
    # pass through a logistic function with tuned steepness so the held-out
    # AUC lands close to the ~0.94 target while still being a genuinely
    # probabilistic (not perfectly separable) label.
    logit = (logit - logit.mean()) / logit.std()
    noise = rng.normal(0, noise_std, size=n)
    combined = logit + noise
    prob = 1 / (1 + np.exp(-steepness * (combined - np.median(combined))))
    label = rng.binomial(1, prob)

    df = X_raw.copy()
    df.insert(0, "patient_id", [f"P{i:04d}" for i in range(n)])
    df["diagnosis"] = label

    meta = {
        "true_biomarkers": true_biomarkers,
        "feature_cols": all_cols,
        "n_patients": n,
    }
    return df, meta


def main():
    df, meta = simulate()
    df.to_csv("data/multiomics_cohort.csv", index=False)
    with open("data/true_biomarkers.txt", "w") as f:
        f.write("\n".join(meta["true_biomarkers"]))
    print(f"Simulated {meta['n_patients']} patients x {len(meta['feature_cols'])} features "
          f"({N_GENOMICS} genomics + {N_TRANSCRIPTOMICS} transcriptomics + {N_PROTEOMICS} proteomics)")
    print(f"Injected {len(meta['true_biomarkers'])} truly predictive biomarkers: "
          f"{meta['true_biomarkers']}")
    print(f"Diagnosis-positive rate: {df['diagnosis'].mean():.3f}")


if __name__ == "__main__":
    main()
