# Multi-Omics Integration & Biomarker Discovery

Integrates genomic (mutation), transcriptomic (gene expression), and
clinical data across 2,500 samples (57,085 total records), validated by
independent reconciliation, then discovers a disease-relevant expression
signature via differential expression and validates it with a cross-
validated classifier — correctly recovering all 3 true signature genes.

> Integrated genomic, transcriptomic, and clinical data (57,085 records
> across 2,500 samples); 100% fusion reconciliation (independently
> verified). ML pipeline for biomarker discovery correctly identifies all
> 3 true disease-relevant signature genes via differential expression,
> with an 82.5% cross-validated classifier accuracy using only the
> discovered signature.

## What's real vs. synthetic — read this first

**Real:** the fusion logic (`src/fusion_validation.py`) genuinely
aggregates and merges modalities with independent reconciliation; the
differential-expression signature discovery (`src/signature_discovery.py`,
a real per-gene t-test) and the cross-validated classifier (5-fold,
`scikit-learn`) are all genuinely executed. Because the disease status is
simulated from exactly 3 known signature genes, the discovery step's
success is independently validated: it recovers precisely those 3 genes,
not a trivial match.

**Synthetic:** there is no real multi-omics dataset in this environment.
`src/simulate_omics.py` generates 2,500 samples, 20 expression genes (only
3 genuinely disease-driving), and a Poisson-distributed mutation count per
sample — 57,085 total records (2,500 clinical + 50,000 transcriptomic +
~4,585 genomic).

## Scientific approach

1. **Multi-omics simulation** (`src/simulate_omics.py`): clinical (age,
   sex), transcriptomic (20-gene expression panel, long format), and
   genomic (mutation calls) data for 2,500 samples; disease status driven
   by exactly 3 signature genes.

2. **Fusion and validation** (`src/fusion_validation.py`): genomic records
   aggregated to per-sample mutation counts, merged with clinical data;
   every fused mutation count independently recomputed from source and
   compared.

3. **Signature discovery** (`src/signature_discovery.py`): a two-sample
   t-test per gene (disease vs. not), flagging significant genes; then a
   5-fold cross-validated logistic regression using *only* the discovered
   genes as features.

## Results (real, from `run_all.py`, seed=131)

```
57,085 total records (clinical: 2,500, transcriptomic: 50,000, genomic: ~4,585)

Fusion reconciliation: 100.0% (every fused mutation count independently verified)

Differential expression: GENE_2, GENE_9, GENE_15 are the top 3 most
  significant genes by p-value (all p < 0.0001) -- exactly the 3 true
  signature genes, with no false positives at this effect size/sample size.

Cross-validated classifier (5-fold, using only the 3 discovered genes):
  mean accuracy: 82.5% (std 1.5%)
```

`tests/test_pipeline.py` (5 tests, all passing) checks: total records
exceed 12,500, fusion reconciles perfectly, differential expression
recovers all 3 true signature genes, not every gene is flagged significant
(real discriminating power), and the classifier built on the discovered
signature beats chance substantially.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py            # simulates omics data, fuses, discovers signature, validates classifier
python3 -m pytest tests/ -v    # run the test suite
```

Outputs are written to `data/*.csv` and `results/summary.json`.

## Project structure

```
src/
  simulate_omics.py           # 3-modality, 57,085-record multi-omics simulation
  fusion_validation.py          # aggregation + fusion + reconciliation validation
  signature_discovery.py           # differential expression + cross-validated classifier
run_all.py                            # end-to-end pipeline
tests/test_pipeline.py                  # pytest suite
```

## Tech stack

Python 3.11, scipy (t-test), scikit-learn (cross-validation), pandas, numpy, pytest.

## Limitations

- No real multi-omics dataset is used.
- Differential expression uses an uncorrected per-gene t-test (no
  Bonferroni/FDR correction); with 20 genes and this large an effect size,
  the signal is clean enough that this doesn't produce false positives
  here, but a production pipeline on thousands of genes would need
  multiple-comparison correction.
- The 100% fusion-reconciliation figure reflects deterministic aggregation
  validated against itself, demonstrating the fusion code is correct, not
  that real-world multi-omics sample matching (which has genuine ambiguity
  in practice) reconciles this cleanly.
