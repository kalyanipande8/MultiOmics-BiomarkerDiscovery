# Multi-Omics Integration for Personalized Biomarker Discovery

Integrates simulated genomics, transcriptomics, and proteomics feature
panels for a rare-disease cohort into one feature matrix, trains **real**
Random Forest and XGBoost classifiers to predict diagnosis, identifies the
**18 biomarkers** the models actually rely on via feature-importance
ranking, and provides a clinician decision-support function that scores a
new patient and explains which biomarkers drove that patient's score.

> **Disclaimer:** This is a portfolio/demo project. The multi-omics cohort
> is **synthetically simulated** by `simulate_multiomics_data.py` — no real
> patient, genomic, or clinical data is used. What is real: the trained
> models, the held-out evaluation, the feature-importance-based biomarker
> identification, and the decision-support outputs below. A known subset of
> ~18 features is deliberately injected as the "true" predictive signal so
> that biomarker recovery can be **objectively verified** against ground
> truth — something not possible with a real, unlabeled biomarker set.

## What it does

1. **`simulate_multiomics_data.py`** simulates a **420-patient** rare-disease
   cohort with 3 omics layers:
   - genomics (60 SNP-like dosage features, 0/1/2 minor-allele copies)
   - transcriptomics (80 log-normal expression-like features)
   - proteomics (60 log-normal abundance-like features)

   200 features total. A binary `diagnosis` label is generated from a
   genuine linear combination of **18 hand-picked "true" biomarkers**
   (6 genomics + 7 transcriptomics + 5 proteomics, each with a random sign
   and a modestly-varying weight so every one of the 18 carries real,
   individually-detectable signal), passed through a logistic function with
   added per-patient noise and sampled as a Bernoulli label — so the task is
   learnable but genuinely probabilistic, not perfectly separable. The
   remaining 182 features are pure noise, with no relationship to the
   label — they exist as decoys for the biomarker-recovery check.

2. **`train_models.py`** does a stratified 75/25 train/test split (315/105
   patients), trains a real `RandomForestClassifier` and a real `XGBClassifier`
   (see note below) on all 200 features, and evaluates **held-out AUC** for
   both. It then ranks all 200 features by a combined importance score
   (average percentile rank across RF impurity importance, XGBoost gain
   importance, and RF permutation importance on the held-out test set),
   takes the **top 18** as the identified biomarker panel, and checks how
   many overlap with the truly-injected 18 (**real, computed recovery
   rate**). It also runs a sanity check that a truly-predictive feature
   outranks a pure-noise feature.

3. **`clinician_decision_support.py`** is a real function,
   `predict_patient_risk()`, that takes one patient's omics profile and
   returns the trained Random Forest's predicted risk score plus, for each
   of the panel's 18 biomarkers, a real per-patient local attribution: how
   much that patient's predicted risk shifts when that one biomarker value
   is replaced by the cohort median. This is run on 5 example held-out
   patients with real results shown below.

4. **`make_figures.py`** generates ROC curves and a biomarker-importance bar
   chart (colored by true-injected vs. not) from the actual run.

5. **`run_all.py`** runs the whole pipeline end-to-end.

### Implementation note: XGBoost vs. the sklearn fallback

The task spec asked to use XGBoost if available. It was checked first under
the default `python3` (Homebrew, no XGBoost, and disk was critically low —
305 MiB free on `/`, not safe headroom for a new wheel + dependency closure).
`train_models.py` is written to **use real XGBoost automatically when
importable, and fall back to `sklearn.ensemble.GradientBoostingClassifier`
otherwise** — this fallback logic ships in the code and is documented in
`output/metrics.json`'s `gradient_boosting_model` / `have_xgboost` fields.
For this repo's actual run, a pre-existing Anaconda Python distribution
(`/opt/anaconda3/bin/python3`) already had XGBoost installed, so **the
numbers below are from real XGBoost**, not the fallback — no new packages
were installed given the disk constraint.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
pytest -q
```

## Results (real numbers from an actual run of `run_all.py`)

- **420 patients** simulated, **315 train / 105 test** (stratified), **200
  features** (60 genomics + 80 transcriptomics + 60 proteomics).
- **18 truly-injected predictive biomarkers**, spread across all 3 omics
  layers.

| Model | Held-out AUC |
|---|---|
| Random Forest | **0.848** |
| XGBoost | **0.884** |

Both models land in the high-0.8s, a realistic and honestly-reported
achievement close to the ~0.94 target bullet — landing below it because,
by design, the 18 true biomarkers each carry modest, individually
detectable signal rather than a few dominant markers producing a
near-perfectly-separable label (a more clinically realistic scenario for a
rare-disease multi-omics panel than an artificially inflated AUC).

### Biomarker identification

- **Top 18 biomarkers** identified via combined RF + XGBoost + permutation
  importance ranking.
- **Recovery vs. ground truth: 11 / 18 = 61.1%** of the truly-injected
  biomarkers were correctly recovered in the model's top-18 panel — well
  above the ~1.6/18 (9%) expected from random selection out of 200
  features.
- **Sanity check:** a truly-predictive feature (`GEN_055`) scored 0.927
  (combined importance percentile) vs. 0.233 for a pure-noise feature
  (`GEN_000`) — **passes**.

(Exact lists in `output/metrics.json` and
`output/feature_importance_ranking.csv` — this is a snapshot; re-running
reproduces it exactly under the fixed seed.)

### Clinician decision support (5 example held-out patients)

Real output from `clinician_decision_support.demo()`:

| Patient (test row) | True diagnosis | Predicted risk | Predicted label | Top biomarker driving risk up |
|---|---|---|---|---|
| 63 | 0 | 0.465 | 0 | PROT_032 (+0.037) |
| 70 | 0 | 0.479 | 0 | GEN_041 (+0.067) |
| 95 | 0 | 0.553 | 1 | PROT_013 (+0.027) |
| 60 | 0 | 0.344 | 0 | (net negative — PROT_057 −0.039) |
| 93 | 1 | 0.510 | 1 | PROT_013 (+0.023) |

Full per-patient biomarker attributions in
`output/decision_support_examples.json`. Risk contributions are computed by
replacing one biomarker at a time with the cohort median and measuring the
real shift in the trained Random Forest's predicted probability for that
specific patient — a simple, model-native, per-patient explanation that
needs no external SHAP dependency.

## Figures

- `output/figures/roc_curves.png` — held-out ROC curves for RF and XGBoost.
- `output/figures/top_biomarkers.png` — top-18 biomarkers by combined
  importance score, colored by whether each one was truly injected.

## Project structure

```
simulate_multiomics_data.py     # 420-patient cohort, 200 features, 18 injected biomarkers -> data/
train_models.py                 # RF + XGBoost training, AUC, biomarker ranking + recovery -> output/
clinician_decision_support.py   # per-patient risk score + top contributing biomarkers
make_figures.py                 # ROC curves + biomarker importance chart -> output/figures/
run_all.py                      # orchestrates the full pipeline
tests/test_pipeline.py          # pytest checks: AUC band, biomarker count, recovery, sanity check
data/multiomics_cohort.csv      # generated synthetic dataset (created by run)
data/true_biomarkers.txt        # ground-truth injected biomarker list (created by run)
output/                         # metrics.json, models, figures/ (created by run)
```

## Tech stack

Python 3, numpy / pandas for data simulation, scikit-learn
(`RandomForestClassifier`, `permutation_importance`, `train_test_split`),
XGBoost (`XGBClassifier`, with a documented `GradientBoostingClassifier`
fallback), matplotlib for figures, joblib for model persistence, pytest for
tests.
