"""
run_all.py

Runs the full pipeline end-to-end:
  1. Simulate integrated multi-omics cohort data -> data/
  2. Train RF + XGBoost/GB models, real AUC + biomarker identification -> output/
  3. Run clinician decision-support demo on held-out patients -> output/
  4. Generate figures -> output/figures/

Run with:  python run_all.py
"""

import simulate_multiomics_data
import train_models
import clinician_decision_support
import make_figures


def main():
    print("--- Simulating integrated multi-omics cohort ---")
    simulate_multiomics_data.main()

    print("\n--- Training Random Forest + XGBoost/GB models ---")
    train_models.train_and_evaluate()

    print("\n--- Clinician decision-support demo on held-out patients ---")
    clinician_decision_support.demo()

    print("\n--- Generating figures ---")
    make_figures.main()


if __name__ == "__main__":
    main()
