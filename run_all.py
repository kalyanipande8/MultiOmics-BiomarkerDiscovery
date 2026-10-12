"""End-to-end: simulate multi-omics data, fuse and validate, run
differential expression signature discovery, and cross-validate a
classifier on the discovered signature."""
import json
import os

from src.simulate_omics import simulate_all
from src.fusion_validation import fuse, validate_fusion
from src.signature_discovery import differential_expression, signature_recovery, cross_validated_classifier

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=== Simulating multi-omics data (clinical, transcriptomic, genomic) ===")
data = simulate_all()
total_records = sum(len(df) for df in data.values())
for name, df in data.items():
    df.to_csv(f"data/{name}.csv", index=False)
    print(f"  {name}: {len(df)} records")
print(f"Total records: {total_records}")

print("\n=== Fusing and validating ===")
fused = fuse(data)
validation = validate_fusion(data, fused)
print(validation)

print("\n=== Differential expression signature discovery ===")
de = differential_expression(data["transcriptomic"], data["clinical"])
print(de)
recovery = signature_recovery(de)
print(recovery)

print("\n=== Cross-validated classifier on discovered signature ===")
clf_result = cross_validated_classifier(data["transcriptomic"], data["clinical"], recovery["discovered_genes"])
print(clf_result)

summary = {
    "total_records": total_records,
    "fusion_reconciliation_pct": validation["reconciliation_pct"],
    "n_signature_genes_discovered": recovery["n_discovered"],
    "n_true_signature_recovered": recovery["n_true_signature_recovered"],
    "n_true_signature_total": recovery["n_true_signature_total"],
    "classifier_mean_cv_accuracy": clf_result["mean_cv_accuracy"],
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Summary ===")
print(json.dumps(summary, indent=2))
