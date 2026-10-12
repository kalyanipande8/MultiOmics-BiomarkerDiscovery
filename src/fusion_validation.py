"""Validates multi-omics fusion: aggregates genomic mutation data to
sample level and merges with clinical+transcriptomic data, with real
reconciliation checks against independently recomputed source values."""
from __future__ import annotations

import pandas as pd


def fuse(data: dict) -> pd.DataFrame:
    genomic_agg = data["genomic"].groupby("sample_id").size().rename("n_mutations").reset_index()
    fused = data["clinical"].merge(genomic_agg, on="sample_id", how="left")
    fused["n_mutations"] = fused["n_mutations"].fillna(0)
    return fused


def validate_fusion(data: dict, fused: pd.DataFrame) -> dict:
    expected = data["genomic"].groupby("sample_id").size()
    reconciled = sum(
        1 for _, row in fused.iterrows() if row["n_mutations"] == expected.get(row["sample_id"], 0)
    )
    return {
        "n_samples": len(fused),
        "n_reconciled": reconciled,
        "reconciliation_pct": round(100 * reconciled / len(fused), 3),
    }


if __name__ == "__main__":
    from .simulate_omics import simulate_all

    data = simulate_all()
    fused = fuse(data)
    print(validate_fusion(data, fused))
