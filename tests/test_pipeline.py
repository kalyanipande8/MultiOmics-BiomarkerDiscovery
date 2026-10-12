import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.simulate_omics import simulate_all, SIGNATURE_GENES
from src.fusion_validation import fuse, validate_fusion
from src.signature_discovery import differential_expression, signature_recovery, cross_validated_classifier


@pytest.fixture(scope="module")
def data():
    return simulate_all()


def test_total_records_exceed_12500(data):
    assert sum(len(df) for df in data.values()) >= 12500


def test_fusion_reconciles_perfectly(data):
    fused = fuse(data)
    result = validate_fusion(data, fused)
    assert result["reconciliation_pct"] == 100.0


def test_differential_expression_recovers_all_true_signature_genes(data):
    de = differential_expression(data["transcriptomic"], data["clinical"])
    recovery = signature_recovery(de)
    assert recovery["n_true_signature_recovered"] == len(SIGNATURE_GENES)


def test_differential_expression_does_not_flag_every_gene(data):
    """Confirms real discriminating power -- not all 20 genes should come
    back significant."""
    de = differential_expression(data["transcriptomic"], data["clinical"])
    assert de["significant"].sum() < len(de)


def test_classifier_on_discovered_signature_beats_chance(data):
    de = differential_expression(data["transcriptomic"], data["clinical"])
    recovery = signature_recovery(de)
    result = cross_validated_classifier(data["transcriptomic"], data["clinical"], recovery["discovered_genes"])
    assert result["mean_cv_accuracy"] > 0.6
