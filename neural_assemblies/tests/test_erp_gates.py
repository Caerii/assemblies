"""Unit tests for ERP violation gates (erp.gates)."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (
    ErpBaseline,
    ErpReadiness,
    ErpThresholds,
    classify_erp_violation,
    default_erp_thresholds,
)


class TestErpGates:
    def test_unready_parser_never_wobbly(self):
        readiness = ErpReadiness(
            n400_ready=False, p600_ready=False, sentences_seen=0,
        )
        v = classify_erp_violation(
            1.0, 1.0, readiness=readiness, baseline=ErpBaseline(),
        )
        assert not v.wobbly

    def test_structural_wobble_high_p600_only(self):
        readiness = ErpReadiness(n400_ready=True, p600_ready=True, sentences_seen=20)
        baseline = ErpBaseline(n400_median=0.9, p600_median=0.2)
        th = ErpThresholds(n400_excess_margin=0.1, p600_excess_margin=0.15)
        v = classify_erp_violation(
            0.92, 0.5,
            readiness=readiness,
            baseline=baseline,
            phrase_stability=0.8,
            thresholds=th,
        )
        assert v.wobbly
        assert v.failure_signature == "structural_wobble"

    def test_grammatical_null_not_wobbly(self):
        readiness = ErpReadiness(n400_ready=True, p600_ready=True, sentences_seen=20)
        baseline = ErpBaseline(n400_median=0.09, p600_median=0.12)
        v = classify_erp_violation(
            0.09, 0.12,
            readiness=readiness,
            baseline=baseline,
            phrase_stability=0.9,
            thresholds=default_erp_thresholds(),
        )
        assert not v.wobbly
