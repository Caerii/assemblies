"""Tests for shared ERP probe layer (erp_probes)."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (
    DEFAULT_CALIBRATION_FRAMES,
    ErpProbeResult,
    critical_position_for_frame,
    run_incremental_erp_probes,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    default_holdout_set,
)


class TestErpProbes:
    def test_critical_position_holdout_subject_frame(self):
        known = ["the", "bird", "sees", "the", "cat"]
        holdout = default_holdout_set()
        pos = critical_position_for_frame("novel_noun", known, holdout_words=holdout)
        assert pos == 1
        assert known[pos] == "bird"

    def test_critical_position_object_frame(self):
        known = ["the", "dog", "chases", "bird"]
        holdout = default_holdout_set()
        pos = critical_position_for_frame("novel_noun", known, holdout_words=holdout)
        assert pos == 3
        assert known[pos] == "bird"

    def test_critical_position_grammatical_defaults_to_last(self):
        known = ["the", "dog", "chases", "cat"]
        pos = critical_position_for_frame("grammatical", known)
        assert pos == 3

    def test_calibration_frames_cover_three_conditions(self):
        labels = {lbl for lbl, _d, _w in DEFAULT_CALIBRATION_FRAMES}
        assert labels == {"grammatical", "category_violation", "novel_noun"}

    def test_run_incremental_erp_probes_returns_probe_results(self):
        parser = EmergentParser(n=3000, k=30, seed=3, fast_training=True)
        for w in ("the", "small", "dog", "runs"):
            parser.register_word(w)
        result, probes = run_incremental_erp_probes(
            parser, ["the", "small", "dog", "runs"],
        )
        assert len(probes) == 4
        assert all(isinstance(p, ErpProbeResult) for p in probes)
        assert "roles" in result
        assert result["wobbly_probes"] is probes

        assert result["erp_protocol"]["observation_version"] == "existing-context-v1"
