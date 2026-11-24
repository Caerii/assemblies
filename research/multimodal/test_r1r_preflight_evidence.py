"""R1r smoke contract with an explicitly fake backend, not native evidence."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from research.multimodal import native_execution_preflight as P


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeModel:
    evidence_mode = "correct"

    def __init__(self, cfg, beta, presentations, teacher_gain):
        self.beta = beta
        self.target = None
        self.state = "initial"
        self.clamp_diagnostics = []

    def state_fingerprint(self):
        return self.state

    def train(self, item, role, cue, target):
        self.target = target.copy()
        if self.beta > 0:
            self.state = "changed"
        count = (7 if self.beta else 0)
        proof = {
            "oracle": "native_numpy_explicit_float32_multiplicative",
            "rounds": 3,
            "changed_edges_full_graph": count,
            "before_sha256": "before",
            "after_sha256": "after" if self.beta else "before",
        }
        if self.evidence_mode == "missing":
            proof = None
        elif self.evidence_mode == "incorrect":
            proof["changed_edges_full_graph"] = 0 if self.beta else 8
        self.clamp_diagnostics.append({
            "teacher_target_overlap": 1.0,
            "active_edge_changed": count,
            "active_edge_delta_sum": float(count),
            "exact_synapse_law": proof,
        })

    def predict(self, item, role, cue):
        return self.target.copy()


def install(monkeypatch, mode="correct"):
    class Model(FakeModel):
        evidence_mode = mode

    monkeypatch.setitem(sys.modules, "research.multimodal.zero_shot_gate",
                        SimpleNamespace(Config=Config))
    monkeypatch.setitem(sys.modules, "research.multimodal.native_teacher_drive",
                        SimpleNamespace(NativeTeacherDriven=Model))


def test_preflight_requires_certified_law(monkeypatch):
    install(monkeypatch, "correct")
    result = P.numeric_smoke()
    assert result["plastic"]["exact_synapse_law"]["rounds"] == 3
    assert result["nonplastic"]["exact_synapse_law"]["changed_edges_full_graph"] == 0
    assert result["plastic"]["full_graph_immutable_on_read"]


def test_preflight_rejects_missing_law(monkeypatch):
    install(monkeypatch, "missing")
    with pytest.raises(P.NativePreflightError, match="evidence missing"):
        P.numeric_smoke()


def test_preflight_rejects_incorrect_law(monkeypatch):
    install(monkeypatch, "incorrect")
    with pytest.raises(P.NativePreflightError, match="did not change"):
        P.numeric_smoke()
