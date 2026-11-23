"""R1i fail-closed synaptic evidence gate. Does not execute the native Brain."""
import pytest

from research.multimodal import native_teacher_controls as N
from research.multimodal.zero_shot_gate import Config


def _fake_result():
    return {
        "dataset": {"train": [(0, "SUBJ"), (1, "SUBJ")]},
        "conditions": [
            {
                "condition": arm,
                "weight_audit_available": True,
                "clamp_diagnostics": [
                    {"active_edge_changed": 0 if arm == "no_plasticity" else 4}
                    for _ in range(2)
                ],
            }
            for arm in ("correct_teacher", "shuffled_teacher", "no_plasticity")
        ],
        "limitations": [],
    }


def test_accepts_complete_audit(monkeypatch):
    monkeypatch.setattr(N, "run_conditions", lambda cfg, factory: _fake_result())
    result = N.run(Config(contents=4))
    assert result["status"] == "native_r1i_teacher_controls_unvalidated"


def test_missing_fingerprint_fails(monkeypatch):
    result = _fake_result()
    result["conditions"][0]["weight_audit_available"] = False
    monkeypatch.setattr(N, "run_conditions", lambda cfg, factory: result)
    with pytest.raises(AssertionError, match="unavailable"):
        N.run(Config(contents=4))


def test_missing_step_diagnostics_fails(monkeypatch):
    result = _fake_result()
    result["conditions"][1]["clamp_diagnostics"].pop()
    monkeypatch.setattr(N, "run_conditions", lambda cfg, factory: result)
    with pytest.raises(AssertionError, match="missing"):
        N.run(Config(contents=4))


def test_beta_zero_write_fails(monkeypatch):
    result = _fake_result()
    result["conditions"][-1]["clamp_diagnostics"][0]["active_edge_changed"] = 1
    monkeypatch.setattr(N, "run_conditions", lambda cfg, factory: result)
    with pytest.raises(AssertionError, match="beta=0"):
        N.run(Config(contents=4))
