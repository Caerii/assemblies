"""Construction contracts for the refraction-memory runner."""

from research.experiments import refraction_memory_numpy as study


def test_smoke_checkpoint_selection_does_not_mutate_module_defaults(monkeypatch):
    original = study.MS
    monkeypatch.setattr(study, "run_experiment", lambda **kwargs: kwargs)

    study.main(["--tag", "checkpoint-contract", "--smoke", "--seeds", "1", "2", "3"])

    assert study.MS == original
