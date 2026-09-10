"""A training round-trip score must name its schedule and respond to a null."""
import pytest

from neural_assemblies.programs.pnas_extended import run_pnas_reciprocal


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_roundtrip_responds_to_learning_disabled_control(seed):
    trained = run_pnas_reciprocal(seed=seed, recurrent=True)
    null = run_pnas_reciprocal(seed=seed, recurrent=True, beta=0.)
    assert trained.parameters["recurrent"] is True
    assert trained.parameters["engine"] == "numpy_sparse"
    # Instrument sensitivity, not a statistical adoption bar or recall theorem.
    assert trained.reciprocal_restore_overlap > null.reciprocal_restore_overlap + .5


def test_disabled_source_recurrence_is_recorded():
    result = run_pnas_reciprocal(recurrent=False)
    assert result.parameters["recurrent"] is False
