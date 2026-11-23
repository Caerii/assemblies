"""Reproducibility and paired seed coverage for synthetic structural sweep."""
import pytest
from research.multimodal.baseline_sweep import Sweep, run, BASELINES


def test_seed_grid_and_models():
    spec = Sweep(geometry=((64, 4), (128, 8)), seeds=(3, 4), contents=6)
    result = run(spec)
    assert len(result["rows"]) == 2 * 2 * len(BASELINES)
    assert len(result["summary"]) == 2 * len(BASELINES)
    assert {(r["n"], r["k"], r["dataset_seed"]) for r in result["rows"]} == {
        (64, 4, 3), (64, 4, 4), (128, 8, 3), (128, 8, 4)}
    assert result == run(spec)


def test_control_inequalities():
    x = run(Sweep(geometry=((128, 8),), seeds=(0, 1, 2, 3), contents=8))
    metrics = {r["model"]: r for r in x["summary"]}
    assert metrics["shared_shift"]["heldout_exact"]["mean"] == 1.
    assert metrics["hebbian_pairs"]["train_exact"]["mean"] == 1.
    assert metrics["exact_pair_memory"]["heldout_exact"]["mean"] == 0.
    assert metrics["identity_copy"]["heldout_exact"]["mean"] == 0.


@pytest.mark.parametrize("spec", [
    Sweep(seeds=()), Sweep(seeds=(0, 0)), Sweep(geometry=()),
    Sweep(seeds=(1729,)), Sweep(geometry=((128, 8), (128, 8))),
    Sweep(geometry=((8, 9),)),
])
def test_bad_grid_rejected(spec):
    with pytest.raises(ValueError):
        run(spec)
