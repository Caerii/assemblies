"""Convergence helpers reject schedules that cannot express convergence."""

import pytest

from neural_assemblies.assembly_calculus.ops import learn_assembly
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=23, engine="numpy_sparse")
    brain.add_stimulus("s", 10)
    brain.add_area("A", 100, 10, 0.1)
    return brain


@pytest.mark.parametrize("kwargs", [
    {"max_epochs": 0}, {"project_rounds": 0},
    {"stability_window": 1}, {"stability_window": 0},
    {"max_epochs": 1.5}, {"project_rounds": True},
])
def test_learning_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        learn_assembly(_brain(), "s", "A", **kwargs)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_learning_rejects_invalid_convergence(value):
    with pytest.raises(ValueError, match="convergence"):
        learn_assembly(_brain(), "s", "A", convergence=value)
