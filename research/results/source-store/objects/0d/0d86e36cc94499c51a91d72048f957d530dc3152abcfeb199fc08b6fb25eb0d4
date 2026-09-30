"""Explicit-pattern learning validates its source and schedule before mutation."""

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.ops import learn_assembly_from_pattern
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=37, engine="numpy_sparse")
    brain.add_area("SRC", 20, 5, 0.1)
    brain.add_area("DST", 50, 5, 0.1)
    return brain


@pytest.mark.parametrize("pattern", [np.zeros(20), np.zeros((20, 1)), np.ones(19)])
def test_pattern_learning_rejects_empty_or_wrong_shape(pattern):
    with pytest.raises(ValueError, match="pattern"):
        learn_assembly_from_pattern(_brain(), "SRC", pattern, "DST")


def test_pattern_learning_rejects_unknown_area_before_mutation():
    with pytest.raises(KeyError, match="source area"):
        learn_assembly_from_pattern(_brain(), "TYPO", np.ones(20), "DST")


@pytest.mark.parametrize("kwargs", [
    {"max_epochs": 0}, {"project_rounds": 0}, {"stability_window": 1},
    {"tau": 1.1}, {"recurrent": 1},
])
def test_pattern_learning_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        learn_assembly_from_pattern(_brain(), "SRC", np.ones(20), "DST", **kwargs)
