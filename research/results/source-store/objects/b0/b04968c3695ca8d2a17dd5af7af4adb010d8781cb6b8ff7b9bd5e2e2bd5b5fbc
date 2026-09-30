"""Binding diagnostics must not silently discard misspelled area names."""

import pytest

from neural_assemblies.assembly_calculus.binding import input_drive, recall
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=11, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    return brain


def test_recall_rejects_unknown_area_instead_of_returning_none():
    with pytest.raises(KeyError, match="source area"):
        recall(_brain(), sources=["TYPO"], target_area="DST")
    with pytest.raises(KeyError, match="target area"):
        recall(_brain(), sources=["SRC"], target_area="TYPO")


def test_input_drive_rejects_unknown_areas_instead_of_returning_empty_mapping():
    with pytest.raises(KeyError, match="source area"):
        input_drive(_brain(), sources=["TYPO"], target_areas=["DST"])
    with pytest.raises(KeyError, match="target area"):
        input_drive(_brain(), sources=["SRC"], target_areas=["TYPO"])
