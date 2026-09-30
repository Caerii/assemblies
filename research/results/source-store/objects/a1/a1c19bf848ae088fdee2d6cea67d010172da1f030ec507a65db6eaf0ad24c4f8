"""Sequence novelty references must inhabit the recalled area's domain."""

import pytest

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import ordered_recall
from neural_assemblies.assembly_calculus.tracing.operations import ordered_recall_trace
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=71, engine="numpy_sparse")
    brain.add_stimulus("cue", 10)
    brain.add_area("A", 100, 10, 0.1, refractory_period=2)
    return brain


@pytest.mark.parametrize("recall", [ordered_recall, ordered_recall_trace])
def test_known_assemblies_reject_cross_area_snapshots(recall):
    with pytest.raises(ValueError, match="recall area"):
        recall(_brain(), "A", "cue", known_assemblies=[Assembly("OTHER", range(10))])


@pytest.mark.parametrize("recall", [ordered_recall, ordered_recall_trace])
def test_known_assemblies_reject_malformed_entries(recall):
    with pytest.raises(TypeError, match="Assembly snapshots"):
        recall(_brain(), "A", "cue", known_assemblies=["not-an-assembly"])
