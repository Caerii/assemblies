"""Overlap must not infer an index space from a mixed object/raw call."""

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.core.index_spaces import NeuronIds


def test_mixed_assembly_and_raw_array_is_rejected():
    assembly = Assembly("A", NeuronIds(np.array([1, 2], dtype=np.uint32)))
    with pytest.raises(TypeError, match="same-space arrays"):
        overlap(assembly, np.array([1, 2], dtype=np.uint32))
    with pytest.raises(TypeError, match="same-space arrays"):
        overlap(np.array([1, 2], dtype=np.uint32), assembly)


def test_assembly_overlap_and_raw_pair_remain_valid():
    left = Assembly("A", NeuronIds(np.array([1, 2], dtype=np.uint32)))
    right = Assembly("B", NeuronIds(np.array([2, 3], dtype=np.uint32)))
    assert overlap(left, right) == 0.5
    assert overlap(np.array([1, 2], dtype=np.uint32),
                   np.array([2, 3], dtype=np.uint32)) == 0.5


@pytest.mark.parametrize("values", [[1, 1], np.array([1.0, 2.0])])
def test_overlap_rejects_malformed_raw_winners(values):
    with pytest.raises(ValueError):
        overlap(values, np.array([1, 2], dtype=np.uint32))


def test_assembly_rejects_duplicate_neuron_ids():
    with pytest.raises(ValueError, match="duplicates"):
        Assembly("A", [1, 1])
