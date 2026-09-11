"""Weight normalization must be a live mutation or an explicit rejection."""

import pytest
import numpy as np

from neural_assemblies import Brain


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_explicit"])
def test_engines_without_mutable_normalization_reject(engine):
    brain = Brain(engine=engine, norm_init=False)
    brain.add_area("A", 20, 2, 0.1)
    with pytest.raises(NotImplementedError, match="weight normalization"):
        brain.normalize_weights("A")


def test_sparse_normalization_remains_a_live_operation():
    brain = Brain(engine="numpy_sparse", norm_init=False)
    brain.add_area("A", 20, 2, 0.1)
    brain.materialize_area("A")
    weights = brain.connectomes["A"]["A"].weights
    indices = np.arange(20)
    weights[np.ix_(indices, indices)] = np.full((20, 20), 2.0, dtype=np.float32)
    brain.normalize_weights("A")
    weights = brain.connectomes["A"]["A"].weights
    sums = np.asarray(weights.sum(axis=0)).ravel()
    assert np.allclose(sums[sums > 0], 1.0)
