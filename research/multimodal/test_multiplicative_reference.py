"""Reproducible contract tests for the standalone multiplicative fiber.

Run from repository root:
    uv run python -m pytest research/multimodal/test_multiplicative_reference.py -q
"""
import numpy as np
import pytest

from research.multimodal.multiplicative_reference import (
    MultiplicativeFiber, SparseMultiplicativeFiber,
)


@pytest.mark.parametrize("n,k", [(24, 3), (64, 8), (129, 11)])
def test_exact_sparse_dense_parity(n, k):
    rng = np.random.default_rng(729 + n)
    initial = (rng.random((n, n)) < .18).astype(np.float32)
    dense = MultiplicativeFiber(initial, k, beta=.125, w_max=4.)
    sparse = SparseMultiplicativeFiber(initial, k, beta=.125, w_max=4.)
    for i in range(100):
        src = rng.choice(n, k, replace=False).astype(np.uint32)
        dst = rng.choice(n, k, replace=False).astype(np.uint32)
        rounds = 2 if i % 11 == 0 else 1
        dense.cofire(src, dst, rounds)
        sparse.cofire(src, dst, rounds)
        np.testing.assert_array_equal(dense.weights, sparse.dense())
        np.testing.assert_array_equal(dense.drive(src), sparse.drive(src))
        np.testing.assert_array_equal(dense.predict(src), sparse.predict(src))


def test_zero_synapses_remain_zero():
    matrix = np.zeros((12, 12), dtype=np.float32)
    matrix[0, 3] = 1
    m = MultiplicativeFiber(matrix, k=2, beta=.5)
    m.cofire([0, 1], [3, 4], rounds=5)
    assert np.count_nonzero(m.weights) == 1
    assert float(m.weights[0, 3]) > 1


def test_read_only_observation_does_not_mutate():
    m = MultiplicativeFiber(np.eye(10, dtype=np.float32), 2, read_only=True)
    before = m.weights.copy()
    m.predict([0, 1])
    np.testing.assert_array_equal(before, m.weights)
    with pytest.raises(PermissionError):
        m.cofire([0, 1], [2, 3])
    np.testing.assert_array_equal(before, m.weights)


def test_clipping_and_deterministic_ties():
    m = MultiplicativeFiber(np.ones((10, 10), dtype=np.float32), 2, beta=.5, w_max=2.)
    assert list(m.predict([0, 1])) == [0, 1]
    m.cofire([0, 1], [2, 3], rounds=10)
    assert m.weights[0, 2] == 2.
    assert m.weights[0, 9] == 1.


@pytest.mark.parametrize("source", [[1, 1], [-1, 2], [1], [1, 99]])
def test_invalid_source_rejected(source):
    m = MultiplicativeFiber(np.eye(10, dtype=np.float32), k=2)
    with pytest.raises(ValueError):
        m.predict(source)
