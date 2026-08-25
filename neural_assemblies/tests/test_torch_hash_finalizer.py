"""A third torch-engine defect of the familiar shape: the mirror missed a fix.

``kernels/implicit.py`` documents at length why every hash in this project ends
in murmur3's ``fmix32``, and records that the finalizer was added to all four
sites there and both in ``cuda_engine.py`` "together, on purpose", because
"fixing a subset would leave the engine and the kernels disagreeing about which
synapses exist, which is worse than a uniform bias".

``torch_engine/_hash.py`` was that subset. It carried the docstring "Same hash
function as cuda_engine._hash_bernoulli_2d" and contained no finalizer, so the
torch engine and the numpy/rust engine disagreed about 9.5% of cells while
every density check stayed green.

The assertions here are "torch does what numpy does" -- pinned against the
numpy engine's own ``_seeding.hash_bernoulli_2d`` rather than against a
transcription of the hash, because a mirror that agrees with my re-derivation
and disagrees with the engine is the failure being tested for.

Skipped without CUDA. The default interpreter on this machine carries a
CPU-only torch build, so these only run under the project's uv venv.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from neural_assemblies.core.numpy_engine import _seeding as np_seed   # noqa: E402
from neural_assemblies.core.torch_engine import _hash as t_hash       # noqa: E402

N = 1024
P = 0.05
SEED = 0x5EED1234


def _torch_block(n=N, p=P, seed=SEED):
    return t_hash.hash_bernoulli_2d(
        0, n, 0, n, seed, p, device='cuda').float().cpu().numpy()


def _dispersion(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return counts.var() / counts.mean()


# -- the property that was actually violated -------------------------------

def test_torch_hash_matches_the_numpy_engine_cell_for_cell():
    """The two engines must agree about WHICH SYNAPSES EXIST."""
    got = _torch_block().astype(bool)
    ref = np_seed.hash_bernoulli_2d(0, N, 0, N, SEED, P,
                                    finalize=True).astype(bool)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"torch and numpy disagree about {(got != ref).mean():.4%} of cells")


def test_torch_hash_is_not_the_unfinalized_hash():
    """Fails loudly if someone 'simplifies' the finalizer away again."""
    got = _torch_block().astype(bool)
    raw = np_seed.hash_bernoulli_2d(0, N, 0, N, SEED, P,
                                    finalize=False).astype(bool)
    agree = float((got == raw).mean())
    assert agree < 0.99, (
        "torch_engine/_hash.py is producing the UNFINALIZED hash again "
        f"(agrees with raw at {agree:.6f}); see _hash._fmix32")


# -- the mechanism the bias broke ------------------------------------------

def test_in_degree_is_binomial_not_nearly_constant():
    """``norm_init`` divides by in-degree, so constant in-degree degenerates it.

    The raw hash gave column dispersion ~0.015 where Binomial says ~1-p. The
    density was right the whole time, which is why this survived.
    """
    got = _torch_block()
    assert _dispersion(got.sum(axis=0)) > 0.5
    assert got.mean() == pytest.approx(P, abs=0.005)


def test_adjacent_cells_are_not_anticorrelated():
    """The raw hash correlated neighbouring synapses at about -0.05."""
    got = _torch_block()
    r = float(np.corrcoef(got[:-1].ravel(), got[1:].ravel())[0, 1])
    assert abs(r) < 0.02


# -- the seed derivation feeding it ----------------------------------------

def test_pair_seed_derivation_agrees_with_the_numpy_engine():
    """A matching hash on a mismatched seed would still build a different graph."""
    for src, tgt in (("A", "B"), ("LEX", "SYN"), ("x", "x")):
        assert (t_hash.fnv1a_pair_seed(7, src, tgt)
                == np_seed.fnv1a_pair_seed(7, src, tgt))


def test_stim_counts_agree_with_the_numpy_engine():
    """The stimulus hash is a separate site and was separately unfinalized."""
    got = t_hash.hash_stim_counts(64, 0, 512, SEED, P,
                                  device='cuda').float().cpu().numpy()
    ref = np.asarray(np_seed.hash_stim_counts(64, 0, 512, SEED, P),
                     dtype=np.float64)
    assert np.array_equal(got.astype(np.int64), ref.astype(np.int64))
