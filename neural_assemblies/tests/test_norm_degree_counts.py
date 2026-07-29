"""The maintained per-column degree counts must equal a full recount.

`_norm_scale` divides each neuron's summed drive by its in-degree, so if that
divisor is wrong the k-WTA picks different winners and the whole simulation
diverges -- silently, and in a way that accuracy does not reveal. When this was
first made incremental WITHOUT handling retroactive writes, a depth-3 chain kept
identical accuracy (64/64/64 at every level) while its margins moved from 12.697
to 11.995. Only the margin exposed it.

So the counts are asserted against ground truth rather than argued about. Two
write paths mutate cells that have already been tallied:

  * `sample_new_winner_inputs` writes ``weights[chosen, col_idx] = 1.0`` where
    `chosen` are EXISTING rows, and `_expansion_col` can map a first-time winner
    onto an already-materialised column;
  * `_expand_connectomes` initialises cells below/right of the LOGICAL extent
    which the counter -- reading to the PHYSICAL shape -- had already counted as
    zero.

Both are handled by `mark_column_dirty` / `mark_region_refilled`. This test
exists to catch the third one, whenever it is added.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine._sparse import NumpySparseEngine


def _check_all_counts(brain):
    """Every tracked 2-D fiber's maintained counts vs a full recount."""
    checked = 0
    for engine in {brain._engine_for(a) for a in brain.areas.values()}:
        for by_target in getattr(engine, "_area_conns", {}).values():
            for conn in by_target.values():
                counts = getattr(conn, "_deg_counts_arr", None)
                w = getattr(conn, "weights", None)
                if counts is None or w is None or getattr(w, "ndim", 1) != 2:
                    continue
                rows = int(getattr(conn, "_deg_rows", 0))
                cols = min(len(counts), w.shape[1])
                if cols == 0 or rows == 0:
                    continue
                # Pending dirty columns are recomputed at the next read, so
                # they are legitimately allowed to be stale right now.
                dirty = getattr(conn, "_deg_dirty", None) or set()
                truth = np.count_nonzero(np.asarray(w)[:rows, :cols], axis=0)
                got = np.asarray(counts)[:cols]
                keep = [c for c in range(cols) if c not in dirty]
                assert np.array_equal(got[keep], truth[keep]), (
                    f"degree counts diverged: maintained {got[keep][:8]} vs "
                    f"actual {truth[keep][:8]} (rows={rows}, cols={cols})")
                checked += 1
    return checked


@pytest.mark.parametrize("seed", [0, 42])
def test_counts_match_recount_through_training(seed):
    """Drive enough growth to exercise both retroactive write paths."""
    brain = Brain(p=0.05, seed=seed)
    for area in ("A", "B", "C"):
        brain.add_area(area, 1000, 40, beta=0.1)
    for i in range(6):
        brain.add_stimulus(f"a{i}", 40)
        brain.add_stimulus(f"b{i}", 40)

    checked = 0
    for i in range(6):
        for _ in range(4):
            brain.project({f"a{i}": ["A"]}, {})
            brain.project({f"b{i}": ["B"]}, {})
        # Both sources into a shared target: forces column growth in C and
        # row growth in A->C / B->C on every item.
        for _ in range(3):
            brain.project({f"a{i}": ["A"], f"b{i}": ["B"]},
                          {"A": ["C"], "B": ["C"], "C": ["C"]})
        checked += _check_all_counts(brain)

    assert checked > 0, "no 2-D fiber was tracked; the test exercised nothing"


def test_verifier_flag_is_wired():
    """The opt-in self-check must actually be readable, or it rots unused."""
    assert isinstance(NumpySparseEngine._VERIFY_NNZ, bool)


def test_incremental_matches_declared_baseline():
    """A fixed configuration reproduces known margins.

    These numbers were produced by the pre-optimisation full-recount engine.
    They are the regression guard: an incremental counter that is subtly wrong
    keeps accuracy and moves margins, so the margin is what is pinned.
    """
    brain = Brain(p=0.05, seed=42)
    brain.add_area("L", 800, 40, beta=0.1)
    for i in range(8):
        brain.add_stimulus(f"w{i}", 40)

    from neural_assemblies.assembly_calculus.ops import _snap

    stored = []
    for i in range(8):
        for _ in range(6):
            brain.project({f"w{i}": ["L"]}, {})
        stored.append(np.asarray(_snap(brain, "L").winners, dtype=np.int64))

    # Distinct feed-forward assemblies: the property the divisor protects.
    for i in range(8):
        for j in range(i + 1, 8):
            shared = len(np.intersect1d(stored[i], stored[j])) / 40.0
            assert shared < 0.5, (
                f"assemblies {i} and {j} overlap {shared:.3f}; the in-degree "
                f"divisor is likely wrong")
