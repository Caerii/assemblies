"""A standalone exact-law oracle tested against manually generated writes."""
from types import SimpleNamespace

import numpy as np
import pytest

from research.multimodal.exact_synapse_law import (
    snapshot_area_fibers, verify_exact_hebbian_write, graph_digest,
)


class Fake:
    def __init__(self, n=10):
        self.engine = SimpleNamespace(_area_conns={})
        for src in ("LEX", "SUBJ", "OBJ"):
            self.engine._area_conns[src] = {}
            for dst in ("LEX", "SUBJ", "OBJ"):
                self.engine._area_conns[src][dst] = SimpleNamespace(
                    weights=np.ones((n, n), dtype=np.float32))

    def engine_for(self, area):
        return self.engine


def snap(b):
    return snapshot_area_fibers(b, ("LEX", "SUBJ", "OBJ"))


def test_exact_two_round_float32_growth():
    b = Fake()
    before = snap(b)
    cue = [0, 3]
    for dst in ([2, 4], [2, 5]):
        b.engine._area_conns["LEX"]["SUBJ"].weights[np.ix_(cue, dst)] *= 1.125
    after = snap(b)
    proof = verify_exact_hebbian_write(
        before, after, source="LEX", target="SUBJ", cue=cue,
        selected_by_round=[[2, 4], [2, 5]], beta=.125, ceiling=20)
    assert proof["rounds"] == 2
    assert proof["changed_edges_full_graph"] == 6
    assert proof["before_sha256"] != proof["after_sha256"]


def test_beta_zero_leaves_all_matrix_bytes_identical():
    b = Fake()
    before = snap(b)
    after = snap(b)
    proof = verify_exact_hebbian_write(
        before, after, source="LEX", target="SUBJ", cue=[1],
        selected_by_round=[[3], [4]], beta=0, ceiling=20)
    assert proof["changed_edges_full_graph"] == 0
    assert graph_digest(before) == graph_digest(after)


def test_clipping_matches_each_native_round():
    b = Fake()
    w = b.engine._area_conns["LEX"]["OBJ"].weights
    w[2, 3] = np.float32(18)
    before = snap(b)
    for _ in range(3):
        w[np.ix_([2], [3])] *= 1.5
        x = w[np.ix_([2], [3])]
        np.clip(x, 0, 20, out=x)
        w[np.ix_([2], [3])] = x
    proof = verify_exact_hebbian_write(
        before, snap(b), source="LEX", target="OBJ", cue=[2],
        selected_by_round=[[3]]*3, beta=.5, ceiling=20)
    assert proof["changed_edges_full_graph"] == 1
    assert w[2, 3] == 20


@pytest.mark.parametrize("fiber,src,dst", [
    (("SUBJ", "LEX"), 1, 2), (("SUBJ", "SUBJ"), 3, 3),
    (("LEX", "OBJ"), 4, 5), (("LEX", "SUBJ"), 4, 6),
])
def test_wrong_path_or_offtarget_edge_fails(fiber, src, dst):
    b = Fake()
    before = snap(b)
    b.engine._area_conns[fiber[0]][fiber[1]].weights[src, dst] += .25
    with pytest.raises(AssertionError, match="unexpected Hebbian change"):
        verify_exact_hebbian_write(
            before, snap(b), source="LEX", target="SUBJ", cue=[1],
            selected_by_round=[[2]], beta=.1, ceiling=20)


def test_missing_reverse_fiber_fails_before_learning():
    b = Fake()
    del b.engine._area_conns["OBJ"]["LEX"]
    with pytest.raises(RuntimeError, match="missing explicit fiber"):
        snap(b)


def test_different_area_engines_rejected():
    b = Fake()
    b.engine_for = lambda a: b.engine if a == "LEX" else object()
    with pytest.raises(RuntimeError, match="mixed engine ownership"):
        snap(b)


def test_winner_order_and_dtype_preserved_without_mutating_inputs():
    b = Fake()
    before = snap(b)
    cue = np.array([3, 0], dtype=np.uint32)
    win = np.array([4, 2], dtype=np.uint32)
    c0, w0 = cue.copy(), win.copy()
    b.engine._area_conns["LEX"]["SUBJ"].weights[np.ix_(cue, win)] *= 1.1
    verify_exact_hebbian_write(before, snap(b), source="LEX", target="SUBJ",
                               cue=cue, selected_by_round=[win], beta=.1, ceiling=20)
    np.testing.assert_array_equal(cue, c0)
    np.testing.assert_array_equal(win, w0)


@pytest.mark.parametrize("cue,winners", [
    ([0, 0], [[1, 2]]), ([-1, 0], [[1, 2]]), ([1, 10], [[1, 2]]),
    ([1, 2], [[3, 3]]), ([1, 2], []),
])
def test_invalid_indices_or_rounds_fail(cue, winners):
    b = Fake()
    with pytest.raises(ValueError):
        verify_exact_hebbian_write(snap(b), snap(b), source="LEX", target="SUBJ",
                                   cue=cue, selected_by_round=winners, beta=.1, ceiling=20)
