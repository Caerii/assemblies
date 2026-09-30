"""The wired prune must produce BIT-IDENTICAL brains, or it is not an
optimisation -- it is a silent change to the science.

`test_kwta_prune.py` proves the algorithm exact against brute force. This
proves the ENGINE agrees with itself: same seeds, same protocol, prune on and
off, compared on the winners AND on the connectome, not on a summary. A
summary can agree while the brains diverge -- that is how a wrong fast path
survives review.

The guards get the same treatment. A guard that is merely documented is a
guard that will be wrong after the next refactor, so each one is asserted to
DECLINE, by checking the engine's own hit counter rather than by trusting the
output to look the same (it would look the same either way, which is the
problem).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain

N, K, BETA, P = 1500, 40, 0.10, 0.50
AREA = "A"


def _run(prune, seed, T=20, M=3, norm_init=False, scaling=False,
         noise=0.0):
    random.seed(seed)
    np.random.seed(seed)
    b = Brain(p=P, seed=seed, engine="numpy_sparse", recurrent_projection=True,
              norm_init=norm_init, synaptic_scaling=scaling)
    b.add_area(AREA, N, K, BETA)
    eng = b._engine_for(b.areas[AREA])
    eng.kwta_prune = prune
    if noise:
        # ON THE ENGINE'S AREA, not the Brain's. They are different objects,
        # and setting the Brain's would have left the guard untested while the
        # test still passed -- a guard test that cannot fail is not a test.
        eng._areas[AREA].input_noise_std = noise
    stims = []
    for i in range(M):
        s = f"s{i}"
        b.add_stimulus(s, K)
        stims.append(s)
    for s in stims:
        b.inhibit_areas([AREA])
        for _ in range(T):
            b.project({s: [AREA]}, {AREA: [AREA]}, )
    conn = eng._area_conns[AREA][AREA]
    rows = int(eng.materialized_count(AREA) or 0)
    w = np.asarray(conn.weights)[:rows, :rows].copy()
    return {
        "winners": np.asarray(b.areas[AREA].winners).copy(),
        "rows": rows,
        "weights": w,
        "hits": getattr(eng, "_prune_hits", 0),
        "misses": getattr(eng, "_prune_misses", 0),
    }


def _assert_identical(a, b, why):
    """The contract: same neurons recruited, same winner SET, same connectome.

    Winner ORDER is deliberately not asserted -- see the module docstring of
    `_kwta_prune`. `argpartition`/`argsort` are unstable, so the position of
    an EXACTLY TIED winner depends on values elsewhere in the array, including
    the slots the prune left alone. Measured: two columns at 738.114563 each,
    returned in opposite order, same set, same connectome. Asserting order
    would be asserting a property the unpruned engine does not itself have.
    """
    assert a["rows"] == b["rows"], f"{why}: materialized {a['rows']} vs {b['rows']}"
    assert sorted(a["winners"].tolist()) == sorted(b["winners"].tolist()), (
        f"{why}: winner SET differs")
    assert a["weights"].shape == b["weights"].shape, f"{why}: shape"
    assert np.array_equal(a["weights"], b["weights"]), (
        f"{why}: connectome differs at "
        f"{int(np.argmax(np.abs(a['weights'] - b['weights'])))}")


class TestBitIdentical:
    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_pruned_brain_is_bit_identical(self, seed):
        off, on = _run(False, seed), _run(True, seed)
        _assert_identical(off, on, "prune on vs off")

    def test_the_prune_actually_fired(self):
        """Bit-identity is worthless if the fast path never ran -- that is the
        silent-no-op shape, and it would make every assertion above vacuous."""
        on = _run(True, 42)
        assert on["hits"] > 0, (
            f"prune never fired (hits={on['hits']}, misses={on['misses']}); "
            f"the identity tests prove nothing")

    def test_identical_across_depths_including_before_it_can_apply(self):
        """T=4 is below `p*(1+beta)^T > 1` so the prune must DECLINE; T=20 is
        above it so it must fire. Both must give the same brain as prune off."""
        for T, want_fire in ((4, False), (20, True)):
            off, on = _run(False, 42, T=T), _run(True, 42, T=T)
            _assert_identical(off, on, f"T={T}")
            assert (on["hits"] > 0) is want_fire, (
                f"T={T}: hits={on['hits']} misses={on['misses']}")


class TestGuardsDecline:
    """Each guard must DECLINE, checked on the counter. Checking the output
    would pass either way, which is exactly what makes a broken guard silent."""

    def test_norm_init_declines(self):
        on = _run(True, 42, norm_init=True)
        assert on["hits"] == 0, on
        _assert_identical(_run(False, 42, norm_init=True), on, "norm_init")

    def test_synaptic_scaling_declines(self):
        on = _run(True, 42, scaling=True)
        assert on["hits"] == 0, on
        _assert_identical(_run(False, 42, scaling=True), on, "scaling")

    def test_input_noise_declines(self):
        """Additive noise can lift ANY column, including one whose bound had
        already lost."""
        on = _run(True, 42, noise=0.5)
        assert on["hits"] == 0, on

    def test_record_activation_declines(self):
        """The pre-kWTA snapshot reads the FULL drive vector, so a pruned
        vector would hand the caller zeros and look like a measurement.

        Asserted against the UNPRUNED snapshot rather than against a
        non-zero count: a pruned vector still has the stimulus term in every
        slot, so counting non-zeros would pass while the area contributions
        were missing.
        """
        def snap(prune):
            random.seed(1)
            np.random.seed(1)
            b = Brain(p=P, seed=1, engine="numpy_sparse",
                      recurrent_projection=True, norm_init=False)
            b.add_area(AREA, N, K, BETA)
            eng = b._engine_for(b.areas[AREA])
            eng.kwta_prune = prune
            b.add_stimulus("s", K)
            for _ in range(20):
                b.project({"s": [AREA]}, {AREA: [AREA]})
            before = getattr(eng, "_prune_hits", 0)
            res = eng.project_into(AREA, from_stimuli=["s"],
                                   from_areas=[AREA],
                                   plasticity_enabled=False,
                                   record_activation=True)
            fired = getattr(eng, "_prune_hits", 0) != before
            return np.asarray(res.pre_kwta_inputs), fired

        off, off_fired = snap(False)
        on, on_fired = snap(True)
        assert not on_fired, "pruned while record_activation was on"
        assert not off_fired
        assert np.array_equal(off, on), "snapshot differs under the prune"


class TestSupportMaintenance:
    def test_dropping_the_support_is_safe_not_wrong(self):
        """Losing the index must only cost accuracy of the BOUND, never the
        answer. Anything that renumbers compact indices calls
        `drop_potentiated_support`, and the brain produced afterwards must
        still match the unpruned one.

        Note the prune can still fire immediately after a drop: the STIMULUS
        term is evaluated exactly and can carry `tau` on its own. That is
        correct, not a leak -- the bound does not depend on the index being
        complete, only on it never over-claiming.
        """
        def run(prune, drop):
            random.seed(9)
            np.random.seed(9)
            b = Brain(p=P, seed=9, engine="numpy_sparse",
                      recurrent_projection=True, norm_init=False)
            b.add_area(AREA, N, K, BETA)
            eng = b._engine_for(b.areas[AREA])
            eng.kwta_prune = prune
            b.add_stimulus("s", K)
            for i in range(20):
                if drop and i == 10:
                    eng.drop_potentiated_support()
                b.project({"s": [AREA]}, {AREA: [AREA]})
            rows = int(eng.materialized_count(AREA) or 0)
            w = np.asarray(eng._area_conns[AREA][AREA].weights)[:rows, :rows]
            return sorted(np.asarray(b.areas[AREA].winners).tolist()), w.copy()

        base_w, base_m = run(False, False)
        drop_w, drop_m = run(True, True)
        assert base_w == drop_w
        assert np.array_equal(base_m, drop_m)

    def test_support_is_not_built_when_the_prune_is_off(self):
        """An engine that never prunes must not pay maintenance."""
        random.seed(4)
        np.random.seed(4)
        b = Brain(p=P, seed=4, engine="numpy_sparse", recurrent_projection=True,
                  norm_init=False)
        b.add_area(AREA, N, K, BETA)
        eng = b._engine_for(b.areas[AREA])
        b.add_stimulus("s", K)
        for _ in range(10):
            b.project({"s": [AREA]}, {AREA: [AREA]})
        assert not getattr(eng, "_pot_support", {})


def test_default_is_off():
    """Every result in this repository was measured without the prune. It ships
    off until a census says the organ agrees byte-for-byte."""
    b = Brain(p=P, seed=0, engine="numpy_sparse")
    b.add_area(AREA, N, K, BETA)
    assert not getattr(b._engine_for(b.areas[AREA]), "kwta_prune", False)
