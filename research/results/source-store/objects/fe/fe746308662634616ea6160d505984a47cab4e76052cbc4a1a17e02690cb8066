"""An area's size and a fiber's column extent are ONE invariant, or a bug.

#69 / #41 / #47. A lazily materialized fiber has columns for the target's
neurons. When the area grows through some OTHER fiber, this one is not expanded
with it, and its last columns become allocated-but-uninitialised zeros. Nothing
raises: reads still return numbers, k-WTA still returns k winners.

WHAT IS PINNED HERE, and why each one is not redundant:

1. `fiber_extent` / `materialized_count` have ONE defined meaning, returning
   None (not zero) where the question does not apply. The three quantities a
   caller used to choose between by hand -- `area.w`, `conn._log_cols`,
   `weights.shape[1]` -- read 247 / 213 / 367 for the same fiber at the same
   instant on one recurrent protocol at n=1000, and a ratio measured through
   them moved 1.87 / 1.65 / 2.84. The choice was silent and load-bearing.

2. `read_only()` holds the invariant. MEASURED: connectome byte-identical in
   6/6 trials, d(w)=0, d(extent)=0.

3. `frozen()` BREAKS it, and that is pinned as a KNOWN DEFECT rather than
   asserted away. frozen() stops plasticity, not recruitment, so a probe under
   it grew the area by ~26 neurons and added 750 nonzeros of weight exactly 1.0
   to the self fiber -- materialization, not Hebbian learning. If that test
   starts failing, frozen()'s semantics changed and every probe written against
   it needs re-reading.

Without (3) the suite would pass just as well if the invariant were vacuous,
which is the failure mode that makes a determinism test useless.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import fiber_census, FiberState, Verdict

N, K, P, BETA, ROUNDS = 400, 20, 0.05, 0.05, 6


def _strict_probes(b, on):
    """Pin the flag EXPLICITLY on every engine.

    These tests assert both states, so inheriting NEURAL_ASSEMBLIES_STRICT_PROBES
    from the environment would make half of them fail whenever the suite is run
    with the instrument on -- which is exactly when they are most needed.
    """
    for eng in b._all_engines():
        eng._strict_probes = bool(on)
    return b


def _brain(strict=False, **kw):
    b = Brain(p=P, seed=3, engine="numpy_sparse", norm_init=True, **kw)
    b.add_stimulus("S1", K)
    b.add_stimulus("S2", K)
    b.add_area("A", N, K, BETA)
    return _strict_probes(b, strict)


def _trained():
    b = _brain()
    b.project({"S1": ["A"]}, {})
    for _ in range(ROUNDS):
        b.project({"S1": ["A"]}, {"A": ["A"]})
    return b


def _eng(b):
    return b._engine_for(b.areas["A"])


def _fibers(b):
    return [f for f in fiber_census(b) if isinstance(f, FiberState)]


class TestAccessorContract:

    def test_lazy_fiber_reports_an_integer_extent(self):
        b = _trained()
        assert _eng(b).fiber_extent("A", "A") is not None
        assert _eng(b).materialized_count("A") > 0

    def test_absent_fiber_is_none_not_zero(self):
        """None means NOT APPLICABLE. Zero would mean 'no columns', which is a
        real and different state -- a dead fiber."""
        b = _trained()
        assert _eng(b).fiber_extent("A", "nope") is None
        assert _eng(b).materialized_count("nope") is None

    def test_dense_area_has_no_watermark(self):
        """An explicit area allocates all n up front, so there is nothing for
        the extent to disagree with and the invariant must be VACUOUS rather
        than trivially violated."""
        b = Brain(p=P, seed=3, engine="numpy_sparse", norm_init=True)
        b.add_stimulus("S1", K)
        b.add_explicit_area("D", N, K, BETA)
        b.project({"S1": ["D"]}, {})
        eng = b._engine_for(b.areas["D"])
        assert eng.fiber_extent("D", "D") is None
        assert all(f.extent_desync == 0 for f in _fibers(b))

    def test_extent_never_exceeds_physical_columns(self):
        b = _trained()
        conn = _eng(b)._area_conns["A"]["A"]
        assert _eng(b).fiber_extent("A", "A") <= conn.weights.shape[1]


class TestInvariantUnderProbes:

    def test_training_leaves_the_invariant_intact(self):
        b = _trained()
        assert all(f.extent_desync == 0 for f in _fibers(b)), (
            "ordinary training desynced a fiber from its target -- growth and "
            "expansion have come apart on the normal path")

    def test_read_only_holds_the_invariant(self):
        b = _trained()
        with b.read_only():
            b.project({"S2": ["A"]}, {"A": ["A"]})
        assert all(f.extent_desync == 0 for f in _fibers(b))

    def test_read_only_does_not_grow_the_area(self):
        """#41 as titled. MEASURED: 0 growth, connectome byte-identical."""
        b = _trained()
        before = _eng(b).materialized_count("A")
        w0 = np.asarray(_eng(b)._area_conns["A"]["A"].weights).copy()
        with b.read_only():
            b.project({"S2": ["A"]}, {"A": ["A"]})
        assert _eng(b).materialized_count("A") == before
        assert np.array_equal(np.asarray(_eng(b)._area_conns["A"]["A"].weights),
                              w0), "read_only() mutated the connectome"

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN DEFECT, pinned deliberately: frozen() stops plasticity but NOT "
        "recruitment, so a probe under it grows the area and desyncs the self "
        "fiber. Use read_only() for reads. If this XPASSes, frozen() changed "
        "and every probe written against it must be re-read."))
    def test_frozen_probe_holds_the_invariant(self):
        b = _trained()
        with b.frozen():
            b.project({"S2": ["A"]}, {})
        assert all(f.extent_desync == 0 for f in _fibers(b))

    def test_frozen_probe_grows_the_area(self):
        """The positive statement of the same fact, so the MECHANISM is pinned
        and not merely its symptom: recruitment happens while plasticity is off.

        Note the fiber is NOT a source here, which is exactly why it is left
        behind: `_expand_connectomes` only expands fibers that participated in
        the projection, so the area gains neurons that this fiber has no column
        for. Measured 82 -> 101 materialized with the extent stuck at 82.
        """
        b = _trained()
        before = _eng(b).materialized_count("A")
        ext_before = _eng(b).fiber_extent("A", "A")
        with b.frozen():
            b.project({"S2": ["A"]}, {})
        assert _eng(b).materialized_count("A") > before, (
            "frozen() no longer recruits -- if this is intended it makes "
            "read_only()'s no-recruitment guarantee redundant, and the "
            "xfail above should now XPASS")
        assert _eng(b).fiber_extent("A", "A") == ext_before, (
            "the self fiber expanded even though it was not a source")

    def test_frozen_probe_writes_synapses_when_the_fiber_IS_driven(self):
        """The other half: drive the fiber and it materializes NEW WEIGHT while
        plasticity is off. Measured at n=1000: +750 nonzeros in one probe, all
        of value 1.0 -- recruitment wiring, not Hebbian potentiation.

        Both halves are needed. Without this one, "frozen() is safe as long as
        you drive every fiber" would look true.
        """
        b = _trained()
        conn = _eng(b)._area_conns["A"]["A"]
        before = np.asarray(conn.weights).copy()
        with b.frozen():
            for _ in range(3):
                b.project({"S2": ["A"]}, {"A": ["A"]})
        after = np.asarray(_eng(b)._area_conns["A"]["A"].weights)
        r, c = before.shape
        assert int((after[:r, :c] != 0).sum()) > int((before != 0).sum()), (
            "expected frozen() to materialize new synapses into the driven "
            "fiber; if it no longer does, probe contamination is closed and "
            "the frozen()/read_only() distinction can be revisited")
        old = after[:r, :c]
        fresh = old[(before == 0) & (old != 0)]
        assert np.allclose(fresh, fresh.flat[0]), (
            "new entries should all be the same untouched initial weight -- "
            f"varied values would mean plasticity ran: {np.unique(fresh)[:5]}")


class TestStrictProbeMode:
    """NEURAL_ASSEMBLIES_STRICT_PROBES=1 turns "which of these 50 frozen()
    sites is a contaminating probe?" into something you can RUN.

    Opt-in, because switching a site to read_only() changes its numbers --
    suppressing recruitment moves what the probe reads. So it is a measurement
    instrument, not a policy.
    """

    def _strict(self, b):
        return _strict_probes(b, True)

    def test_recruiting_under_frozen_raises(self):
        b = self._strict(_trained())
        with pytest.raises(RuntimeError, match="STRICT PROBES"):
            with b.frozen():
                b.project({"S2": ["A"]}, {})

    def test_the_message_says_what_to_do(self):
        b = self._strict(_trained())
        with pytest.raises(RuntimeError, match="read_only"):
            with b.frozen():
                b.project({"S2": ["A"]}, {})

    def test_read_only_is_silent_under_strict_mode(self):
        """The whole point: the correct probe must not trip the guard, or the
        guard is just noise."""
        b = self._strict(_trained())
        with b.read_only():
            b.project({"S2": ["A"]}, {"A": ["A"]})

    def test_ordinary_training_is_silent_under_strict_mode(self):
        """Recruitment WITH plasticity on is how the brain is built. Flagging
        it would make the mode unusable."""
        b = self._strict(_brain())
        b.project({"S1": ["A"]}, {})
        for _ in range(ROUNDS):
            b.project({"S1": ["A"]}, {"A": ["A"]})

    def test_the_flag_can_be_turned_off(self):
        """Not asserted: that it is OFF by default -- the suite is deliberately
        run with NEURAL_ASSEMBLIES_STRICT_PROBES=1 as an instrument, and a test
        that pins the environment would fail exactly then."""
        b = _strict_probes(_trained(), False)
        with b.frozen():
            b.project({"S2": ["A"]}, {})   # must not raise


class TestCensusReportsIt:

    def _desyncs(self, b):
        # Verdicts require DECLARING the pathway, matching the existing census
        # contract: a plain census returns records, a declaration returns
        # judgements. Without `driven` the desync is still on the record as
        # `.extent_desync`.
        return [v for v in fiber_census(b, driven={"A": ["A"]})
                if isinstance(v, Verdict) and "EXTENT DESYNC" in v.detail]

    def test_desync_is_reported_as_a_verdict_not_just_a_number(self):
        b = _trained()
        with b.frozen():
            b.project({"S2": ["A"]}, {})
        assert self._desyncs(b), "census saw a desynced fiber and said nothing"

    def test_clean_brain_yields_no_desync_verdict(self):
        """The check must be able to stay silent, or it is not evidence."""
        assert not self._desyncs(_trained())

    def test_plain_census_still_returns_only_records(self):
        """Regression: adding the verdict must not change what a census
        RETURNS. Every existing caller does `[f for f in fiber_census(b) if
        f.dead]`, and a Verdict has no `.dead`."""
        b = _trained()
        with b.frozen():
            b.project({"S2": ["A"]}, {})
        assert all(isinstance(f, FiberState) for f in fiber_census(b))
        assert any(f.extent_desync for f in fiber_census(b))
