"""The parser paper's structural violation signals (TACL 2021 sections 4.3, 6).

WHAT THESE TESTS ARE FOR. Our shipped P600 detector is a continuous quantity
compared against a fixed margin, and it is DEAD -- the margin sits 11.9x above
the largest excess ever observed, because the underlying quantity was redefined
and the constant was not (#104, research/notes/language/erp_metric_is_clipped.md). The
detectors in `parse_errors` cannot fail that way: two are boolean and the third
is a set comparison, and every threshold is DERIVED from (n, k).

So the tests that matter most here are the ones on the NULLS. A threshold that
silently stops matching its substrate is the exact defect being designed out,
and `test_thresholds_track_the_substrate` is the guard for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.parse_errors import (
    EmptyProject,
    chance_overlap,
    empty_project,
    lexical_readout,
    nonsense_threshold,
    overlap_sd,
)
from neural_assemblies.core.inhibition import (
    DISINHIBIT,
    InhibitionState,
    apply_rule,
    fiber_rule,
)


def _asm(area: str, ids) -> Assembly:
    return Assembly(area=area, winners=np.array(sorted(ids), dtype=np.uint32))


class TestDerivedNulls:
    """No constant in this module may be fitted to a particular brain."""

    def test_chance_is_k_over_n(self):
        assert chance_overlap(3000, 30) == pytest.approx(0.01)
        assert chance_overlap(1000, 50) == pytest.approx(0.05)

    def test_chance_is_degenerate_only_where_it_should_be(self):
        assert chance_overlap(0, 30) == 0.0
        assert chance_overlap(3000, 0) == 0.0
        # k == n means every draw is the whole area, so overlap is certain.
        assert chance_overlap(30, 30) == pytest.approx(1.0)

    def test_thresholds_track_the_substrate(self):
        """THE POINT OF THE WHOLE MODULE.

        #104 exists because a threshold was tuned for P600 ~ 5.0 and kept when
        the quantity became bounded in [0,1]. A derived threshold cannot do
        that: change the area and it moves with it. Asserted as a RELATION, not
        as two numbers, because pinning the numbers would recreate the defect.
        """
        small = nonsense_threshold(1000, 30)
        large = nonsense_threshold(100000, 30)
        assert large < small, (
            "a bigger area makes accidental overlap RARER, so the bar to call "
            f"a cap a word must fall: n=1e3 -> {small:.5f}, n=1e5 -> {large:.5f}")
        for n in (1000, 3000, 100000):
            assert nonsense_threshold(n, 30) > chance_overlap(n, 30), (
                "the threshold must sit strictly above the null it is derived "
                "from, or every random cap reads as a word")

    def test_threshold_is_far_above_chance_and_far_below_a_real_match(self):
        """The operating point, stated as a window rather than a constant."""
        thr = nonsense_threshold(3000, 30)
        assert thr > 5 * chance_overlap(3000, 30)
        assert thr < 0.25, (
            f"threshold {thr:.4f} has climbed into the range of genuine "
            f"lexical matches (measured 0.25+), so real words would read as "
            f"nonsense")

    def test_sd_shrinks_as_the_area_grows(self):
        assert overlap_sd(100000, 30) < overlap_sd(1000, 30)
        assert overlap_sd(1, 30) == 0.0


class TestNonsenseAssembly:
    """`getWord()` failing IS the paper's nonsense-assembly error."""

    def test_an_exact_lexical_cap_reads_as_its_word(self):
        lex = {"dog": _asm("LEX", range(30)), "cat": _asm("LEX", range(100, 130))}
        out = lexical_readout(_asm("LEX", range(30)), lex, n=3000)
        assert out.word == "dog"
        assert not out.is_nonsense
        assert out.best_overlap == pytest.approx(1.0)

    def test_a_random_cap_is_NONSENSE_not_a_nearest_word(self):
        """The behaviour the AC substrate otherwise cannot express.

        `readout()` and `readout_all()` always name a best match, so a cap that
        is no word at all still comes back labelled -- the totalising habit that
        makes apparatus defects look like results. This must return None.
        """
        lex = {"dog": _asm("LEX", range(30)), "cat": _asm("LEX", range(100, 130))}
        rng = np.random.default_rng(0)
        junk = _asm("LEX", rng.choice(3000, size=30, replace=False))
        out = lexical_readout(junk, lex, n=3000)
        assert out.is_nonsense, (
            f"a random cap read as {out.word!r} at overlap {out.best_overlap:.4f} "
            f"(threshold {out.threshold:.4f})")
        assert out.word is None

    def test_a_partial_cap_still_reads_when_it_beats_the_null(self):
        """Detection must survive a degraded assembly, or it is useless.

        Half the neurons of "dog" plus half junk is 0.5 overlap -- 50x chance.
        A detector that only accepts exact caps would call every real parse
        nonsense.
        """
        lex = {"dog": _asm("LEX", range(30)), "cat": _asm("LEX", range(100, 130))}
        degraded = _asm("LEX", list(range(15)) + list(range(2000, 2015)))
        out = lexical_readout(degraded, lex, n=3000)
        assert out.word == "dog"
        assert out.times_chance > 10

    def test_empty_lexicon_is_nonsense_not_a_crash(self):
        out = lexical_readout(_asm("LEX", range(30)), {}, n=3000)
        assert out.is_nonsense and out.best_overlap == 0.0

    def test_ties_break_by_NAME_not_by_dict_order(self):
        """#80 was three set-iteration sites; this is the same trap in a readout.

        Two words with identical overlap must resolve the same way regardless of
        insertion order, or the detector is process-dependent.
        """
        a = {"alpha": _asm("LEX", range(30)), "beta": _asm("LEX", range(30))}
        b = {"beta": _asm("LEX", range(30)), "alpha": _asm("LEX", range(30))}
        assert lexical_readout(_asm("LEX", range(30)), a, n=3000).word == \
               lexical_readout(_asm("LEX", range(30)), b, n=3000).word == "alpha"

    def test_margin_reports_ambiguity(self):
        lex = {"dog": _asm("LEX", range(30)), "cat": _asm("LEX", range(30))}
        out = lexical_readout(_asm("LEX", range(30)), lex, n=3000)
        assert out.margin == pytest.approx(0.0), (
            "two words with identical assemblies must report ZERO margin -- an "
            "unambiguous-looking readout over a degenerate lexicon is exactly "
            "the fake-perfect signature")


class TestEmptyProject:
    """The paper's example: after an intransitive verb, a noun has nowhere to go."""

    AREAS = ("LEX", "SUBJ", "VERB", "OBJ")

    def _state_with_winners(self, opened):
        state = InhibitionState(self.AREAS, initial_areas=self.AREAS)
        for a1, a2 in opened:
            apply_rule(state, fiber_rule(DISINHIBIT, a1, a2))
        return state

    class _Brain:
        """Minimal stand-in: project_map only asks which areas have winners."""

        class _A:
            def __init__(self, has):
                self.winners = np.arange(30) if has else np.array([], dtype=int)

        def __init__(self, active):
            self.areas = {n: self._A(n in active) for n in
                          ("LEX", "SUBJ", "VERB", "OBJ")}

    def test_an_open_route_is_not_empty(self):
        state = self._state_with_winners([("LEX", "SUBJ")])
        out = empty_project(state, self._Brain({"LEX"}), "LEX")
        assert not out.empty
        assert "SUBJ" in out.lex_targets

    def test_all_fibers_closed_is_EMPTY_PROJECT(self):
        """'the dogs lived' + 'cats': OBJ was never disinhibited."""
        state = self._state_with_winners([])
        out = empty_project(state, self._Brain({"LEX"}), "LEX")
        assert out.empty
        assert out.lex_targets == ()
        assert out.detected_by == "gating"

    def test_LEX_to_LEX_alone_does_NOT_count_as_a_destination(self):
        """The reference treats LEX->LEX as bookkeeping, not a target.

        `check_war_of_fibers` bounds LEX's targets at TWO *because* the
        self-projection is expected. Counting it here would make every state
        look non-empty and the detector would never fire -- which is precisely
        how #104's detector died.
        """
        state = self._state_with_winners([("LEX", "LEX")])
        out = empty_project(state, self._Brain({"LEX"}), "LEX")
        assert out.empty, (
            f"LEX->LEX was counted as a destination: targets={out.lex_targets}")

    def test_a_closed_TARGET_AREA_also_produces_empty_project(self):
        """Area inhibition, not just fiber inhibition, must be able to cause it.

        The paper's mechanism is an area (OBJ) that was never disinhibited, so
        a detector that only watched fibers would miss the paper's own example.
        """
        state = InhibitionState(self.AREAS, initial_areas=("LEX",))
        apply_rule(state, fiber_rule(DISINHIBIT, "LEX", "OBJ"))
        out = empty_project(state, self._Brain({"LEX"}), "LEX")
        assert out.empty, (
            f"OBJ is inhibited but LEX still reaches {out.lex_targets}")

    def test_result_is_a_dataclass_not_a_bare_bool(self):
        """A bare bool cannot say HOW it was established.

        `detected_by` distinguishes gating (what we can see today) from drive (a
        projection that ran and produced nothing, which our total k-WTA cannot
        report). Collapsing them would hide a known limit.
        """
        out = empty_project(self._state_with_winners([]),
                            self._Brain({"LEX"}), "LEX")
        assert isinstance(out, EmptyProject)
        assert out.detected_by in ("gating", "drive")


class TestTryProjectOnARealBrain:
    """The stability criterion, measured -- including the case that kills it.

    THE FIRST IMPLEMENTATION OF THIS PASSED AND WAS WORTHLESS. It fired
    `{A: [B]}` twice with no self-fiber, so the second firing saw an identical
    input and returned an identical cap: jaccard 1.0000 for a trained pathway
    AND for an untrained one. The test that caught it is
    `test_an_untrained_pathway_is_NOT_an_assembly` -- the true-negative arm.
    A detector is only worth what its negative case proves.
    """

    N, K = 3000, 30

    def _brain(self, *, materialize: bool):
        from neural_assemblies.assembly_calculus.ops import project
        from neural_assemblies.core.brain import Brain

        b = Brain(p=0.05, save_winners=True, seed=1, engine="numpy_sparse")
        b.add_stimulus("s", self.K)
        for a in ("A", "B", "C"):
            b.add_area(a, self.N, self.K, 0.1)
        project(b, "s", "A", rounds=10)
        for _ in range(10):                 # train A -> B; leave A -> C alone
            b.project({}, {"A": ["B"]})
        if materialize:
            for a in ("A", "B", "C"):
                b._engine_for(b.areas[a]).materialize_area(a)
        return b

    def test_a_trained_pathway_IS_an_assembly(self):
        from neural_assemblies.assembly_calculus.parse_errors import try_project

        b = self._brain(materialize=True)
        assert try_project(b, "A", "B") is not None

    def test_an_untrained_pathway_is_NOT_an_assembly(self):
        """The true negative. Measured jaccard 0.3000 against 1.0000 trained."""
        from neural_assemblies.assembly_calculus.parse_errors import (
            assembly_stability, try_project,
        )

        b = self._brain(materialize=True)
        st = assembly_stability(b, "A", "C")
        assert st.trustworthy, f"pool {st.pool} <= k {st.k}"
        assert not st.stable, (
            f"an UNTRAINED pathway produced a stable cap (jaccard "
            f"{st.jaccard:.4f}). Either the self-recurrence on the repeated "
            f"firing has been dropped -- which makes the criterion a no-op -- "
            f"or the target is not really untrained")
        assert try_project(b, "A", "C") is None

    def test_the_two_arms_are_actually_separated(self):
        """Assert the CONTRAST, not either arm's number.

        A trained pathway scoring 1.000 is only meaningful next to an untrained
        one that does not. Pinning 0.3000 would pin a realization; the model
        claim is that self-reinforcement separates them.
        """
        from neural_assemblies.assembly_calculus.parse_errors import (
            assembly_stability,
        )

        b = self._brain(materialize=True)
        trained = assembly_stability(b, "A", "B").jaccard
        untrained = assembly_stability(b, "A", "C").jaccard
        assert trained > untrained + 0.3, (
            f"trained {trained:.4f} vs untrained {untrained:.4f} -- the "
            f"stability criterion no longer separates a real assembly from an "
            f"arbitrary top-k")

    def test_an_UNDER_MATERIALIZED_target_refuses_to_answer(self):
        """A cold target cannot produce a measured Stability result."""
        from neural_assemblies.assembly_calculus.parse_errors import assembly_stability, try_project
        b = self._brain(materialize=False)
        pool = b._engine.materialized_count('C')
        for probe in (assembly_stability, try_project):
            with pytest.raises(ValueError, match='materialized'):
                probe(b, 'A', 'C')
            assert b._engine.materialized_count('C') == pool

    def test_full_but_no_alternatives_is_measured_and_untrustworthy(self):
        """pool == k can run, but a perfect score is still not evidence."""
        from neural_assemblies import Brain
        from neural_assemblies.assembly_calculus.parse_errors import assembly_stability, try_project
        b = Brain(p=1, engine='numpy_explicit', norm_init=False)
        for name in ('A', 'C'):
            b.add_area(name, 4, 4)
        b.areas['A'].winners = np.arange(4, dtype=np.uint32)
        st = assembly_stability(b, 'A', 'C')
        assert st.stable and st.jaccard == 1
        assert st.pool == st.k == 4 and not st.trustworthy
        with pytest.raises(ValueError, match='vacuous'):
            try_project(b, 'A', 'C')


@pytest.mark.parametrize('rounds', [0, -1, True, 1.5])
def test_stability_rejects_invalid_rounds_before_observation(rounds):
    from types import SimpleNamespace
    from neural_assemblies.assembly_calculus.parse_errors import assembly_stability
    def fail():
        pytest.fail('invalid rounds reached neural observation')
    with pytest.raises(ValueError, match='rounds'):
        assembly_stability(SimpleNamespace(read_only=fail), 'A', 'C', rounds=rounds)
