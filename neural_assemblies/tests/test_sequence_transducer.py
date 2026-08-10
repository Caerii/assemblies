"""Tests for the induced-state transducer organ.

THE CLOCK IS THE THING WORTH TESTING. `tick` is the only call that may advance
the machine; `write` and `emit` read a FIXED arc. If `write` advanced the state,
`rounds` would silently mean "run the state forward this many times per word"
and every training-strength parameter inherited from #14 would be measuring
something else. That is not visible in a score -- it would just make the state
wrong -- so it is asserted directly.

Every behavioural case carries its untrained control, per the rule that a test
a degenerate arm also passes is not a test.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import assembly_overlap, read_assembly
from neural_assemblies.programs.sequence_transducer import SequenceTransducer

VOCAB = ["a", "b", "c", "d"]


def _build(seed=42, *, beta=0.1, organ_p=0.4, n=2000, n_arc=800, k=40):
    b = Brain(p=0.05, save_winners=True, seed=seed, engine="numpy_sparse")
    t = SequenceTransducer(b, VOCAB, n=n, n_arc=n_arc, k=k, beta=beta,
                           organ_p=organ_p, prefix="_t")
    t.ground(rounds=5)
    return b, t


def _ov(a, b):
    """Overlap in the NEURON-ID space -- see `assembly_overlap`'s contract.

    Reading `area.winners` for two prefixes and intersecting them is invalid
    even though both sides are compact: recruitment between the two reads
    renumbers the space, and `frozen()` (unlike `probe()`) allows recruitment.
    Caught by `test_index_space_ratchet`, which is why the ratchet is run at
    the START of a unit.
    """
    return assembly_overlap(a, b)


def _arc_and_state(t, first):
    """Arc and state assemblies along the prefix ``(first, 'b')``, as sets.

    Callers wrap this in ``frozen()``, NOT ``probe()``. Probe suppresses
    RECRUITMENT, which is right for reading a trained organ but WRONG for an
    area training left unmaterialised: with nothing to recruit it answers from
    the handful of neurons it has and returns the SAME winners for perfectly
    disjoint inputs. Measured here first as an arc overlap of 1.000 against a
    LEX overlap of 0.000 and misread as a sampler defect; the two engines agree
    once recruitment is allowed. See [[probe-isolation-required]] for the
    opposite error, which is the more common one.
    """
    t.reset()
    arcs, states = [], []
    for w in (first, "b"):
        t.tick(w)
        arcs.append(read_assembly(t.brain, t.arc_area))
        t.emit()
        states.append(np.asarray(t.state().winners))
    return arcs, states


class TestClock(unittest.TestCase):

    def test_write_does_not_advance_the_state(self):
        """Repeated writes strengthen; they must not step the machine.

        The arc is unchanged across these calls, so the state re-settles onto
        the same assembly each time. Anything else means `rounds` is running
        the transducer forward.
        """
        _, t = _build()
        t.reset()
        t.tick("a")
        t.write("b", rounds=1)
        first = t.state()
        t.write("b", rounds=1)
        second = t.state()
        self.assertEqual(overlap(first, second), 1.0)

    def test_tick_advances_the_arc(self):
        """The control for the above: a new word must move the machine.

        ASSERTED ON THE ARC, NOT THE STATE, and that is a finding rather than a
        convenience. On this substrate the state does NOT move -- overlap 1.00
        between consecutive steps of an UNTRAINED organ, with beta on
        arc -> state set to 0 and with a single write round, so no learning of
        any kind is involved. Two arcs overlapping 0.35 produce states
        overlapping 0.97.

        The cause is upstream of this class. On an unmaterialised target the
        sparse candidate sampler AMPLIFIES input overlap about twofold against
        `numpy_exact` (0.75 -> 0.725 vs 0.325), and pre-materialising the
        target removes most of it (0.175 -> 0.075 at an input overlap of 0.35).
        Amplification per step compounds along a sequence. See
        [[sampler-merges-at-low-load]] and [[sampler-is-the-whole-discrepancy]].
        """
        _, t = _build()
        t.reset()
        t.tick("a")
        before = read_assembly(t.brain, t.arc_area)
        t.write("b", rounds=3)
        t.tick("c")
        after = read_assembly(t.brain, t.arc_area)
        self.assertLess(_ov(before, after), 1.0)

    def test_reset_clears_every_area(self):
        b, t = _build()
        t.reset()
        t.tick("a")
        t.write("b")
        t.reset()
        for area in (t.lex_area, t.arc_area, t.state_area, t.out_area):
            self.assertEqual(len(b.areas[area].winners), 0, area)


class TestState(unittest.TestCase):

    def _trained(self, reps=20, **kw):
        b, t = _build(**kw)
        for _ in range(reps):
            t.train_sentence(["a", "b", "d"], rounds=3)
            t.train_sentence(["c", "b", "a"], rounds=3)
        return b, t

    def test_training_sharpens_the_arc_at_the_first_position(self):
        """The conjunction itself learns: distinct words give distinct arcs,
        and training makes them MORE distinct. Untrained is the control."""
        b, t = _build()
        with b.frozen():
            untrained = _ov(_arc_and_state(t, "a")[0][0],
                            _arc_and_state(t, "c")[0][0])
        b, t = self._trained()
        with b.frozen():
            trained = _ov(_arc_and_state(t, "a")[0][0],
                          _arc_and_state(t, "c")[0][0])
        self.assertLess(trained, untrained)

    def test_the_state_collapses_while_the_arc_does_not(self):
        """Where the organ actually stands, and the split is the finding.

        Both prefixes end in "b", so the arc at position 1 can separate them
        only THROUGH the state. Measured at toy scale over 5 seeds, mean +/-
        95% CI, at the registered beta::

            arc@0    0.035 +/- 0.035     the conjunction works
            state@0  0.985 +/- 0.028     the state is one attractor
            arc@1    0.975 +/- 0.022     so history does not survive one step

        WHAT IS NOT ASSERTED, AND WHY. Lowering beta on arc -> state alone
        looked like a clean fix at seed 42 (arc@1 fell to 0.00, monotonically
        in beta). It does not survive seeding: at beta=0 the same statistic is
        0.580 +/- 0.617 over 5 seeds -- a CI spanning the range. The trend is
        real enough to register as a question and nowhere near enough to assert
        as a mechanism, so this test pins only the collapse.
        See [[ensemble-not-realization]].
        """
        b, t = self._trained()
        with b.frozen():
            arcs_a, states_a = _arc_and_state(t, "a")
            arcs_c, states_c = _arc_and_state(t, "c")
        self.assertLess(_ov(arcs_a[0], arcs_c[0]), 0.3)
        self.assertGreater(_ov(states_a[0], states_c[0]), 0.8)
        self.assertGreater(_ov(arcs_a[1], arcs_c[1]), 0.8)

    def test_learning_on_arc_to_state_is_what_collapses_the_state(self):
        """MECHANISM, pinned here so it cannot change silently.

        The state area has no self fiber, so #14's collapse channel is absent
        ([[recurrence-is-the-collapse-channel]]) -- and the state collapses
        anyway. Setting beta to 0 on arc -> state ALONE, which makes it a fixed
        random projection, restores separation. So the collapse is hub
        formation on the FEED-FORWARD fiber that carries the transition, not
        recurrence.

        Toy scale, one seed: this holds the mechanism in place, it does not
        establish it. The seeded measurement is the experiment.
        """
        b, t = self._trained()
        with b.frozen():
            collapsed = _ov(_arc_and_state(t, "a")[1][0],
                            _arc_and_state(t, "c")[1][0])

        b, t = _build()
        b.update_plasticity(t.arc_area, t.state_area, 0.0)
        for _ in range(20):
            t.train_sentence(["a", "b", "d"], rounds=3)
            t.train_sentence(["c", "b", "a"], rounds=3)
        with b.frozen():
            fixed_fiber = _ov(_arc_and_state(t, "a")[1][0],
                              _arc_and_state(t, "c")[1][0])

        self.assertGreater(collapsed, 0.8)
        self.assertLess(fixed_fiber, collapsed)

    def test_state_self_fiber_is_never_driven(self):
        """[[recurrence-is-the-collapse-channel]]. ``add_area`` pre-creates a
        connectome for every pair, so asserting the OBJECT is absent tests
        nothing -- it is there. What matters is that no projection ever drives
        it, which shows up as a fiber that never gains weight."""
        b, t = self._trained(reps=6)
        conn = b._engine._area_conns[t.state_area][t.state_area]
        total = (conn.weights.sum() if conn.sparse
                 else float(np.sum(np.asarray(conn.weights))))
        self.assertEqual(float(total), 0.0)


class TestReadout(unittest.TestCase):

    def test_untrained_transducer_does_not_rank_a_successor(self):
        """The degenerate control: an untrained arc -> OUT knows no successor.

        THE VARIATION HAS TO BE ACROSS BRAINS. Repeating one cue on one brain
        inside a read context is ONE draw measured many times -- the emitted
        assembly is identical every pass, so a hair's-breadth win becomes 20/20
        and reads as a real effect. This test scored 19/20 that way before the
        loop was rewritten to vary the seed and the cue.
        """
        rng = random.Random(0)
        hits = trials = 0
        for seed in (1, 2, 3, 4, 5):
            b, t = _build(seed=seed)
            with b.frozen():
                for cue, nxt in (("a", "b"), ("b", "c"), ("c", "d")):
                    t.reset()
                    t.tick(cue)
                    hits += t.rank(t.emit(), rng)[0] == nxt
                    trials += 1
        self.assertLessEqual(hits / trials, 0.5,
                             "untrained readout produced a successor")

    def test_signatures_are_neuron_ids_and_survive_growth(self):
        """Stored in neuron IDs, so training between grounding and readout
        cannot invalidate them. See [[two-index-spaces-compact-vs-neuron-id]]."""
        b, t = _build()
        before = {w: np.array(t.out_signature[w].winners, copy=True)
                  for w in VOCAB}
        for _ in range(4):
            t.train_sentence(["a", "b", "c", "d"], rounds=3)
        for w in VOCAB:
            self.assertTrue(np.array_equal(before[w],
                                           t.out_signature[w].winners))
        self.assertGreater(b.areas[t.out_area].w, 0)


class TestRegime(unittest.TestCase):

    def test_organ_p_applies_to_the_four_organ_fibers(self):
        """[[SEQ-ORGAN-EMBEDS]]: the organ carries its own density, and the
        stimulus fibers deliberately stay at ambient so word grounding matches
        the study this is compared against."""
        b, t = _build(organ_p=0.4)
        for src, dst in ((t.lex_area, t.arc_area),
                         (t.state_area, t.arc_area),
                         (t.arc_area, t.state_area),
                         (t.arc_area, t.out_area)):
            self.assertAlmostEqual(
                b._engine._area_conns[src][dst].p, 0.4, places=6,
                msg=f"{src}->{dst}")
        stim = b._engine._stim_conns[t._s_stim["a"]][t.lex_area]
        self.assertAlmostEqual(stim.p, 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
