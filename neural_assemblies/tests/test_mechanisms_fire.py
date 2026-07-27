"""Assert that wired-in mechanisms actually FIRE, not merely that they are called.

Every failure this file guards against was found the hard way, and each one
looked like working code: the call happened, no exception was raised, and the
mechanism did nothing.

  * ``add_mutual_inhibition`` silently no-ops unless two or more areas of the
    group are targets of the SAME ``project()`` call -- so a caller that
    projects into one area at a time gets no inhibition and no warning.
  * ``StatePredictionMixin`` trained happily while every state -> PREDICTION
    fiber stayed empty, giving retrieval overlap of exactly 0.0 for every
    prefix (see test_state_prediction.py).
  * ``Brain(seed=)`` was not reproducible: global RNG leaked between
    constructions, so borderline experiments flipped between runs.
  * ``Connectome.update_weights`` multiplied without a ceiling, overflowing
    float32 after ~120 training sentences.

"Is it called?" is cheap to check and nearly worthless. These assert observable
consequences instead.
"""
import copy

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.connectome import Connectome


class TestSeedReproducibility:
    """Brain(seed=) must depend only on the seed, not on execution history."""

    def _sig(self, brain):
        total = 0.0
        eng = brain._engine
        for _src, d in sorted(eng._area_conns.items()):
            for _tgt, conn in sorted(d.items()):
                w = getattr(conn, "weights", None)
                if w is not None and getattr(w, "size", 0):
                    total += float(np.asarray(w).sum())
        return round(total, 4)

    def _build(self, seed=7):
        b = Brain(p=0.05, seed=seed, norm_init=False)
        b.add_explicit_area("PHON", 300, 50, 0.1)
        b.add_area("A", 500, 50, 0.1)
        b.add_area("B", 500, 50, 0.1)
        return b

    def test_same_seed_same_wiring_within_a_process(self):
        # The original bug: consecutive builds read a shared global stream, so
        # build 1 and build 3 matched but build 2 and 4 differed (alternating).
        sigs = [self._sig(self._build()) for _ in range(4)]
        assert len(set(sigs)) == 1, (
            f"Brain(seed=) is not reproducible across builds: {sigs}. "
            "Something is drawing from a global RNG -- trap np.random.* and "
            "np.random.default_rng before importing to find the caller.")

    def test_different_seeds_differ(self):
        """Guards the opposite failure: a 'fix' that makes everything constant."""
        assert self._sig(self._build(seed=1)) != self._sig(self._build(seed=2))

    def test_no_global_rng_during_construction(self):
        calls = []
        real = np.random.binomial
        np.random.binomial = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            self._build()
        finally:
            np.random.binomial = real
        assert not calls, (
            f"{len(calls)} global np.random.binomial calls during construction; "
            "wiring must draw from a seeded generator")


class TestMutualInhibitionFires:
    """It only fires when >=2 group areas are targets of the SAME project()."""

    def _brain(self):
        b = Brain(p=0.1, seed=3, norm_init=False)
        b.add_stimulus("s", 40)
        for a in ("R1", "R2"):
            b.add_area(a, 400, 40, 0.1)
        b.project({"s": ["R1"]}, {})
        b.project({"s": ["R2"]}, {})
        return b

    def test_suppresses_loser_when_co_targeted(self):
        b = self._brain()
        b.add_mutual_inhibition(["R1", "R2"])
        b.project({"s": ["R1", "R2"]}, {})
        alive = [a for a in ("R1", "R2") if len(b.areas[a].winners) > 0]
        assert len(alive) == 1, (
            "mutual inhibition did not suppress a loser when both areas were "
            f"co-targeted (alive={alive})")

    def test_documents_the_single_target_no_op(self):
        """Pins the trap: projecting one area at a time inhibits NOTHING.

        This is not desired behaviour, it is a property callers must know --
        `_apply_mutual_inhibition` skips any group with fewer than two entries
        in `activation_scores`, and `activation_scores` only holds areas that
        were targets of the call.
        """
        b = self._brain()
        b.add_mutual_inhibition(["R1", "R2"])
        b.project({"s": ["R1"]}, {})      # only ONE group member targeted
        both_alive = all(len(b.areas[a].winners) > 0 for a in ("R1", "R2"))
        assert both_alive, (
            "single-target projection now inhibits -- if this changed "
            "deliberately, update the callers that relied on the no-op")


class TestMutualInhibitionIsReachedInProduction:
    """The missing half of the audit: does the PARSER ever invoke it?

    The two tests above establish that the mechanism suppresses correctly when
    co-targeted, and that single-target projection no-ops. Neither asks whether
    the emergent parser ever co-targets a group -- and it does not. Measured on
    a SENTENCES-depth parser: 1361 `project()` calls during training and 12
    during `parse()` reach `_apply_mutual_inhibition`, and in ZERO of them does
    any group have two or more members in `activation_scores`.

    So the paper's "the three ROLE areas are in mutual inhibition; this is the
    only use of interarea inhibition in our model" (Mitropolsky & Papadimitriou
    2025 sec. 2.3) is registered but dormant, and role exclusivity is carried
    entirely by the Python `inhibited` set in `_assign_roles_neural`. Wiring the
    groups up made the mechanism available, not operational.

    xfail rather than deleted: making this pass is a design change (the parser
    must project into competing role areas in one call), tracked as task #24.
    Leaving it here means the gap cannot be forgotten, and the day a caller
    starts co-targeting, this flips to xpass and says so.
    """

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="EmergentParser never co-targets a mutual-inhibition group, so "
               "the paper's inter-area inhibition never runs; role exclusivity "
               "is symbolic (the Python `inhibited` set). Task #24.",
        strict=False,
    )
    def test_parser_actually_co_targets_a_group(self):
        import os

        os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
        from neural_assemblies.assembly_calculus.emergent.parser import (
            EmergentParser,
        )
        from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
            create_training_sentences,
        )

        fired = {"n": 0, "seen": 0}
        orig = Brain._apply_mutual_inhibition

        def spy(self, activation_scores):
            fired["seen"] += 1
            scores = activation_scores
            for group in self._mutual_inhibition_groups:
                if len([n for n in group if n in scores]) > 1:
                    fired["n"] += 1
            return orig(self, activation_scores)

        Brain._apply_mutual_inhibition = spy
        try:
            p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
            p.train(create_training_sentences())
            p.parse(["the", "dog", "chases", "the", "cat"])
        finally:
            Brain._apply_mutual_inhibition = orig

        assert fired["n"] > 0, (
            f"mutual inhibition never fired: {fired['seen']} project() calls "
            f"reached it, none co-targeted a group. The mechanism is dormant, "
            f"so role exclusivity is symbolic rather than neural."
        )


class TestConnectomeSaturation:
    def test_update_weights_respects_w_max(self):
        c = Connectome(20, 20, p=1.0, sparse=False, rng=np.random.default_rng(0))
        pre = list(range(20))
        for _ in range(500):                     # would overflow float32 unclamped
            c.update_weights(pre, pre, beta=0.5, w_max=20.0)
        w = np.asarray(c.weights)
        assert np.isfinite(w).all(), "weights overflowed despite w_max"
        assert w.max() <= 20.0 + 1e-6

    def test_unclamped_still_available(self):
        """w_max=None must remain opt-out, not silently clamped."""
        c = Connectome(5, 5, p=1.0, sparse=False, rng=np.random.default_rng(0))
        pre = list(range(5))
        for _ in range(20):
            c.update_weights(pre, pre, beta=0.5)
        assert np.asarray(c.weights).max() > 20.0


def test_brain_is_deep_copyable():
    """Brain is deep-copied in parity/trace paths; a stored module breaks it."""
    b = Brain(p=0.05, seed=1, norm_init=False)
    b.add_stimulus("s", 20)
    b.add_area("A", 200, 20, 0.1)
    b.project({"s": ["A"]}, {})
    clone = copy.deepcopy(b)
    assert len(clone.areas["A"].winners) == len(b.areas["A"].winners)
