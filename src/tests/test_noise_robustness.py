"""
Noise robustness characterization (Q10): recovery from corrupted assemblies.

Proper protocol (stimulus-free recovery):
    1. Train: project stimulus -> area for ROUNDS (with recurrence)
    2. Remove stimulus: only self-projection henceforth
    3. Inject noise: replace a fraction of winners with random neurons
    4. Recover: run autonomous self-projection for RECOVERY_ROUNDS
    5. Measure: overlap of recovered assembly with original

This is a falsifiable test of the attractor hypothesis: a well-trained
assembly should be recoverable from partial corruption via attractor
dynamics alone, without stimulus input.

References:
    PRIORITIES_AND_GAPS.md Q10 (noise robustness redesign).
    Papadimitriou et al. PNAS 2020 (attractor dynamics).
"""

import copy
import random
import time

import numpy as np
import pytest

from src.core.brain import Brain
from src.assembly_calculus import (
    Assembly, overlap, chance_overlap, project, build_lexicon,
    fuzzy_readout,
)
from src.assembly_calculus.ops import _snap


N = 10000
K = 100
P = 0.05
BETA = 0.1
ROUNDS = 10
RECOVERY_ROUNDS = 8
SEED = 42


@pytest.fixture(autouse=True)
def _timer(request):
    t0 = time.perf_counter()
    yield
    print(f"  [{time.perf_counter() - t0:.3f}s]")


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


def _inject_noise(brain, area_name, fraction, rng):
    """Replace *fraction* of current winners with other fired neurons.

    Works in compact index space (same as pattern_complete). Replaces
    chosen winners with randomly-selected indices from the ever-fired
    population that are not currently winners.

    Args:
        brain: Brain instance.
        area_name: Target area name.
        fraction: Fraction of winners to corrupt (0.0 to 1.0).
        rng: random.Random instance.

    Returns:
        (original_snap, corrupted_snap) tuple.
    """
    original = _snap(brain, area_name)
    compact_winners = list(brain.areas[area_name].winners)
    k = len(compact_winners)
    n_corrupt = int(k * fraction)

    w = brain._engine.get_num_ever_fired(area_name)
    current_set = set(int(x) for x in compact_winners)
    available = [i for i in range(w) if i not in current_set]

    if len(available) < n_corrupt:
        # Not enough non-winners — use what we have
        n_corrupt = len(available)

    # Choose which winners to replace
    replace_positions = rng.sample(range(k), n_corrupt)
    replacements = rng.sample(available, n_corrupt)

    noisy_winners = list(compact_winners)
    for pos, repl in zip(replace_positions, replacements):
        noisy_winners[pos] = repl

    brain.areas[area_name].winners = np.array(noisy_winners, dtype=np.uint32)

    corrupted = _snap(brain, area_name)
    return original, corrupted


# ======================================================================
# 1. Basic noise recovery (stimulus-free)
# ======================================================================

class TestBasicNoiseRecovery:
    """Test recovery from noise injection using autonomous recurrence."""

    def test_recovery_from_mild_noise(self):
        """Assembly recovers from 20% corruption via self-projection.

        Theory: at fraction=0.2, 80% of the attractor basin is intact,
        so self-projection should recover the full assembly.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)
        rng = random.Random(SEED + 1)
        original, corrupted = _inject_noise(b, "A", fraction=0.2, rng=rng)

        # Verify corruption was applied
        assert overlap(original, corrupted) < 0.95, (
            f"Corruption should reduce overlap: {overlap(original, corrupted):.3f}"
        )

        # Autonomous recovery (no stimulus)
        for _ in range(RECOVERY_ROUNDS):
            b.project({}, {"A": ["A"]})
        recovered = _snap(b, "A")

        recovery = overlap(original, recovered)
        assert recovery > 0.6, (
            f"Recovery from 20% noise: overlap={recovery:.3f}"
        )

    def test_recovery_from_moderate_noise(self):
        """Assembly recovery degrades gracefully at 50% corruption."""
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)
        rng = random.Random(SEED + 2)
        original, _ = _inject_noise(b, "A", fraction=0.5, rng=rng)

        for _ in range(RECOVERY_ROUNDS):
            b.project({}, {"A": ["A"]})
        recovered = _snap(b, "A")

        recovery = overlap(original, recovered)
        chance = chance_overlap(K, N)
        assert recovery > chance * 3, (
            f"Recovery from 50% noise should be well above chance: "
            f"overlap={recovery:.3f}, chance={chance:.3f}"
        )

    def test_no_stimulus_during_recovery(self):
        """Verify protocol: stimulus is truly absent during recovery.

        This is the key fix from the invalid Q10 test. Recovery uses
        ONLY project({}, {A: [A]}) — no stimulus input.
        """
        b = _make_brain()
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)

        project(b, "stim", "A", rounds=ROUNDS)
        rng = random.Random(SEED + 3)
        original, _ = _inject_noise(b, "A", fraction=0.3, rng=rng)

        # Recovery uses ONLY self-projection — empty stim dict
        for _ in range(RECOVERY_ROUNDS):
            b.project({}, {"A": ["A"]})

        recovered = _snap(b, "A")
        assert len(recovered.winners) == K, (
            f"Recovery should produce K={K} winners"
        )
        # The test passing with stimulus-free recovery validates the protocol
        recovery = overlap(original, recovered)
        assert recovery > chance_overlap(K, N) * 2, (
            f"Stimulus-free recovery should exceed chance: overlap={recovery:.3f}"
        )


# ======================================================================
# 2. Systematic sweep
# ======================================================================

class TestNoiseSweep:
    """Systematic sweep: corruption fraction vs recovery overlap."""

    def test_monotonic_degradation(self):
        """Recovery decreases monotonically with corruption fraction.

        Sweeps fraction from 0.1 to 0.9 and checks that recovery
        is non-increasing (with tolerance for stochasticity).
        """
        fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
        recoveries = {}

        for frac in fractions:
            b = _make_brain()
            b.add_stimulus("stim", K)
            b.add_area("A", N, K, BETA)

            project(b, "stim", "A", rounds=ROUNDS)
            rng = random.Random(SEED + int(frac * 100))
            original, _ = _inject_noise(b, "A", fraction=frac, rng=rng)

            for _ in range(RECOVERY_ROUNDS):
                b.project({}, {"A": ["A"]})
            recovered = _snap(b, "A")
            recoveries[frac] = overlap(original, recovered)

        # Check monotonic degradation (with 0.15 tolerance for noise)
        for i in range(len(fractions) - 1):
            f_lo, f_hi = fractions[i], fractions[i + 1]
            assert recoveries[f_lo] >= recoveries[f_hi] - 0.15, (
                f"Recovery should degrade: frac={f_lo} -> {recoveries[f_lo]:.3f}, "
                f"frac={f_hi} -> {recoveries[f_hi]:.3f}"
            )

    def test_readout_survives_low_noise(self):
        """Lexicon readout survives 10% noise corruption after recovery.

        Builds a small lexicon, corrupts each word's assembly with 10%
        noise, runs autonomous recovery, and verifies fuzzy_readout
        still returns the correct word.
        """
        b = _make_brain()
        words = ["alpha", "beta", "gamma"]
        stim_map = {}
        for w in words:
            stim_name = f"stim_{w}"
            b.add_stimulus(stim_name, K)
            stim_map[w] = stim_name
        b.add_area("LEX", N, K, BETA)

        lexicon = build_lexicon(b, "LEX", words, stim_map, rounds=ROUNDS)

        for word in words:
            # Re-project word to set its assembly
            project(b, stim_map[word], "LEX", rounds=ROUNDS)

            rng = random.Random(SEED + hash(word) % 10000)
            original, _ = _inject_noise(b, "LEX", fraction=0.1, rng=rng)

            # Autonomous recovery
            for _ in range(RECOVERY_ROUNDS):
                b.project({}, {"LEX": ["LEX"]})

            recovered = _snap(b, "LEX")
            decoded = fuzzy_readout(recovered, lexicon, threshold=0.3)
            assert decoded == word, (
                f"Low noise (10%): '{word}' should decode after recovery, "
                f"got '{decoded}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
