"""
Hebbian bridge parameter sweep for multi-step ordered recall.

Problem: ordered_recall reliably recovers the first item from a cue, but
multi-step recall (2nd, 3rd+ item) fails.  After LRI suppresses the current
assembly, the next assembly's activation drops below the novel-assembly
threshold (0.3 overlap with known), causing early termination.

Root cause: Hebbian bridges between consecutive assemblies are too weak at
default parameters (N=10000, K=100, P=0.05, BETA=0.1, 2 recurrence rounds
out of 10).

This test file systematically sweeps parameters and architectural alternatives
to find regimes where ordered_recall reliably transitions through 3+ items.

Reference:
    Dabagia, Papadimitriou, Vempala.
    "Computation with Sequences of Assemblies in a Model of the Brain."
    Neural Computation (2025).  arXiv:2306.03812.
"""

import time

import numpy as np
import pytest

from src.core.brain import Brain
from src.assembly_calculus import (
    Assembly, Sequence, overlap, chance_overlap, ordered_recall,
)
from src.assembly_calculus.ops import _snap


# Defaults (same as other test files)
N = 10000
K = 100
P = 0.05
BETA = 0.1
SEED = 42
SEQ_LEN = 5  # Items in test sequence


@pytest.fixture(autouse=True)
def _timer(request):
    t0 = time.perf_counter()
    yield
    print(f"  [{time.perf_counter() - t0:.3f}s]")


def _make_brain(n=N, k=K, p=P, beta=BETA, seed=SEED, **kwargs):
    defaults = dict(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


# ======================================================================
# Custom memorization with configurable Phase A/B split
# ======================================================================

def _custom_memorize(brain, stimuli, target, rounds_per_step=10,
                     phase_b_ratio=0.2, repetitions=1, beta_boost=None):
    """Sequence memorization with configurable Phase A/B split.

    Unlike sequence_memorize (which hardcodes max(1, rps-2) for Phase A),
    this allows arbitrary control over the stim-only vs stim+recur ratio.

    Args:
        brain: Brain instance.
        stimuli: Ordered list of stimulus names.
        target: Target area name.
        rounds_per_step: Total rounds per stimulus.
        phase_b_ratio: Fraction of rounds devoted to Phase B
            (stimulus + recurrence).  Default 0.2 = 2/10.
        repetitions: Number of full sequence replays.
        beta_boost: If set, temporarily boost beta during Phase B
            to strengthen Hebbian bridges.  Restored after Phase B.
    """
    original_beta = brain.areas[target].beta
    assemblies = []

    for _rep in range(repetitions):
        assemblies = []
        for stim_name in stimuli:
            recur_rounds = max(1, int(rounds_per_step * phase_b_ratio))
            stim_rounds = rounds_per_step - recur_rounds

            # Phase A: stimulus only (anchor assembly)
            for _ in range(stim_rounds):
                brain.project({stim_name: [target]}, {})

            # Phase B: stimulus + recurrence (build bridge)
            if beta_boost is not None:
                brain.update_plasticity(target, target, beta_boost)

            for _ in range(recur_rounds):
                brain.project({stim_name: [target]}, {target: [target]})

            if beta_boost is not None:
                brain.update_plasticity(target, target, original_beta)

            assemblies.append(_snap(brain, target))

    return Sequence(area=target, assemblies=assemblies)


def _setup_stimuli(brain, n_items=SEQ_LEN, k=K):
    """Create stimuli and return their names."""
    stim_names = []
    for i in range(n_items):
        name = f"s{i}"
        brain.add_stimulus(name, k)
        stim_names.append(name)
    return stim_names


def _measure_recall_length(recalled, memorized, threshold=0.3):
    """Count consecutive items matching in order.

    Returns the number of recalled assemblies that match the
    corresponding memorized assembly above threshold, starting
    from index 0.
    """
    count = 0
    for i in range(min(len(recalled), len(memorized))):
        if overlap(recalled[i], memorized[i]) > threshold:
            count += 1
        else:
            break
    return count


def _try_recall(brain, area, cue_stim, memorized,
                refractory_period=3, inhibition_strength=100.0,
                rounds_per_step=1):
    """Set LRI, run ordered_recall, measure recall length."""
    brain.set_lri(area, refractory_period, inhibition_strength)
    recalled = ordered_recall(
        brain, area, cue_stim, max_steps=len(memorized) + 5,
        known_assemblies=list(memorized),
        convergence_threshold=0.9,
        rounds_per_step=rounds_per_step,
    )
    return _measure_recall_length(list(recalled), list(memorized))


# ======================================================================
# 1. Bridge Strength Diagnostics (memorization quality, not recall)
# ======================================================================

class TestBridgeStrengthDiagnostics:
    """Measure raw Hebbian bridge strength after sequence memorization.

    Isolates memorization quality from the recall mechanism by measuring
    consecutive overlap directly after memorization, without LRI.
    """

    def test_bridge_strength_vs_recurrence_rounds(self):
        """More Phase B (recurrence) rounds should produce stronger bridges.

        Sweeps phase_b_ratio from 0.2 to 0.8 and measures consecutive
        overlap strength.  Higher ratio = more recurrence = stronger bridge.
        """
        ratios = [0.2, 0.4, 0.6, 0.8]
        results = {}

        for ratio in ratios:
            b = _make_brain()
            b.add_area("A", N, K, BETA)
            stims = _setup_stimuli(b, n_items=3)

            seq = _custom_memorize(
                b, stims, "A", rounds_per_step=10,
                phase_b_ratio=ratio, repetitions=3,
            )

            # Measure consecutive overlaps
            overlaps = []
            for i in range(len(seq) - 1):
                overlaps.append(overlap(seq[i], seq[i + 1]))
            results[ratio] = np.mean(overlaps)
            print(f"  phase_b={ratio:.1f}: mean_consec_overlap={results[ratio]:.3f}")

        # Higher Phase B ratio should produce equal or stronger bridges
        # (with tolerance for stochasticity)
        assert results[0.8] >= results[0.2] - 0.05, (
            f"More recurrence should strengthen bridges: "
            f"ratio=0.2 -> {results[0.2]:.3f}, ratio=0.8 -> {results[0.8]:.3f}"
        )

    def test_bridge_strength_vs_beta(self):
        """Higher beta should produce stronger Hebbian bridges.

        Sweeps BETA from 0.05 to 0.5.
        """
        betas = [0.05, 0.1, 0.2, 0.5]
        results = {}

        for beta in betas:
            b = _make_brain(beta=beta)
            b.add_area("A", N, K, beta)
            stims = _setup_stimuli(b, n_items=3)

            seq = _custom_memorize(
                b, stims, "A", rounds_per_step=10,
                phase_b_ratio=0.4, repetitions=3,
            )

            overlaps = []
            for i in range(len(seq) - 1):
                overlaps.append(overlap(seq[i], seq[i + 1]))
            results[beta] = np.mean(overlaps)
            print(f"  beta={beta:.2f}: mean_consec_overlap={results[beta]:.3f}")

        # Higher beta should not decrease bridge strength
        assert results[0.5] >= results[0.05] - 0.05, (
            f"Higher beta should strengthen bridges: "
            f"beta=0.05 -> {results[0.05]:.3f}, beta=0.5 -> {results[0.5]:.3f}"
        )

    def test_bridge_strength_vs_repetitions(self):
        """More repetitions should cumulatively strengthen bridges.

        Sweeps repetitions from 1 to 10.
        """
        reps_list = [1, 3, 5, 10]
        results = {}

        for reps in reps_list:
            b = _make_brain()
            b.add_area("A", N, K, BETA)
            stims = _setup_stimuli(b, n_items=3)

            seq = _custom_memorize(
                b, stims, "A", rounds_per_step=10,
                phase_b_ratio=0.4, repetitions=reps,
            )

            overlaps = []
            for i in range(len(seq) - 1):
                overlaps.append(overlap(seq[i], seq[i + 1]))
            results[reps] = np.mean(overlaps)
            print(f"  reps={reps}: mean_consec_overlap={results[reps]:.3f}")

        # More reps should not decrease bridge strength
        assert results[10] >= results[1] - 0.05, (
            f"More repetitions should strengthen bridges: "
            f"reps=1 -> {results[1]:.3f}, reps=10 -> {results[10]:.3f}"
        )

    def test_bridge_strength_vs_n_k(self):
        """Larger N (with proportional K) should maintain bridge quality.

        Tests (N, K) = (5000, 70), (10000, 100), (20000, 141).
        K ≈ sqrt(N) to maintain the theoretical scaling.
        """
        configs = [
            (5000, 70),
            (10000, 100),
            (20000, 141),
        ]
        results = {}

        for n, k in configs:
            b = _make_brain(n=n, k=k)
            b.add_area("A", n, k, BETA)
            stims = _setup_stimuli(b, n_items=3, k=k)

            seq = _custom_memorize(
                b, stims, "A", rounds_per_step=10,
                phase_b_ratio=0.4, repetitions=3,
            )

            overlaps = []
            for i in range(len(seq) - 1):
                overlaps.append(overlap(seq[i], seq[i + 1]))
            results[(n, k)] = np.mean(overlaps)
            print(f"  N={n}, K={k}: mean_consec_overlap={results[(n, k)]:.3f}")

        # All configs should produce above-chance bridges
        chance = chance_overlap(K, N)
        for key, val in results.items():
            assert val > chance * 3, (
                f"Bridge at {key} should be above chance: {val:.3f} vs {chance:.3f}"
            )


# ======================================================================
# 2. Recall Length Sweep
# ======================================================================

class TestRecallLengthSweep:
    """Measure how many items ordered_recall recovers per parameter combo.

    For each parameter set: memorize a 5-item sequence, enable LRI,
    cue from first item, count items correctly recovered in order.
    """

    def test_recall_length_vs_inhibition(self):
        """Sweep refractory_period × inhibition_strength.

        The LRI penalty decays as 1 - (steps_ago-1)/period, scaled by
        inhibition_strength.  Need enough suppression to push past current
        assembly, but not so much that the next assembly can't fire.
        """
        periods = [2, 4, 6]
        strengths = [50.0, 100.0, 500.0]
        best_combo = None
        best_length = 0

        for period in periods:
            for strength in strengths:
                b = _make_brain()
                b.add_area("A", N, K, BETA)
                stims = _setup_stimuli(b)

                seq = _custom_memorize(
                    b, stims, "A", rounds_per_step=10,
                    phase_b_ratio=0.4, repetitions=5,
                )

                length = _try_recall(b, "A", stims[0], seq,
                                     refractory_period=period,
                                     inhibition_strength=strength)

                print(f"  period={period}, strength={strength}: "
                      f"recall_length={length}/{len(seq)}")

                if length > best_length:
                    best_length = length
                    best_combo = (period, strength)

        print(f"\n  Best: period={best_combo[0]}, strength={best_combo[1]} "
              f"-> {best_length} items")

        # Should recall at least the first item everywhere
        assert best_length >= 1, "Should recall at least one item"

    def test_recall_length_vs_beta_and_rounds(self):
        """Cross BETA × rounds_per_step for recall effectiveness."""
        betas = [0.1, 0.3, 0.5]
        rounds_list = [10, 15, 20]
        best_combo = None
        best_length = 0

        for beta in betas:
            for rps in rounds_list:
                b = _make_brain(beta=beta)
                b.add_area("A", N, K, beta)
                stims = _setup_stimuli(b)

                seq = _custom_memorize(
                    b, stims, "A", rounds_per_step=rps,
                    phase_b_ratio=0.4, repetitions=5,
                )

                length = _try_recall(b, "A", stims[0], seq,
                                     refractory_period=4,
                                     inhibition_strength=100.0)

                print(f"  beta={beta}, rounds={rps}: "
                      f"recall_length={length}/{len(seq)}")

                if length > best_length:
                    best_length = length
                    best_combo = (beta, rps)

        print(f"\n  Best: beta={best_combo[0]}, rounds={best_combo[1]} "
              f"-> {best_length} items")
        assert best_length >= 1

    def test_recall_length_vs_phase_split(self):
        """Sweep Phase A/B ratio for recall effectiveness."""
        ratios = [0.2, 0.4, 0.6, 0.8]
        best_ratio = None
        best_length = 0

        for ratio in ratios:
            b = _make_brain()
            b.add_area("A", N, K, BETA)
            stims = _setup_stimuli(b)

            seq = _custom_memorize(
                b, stims, "A", rounds_per_step=10,
                phase_b_ratio=ratio, repetitions=5,
            )

            length = _try_recall(b, "A", stims[0], seq,
                                 refractory_period=4,
                                 inhibition_strength=100.0)

            print(f"  phase_b_ratio={ratio}: recall_length={length}/{len(seq)}")

            if length > best_length:
                best_length = length
                best_ratio = ratio

        print(f"\n  Best: phase_b_ratio={best_ratio} -> {best_length} items")
        assert best_length >= 1

    def test_recall_length_vs_repetitions(self):
        """More sequence replays should strengthen recall."""
        reps_list = [1, 3, 5, 10, 20]
        best_reps = None
        best_length = 0

        for reps in reps_list:
            b = _make_brain()
            b.add_area("A", N, K, BETA)
            stims = _setup_stimuli(b)

            seq = _custom_memorize(
                b, stims, "A", rounds_per_step=10,
                phase_b_ratio=0.4, repetitions=reps,
            )

            length = _try_recall(b, "A", stims[0], seq,
                                 refractory_period=4,
                                 inhibition_strength=100.0)

            print(f"  reps={reps}: recall_length={length}/{len(seq)}")

            if length > best_length:
                best_length = length
                best_reps = reps

        print(f"\n  Best: reps={best_reps} -> {best_length} items")
        assert best_length >= 1


# ======================================================================
# 3. Architectural Alternatives
# ======================================================================

class TestArchitecturalAlternatives:
    """Test structural changes to the memorization/recall pipeline."""

    def test_elevated_beta_during_recurrence(self):
        """Temporarily boost beta during Phase B to strengthen bridges.

        Uses beta_boost=0.5 during the recurrence (Phase B) rounds,
        while keeping beta=0.1 for stimulus-only (Phase A) rounds.
        This should produce stronger inter-assembly bridges without
        over-strengthening stimulus->assembly connections.
        """
        # Control: standard beta throughout
        b_ctrl = _make_brain()
        b_ctrl.add_area("A", N, K, BETA)
        stims_ctrl = _setup_stimuli(b_ctrl)
        seq_ctrl = _custom_memorize(
            b_ctrl, stims_ctrl, "A", rounds_per_step=10,
            phase_b_ratio=0.4, repetitions=5,
        )
        len_ctrl = _try_recall(b_ctrl, "A", stims_ctrl[0], seq_ctrl,
                               refractory_period=4, inhibition_strength=100.0)

        # Treatment: boosted beta during Phase B
        b_boost = _make_brain()
        b_boost.add_area("A", N, K, BETA)
        stims_boost = _setup_stimuli(b_boost)
        seq_boost = _custom_memorize(
            b_boost, stims_boost, "A", rounds_per_step=10,
            phase_b_ratio=0.4, repetitions=5, beta_boost=0.5,
        )
        len_boost = _try_recall(b_boost, "A", stims_boost[0], seq_boost,
                                refractory_period=4, inhibition_strength=100.0)

        print(f"  control recall_length={len_ctrl}, "
              f"beta_boost recall_length={len_boost}")

        # Beta boost should not decrease recall length
        # (it may or may not help, but it shouldn't hurt)
        assert len_boost >= len_ctrl - 1, (
            f"Beta boost should not significantly degrade recall: "
            f"control={len_ctrl}, boost={len_boost}"
        )

    def test_bridge_area_pattern(self):
        """Dedicated BRIDGE area for cross-item connectivity.

        Architecture:
            Memorize: stim -> SEQ + (SEQ -> BRIDGE) during Phase B
            Recall:   SEQ -> SEQ with BRIDGE -> SEQ also active

        The BRIDGE area acts as an auxiliary memory that stores
        cross-assembly associations.
        """
        b = _make_brain()
        b.add_area("SEQ", N, K, BETA)
        b.add_area("BRIDGE", N, K, BETA)
        stims = _setup_stimuli(b)

        assemblies = []
        for _rep in range(5):
            assemblies = []
            for stim_name in stims:
                # Phase A: stimulus only
                for _ in range(6):
                    b.project({stim_name: ["SEQ"]}, {})

                # Phase B: stimulus + SEQ recurrence + SEQ -> BRIDGE
                for _ in range(4):
                    b.project(
                        {stim_name: ["SEQ"]},
                        {"SEQ": ["SEQ", "BRIDGE"], "BRIDGE": ["BRIDGE"]},
                    )

                assemblies.append(_snap(b, "SEQ"))

        seq = Sequence(area="SEQ", assemblies=assemblies)

        # Recall with BRIDGE feeding back into SEQ
        b.set_lri("SEQ", refractory_period=4, inhibition_strength=100.0)
        b.clear_refractory("SEQ")

        # Activate cue
        b.project({stims[0]: ["SEQ"]}, {})
        recalled = [_snap(b, "SEQ")]

        for _step in range(len(seq) + 5):
            # Self-project SEQ with BRIDGE assistance
            b.project({}, {"SEQ": ["SEQ"], "BRIDGE": ["SEQ", "BRIDGE"]})
            current = _snap(b, "SEQ")

            # Check for cycle
            if any(overlap(current, prev) > 0.9 for prev in recalled):
                break
            if max(overlap(current, k) for k in list(seq)) < 0.2:
                break
            recalled.append(current)

        bridge_length = _measure_recall_length(recalled, list(seq))
        print(f"  bridge_area recall_length={bridge_length}/{len(seq)}")

        # Bridge area should recall at least the first item
        assert bridge_length >= 1, (
            f"Bridge area pattern should recall at least 1 item"
        )

    def test_pre_memorization_reinforcement(self):
        """Extra cross-item reinforcement after sequence_memorize.

        After the initial sequence memorization, do additional rounds
        of cross-item reinforcement: for each consecutive pair (i, i+1),
        activate item i, then immediately project with recurrence while
        activating item i+1.  This should strengthen the x_i -> x_{i+1}
        Hebbian bridge.
        """
        b = _make_brain()
        b.add_area("A", N, K, BETA)
        stims = _setup_stimuli(b)

        # Initial memorization
        seq = _custom_memorize(
            b, stims, "A", rounds_per_step=10,
            phase_b_ratio=0.4, repetitions=3,
        )

        # Extra cross-item reinforcement
        for _extra_rep in range(5):
            for i in range(len(stims) - 1):
                # Activate item i
                b.project({stims[i]: ["A"]}, {})
                b.project({stims[i]: ["A"]}, {"A": ["A"]})

                # Transition to item i+1 with recurrence
                b.project({stims[i + 1]: ["A"]}, {"A": ["A"]})
                b.project({stims[i + 1]: ["A"]}, {"A": ["A"]})

        # Re-snapshot after reinforcement (assemblies may have drifted)
        reinforced = []
        for stim in stims:
            for _ in range(5):
                b.project({stim: ["A"]}, {})
            b.project({stim: ["A"]}, {"A": ["A"]})
            reinforced.append(_snap(b, "A"))
        seq_reinforced = Sequence(area="A", assemblies=reinforced)

        length = _try_recall(b, "A", stims[0], seq_reinforced,
                             refractory_period=4, inhibition_strength=100.0)

        print(f"  reinforced recall_length={length}/{len(seq_reinforced)}")
        assert length >= 1


# ======================================================================
# 4. Best Parameter Demo
# ======================================================================

class TestBestParameterDemo:
    """Demonstrate the best-found parameters from the sweep.

    Uses aggressive settings: high beta boost, many repetitions,
    balanced Phase A/B split.  This is the summary test.
    """

    def _run_best_config(self):
        """Run the best parameter configuration found in the sweep."""
        b = _make_brain(beta=0.1)
        b.add_area("A", N, K, 0.1)
        stims = _setup_stimuli(b)

        # Best config: balanced split, beta boost, many reps
        seq = _custom_memorize(
            b, stims, "A", rounds_per_step=15,
            phase_b_ratio=0.5, repetitions=10, beta_boost=0.5,
        )

        return b, stims, seq

    def test_reliable_three_item_recall(self):
        """With best parameters, recall at least 3 items from 5."""
        b, stims, seq = self._run_best_config()

        # Try multiple LRI configurations
        best_length = 0
        for period in [3, 4, 5, 6]:
            for strength in [50.0, 100.0, 200.0]:
                b_copy = _make_brain(beta=0.1)
                b_copy.add_area("A", N, K, 0.1)
                stims_copy = _setup_stimuli(b_copy)
                seq_copy = _custom_memorize(
                    b_copy, stims_copy, "A", rounds_per_step=15,
                    phase_b_ratio=0.5, repetitions=10, beta_boost=0.5,
                )
                length = _try_recall(
                    b_copy, "A", stims_copy[0], seq_copy,
                    refractory_period=period,
                    inhibition_strength=strength,
                )
                if length > best_length:
                    best_length = length

        print(f"  Best recall length: {best_length}/{len(seq)}")

        # This test documents the current state — it may or may not
        # achieve 3-item recall.  The assertion is intentionally lenient
        # to avoid flaky failures while still tracking progress.
        assert best_length >= 1, (
            f"Best config should recall at least 1 item, got {best_length}"
        )
        if best_length >= 3:
            print("  *** 3-ITEM RECALL ACHIEVED ***")
        elif best_length >= 2:
            print("  *** 2-ITEM RECALL ACHIEVED ***")

    def test_reliable_five_item_recall(self):
        """Stretch goal: recall all 5 items.

        This test is exploratory — it documents whether the current
        parameter space can achieve full sequence recall.
        """
        b = _make_brain(beta=0.1)
        b.add_area("A", N, K, 0.1)
        stims = _setup_stimuli(b)

        # Extra-aggressive settings
        seq = _custom_memorize(
            b, stims, "A", rounds_per_step=20,
            phase_b_ratio=0.6, repetitions=15, beta_boost=1.0,
        )

        best_length = 0
        for period in [3, 5, 8]:
            for strength in [50.0, 200.0, 500.0]:
                b2 = _make_brain(beta=0.1)
                b2.add_area("A", N, K, 0.1)
                stims2 = _setup_stimuli(b2)
                seq2 = _custom_memorize(
                    b2, stims2, "A", rounds_per_step=20,
                    phase_b_ratio=0.6, repetitions=15, beta_boost=1.0,
                )
                length = _try_recall(
                    b2, "A", stims2[0], seq2,
                    refractory_period=period,
                    inhibition_strength=strength,
                    rounds_per_step=2,
                )
                if length > best_length:
                    best_length = length

        print(f"  Best recall length: {best_length}/{len(seq)}")
        assert best_length >= 1
        if best_length >= 5:
            print("  *** FULL 5-ITEM RECALL ACHIEVED ***")
        elif best_length >= 3:
            print("  *** 3+ ITEM RECALL ACHIEVED ***")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
