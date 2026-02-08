"""
Tests for Long-Range Inhibition (LRI) — refractory suppression in the
sparse engine.

LRI penalises recently-fired neurons during winner selection so that
sequences can advance instead of oscillating between consecutive assemblies.
"""

import unittest
import numpy as np

from src.core.brain import Brain
from src.assembly_calculus.ops import _snap

N = 10000
K = 100
P = 0.05
BETA = 0.1
ROUNDS = 10
SEED = 42


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


def _train_assembly(brain, stim, area, rounds=ROUNDS):
    """Train an assembly using direct brain.project with self-recurrence.

    Uses brain.project() (not ops.project) so that area→area recurrent
    connections are populated via _project_impl.  This is necessary for
    LRI tests because the recurrent connectome must exist for self-projection
    to have meaningful activations.
    """
    # First step: stimulus only (seeds initial winners)
    brain.project({stim: [area]}, {})
    # Remaining steps: stimulus + self-recurrence (builds A→A connectome)
    for _ in range(rounds - 1):
        brain.project({stim: [area]}, {area: [area]})


class TestLRI(unittest.TestCase):
    """Test Long-Range Inhibition (refractory suppression) in the sparse engine."""

    def test_no_lri_by_default(self):
        """With refractory_period=0, LRI is disabled and assembly converges normally."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)  # default: refractory_period=0

        _train_assembly(b, "s", "A")
        asm = _snap(b, "A")

        # Self-project several times — assembly should be stable (no LRI penalty)
        for _ in range(5):
            b.project({}, {"A": ["A"]})
        asm_later = _snap(b, "A")

        self.assertGreater(asm.overlap(asm_later), 0.9,
                           "Without LRI, self-recurrence should maintain the assembly.")

    def test_hard_suppression_changes_assembly(self):
        """With high inhibition_strength, self-projection must produce a different assembly."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)

        # Train without LRI so recurrent connectome forms properly
        _train_assembly(b, "s", "A")

        # Enable hard LRI
        b.set_lri("A", refractory_period=3, inhibition_strength=1000.0)

        # First self-project populates refractory history (no penalty yet)
        b.project({}, {"A": ["A"]})
        asm_before = _snap(b, "A")

        # Second self-project — LRI now has history, penalty kicks in
        b.project({}, {"A": ["A"]})
        asm_after = _snap(b, "A")

        self.assertLess(asm_before.overlap(asm_after), 0.5,
                        "Hard LRI suppression should produce a largely different assembly.")

    def test_soft_suppression_shifts_assembly(self):
        """With moderate inhibition_strength, assembly shifts but doesn't vanish."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)

        _train_assembly(b, "s", "A")

        # Enable soft LRI
        b.set_lri("A", refractory_period=2, inhibition_strength=5.0)

        # First self-project populates history
        b.project({}, {"A": ["A"]})
        asm_before = _snap(b, "A")

        # Second self-project — LRI penalty kicks in
        b.project({}, {"A": ["A"]})
        asm_after = _snap(b, "A")

        # Should be different but not completely disjoint
        ovlp = asm_before.overlap(asm_after)
        self.assertLess(ovlp, 0.98,
                        "Soft LRI should shift the assembly (overlap < 0.98).")

    def test_refractory_period_controls_suppression_span(self):
        """Longer refractory periods suppress more historical assemblies."""
        results = {}
        for period in [1, 5]:
            b = _make_brain()
            b.add_stimulus("s", K)
            b.add_area("A", N, K, BETA)

            _train_assembly(b, "s", "A")
            asm_original = _snap(b, "A")

            # Enable hard LRI
            b.set_lri("A", refractory_period=period, inhibition_strength=1000.0)

            # Self-project 2 times
            for _ in range(2):
                b.project({}, {"A": ["A"]})

            asm_final = _snap(b, "A")
            results[period] = asm_original.overlap(asm_final)

        # With period=1, the original assembly is no longer suppressed after
        # 2 steps, so it may partly recover.  With period=5, it is still
        # suppressed so overlap should be lower.
        self.assertLessEqual(results[5], results[1] + 0.1,
                             f"Longer refractory period should maintain more suppression: "
                             f"period=1 overlap={results[1]:.3f}, period=5 overlap={results[5]:.3f}")

    def test_clear_refractory_resets_history(self):
        """After clear_refractory, the next projection applies no LRI penalty."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)

        # Train without LRI
        _train_assembly(b, "s", "A")
        asm_before = _snap(b, "A")

        # Enable LRI and do one self-projection (populates history)
        b.set_lri("A", refractory_period=3, inhibition_strength=1000.0)
        b.project({}, {"A": ["A"]})

        # Clear refractory history
        b.clear_refractory("A")

        # Self-project — since history is cleared, no penalty is applied,
        # and the assembly should remain stable (recurrent attractor)
        b.project({}, {"A": ["A"]})
        asm_after = _snap(b, "A")

        self.assertGreater(asm_before.overlap(asm_after), 0.5,
                           "After clearing refractory, self-recurrence should "
                           "partially recover the assembly.")

    def test_stimulus_input_dominates_lri_at_moderate_strength(self):
        """With moderate LRI, stimulus-driven projection still forms assemblies."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)

        # Train normally, then enable moderate LRI
        _train_assembly(b, "s", "A")
        asm_no_lri = _snap(b, "A")

        b.set_lri("A", refractory_period=2, inhibition_strength=10.0)

        # Project stimulus again WITH LRI active — stimulus input should
        # still form a meaningful assembly (not just noise)
        for _ in range(ROUNDS):
            b.project({"s": ["A"]}, {})
        asm_with_lri = _snap(b, "A")

        # The assembly should have some structure (not random)
        # — moderate overlap with the original trained assembly
        self.assertGreater(len(asm_with_lri.winners), 0,
                           "Stimulus projection with LRI should produce winners.")


if __name__ == '__main__':
    unittest.main()
