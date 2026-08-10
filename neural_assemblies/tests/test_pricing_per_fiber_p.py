"""Per-fiber ``p`` arithmetic, and the bit-identity that makes it safe to ship.

The whole refactor rests on one property: a brain where every fiber shares a
density must behave EXACTLY as it did before per-fiber ``p`` existed. Not
"within tolerance" -- exactly, because the sampler's binomial-quantile cache is
keyed on the float and a rounding difference re-draws every candidate.
"""

from __future__ import annotations

import unittest

from neural_assemblies.core._pricing import candidate_divisor, effective_binomial


class TestEffectiveBinomial(unittest.TestCase):
    def test_homogeneous_is_returned_unchanged(self):
        """Homogeneous input must return its own inputs, not re-derive them.

        ``1 - var/mu`` evaluates to 0.19999999999999996 for p=0.2, so deriving
        the answer would perturb every existing draw by a rounding error.
        """
        for p in (0.05, 0.1, 0.2, 0.4):
            n_eff, p_eff = effective_binomial([70, 70, 50], [p, p, p])
            self.assertEqual(n_eff, 190)
            self.assertEqual(p_eff, p, "homogeneous p must be returned exactly")

    def test_matches_the_first_two_moments(self):
        sizes, ps = [70, 70], [0.05, 0.4]
        n_eff, p_eff = effective_binomial(sizes, ps)
        mu = sum(a * q for a, q in zip(sizes, ps))
        # integer rounding of n_eff costs a little; the mean must still land
        self.assertAlmostEqual(n_eff * p_eff, mu, delta=0.5)
        self.assertGreater(p_eff, min(ps))
        self.assertLess(p_eff, max(ps))

    def test_degenerate_inputs_do_not_produce_an_invalid_draw(self):
        self.assertEqual(effective_binomial([], []), (0, 0.0))
        self.assertEqual(effective_binomial([70, 70], [0.0, 0.0])[0], 140)
        n_eff, p_eff = effective_binomial([70, 70], [1.0, 0.5])
        self.assertTrue(0.0 < p_eff <= 1.0)
        self.assertGreater(n_eff, 0)


class TestCandidateDivisor(unittest.TestCase):
    SIZES = [70, 70, 50]
    POPS = [1000, 500, 500]

    def test_scalar_and_uniform_sequence_agree_exactly(self):
        for p in (0.05, 0.2, 0.4):
            self.assertEqual(
                candidate_divisor(p, 500, self.SIZES, self.POPS),
                candidate_divisor([p] * 3, 500, self.SIZES, self.POPS),
                "a uniform per-fiber list must price exactly as the scalar")

    def test_per_fiber_p_moves_the_divisor_between_the_scalar_bounds(self):
        lo = candidate_divisor(0.05, 500, self.SIZES, self.POPS)
        hi = candidate_divisor(0.4, 500, self.SIZES, self.POPS)
        mixed = candidate_divisor([0.05, 0.4, 0.4], 500, self.SIZES, self.POPS)
        self.assertGreater(mixed, lo)
        self.assertLess(mixed, hi)

    def test_denser_fiber_raises_the_price_of_candidates(self):
        """A fiber wired more densely delivers more drive, so candidates sampled
        against it must be divided by more, or they outbid real incumbents."""
        base = candidate_divisor([0.2, 0.2, 0.2], 500, self.SIZES, self.POPS)
        denser = candidate_divisor([0.4, 0.2, 0.2], 500, self.SIZES, self.POPS)
        self.assertGreater(denser, base)

    def test_mismatched_lengths_raise_rather_than_silently_recycle(self):
        with self.assertRaises(ValueError):
            candidate_divisor([0.2, 0.2], 500, self.SIZES, self.POPS)


if __name__ == "__main__":
    unittest.main()
