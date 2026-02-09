"""
Tests for fuzzy readout: assembly → word mapping.

Based on:
    Mitropolsky & Papadimitriou (2023).
    "The Architecture of a Biologically Plausible Language Organ."
"""

import unittest
import numpy as np

from src.core.brain import Brain
from src.assembly_calculus.assembly import Assembly, overlap
from src.assembly_calculus.readout import (
    fuzzy_readout, readout_all, build_lexicon, Lexicon,
)
from src.assembly_calculus.ops import project, _snap


N = 10000
K = 100
P = 0.05
BETA = 0.1
SEED = 42


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


class TestFuzzyReadout(unittest.TestCase):
    """Test fuzzy_readout and readout_all."""

    def _build_simple_lexicon(self):
        """Build a brain with 3 words and return (brain, area, lexicon)."""
        b = _make_brain()
        words = ["cat", "dog", "fish"]
        stim_map = {}
        for w in words:
            stim_name = f"stim_{w}"
            b.add_stimulus(stim_name, K)
            stim_map[w] = stim_name
        b.add_area("LEX", N, K, BETA)

        lexicon = build_lexicon(b, "LEX", words, stim_map, rounds=10)
        return b, "LEX", lexicon, stim_map

    def test_readout_matches_trained_word(self):
        """Projecting a word's stimulus should read back as that word."""
        b, area, lexicon, stim_map = self._build_simple_lexicon()

        # Re-project "cat" and check readout
        asm = project(b, stim_map["cat"], area, rounds=10)
        result = fuzzy_readout(asm, lexicon, threshold=0.5)
        self.assertEqual(result, "cat",
                         f"Expected 'cat', got {result!r}")

    def test_readout_none_below_threshold(self):
        """A random assembly should not match any word."""
        _, area, lexicon, _ = self._build_simple_lexicon()

        # Create a random assembly (no relation to any word)
        rng = np.random.default_rng(99)
        random_winners = rng.choice(N, size=K, replace=False).astype(np.uint32)
        random_asm = Assembly(area, random_winners)

        result = fuzzy_readout(random_asm, lexicon, threshold=0.5)
        self.assertIsNone(result,
                          f"Random assembly should not match any word, got {result!r}")

    def test_readout_all_sorted(self):
        """readout_all should return all words sorted by overlap descending."""
        b, area, lexicon, stim_map = self._build_simple_lexicon()

        asm = project(b, stim_map["dog"], area, rounds=10)
        results = readout_all(asm, lexicon)

        self.assertEqual(len(results), 3)
        # Should be sorted descending
        overlaps = [ov for _, ov in results]
        self.assertEqual(overlaps, sorted(overlaps, reverse=True))
        # Top result should be "dog"
        self.assertEqual(results[0][0], "dog")

    def test_build_lexicon_distinct(self):
        """Each word should get a distinct assembly in the lexicon."""
        _, _, lexicon, _ = self._build_simple_lexicon()

        words = list(lexicon.keys())
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                ov = overlap(lexicon[words[i]], lexicon[words[j]])
                self.assertLess(ov, 0.3,
                                f"'{words[i]}' and '{words[j]}' should be "
                                f"distinct (overlap={ov:.3f}).")

    def test_empty_lexicon_returns_none(self):
        """fuzzy_readout on empty lexicon should return None."""
        rng = np.random.default_rng(0)
        asm = Assembly("A", rng.choice(N, size=K, replace=False).astype(np.uint32))
        result = fuzzy_readout(asm, {}, threshold=0.5)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
