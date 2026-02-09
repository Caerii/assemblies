"""
Integration tests that prove computational value of the assembly calculus stack.

Each test demonstrates a non-trivial computational capability. The "computer"
is 10k spiking neurons with Hebbian plasticity — assemblies encode information
as distributed, noise-tolerant patterns that support pattern completion,
readout, and composable state machines.

Demonstrates:
    1. Lexicon readout: assemblies encode/decode a vocabulary with noise rejection
    2. FSM computation: parity checker with neural state encoding
    3. PFA generation: neural coin flip produces genuinely stochastic output
    4. End-to-end pipeline: stimulus -> assembly -> readout -> FSM -> state
    5. Merge/composition: conjunctive representations from multiple sources
"""

import unittest
from collections import Counter

import numpy as np

from src.core.brain import Brain
from src.assembly_calculus import (
    project, reciprocal_project, merge, pattern_complete, overlap,
    chance_overlap, Assembly,
    fuzzy_readout, readout_all, build_lexicon,
    FSMNetwork, PFANetwork, RandomChoiceArea,
)


N = 10000
K = 100
P = 0.05
BETA = 0.05
SEED = 42


def _brain(**kw):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kw)
    return Brain(**defaults)


# ======================================================================
# 1. LEXICON: Neural assemblies can encode and decode a vocabulary
# ======================================================================

class TestLexiconComputation(unittest.TestCase):
    """build_lexicon + fuzzy_readout forms a working encode/decode channel:
    project a word, read it back, get the same word. This is the foundation
    for all symbolic computation on top of assemblies."""

    def setUp(self):
        self.brain = _brain()
        self.area = "lang"
        self.brain.add_area(self.area, N, K, BETA)

        self.words = ["cat", "dog", "run", "big", "the"]
        self.stim_map = {}
        for w in self.words:
            stim = f"stim_{w}"
            self.brain.add_stimulus(stim, K)
            self.stim_map[w] = stim

        self.lexicon = build_lexicon(
            self.brain, self.area, self.words, self.stim_map, rounds=10
        )

    def test_all_words_roundtrip(self):
        """Every word survives encode -> decode."""
        errors = []
        for word in self.words:
            asm = project(self.brain, self.stim_map[word], self.area, rounds=10)
            decoded = fuzzy_readout(asm, self.lexicon, threshold=0.5)
            if decoded != word:
                errors.append(f"{word} -> {decoded}")
        self.assertEqual(errors, [],
                         f"Roundtrip failures: {errors}")

    def test_lexicon_entries_are_distinct(self):
        """All word assemblies should have pairwise overlap near chance."""
        chance = chance_overlap(K, N)
        high_overlaps = []
        for i, w1 in enumerate(self.words):
            for w2 in self.words[i+1:]:
                ov = overlap(self.lexicon[w1], self.lexicon[w2])
                if ov > chance * 5:
                    high_overlaps.append((w1, w2, ov))
        self.assertEqual(high_overlaps, [],
                         f"Words too similar: {high_overlaps}")

    def test_pattern_completion_preserves_readout(self):
        """Half-corrupted assembly should still decode to correct word."""
        project(self.brain, self.stim_map["cat"], self.area, rounds=10)
        recovered, recovery = pattern_complete(
            self.brain, self.area, fraction=0.5, rounds=5, seed=0
        )
        decoded = fuzzy_readout(recovered, self.lexicon, threshold=0.3)
        self.assertEqual(decoded, "cat",
                         f"Pattern completion lost 'cat' (recovery={recovery:.2f}, "
                         f"decoded={decoded})")

    def test_noise_rejected_as_improper_parse(self):
        """Random activation should NOT match any word (improper parse)."""
        rng = np.random.default_rng(99)
        fake_winners = rng.choice(N, size=K, replace=False).astype(np.uint32)
        noise_asm = Assembly(self.area, fake_winners)
        decoded = fuzzy_readout(noise_asm, self.lexicon, threshold=0.5)
        self.assertIsNone(decoded,
                          f"Random noise decoded as '{decoded}'")


# ======================================================================
# 2. FSM: A binary parity checker with neural state encoding
# ======================================================================

class TestFSMParityChecker(unittest.TestCase):
    """Parity checker: the simplest non-trivial FSM. Requires maintaining
    discrete state (even/odd) across multi-step computation. The neural
    value: states are represented as robust assemblies that support
    pattern completion and composition with other neural components.

    FSM:
        States: even, odd
        Alphabet: {0, 1}
        Transitions:
            (even, 0) -> even
            (even, 1) -> odd
            (odd, 0)  -> odd
            (odd, 1)  -> even
        Initial: even
    """

    def setUp(self):
        self.brain = _brain()
        self.fsm = FSMNetwork(
            self.brain,
            states=["even", "odd"],
            symbols=["0", "1"],
            transitions=[
                ("even", "0", "even"),
                ("even", "1", "odd"),
                ("odd", "0", "odd"),
                ("odd", "1", "even"),
            ],
            initial_state="even",
            n=N, k=K, beta=BETA, rounds=10,
        )

    def test_empty_string_is_even(self):
        """No input -> parity is even."""
        self.assertEqual(self.fsm.current_state, "even")

    def test_single_one_is_odd(self):
        """'1' has odd parity."""
        self.fsm.step("1")
        self.assertEqual(self.fsm.current_state, "odd")

    def test_two_ones_is_even(self):
        """'11' has even parity."""
        trajectory = self.fsm.run(["1", "1"])
        self.assertEqual(trajectory[-1], "even",
                         f"'11' should be even, got trajectory {trajectory}")

    def test_mixed_string(self):
        """'10110' has 3 ones -> odd parity."""
        trajectory = self.fsm.run(["1", "0", "1", "1", "0"])
        expected_states = ["odd", "odd", "even", "odd", "odd"]
        self.assertEqual(trajectory, expected_states,
                         f"Parity trajectory mismatch: {trajectory}")

    def test_long_string_parity(self):
        """'11001011' has 5 ones -> odd."""
        bits = list("11001011")
        trajectory = self.fsm.run(bits)

        parity = 0
        expected = []
        for b in bits:
            parity ^= int(b)
            expected.append("odd" if parity else "even")

        self.assertEqual(trajectory, expected,
                         f"Long parity mismatch:\n"
                         f"  input:    {bits}\n"
                         f"  expected: {expected}\n"
                         f"  got:      {trajectory}")

    def test_reset_clears_state(self):
        """After processing, reset should return to even."""
        self.fsm.run(["1", "1", "1"])
        self.assertEqual(self.fsm.current_state, "odd")
        self.fsm.reset()
        self.assertEqual(self.fsm.current_state, "even")

    def test_state_assemblies_are_distinct(self):
        """State assemblies should be non-overlapping neural patterns."""
        ov = overlap(self.fsm.state_lexicon["even"],
                     self.fsm.state_lexicon["odd"])
        chance = chance_overlap(K, N)
        self.assertLess(ov, chance * 5,
                        f"State assemblies too similar: overlap={ov:.3f}")

    def test_state_assembly_survives_pattern_completion(self):
        """The current state assembly should be recoverable from 50% corruption.
        This proves the neural state encoding has real attractor dynamics."""
        # Drive FSM to 'odd'
        self.fsm.step("1")
        self.assertEqual(self.fsm.current_state, "odd")

        # Corrupt and recover the state area
        recovered, recovery = pattern_complete(
            self.brain, self.fsm.state_area, fraction=0.5, rounds=5, seed=0
        )

        # The recovered assembly should match the 'odd' state
        decoded = fuzzy_readout(recovered, self.fsm.state_lexicon, threshold=0.3)
        self.assertEqual(decoded, "odd",
                         f"State pattern completion failed: recovery={recovery:.2f}, "
                         f"decoded={decoded}")


# ======================================================================
# 3. PFA: Neural coin flip produces stochastic output
# ======================================================================

class TestPFAGeneration(unittest.TestCase):
    """Proves that neural attractor competition produces genuinely
    stochastic output: two trained attractors compete after mixed
    initialization, and the outcome varies with the random seed.

    PFA:
        q0 --a--> q1  (prob 0.5)
        q0 --a--> q2  (prob 0.5)
        q1 --b--> q0  (prob 1.0)
        q2 --b--> q0  (prob 1.0)
    """

    def test_pfa_generates_both_paths(self):
        """Over many runs, both q1 and q2 should be reached from q0."""
        b = _brain()
        pfa = PFANetwork(
            b,
            states=["q0", "q1", "q2"],
            symbols=["a", "b"],
            transitions=[
                ("q0", "a", "q1", 0.5),
                ("q0", "a", "q2", 0.5),
                ("q1", "b", "q0", 1.0),
                ("q2", "b", "q0", 1.0),
            ],
            initial_state="q0",
            n=N, k=K, beta=BETA, rounds=10,
        )

        results = Counter()
        for i in range(40):
            pfa.reset()
            state = pfa.step("a", seed=i * 11)
            results[state] += 1

        self.assertGreater(results.get("q1", 0), 0,
                           f"q1 never reached: {dict(results)}")
        self.assertGreater(results.get("q2", 0), 0,
                           f"q2 never reached: {dict(results)}")

    def test_pfa_deterministic_path_always_works(self):
        """Deterministic transitions (prob=1.0) should always succeed."""
        b = _brain()
        pfa = PFANetwork(
            b,
            states=["q0", "q1", "q2"],
            symbols=["a", "b"],
            transitions=[
                ("q0", "a", "q1", 0.5),
                ("q0", "a", "q2", 0.5),
                ("q1", "b", "q0", 1.0),
                ("q2", "b", "q0", 1.0),
            ],
            initial_state="q0",
            n=N, k=K, beta=BETA, rounds=10,
        )
        # Regardless of which state 'a' takes us to,
        # 'b' should always return to q0
        pfa.step("a", seed=42)
        state = pfa.step("b", seed=42)
        self.assertEqual(state, "q0",
                         f"Deterministic 'b' should return to q0, got {state}")

    def test_coin_flip_is_genuinely_random(self):
        """The RandomChoiceArea should produce different outcomes for
        different seeds, proving neural stochasticity."""
        b = _brain()
        coin = RandomChoiceArea(b, n=N, k=K, beta=BETA)

        outcomes = set()
        for seed in range(20):
            result = coin.flip(bias=0.5, rounds=10, seed=seed)
            outcomes.add(result)
            if len(outcomes) == 2:
                break

        self.assertEqual(len(outcomes), 2,
                         f"Coin flip always returned same value: {outcomes}")


# ======================================================================
# 4. END-TO-END: Stimulus -> Assembly -> Readout -> FSM -> State
# ======================================================================

class TestEndToEndPipeline(unittest.TestCase):
    """The full pipeline: encode words as assemblies, decode them via
    readout, and feed the decoded symbols into an FSM. This proves
    the stack composes: lexicon + readout + FSM form a working
    neural computation pipeline.

    This is the minimal viable "neural compiler" loop.
    """

    def test_lexicon_drives_fsm(self):
        """Symbols encoded as assemblies, decoded via readout,
        correctly drive FSM state transitions."""
        b = _brain()

        # Build a small lexicon in one area
        lex_area = "lex"
        b.add_area(lex_area, N, K, BETA)
        symbols = ["a", "b"]
        stim_map = {}
        for s in symbols:
            stim = f"inp_{s}"
            b.add_stimulus(stim, K)
            stim_map[s] = stim
        lexicon = build_lexicon(b, lex_area, symbols, stim_map, rounds=10)

        # Build FSM: q0 -a-> q1, q1 -b-> q0
        fsm = FSMNetwork(
            b,
            states=["q0", "q1"],
            symbols=["a", "b"],
            transitions=[
                ("q0", "a", "q1"),
                ("q1", "b", "q0"),
            ],
            initial_state="q0",
            n=N, k=K, beta=BETA, rounds=10,
        )

        # Pipeline: project stimulus -> decode -> FSM step
        input_sequence = ["a", "b", "a"]
        expected_trajectory = ["q1", "q0", "q1"]

        trajectory = []
        for sym in input_sequence:
            # Encode: project stimulus into lexicon area
            asm = project(b, stim_map[sym], lex_area, rounds=10)

            # Decode: readout assembly to symbol name
            decoded = fuzzy_readout(asm, lexicon, threshold=0.5)
            self.assertIsNotNone(decoded,
                                 f"Failed to decode symbol '{sym}'")
            self.assertEqual(decoded, sym,
                             f"Decoded '{decoded}' instead of '{sym}'")

            # Execute: feed decoded symbol into FSM
            new_state = fsm.step(decoded)
            trajectory.append(new_state)

        self.assertEqual(trajectory, expected_trajectory,
                         f"Pipeline trajectory: {trajectory}")

    def test_merge_readout(self):
        """Merged assembly (two concepts) can still be partially
        read out -- both contributing words have above-chance overlap."""
        b = _brain()

        area_a = "concepts_a"
        area_b = "concepts_b"
        area_m = "merged"
        b.add_area(area_a, N, K, BETA)
        b.add_area(area_b, N, K, BETA)
        b.add_area(area_m, N, K, BETA)

        b.add_stimulus("stim_hot", K)
        b.add_stimulus("stim_dog", K)

        # Project each concept into its own area
        project(b, "stim_hot", area_a, rounds=10)
        project(b, "stim_dog", area_b, rounds=10)

        # Merge into a conjunctive representation
        merged_asm = merge(b, area_a, area_b, area_m, rounds=10)

        # The merged assembly should be a real assembly (k neurons)
        self.assertGreaterEqual(len(merged_asm), K * 0.5,
                                f"Merged assembly too small: {len(merged_asm)}")

        # Project each concept alone into merged area and check overlap
        # with the merged assembly -- both should have above-chance overlap
        project(b, "stim_hot", area_a, rounds=10)
        from src.assembly_calculus.ops import _fix, _unfix, _snap
        _fix(b, area_a)
        b.project({}, {area_a: [area_m]})
        _unfix(b, area_a)
        hot_in_merge = _snap(b, area_m)

        project(b, "stim_dog", area_b, rounds=10)
        _fix(b, area_b)
        b.project({}, {area_b: [area_m]})
        _unfix(b, area_b)
        dog_in_merge = _snap(b, area_m)

        ov_hot = overlap(hot_in_merge, merged_asm)
        ov_dog = overlap(dog_in_merge, merged_asm)
        chance = chance_overlap(K, N)

        self.assertGreater(ov_hot, chance * 3,
                           f"'hot' overlap with merge too low: {ov_hot:.3f} "
                           f"(chance={chance:.3f})")
        self.assertGreater(ov_dog, chance * 3,
                           f"'dog' overlap with merge too low: {ov_dog:.3f} "
                           f"(chance={chance:.3f})")


if __name__ == "__main__":
    unittest.main()
