"""
Literature parity tests — pinned seeds and protocol metrics.

Golden values use seed=42, n=5000, k=80 unless noted.
See research/literature/parity/PROTOCOLS.md for full protocols.
"""

import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.compute import EPercentPolicy
from neural_assemblies.assembly_calculus import (
    chance_overlap,
    learn_assembly,
    merge,
    overlap,
    project,
    separate,
    sequence_memorize,
    ordered_recall,
)
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs import (
    RuleParser,
    build_fsm_from_traces,
    learn_fsm_from_sequences,
    learn_separable_classes,
    CoinFlipModel,
    direct_bind,
    measure_directional_asymmetry,
    MinimalTMDemo,
    train_markov_from_sequences,
    BlocksWorldPlanner,
    make_toy_problem,
    make_three_block_problem,
    apply_strips_plan,
    validate_direct_do_calculus,
)


SEED = 42
N = 5000
K = 80
P = 0.05
BETA = 0.1
ROUNDS = 10


def _brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


class TestLiteratureParity:
    """Mechanism and protocol parity against published Assembly Calculus results."""

    def test_project_stability_seed42(self):
        b = _brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)
        asm1 = project(b, "s", "A", rounds=ROUNDS)
        asm2 = project(b, "s", "A", rounds=ROUNDS)
        pers = overlap(asm1, asm2)
        assert pers > 0.85, f"Project persistence {pers:.3f} below 0.85"

    def test_separate_near_chance_seed42(self):
        b = _brain()
        b.add_stimulus("s1", K)
        b.add_stimulus("s2", K)
        b.add_area("A", N, K, BETA)
        _, _, ov = separate(b, "s1", "s2", "A", rounds=ROUNDS)
        chance = chance_overlap(K, N)
        assert ov < chance * 3, f"Separate overlap {ov:.3f} too high vs chance {chance:.3f}"

    def test_merge_responds_above_chance(self):
        b = _brain()
        b.add_stimulus("a", K)
        b.add_stimulus("b", K)
        b.add_area("X", N, K, BETA)
        b.add_area("Y", N, K, BETA)
        b.add_area("C", N, K, BETA)
        project(b, "a", "X", rounds=ROUNDS)
        project(b, "b", "Y", rounds=ROUNDS)
        merged = merge(b, "X", "Y", "C", stim_a="a", stim_b="b", rounds=ROUNDS)
        b.project({"a": ["X"]}, {"X": ["C"], "C": ["C"]})
        for _ in range(ROUNDS - 1):
            b.project({"a": ["X"]}, {"X": ["C"], "C": ["C"]})
        ov = overlap(_snap(b, "C"), merged)
        assert ov > chance_overlap(K, N) * 2

    def test_learn_assembly_converges(self):
        b = _brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)
        asm, epochs, pers = learn_assembly(
            b, "s", "A", max_epochs=12, project_rounds=6, convergence=0.85,
        )
        assert epochs <= 12
        assert pers >= 0.85 or overlap(asm, project(b, "s", "A", rounds=3)) >= 0.8

    def test_epercent_policy_on_area(self):
        b = _brain()
        policy = EPercentPolicy(fraction_of_max=0.4, min_winners=10, e_fraction=0.02)
        b.add_area("A", N, K, BETA, winner_policy=policy)
        assert b.areas["A"].winner_policy == policy
        b.set_competition_policy("A", EPercentPolicy(fraction_of_max=0.5))
        assert b.areas["A"].winner_policy.fraction_of_max == 0.5

    def test_rule_parser_cats_chase_mice(self):
        result = RuleParser(
            language="English", p=0.1, lex_k=20, non_LEX_n=5000,
        ).parse("cats chase mice")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_fsm_learn_from_traces(self):
        traces = [("q0", "a", "q1"), ("q1", "a", "q0")]
        states, symbols, transitions = learn_fsm_from_sequences(traces)
        assert states == ["q0", "q1"]
        assert symbols == ["a"]
        assert ("q0", "a", "q1") in transitions

        b = _brain()
        fsm = build_fsm_from_traces(b, traces, "q0", n=N, k=K, beta=0.05, rounds=8)
        assert fsm.step("a") == "q1"

    def test_tm_demo_unary_increment(self):
        b = _brain()
        tm = MinimalTMDemo(b, n=N, k=K, rounds=6)
        traj = tm.run("11_")
        assert traj[-1] == "q_halt"

    def test_coin_flip_markov_protocol(self):
        traces = [("q0", "flip", "q0")] * 5 + [("q0", "flip", "q1")] * 5
        b = _brain()
        model = CoinFlipModel(b, traces, "q0", n=N, k=K, rounds=6)
        branch = model.sample_branch(bias=0.5, seed=SEED)
        assert branch in (0, 1)

    def test_markov_transition_frequencies_from_traces(self):
        """Dabagia 2024: empirical trace frequencies become transition probs."""
        traces = [("q0", "flip", "q0")] * 5 + [("q0", "flip", "q1")] * 5
        transitions = train_markov_from_sequences(traces)
        probs = {
            (fr, sym, to): p for fr, sym, to, p in transitions
        }
        assert ("q0", "flip", "q0") in probs
        assert ("q0", "flip", "q1") in probs
        assert abs(probs[("q0", "flip", "q0")] - 0.5) < 1e-9
        assert abs(probs[("q0", "flip", "q1")] - 0.5) < 1e-9

    def test_center_embedding_dep_verb_readout(self):
        """Mitropolsky et al. 2022: embedded clause yields DEP-VERB dependencies."""
        simple = RuleParser(
            language="English", p=0.1, lex_k=20, non_LEX_n=5000,
        ).parse("cats chase mice")
        embedded = RuleParser(
            language="English", p=0.1, lex_k=20, non_LEX_n=5000,
        ).parse("cats chase mice that dogs chase cats , love mice")
        assert len(embedded) > len(simple)
        roles = {role for _, _, role in embedded}
        assert "DEP-VERB" in roles

    def test_epwta_variable_size_cap(self):
        """Hoff et al. 2026: winner count bounded by e_fraction of population."""
        e_frac = 0.02
        policy = EPercentPolicy(fraction_of_max=0.35, min_winners=5, e_fraction=e_frac)
        b = _brain()
        b.add_area("A", N, K, BETA, winner_policy=policy)
        b.add_stimulus("s", K)
        asm = project(b, "s", "A", rounds=ROUNDS)
        cap = max(5, int(round(e_frac * N)))
        assert 5 <= len(asm) <= cap

    @pytest.mark.xfail(reason="DIRECT directional binding is vacuous: forward/reverse/do metrics are provably insensitive to the learned CAUSE->BIND connectome (wipe-test + cue-swap + feedforward probes); pre-norm_init values measured degree-hub substrate overlap, not binding. Not re-baselined -- that would pin the artifact.", strict=False)
    def test_direct_binding_asymmetry(self):
        b = _brain()
        b.add_stimulus("cause_s", K)
        b.add_stimulus("effect_s", K)
        b.add_area("CAUSE", N, K, BETA)
        b.add_area("EFFECT", N, K, BETA)
        b.add_area("BIND", N, K, BETA)
        direct_bind(
            b, "CAUSE", "EFFECT", "BIND",
            cause_stim="cause_s", effect_stim="effect_s", rounds=8,
        )
        fwd, rev = measure_directional_asymmetry(b, "CAUSE", "EFFECT", "BIND")
        assert fwd > 0.1
        assert rev > 0.0

    @pytest.mark.xfail(reason="DIRECT directional binding is vacuous: forward/reverse/do metrics are provably insensitive to the learned CAUSE->BIND connectome (wipe-test + cue-swap + feedforward probes); pre-norm_init values measured degree-hub substrate overlap, not binding. Not re-baselined -- that would pin the artifact.", strict=False)
    def test_direct_do_calculus_intervention(self):
        """Kopadi & Kalles 2026: do(effect) preserves cause→bind readout."""
        b = _brain()
        b.add_stimulus("cause_s", K)
        b.add_stimulus("effect_s", K)
        b.add_area("CAUSE", N, K, BETA)
        b.add_area("EFFECT", N, K, BETA)
        b.add_area("BIND", N, K, BETA)
        direct_bind(
            b, "CAUSE", "EFFECT", "BIND",
            cause_stim="cause_s", effect_stim="effect_s", rounds=8,
        )
        fwd, rev, do_fwd = validate_direct_do_calculus(
            b, "CAUSE", "EFFECT", "BIND",
        )
        assert do_fwd >= fwd * 0.45

    def test_coin_ambient_noise_epwta_wiring(self):
        """Dabagia 2024 ambient noise: E%-WTA policy on coin area when noise enabled."""
        traces = [("q0", "flip", "q0")] * 3 + [("q0", "flip", "q1")] * 3
        b = _brain()
        model = CoinFlipModel(
            b, traces, "q0", n=N, k=K, rounds=6, input_noise_std=0.02,
        )
        assert b.areas[model.coin.area_name].input_noise_std == 0.02
        assert b.areas[model.coin.area_name].winner_policy is not None
        assert model.sample_branch(bias=0.5, seed=SEED) in (0, 1)

    def test_coin_empirical_fair_and_biased(self):
        """Dabagia 2024: neural coin-flip bias tracks initialization proportion."""
        traces = [("q0", "flip", "q0")] * 5 + [("q0", "flip", "q1")] * 5
        b = _brain()
        model = CoinFlipModel(b, traces, "q0", n=N, k=K, rounds=6)
        fair0, fair1 = model.empirical_flip_counts(24, bias=0.5, seed_base=100)
        assert fair0 > 0 and fair1 > 0
        bias0, bias1 = model.empirical_flip_counts(20, bias=0.85, seed_base=200)
        assert bias0 > bias1

    def test_pfa_stochastic_both_targets(self):
        """Dabagia 2024/Sequences: 50/50 PFA transitions reach both targets."""
        from collections import Counter
        from neural_assemblies.assembly_calculus.pfa import PFANetwork

        b = _brain()
        transitions = [("q0", "flip", "q0", 0.5), ("q0", "flip", "q1", 0.5)]
        pfa = PFANetwork(
            b, ["q0", "q1"], ["flip"], transitions, "q0",
            n=N, k=K, beta=0.05, rounds=6,
        )
        results = Counter()
        for i in range(30):
            pfa.reset()
            results[pfa.step("flip", seed=SEED + i * 11)] += 1
        assert results["q0"] > 0 and results["q1"] > 0

    def test_sequence_memorize_ordered_recall(self):
        """Dabagia 2025: LRI enables multi-step ordered recall after memorization."""
        b = _brain()
        for i in range(3):
            b.add_stimulus(f"s{i}", K)
        b.add_area("A", N, K, BETA)
        memorized = sequence_memorize(
            b, ["s0", "s1", "s2"], "A",
            rounds_per_step=ROUNDS, repetitions=3,
        )
        b.set_lri("A", refractory_period=3, inhibition_strength=100.0)
        recalled = ordered_recall(
            b, "A", "s0", max_steps=10, known_assemblies=list(memorized),
        )
        assert len(memorized) == 3
        assert len(recalled) >= 1
        assert overlap(recalled[0], memorized[0]) > 0.3

    def test_tm_demo_longer_unary_tape(self):
        """Sequences paper: unary increment TM halts on multi-symbol input."""
        b = _brain()
        tm = MinimalTMDemo(b, n=N, k=K, rounds=6)
        traj = tm.run("111_")
        assert traj[-1] == "q_halt"
        assert len(traj) == 4

    def test_colt_separable_classes(self):
        b = _brain()
        b.add_stimulus("c1", K)
        b.add_stimulus("c2", K)
        b.add_area("CLS", N, K, BETA)
        lex, min_ov = learn_separable_classes(
            b, {"class_a": "c1", "class_b": "c2"}, "CLS", rounds=6,
        )
        assert "class_a" in lex
        assert "class_b" in lex
        assert min_ov < 0.5


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _has_torch_cuda(), reason="torch_sparse requires CUDA")
class TestLiteratureParityGPU:
    def test_sequence_memorize_ordered_recall_torch(self):
        b = _brain(engine="torch_sparse")
        for i in range(3):
            b.add_stimulus(f"s{i}", K)
        b.add_area("A", N, K, BETA)
        memorized = sequence_memorize(
            b, ["s0", "s1", "s2"], "A",
            rounds_per_step=ROUNDS, repetitions=3,
        )
        b.set_lri("A", refractory_period=3, inhibition_strength=100.0)
        recalled = ordered_recall(
            b, "A", "s0", max_steps=10, known_assemblies=list(memorized),
        )
        assert len(recalled) >= 1
        assert overlap(recalled[0], memorized[0]) > 0.3


class TestBlocksPlanning:
    def test_bfs_toy_blocks_plan(self):
        start, goal = make_toy_problem()
        planner = BlocksWorldPlanner(blocks=["A", "B"], n=N, k=K, rounds=6)
        plan = planner.plan(start, goal)
        assert plan is not None
        assert len(plan) >= 1
        assert planner.verify_plan(start, goal, plan)
        end = apply_strips_plan(start, plan)
        assert end.on == goal.on

        b = _brain()
        plan2, traj, final = planner.solve_and_run(b, start, goal)
        assert plan2 is not None
        assert len(traj) >= 2
        assert final is not None
        assert final.on == goal.on

    def test_three_block_plan(self):
        start, goal = make_three_block_problem()
        planner = BlocksWorldPlanner(blocks=["A", "B", "C"], n=N, k=K, rounds=5)
        plan = planner.plan(start, goal)
        assert plan is not None
        assert planner.verify_plan(start, goal, plan)
        end = apply_strips_plan(start, plan)
        assert end.on == goal.on

    def test_three_block_neural_execution(self):
        """AAAI 2022: neural FSM scaffold executes a valid 3-block plan."""
        start, goal = make_three_block_problem()
        planner = BlocksWorldPlanner(blocks=["A", "B", "C"], n=N, k=K, rounds=5)
        b = _brain()
        plan, traj, final = planner.solve_and_run(b, start, goal)
        assert plan is not None
        assert planner.verify_plan(start, goal, plan)
        assert len(traj) >= len(plan) + 1
        assert traj[0] == "search"
        assert final is not None
        assert final.on == goal.on

    def test_four_block_plan_and_neural_run(self):
        from neural_assemblies.programs import make_four_block_problem

        start, goal = make_four_block_problem()
        planner = BlocksWorldPlanner(blocks=["A", "B", "C", "D"], n=N, k=K, rounds=5)
        plan = planner.plan(start, goal)
        assert plan is not None
        assert planner.verify_plan(start, goal, plan)
        b = _brain()
        _, traj, final = planner.solve_and_run(b, start, goal)
        assert final is not None
        assert final.on == goal.on
        assert len(traj) >= 2


@pytest.fixture(scope="module")
def nemo_parser():
    """Train emergent NEMO parser once (mitropolsky2025simulated protocol)."""
    from neural_assemblies.assembly_calculus.emergent import EmergentParser

    parser = EmergentParser(
        n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS,
    )
    parser.train()
    return parser


class TestLiteratureParityNEMO2025:
    """mitropolsky2025simulated — grounded emergent parser curriculum."""

    _MODALITY_TO_POS = {
        "visual": "NOUN",
        "motor": "VERB",
        "properties": "ADJ",
        "spatial": "PREP",
        "social": "PRON",
        "temporal": "ADV",
        "none": "DET",
    }

    _PINNED_ROLES = [
        (["the", "dog", "runs"], {"dog": "AGENT"}),
        (["the", "cat", "chases", "the", "bird"], {"cat": "AGENT", "bird": "PATIENT"}),
        (["she", "sees", "the", "bird"], {"she": "AGENT", "bird": "PATIENT"}),
    ]

    def test_pos_classification_at_least_80_percent(self, nemo_parser):
        from neural_assemblies.assembly_calculus.emergent.core.grounding import VOCABULARY

        correct = 0
        for word, ctx in VOCABULARY.items():
            expected = self._MODALITY_TO_POS[ctx.dominant_modality]
            actual, _ = nemo_parser.classify_word(word)
            if actual == expected:
                correct += 1
        accuracy = correct / len(VOCABULARY)
        assert accuracy >= 0.80, f"POS accuracy {accuracy:.0%} < 80%"

    @pytest.mark.parametrize("words,expected_roles", _PINNED_ROLES)
    def test_pinned_role_binding(self, nemo_parser, words, expected_roles):
        roles = nemo_parser.parse(words)["roles"]
        for word, role in expected_roles.items():
            assert roles.get(word) == role, f"{word}: got {roles.get(word)!r}, want {role!r}"

    def test_novel_generalization_bird_chases_boy(self, nemo_parser):
        roles = nemo_parser.parse(["the", "bird", "chases", "the", "boy"])["roles"]
        assert roles.get("bird") == "AGENT"
        assert roles.get("boy") == "PATIENT"
        assert roles.get("chases") == "ACTION"

    def test_word_order_svo(self, nemo_parser):
        from neural_assemblies.assembly_calculus.emergent.parser import EvaluationSuite

        result = EvaluationSuite(nemo_parser).evaluate_word_order(target="SVO")
        assert result["correct"] is True
        assert result["inferred"] == "SVO"


class TestColt2022Halfspace:
    """[COLT22] Theorem 6 (Learning Linear Thresholds), the paper's own protocol.

    Ported from ``.reference/mdabagia-learning-with-assemblies/Halfspace.ipynb``.
    Asserts the theorem's OWN bounds -- a fresh D+ sample's cap overlaps at
    least 3k/4 of A*, a D- sample's at most k/4 -- which hold here with wide
    margin (measured ~99 and ~10 against 75 and 25), so this tests the effect
    rather than the seed set.

    This replaces nothing: the older ``test_colt_separable_classes`` asserts
    ``min_ov < 0.5``, which is satisfiable without any generalisation at all.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from neural_assemblies.programs.colt_halfspace_numpy import (
            run_colt_halfspace,
        )
        return run_colt_halfspace()

    def test_positive_cap_overlaps_at_least_three_quarters_k(self, result):
        k = result.parameters["cap_size"]
        assert result.pos_overlap >= 0.75 * k, (
            f"D+ cap overlaps A* in {result.pos_overlap:.2f} neurons, "
            f"below the theorem's 3k/4 = {0.75 * k:.0f}"
        )

    def test_negative_cap_overlaps_at_most_a_quarter_k(self, result):
        k = result.parameters["cap_size"]
        assert result.neg_overlap <= 0.25 * k, (
            f"D- cap overlaps A* in {result.neg_overlap:.2f} neurons, "
            f"above the theorem's k/4 = {0.25 * k:.0f}"
        )

    def test_every_seed_separates_not_just_the_mean(self, result):
        k = result.parameters["cap_size"]
        for i, (p, n) in enumerate(zip(result.per_seed_pos, result.per_seed_neg)):
            assert p >= 0.75 * k and n <= 0.25 * k, (
                f"seed index {i} does not separate: D+ {p:.2f}, D- {n:.2f}"
            )

    def test_classes_share_support(self, result):
        """The claim is non-trivial only because the supports overlap.

        Both distributions draw Bernoulli(k/n) over the coordinates outside the
        halfspace block, so a positive-looking example can be drawn from D-.
        If negatives carried their own disjoint block this would be trivial.
        """
        assert result.n_on_neg == 0
        assert result.n_on_pos > 0

    def test_golden_matches_recorded(self, result):
        import json
        from pathlib import Path

        golden_path = (Path(__file__).resolve().parents[2] / "research"
                       / "literature" / "parity" / "golden"
                       / "colt2022_halfspace.json")
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        assert result.pos_overlap == pytest.approx(
            golden["metrics"]["pos_overlap"], abs=0.01)
        assert result.neg_overlap == pytest.approx(
            golden["metrics"]["neg_overlap"], abs=0.01)
