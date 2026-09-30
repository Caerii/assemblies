"""Golden-file parity tests for recorded literature protocols."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from neural_assemblies.assembly_calculus.ops import project
from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.learn import classify, learn_separable_classes

_REPO = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO / "research" / "literature" / "parity" / "golden"


def _load(name: str) -> dict:
    path = _GOLDEN_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _require_golden_dataset(result, golden) -> None:
    expected = golden.get("metrics", {}).get("data_source")
    if expected is not None and result.data_source != expected:
        pytest.skip(
            f"golden requires {expected}; producer resolved {result.data_source}. "
            "Synthetic fallback is an API smoke fixture, not golden evidence."
        )


def _verify_or_skip_missing_dataset(protocol_id):
    from neural_assemblies.parity.runner import verify_protocol
    from neural_assemblies.programs.colt_mnist_data import DatasetUnavailable

    try:
        return verify_protocol(protocol_id)
    except DatasetUnavailable as exc:
        pytest.skip(str(exc))


class TestColt2022Golden:
    def test_ten_class_separable_matches_golden(self):
        g = _load("colt2022_mnist.json")
        p = g["parameters"]
        brain = Brain(p=p["p"], save_winners=True, seed=p["seed"], engine="numpy_sparse")
        brain.add_area("CLASS", p["n"], p["k"], p["beta"])
        class_stimuli = {str(i): f"s{i}" for i in range(p["n_classes"])}
        for stim in class_stimuli.values():
            brain.add_stimulus(stim, p["k"])
        lexicon, min_ov = learn_separable_classes(
            brain, class_stimuli, "CLASS", rounds=p["rounds"],
        )
        correct = sum(
            1
            for label, stim in class_stimuli.items()
            if classify(lexicon, project(brain, stim, "CLASS", rounds=3), threshold=0.3) == label
        )
        acc = correct / p["n_classes"]
        assert acc >= g["thresholds"]["train_classify_accuracy_min"]
        assert min_ov <= g["thresholds"]["min_pairwise_overlap_max"]


class TestRetractedCoinGoldens:
    @pytest.mark.parametrize("name", [
        "coin2024_demo.json",
        "coin2024_compete.json",
        "coin2024_softmax.json",
        "coin2024_markov_arc.json",
    ])
    def test_invalid_instruments_cannot_be_consumed_as_goldens(self, name):
        g = _load(name)
        assert "RETRACTED" in g
        assert "metrics" not in g and "expected" not in g
        assert "historical_metrics" in g and "historical_expected" in g


class TestNemo2025ScaffoldGolden:
    def test_scaffold_metrics_match_golden(self):
        g = _load("nemo2025_scaffold.json")
        p = g["parameters"]
        from neural_assemblies.assembly_calculus.scaffold import compare_scaffold_vs_simple

        result = compare_scaffold_vs_simple(
            seq_len=p["seq_len"],
            n=p["n"],
            k=p["k"],
            density=p["p"],
            n_presentations=p["n_presentations"],
            seed=p["seed"],
        )
        tol = g["thresholds"]["metrics_match_tolerance"]
        gm = g["metrics"]
        assert abs(result.simple_last - gm["simple_last_recall"]) < tol
        assert abs(result.scaffold_last - gm["scaffold_last_recall"]) < tol
        assert result.scaffold_beats_simple == gm["scaffold_beats_simple"]


class TestColt2022MnistNotebookGolden:
    def test_notebook_numpy_matches_golden(self):
        g = _load("colt2022_mnist_notebook.json")
        from neural_assemblies.programs.colt_mnist_numpy import run_colt_mnist_numpy

        result = run_colt_mnist_numpy(**g["parameters"])
        _require_golden_dataset(result, g)
        tol = g["thresholds"]["metrics_match_tolerance"]
        assert result.mean_accuracy >= g["thresholds"]["mean_accuracy_min"]
        assert abs(result.mean_accuracy - g["metrics"]["mean_accuracy"]) < tol


@pytest.mark.slow
class TestNemo2025CurriculumGolden:
    def test_curriculum_metrics_match_golden(self):
        g = _load("nemo2025_curriculum.json")
        os.environ["EMERGENT_FAST_TRAINING"] = "1"
        os.environ.pop("EMERGENT_DEV_CURRICULUM", None)

        from neural_assemblies.assembly_calculus.emergent import EmergentParser
        from neural_assemblies.assembly_calculus.emergent.curriculum import CurriculumTrainer
        from neural_assemblies.assembly_calculus.emergent.evaluation import EvaluationSuite

        p = g["parameters"]
        parser = EmergentParser(
            n=p["n"], k=p["k"], p=p["p"], beta=p["beta"],
            seed=42, rounds=p["rounds"],
        )
        trainer = CurriculumTrainer(parser)
        for stage in p["stages"]:
            trainer.train_stage(stage)

        suite = EvaluationSuite(parser)
        wo = suite.evaluate_word_order(target="SVO")
        role_probes = suite.evaluate_roles([
            {"words": ["the", "dog", "runs"], "expected_roles": {"dog": "AGENT"}},
            {"words": ["the", "cat", "chases", "the", "bird"],
             "expected_roles": {"cat": "AGENT", "bird": "PATIENT"}},
        ])

        sent_result = next(r for r in trainer.stage_results if r.stage_name == "SENTENCES")
        sent_acc = sent_result.classification_accuracy
        assert sent_acc >= g["thresholds"]["sentences_classification_min"]
        assert wo["correct"] == g["thresholds"]["word_order_correct"]
        assert role_probes["accuracy"] >= g["thresholds"]["role_probe_accuracy_min"]

        gm = g["metrics"]
        assert abs(sent_acc - gm["stage_metrics"]["SENTENCES"]["classification_accuracy"]) < 0.05
        assert abs(role_probes["accuracy"] - gm["role_probes"]["accuracy"]) < 0.01


class TestDirect2026PearlGolden:
    def test_directional_binding_golden_is_retracted(self):
        g = _load("direct2026_pearl.json")
        assert "RETRACTED" in g
        assert "metrics" not in g and "thresholds" not in g
        assert "historical_metrics" in g and "historical_thresholds" in g


class TestHoff2026SizeDistGolden:
    """Smoke: package E%-WTA params align with Hoff Table 1 qualitative targets."""

    def test_epwta_policy_within_paper_size_band(self):
        g = _load("hoff2026_size_dist.json")
        from neural_assemblies.compute import EPercentPolicy

        targets = g["repo_smoke_targets"]
        n, k = 1000, 80
        brain = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
        policy = EPercentPolicy(
            fraction_of_max=targets["fraction_of_max"],
            e_fraction=targets["e_fraction"],
            min_winners=targets["min_winners"],
        )
        brain.add_area("M", n, k, 0.1, winner_policy=policy)
        brain.add_stimulus("s", k)
        asm = project(brain, "s", "M", rounds=10)
        size = len(asm)
        paper_median = g["table1_epwta_with_feedforward_inhibition"]["beta_0.1"]["size_median_iqr"][0]
        assert 1 <= size <= max(paper_median * 3, k)


class TestNemo2025FsmMod3Golden:
    def test_golden_is_retracted(self):
        """The golden certified a dictionary lookup, and says so.

        It was recorded while `NemoArcFSM.step_symbol` returned
        `self._table[(from_state, symbol)]`, so its metrics came from the
        transition table rather than the network -- the same values are
        returned by an untrained brain and at beta=0. Its parameters also put
        the arc at kp = 2 against a floor of 22.8. Asserting the retraction
        keeps the file from being quietly restored.
        """
        g = _load("nemo2025_fsm_mod3.json")
        assert "RETRACTED" in g, "the mod-3 golden was retracted; see 891a0db"
        assert "expected" not in g and "metrics" not in g, (
            "retracted keys must stay renamed so no test can consume them")

    @pytest.mark.xfail(strict=True, reason=(
        "A1: the FSM decides 5/10 seeds (4/10 with a correct trajectory). "
        "Single transitions are perfect 330/330; the state assembly drifts "
        "along a sequence. Strict, so this fires and forces the golden to be "
        "re-recorded once drift is fixed."))
    def test_mod3_digit_sum_fsm(self):
        from neural_assemblies.programs.mod3_fsm import run_mod3_fsm_demo

        result = run_mod3_fsm_demo(seed=42, presentations=15)
        assert result.positive_accepted
        assert result.negative_rejected


class TestColt2022MnistBrainGolden:
    def test_brain_mnist_smoke(self):
        g = _load("colt2022_mnist_brain.json")
        from neural_assemblies.programs.colt_mnist_brain import run_colt_mnist_brain

        result = run_colt_mnist_brain(**g["parameters"])
        _require_golden_dataset(result, g)
        tol = g["thresholds"]["metrics_match_tolerance"]
        assert result.mean_accuracy >= g["thresholds"]["mean_accuracy_min"]
        assert abs(result.mean_accuracy - g["metrics"]["mean_accuracy"]) < tol


class TestNemo2025FsmMod3NumpyGolden:
    def test_mod3_fsm_numpy_backend(self):
        g = _load("nemo2025_fsm_mod3_numpy.json")
        from neural_assemblies.reference.nemo_numpy.fsm_network import run_mod3_fsm_numpy

        result = run_mod3_fsm_numpy(**g["parameters"])
        assert result["positive_accepted"] == g["expected"]["positive_accepted"]
        assert result["negative_rejected"] == g["expected"]["negative_rejected"]


class TestColt2022MnistHierarchicalGolden:
    def test_hierarchical_mnist_smoke(self):
        g = _load("colt2022_mnist_hierarchical.json")
        from neural_assemblies.programs.colt_mnist_hierarchical import run_colt_mnist_hierarchical

        result = run_colt_mnist_hierarchical(**g["parameters"])
        _require_golden_dataset(result, g)
        tol = g["thresholds"]["metrics_match_tolerance"]
        assert result.mean_accuracy >= g["thresholds"]["mean_accuracy_min"]
        assert abs(result.mean_accuracy - g["metrics"]["mean_accuracy"]) < tol


class TestColt2022MnistNotebookFullGolden:
    @pytest.mark.slow
    def test_notebook_scale_mnist(self):
        result = _verify_or_skip_missing_dataset("colt2022_mnist_notebook_full")
        assert result.passed, result.diffs


class TestColt2022MnistNotebookPaperGolden:
    @pytest.mark.slow
    def test_paper_scale_mnist_n5000(self):
        result = _verify_or_skip_missing_dataset("colt2022_mnist_notebook_paper")
        assert result.passed, result.diffs


class TestPnas2020ReciprocalGolden:
    def test_reciprocal_restore(self):
        from neural_assemblies.parity.runner import verify_protocol

        result = verify_protocol("pnas2020_reciprocal")
        assert result.passed, result.diffs


class TestPnas2020PatternCompleteGolden:
    def test_pattern_complete_50pct(self):
        from neural_assemblies.parity.runner import verify_protocol

        result = verify_protocol("pnas2020_pattern_complete")
        assert result.passed, result.diffs


class TestSeq25MultiRecallGolden:
    def test_first_item_recall(self):
        from neural_assemblies.parity.runner import verify_protocol

        result = verify_protocol("seq25_multi_recall")
        assert result.passed, result.diffs


class TestTacl2021ParserF1Golden:
    def test_english_role_coverage(self):
        from neural_assemblies.parity.runner import verify_protocol

        result = verify_protocol("tacl2021_parser_f1")
        assert result.passed, result.diffs
