"""Tests for robust generalization measurement (holdout lexicon, novel composition)."""

import pytest

# Measured at 468s -- the heaviest file in the suite, dominated by
# run_curriculum_depth_sweep training two extra depths at seed=80. Tagged so the
# inner loop can use `-m "not slow"`; CI should still run the full suite.
# NOTE: this tier is populated from MEASURED timings only. Other files may
# deserve the tag but were not tagged on a guess.
pytestmark = pytest.mark.slow

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    DEFAULT_LEXICON_HOLDOUTS,
    NOVEL_COMPOSITION_PROBES,
    collect_bridge_probes,
    default_holdout_set,
    evaluate_generalization_metrics,
    format_curriculum_sweep_table,
    is_word_in_lexicon,
    run_curriculum_depth_sweep,
    run_generalization_gate,
    summarize_curriculum_sweep,
    train_dialogue_with_holdouts,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
    build_dialogue_stage_schedule,
    exact_training_mode,
)

N, K, ROUNDS = 3000, 30, 6


def _train_dialogue_compiled(*, seed: int = 70, holdout: bool = True) -> EmergentParser:
    parser = EmergentParser(
        n=N,
        k=K,
        seed=seed,
        rounds=ROUNDS,
        fast_training=True,
        vocabulary=build_vocabulary_preset("medium"),
    )
    schedule = build_dialogue_stage_schedule(parser, seed=seed)
    if holdout:
        train_dialogue_with_holdouts(parser, schedule, default_holdout_set())
    else:
        from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
            run_dialogue_stage,
        )

        run_dialogue_stage(parser, schedule)
    return parser


class TestGeneralizationMetrics:
    def test_report_contains_core_fields(self):
        parser = _train_dialogue_compiled(seed=70)
        report = evaluate_generalization_metrics(parser, max_bridge_probes=10)

        for key in (
            "lexicon_holdout",
            "roles",
            "novel_composition",
            "word_order",
            "dialogue",
            "bridge_oov",
            "bridge_seen",
            "composite",
        ):
            assert key in report, f"missing {key}"

        assert report["lexicon_holdout"]["total"] == len(DEFAULT_LEXICON_HOLDOUTS)
        assert 0.0 <= report["composite"] <= 1.0

    def test_holdout_words_not_in_lexicon_after_dialogue(self):
        parser = _train_dialogue_compiled(seed=71)
        for word in default_holdout_set():
            assert not is_word_in_lexicon(parser, word), (
                f"holdout {word!r} should not appear in core lexicons"
            )

    def test_lexicon_holdout_accuracy_floor(self):
        parser = _train_dialogue_compiled(seed=72)
        report = evaluate_generalization_metrics(parser)
        acc = report["lexicon_holdout"]["accuracy"]
        assert acc >= 0.66, (
            f"holdout classification {acc:.1%} < 66%: {report['lexicon_holdout']}"
        )

    def test_holdout_noun_classifies_via_grounding(self):
        parser = _train_dialogue_compiled(seed=73)
        grounding = parser.word_grounding["bird"]
        cat, _ = parser.classify_word("bird", grounding=grounding)
        assert cat == "NOUN", f"'bird' classified as {cat!r}, expected NOUN"

    def test_holdout_verb_classifies_via_grounding(self):
        parser = _train_dialogue_compiled(seed=74)
        grounding = parser.word_grounding["finds"]
        cat, _ = parser.classify_word("finds", grounding=grounding)
        assert cat == "VERB", f"'finds' classified as {cat!r}, expected VERB"

    def test_novel_composition_metric_recorded(self):
        parser = _train_dialogue_compiled(seed=75)
        report = evaluate_generalization_metrics(parser)
        novel = report["novel_composition"]
        assert "accuracy" in novel
        assert 0.0 <= novel["accuracy"] <= 1.0

    def test_oov_bridge_probes_collected_when_holdout_in_corpus(self):
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )
        from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
            create_training_sentences,
        )

        parser = _train_dialogue_compiled(seed=76)
        idx = compile_corpus(parser, create_training_sentences())
        prefixes, expected = collect_bridge_probes(
            idx, parser, default_holdout_set(), oov_only=True, max_probes=20,
        )
        assert len(prefixes) >= 1, "expected OOV bridge probes from training corpus"
        assert all(exp[0] in default_holdout_set() for exp in expected)

    def test_bridge_seen_metrics_computed(self):
        parser = _train_dialogue_compiled(seed=77)
        report = evaluate_generalization_metrics(parser, max_bridge_probes=15)
        seen = report["bridge_seen"]
        assert seen["total"] >= 1
        assert 0.0 <= seen["top5"] <= 1.0

    def test_word_order_reported_after_dialogue(self):
        parser = _train_dialogue_compiled(seed=78)
        report = evaluate_generalization_metrics(parser)
        wo = report["word_order"]
        assert wo["inferred"] in ("SVO", "SOV", "VSO")
        assert 0.0 <= wo["confidence"] <= 1.0


class TestGeneralizationGate:
    def test_compiled_preserves_holdout_generalization(self):
        report = run_generalization_gate(
            n=N, k=K, seed=60, max_bridge_probes=12, qa_subset=8,
        )
        assert report["lexicon_holdout_delta"] >= -0.34, (
            f"compiled holdout regressed vs exact: {report}"
        )
        assert report["composite_delta"] >= -0.30, (
            f"compiled composite generalization regressed: {report}"
        )

    def test_exact_baseline_holdout_still_learns(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=79,
            rounds=ROUNDS,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        schedule = build_dialogue_stage_schedule(parser, seed=79)
        parser._compiled_training_enabled = False
        parser.brain.projection_fidelity = "exact"
        with exact_training_mode():
            train_dialogue_with_holdouts(parser, schedule, default_holdout_set())

        report = evaluate_generalization_metrics(parser, max_bridge_probes=10)
        decomp = report["holdout_decomposition"]
        assert decomp["accuracy_bootstrapped"] >= 0.66

    def test_holdout_decomposition_identifies_signals(self):
        parser = _train_dialogue_compiled(seed=81)
        from neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference import (
            decompose_holdout_classification,
            format_holdout_decomposition,
        )

        decomp = decompose_holdout_classification(parser)
        assert "per_word" in decomp
        assert "bird" in decomp["per_word"]
        for word in decomp["per_word"]:
            info = decomp["per_word"][word]
            assert "failure_mode" in info
            assert "bootstrapped" in info
        text = format_holdout_decomposition(decomp)
        assert "bootstrapped" in text

    def test_sentences_holdout_bootstrap_floor(self, forked_parser):
        parser = forked_parser("SENTENCES", seed=42)
        from neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference import (
            decompose_holdout_classification,
        )

        decomp = decompose_holdout_classification(parser)
        assert decomp["accuracy_bootstrapped"] >= 1.0, (
            f"bootstrap holdout should reach 100% at SENTENCES: {decomp}"
        )


class TestCurriculumDepthSweep:
    def test_sweep_report_structure(self):
        sweep = run_curriculum_depth_sweep(
            depths=["TWO_WORD", "FULL_TRAIN"],
            n=N,
            k=K,
            seed=80,
            max_bridge_probes=8,
            qa_subset=6,
        )
        assert "depths" in sweep
        assert "summary" in sweep
        assert set(sweep["depths"]) == {"TWO_WORD", "FULL_TRAIN"}
        summary = summarize_curriculum_sweep(sweep)
        assert "rows" in summary
        table = format_curriculum_sweep_table(sweep)
        assert "TWO_WORD" in table
        assert "FULL_TRAIN" in table

    def test_full_train_novel_bird_chases_boy(self, forked_parser):
        parser = forked_parser("FULL_TRAIN", seed=42)
        roles = parser.parse(NOVEL_COMPOSITION_PROBES[0]["words"])["roles"]
        assert roles.get("bird") == "AGENT", roles
        assert roles.get("boy") == "PATIENT", roles

    def test_depth_improves_novel_composition_vs_dialogue_fast(self, forked_parser):
        dialogue = forked_parser("DIALOGUE_FAST", seed=42)
        full = forked_parser("FULL_TRAIN", seed=42)
        d_metrics = evaluate_generalization_metrics(dialogue, max_bridge_probes=10)
        f_metrics = evaluate_generalization_metrics(full, max_bridge_probes=10)
        assert f_metrics["novel_composition"]["accuracy"] >= (
            d_metrics["novel_composition"]["accuracy"]
        ), (
            "FULL_TRAIN should not trail DIALOGUE_FAST on novel composition: "
            f"full={f_metrics['novel_composition']} "
            f"dialogue={d_metrics['novel_composition']}"
        )

    def test_sentences_depth_reports_metrics(self, forked_parser):
        parser = forked_parser("SENTENCES", seed=42)
        report = evaluate_generalization_metrics(parser, max_bridge_probes=8)
        assert report["roles"]["accuracy"] >= 0.0
        assert report["lexicon_holdout"]["accuracy"] >= 0.66
