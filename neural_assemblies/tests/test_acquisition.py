"""Tests for ground-up developmental acquisition (babble -> grammar)."""

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"


@pytest.fixture(autouse=True)
def _developmental_curriculum(monkeypatch):
    """Scope the curriculum switch to THIS module's tests.

    THIS LINE USED TO RUN AT IMPORT: `os.environ["EMERGENT_DEV_CURRICULUM"]="1"`.
    pytest imports every collected module before running anything, so
    `pytest neural_assemblies/tests/` set it for the WHOLE SESSION and every
    parser trained afterwards used a different corpus (the flag disables the
    preset skip, so babble + early grammar always run). Running the same files
    by explicit path never imported this module and so never saw it.

    That single line produced a full day of wrong diagnoses: 4 ERP tests failed
    under `pytest tests/` and passed under `pytest <files>`, which was read
    first as cross-test state leakage, then as a cached-vs-fresh parser defect.
    Only cold runs showed it, because a warm run DESERIALIZES a parser instead
    of training one -- which is what made the cache look causal.

    `monkeypatch` restores the previous value after each test, so importing this
    file can no longer reconfigure anybody else's training. The cache key also
    now includes the flag (`sweep._TRAINING_ENV_VARS`) so the two curricula can
    never share an entry, in memory or on disk -- belt and braces, because this
    module is not the only place that could set it.
    """
    monkeypatch.setenv("EMERGENT_DEV_CURRICULUM", "1")

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.acquisition import (
    DEVELOPMENTAL_STAGE_ORDER,
    MisclassificationTarget,
    acquisition_report_to_dict,
    build_adaptive_plan,
    build_remedial_sentences,
    evaluate_stage_gate,
    format_acquisition_report,
    reflect_after_stage,
    run_developmental_acquisition,
)
from neural_assemblies.assembly_calculus.emergent.acquisition.babble import (
    register_early_fuzzy_variants,
    train_babble_stage,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
    StageResult,
)
from neural_assemblies.assembly_calculus.emergent.acquisition.phonology import (
    fuzzy_variants,
    normalize_surface_form,
)
from neural_assemblies.assembly_calculus.emergent.training.perf import (
    developmental_curriculum_enabled,
    should_skip_early_curriculum,
)

N, K = 3000, 30


class TestPhonology:
    def test_fuzzy_variants_produce_misspellings(self):
        variants = fuzzy_variants("dog")
        assert "dog" not in variants
        assert len(variants) >= 1

    def test_normalize_surface_form(self):
        mapping = {"dogg": "dog", "dgg": "dog"}
        assert normalize_surface_form("dogg", mapping) == "dog"
        assert normalize_surface_form("cat", mapping) == "cat"


class TestBabble:
    def test_fuzzy_registration_rejects_unused_seed(self):
        with pytest.raises(ValueError, match="does not use seed"):
            register_early_fuzzy_variants(object(), [], seed=1)

    def test_babble_stage_registers_forms(self):
        parser = EmergentParser(n=N, k=K, seed=1, fast_training=True)
        report = train_babble_stage(parser, n_forms=12, n_utterances=10, seed=1)
        assert report.n_forms == 12
        assert len(parser.babble_forms) == 12
        assert all(f in parser.stim_map for f in parser.babble_forms)


class TestDevelopmentalCurriculum:
    def test_dev_curriculum_enabled_in_tests(self):
        assert developmental_curriculum_enabled()
        assert not should_skip_early_curriculum(500, "DIALOGUE")

    def test_developmental_order_starts_with_babble(self):
        assert DEVELOPMENTAL_STAGE_ORDER[0] == "BABBLE"

    def test_run_developmental_acquisition_to_first_words(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=2, vocabulary=vocab, fast_training=True,
        )
        report = run_developmental_acquisition(
            parser,
            max_stage="FIRST_WORDS",
            seed=2,
        )
        assert report.stages_run[0] == "BABBLE"
        assert "FIRST_WORDS" in report.stages_run
        assert not report.skipped_early
        assert report.babble_forms >= 12
        text = format_acquisition_report(report)
        assert "BABBLE" in text
        assert "observe:" in text

    def test_fuzzy_surface_registered_at_first_words(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=3, vocabulary=vocab, fast_training=True,
        )
        report = run_developmental_acquisition(
            parser,
            max_stage="FIRST_WORDS",
            seed=3,
        )
        assert report.fuzzy_variant_count >= 1
        assert hasattr(parser, "surface_to_canonical")

    def test_register_fuzzy_surface_resolves(self):
        parser = EmergentParser(n=N, k=K, seed=4, fast_training=True)
        parser.register_word("dog")
        parser.register_fuzzy_surface("dog", "dogg")
        assert parser.resolve_surface_word("dogg") == "dog"
        assert parser.word_grounding.get("dogg") == parser.word_grounding.get("dog")


class TestAdaptiveCurriculum:
    def test_build_remedial_adj_frames(self):
        parser = EmergentParser(n=N, k=K, seed=10, fast_training=True)
        for w in ("the", "dog", "runs", "big"):
            parser.register_word(w)
        targets = [
            MisclassificationTarget(word="big", expected="ADJ", predicted="NOUN"),
        ]
        sents = build_remedial_sentences(parser, targets, sentences_per_target=3, seed=1)
        assert len(sents) >= 3
        for s in sents:
            assert "big" in s.words
            idx = s.words.index("big")
            if idx >= 1 and s.words[idx - 1] in ("the", "a"):
                assert idx + 1 < len(s.words)

    def test_build_adaptive_plan_from_hints(self):
        parser = EmergentParser(n=N, k=K, seed=11, fast_training=True)
        reflection = reflect_after_stage(
            parser,
            StageResult(
                stage_name="SENTENCES",
                vocab_size=50,
                classification_accuracy=0.6,
                beta=0.1,
                sentences_trained=40,
                phases_run=["lexicon", "distributional"],
            ),
            holdout_words={"small"},
        )
        assert any(h.action == "remedial_pos" for h in reflection.adaptive_hints)
        trainer = CurriculumTrainer(parser)
        plan = build_adaptive_plan(
            reflection,
            parser,
            trainer._get_stage_words("FIRST_WORDS"),
            holdout_words={"small"},
        )
        assert "distributional" in plan.phases

    def test_adaptive_remediation_runs_after_stage(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=12, vocabulary=vocab, fast_training=True,
        )
        report = run_developmental_acquisition(
            parser,
            max_stage="FIRST_WORDS",
            seed=12,
            adaptive=True,
        )
        assert report.remediations or any(
            r.adaptive_hints for r in report.reflections
        )
        text = format_acquisition_report(report)
        assert "FIRST_WORDS" in text

    def test_adaptive_can_be_disabled(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=13, vocabulary=vocab, fast_training=True,
        )
        report = run_developmental_acquisition(
            parser,
            max_stage="FIRST_WORDS",
            seed=13,
            adaptive=False,
        )
        assert report.remediations == []
        assert all(r.remediation is None for r in report.reflections)


class TestStageGates:
    def test_babble_gate_passes_after_training(self):
        parser = EmergentParser(n=N, k=K, seed=20, fast_training=True)
        train_babble_stage(parser, n_forms=12, seed=20)
        result = StageResult(
            stage_name="BABBLE",
            vocab_size=12,
            classification_accuracy=0.0,
            beta=0.2,
            sentences_trained=0,
            phases_run=["babble"],
        )
        gate = evaluate_stage_gate(parser, result)
        assert gate.passed
        assert gate.checks["babble_forms"]

    def test_two_word_gate_requires_roles_phase(self):
        parser = EmergentParser(n=N, k=K, seed=21, fast_training=True)
        missing = StageResult(
            stage_name="TWO_WORD",
            vocab_size=100,
            classification_accuracy=0.5,
            beta=0.1,
            sentences_trained=50,
            phases_run=["lexicon", "distributional"],
        )
        gate = evaluate_stage_gate(parser, missing)
        assert not gate.passed
        assert "roles" in gate.failures[0]

    def test_acquisition_report_serializes_gates(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=22, vocabulary=vocab, fast_training=True,
        )
        report = run_developmental_acquisition(
            parser,
            max_stage="FIRST_WORDS",
            seed=22,
            gate_enforcement=True,
        )
        payload = acquisition_report_to_dict(report)
        assert payload["stages_run"][0] == "BABBLE"
        assert payload["stages"][-1]["gate"]["passed"] is True

    def test_no_remedial_when_classification_skipped(self):
        import os
        from neural_assemblies.assembly_calculus.emergent.acquisition.adaptive import (
            classification_accuracy_usable,
        )

        assert not classification_accuracy_usable(-1.0)
        os.environ["EMERGENT_SWEEP_MODE"] = "1"
        try:
            vocab = build_vocabulary_preset("core")
            parser = EmergentParser(
                n=N, k=K, seed=23, vocabulary=vocab, fast_training=True,
            )
            report = run_developmental_acquisition(
                parser,
                max_stage="FIRST_WORDS",
                seed=23,
                adaptive=True,
            )
            total_remedial = sum(
                float(r.metrics.get("remedial_sentences", 0.0))
                for r in report.reflections
            )
            assert total_remedial == 0.0
        finally:
            os.environ.pop("EMERGENT_SWEEP_MODE", None)


class TestCDSCorpus:
    def test_first_words_uses_stage1_corpus(self):
        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=30, vocabulary=vocab, fast_training=True,
        )
        trainer = CurriculumTrainer(parser)
        words = trainer._get_stage_words("FIRST_WORDS")
        for w in words[:20]:
            parser.register_word(w.lemma)
        sentences = trainer.generation.generate(
            words, 1, stage_name="FIRST_WORDS",
        )
        flat = {" ".join(s.tokens) for s in sentences}
        assert "ball" in flat
        assert "more milk" in flat or "my ball" in flat


@pytest.mark.slow
class TestDevelopmentalSentences:
    def test_developmental_acquisition_to_sentences(self):
        from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
            default_holdout_set,
        )

        vocab = build_vocabulary_preset("core")
        parser = EmergentParser(
            n=N, k=K, seed=42, vocabulary=vocab, fast_training=True,
        )
        report = run_developmental_acquisition(
            parser,
            max_stage="SENTENCES",
            holdout_words=default_holdout_set(),
            seed=42,
            gate_enforcement=True,
        )
        assert "SENTENCES" in report.stages_run
        assert report.blocked_at_stage is None
        assert report.final_generalization is not None
        metrics = report.final_generalization
        assert metrics["novel_composition"]["accuracy"] >= 0.33
        assert (
            metrics["holdout_decomposition"]["accuracy_bootstrapped"] >= 0.50
        )
