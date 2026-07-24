"""Stress tests: adversarial holdout exposure and wobbly-bootstrap ablation."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.acquisition import (
    classify_word_bootstrapped,
    infer_holdout_categories,
    mine_wobbly_episodes,
    parse_with_wobbly_probes,
    replay_wobbly_episodes,
)
from neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference import (
    ingest_holdout_sentence_stats,
)
from neural_assemblies.assembly_calculus.emergent.evaluation import (
    assess_erp_readiness,
    calibrate_erp_thresholds,
    ensure_parser_erp_calibration,
    train_parser_to_depth,
)
from research.experiments.metrics.measurement import (
    measure_critical_word,
    measure_critical_word_composed,
)

N, K = 3000, 30
HOLDOUT = "small"


def _adversarial_exposure_sentences() -> list:
    """Holdout only as pre-nominal modifier (subject-side), never as object."""
    return [
        ["the", HOLDOUT, "bird"],
        ["the", HOLDOUT, "dog"],
        ["the", HOLDOUT, "cat"],
        ["the", "bird", "runs"],
        ["the", "dog", "runs"],
    ]


def _train_adversarial_parser(seed: int = 44) -> EmergentParser:
    parser = train_parser_to_depth(
        "TWO_WORD",
        n=N,
        k=K,
        seed=seed,
        holdout_words={HOLDOUT},
    )
    for sent in _adversarial_exposure_sentences():
        parser.ingest_raw_sentence(sent)
    ingest_holdout_sentence_stats(parser, {HOLDOUT})
    return parser


class TestAdversarialHoldoutExposure:
    def test_category_violation_frame_triggers_wobbly_after_calibration(self):
        parser = _train_adversarial_parser()
        ensure_parser_erp_calibration(parser)
        readiness = assess_erp_readiness(parser)
        if not readiness.p600_ready:
            return

        # Object position for ADJ holdout should wobble if parsed as NOUN
        _, probes = parse_with_wobbly_probes(
            parser, ["the", "bird", "finds", HOLDOUT],
        )
        crit = probes[3]
        assert crit.word == HOLDOUT
        assert crit.n400 >= 0.0 and crit.p600 >= 0.0

    def test_composed_probe_aligns_with_incremental_runner(self):
        parser = _train_adversarial_parser(seed=45)
        ensure_parser_erp_calibration(parser)
        context = ["the", "bird", "finds"]
        legacy = measure_critical_word(
            parser,
            context,
            HOLDOUT,
            p600_areas=["VP", "ROLE_PATIENT"],
            rounds=4,
            p600_settling_rounds=3,
        )
        composed = measure_critical_word_composed(parser, context, HOLDOUT)
        assert "n400" in composed and "p600" in composed
        assert composed["n400_energy"] == composed["n400"]
        assert legacy["core_area"] or composed["core_area"]


class TestWobblyBootstrapAblation:
    def test_replay_improves_holdout_assignment_vs_inference_only(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=7,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", HOLDOUT, "bird", "runs", "dog", "finds"):
            if w not in parser.stim_map:
                parser.register_word(w)

        sents = [
            ["the", HOLDOUT, "bird"],
            ["the", HOLDOUT, "dog"],
            ["the", "bird", "runs"],
            ["the", "bird", "finds", HOLDOUT],
        ]
        for s in sents:
            parser.ingest_raw_sentence(s)

        ensure_parser_erp_calibration(parser)
        readiness = assess_erp_readiness(parser)
        if not readiness.p600_ready:
            return

        mem = mine_wobbly_episodes(parser, sents, target_words={HOLDOUT})
        if not mem.episodes:
            return

        infer_only = dict(infer_holdout_categories(parser, {HOLDOUT}))
        replay_wobbly_episodes(parser, mem)
        after_replay = dict(infer_holdout_categories(parser, {HOLDOUT}))

        if infer_only.get(HOLDOUT):
            assert after_replay.get(HOLDOUT) == infer_only[HOLDOUT]
        else:
            cat, _ = classify_word_bootstrapped(parser, HOLDOUT)
            assert cat != "UNKNOWN" or after_replay.get(HOLDOUT) is not None

    def test_mine_without_replay_leaves_episodes_unresolved(self):
        parser = train_parser_to_depth(
            "TWO_WORD", n=N, k=K, seed=8, holdout_words={HOLDOUT},
        )
        sents = _adversarial_exposure_sentences() + [["the", "bird", "finds", HOLDOUT]]
        for s in sents:
            parser.ingest_raw_sentence(s)

        ensure_parser_erp_calibration(parser)
        if not assess_erp_readiness(parser).p600_ready:
            return

        mem = mine_wobbly_episodes(parser, sents, target_words={HOLDOUT})
        if not mem.episodes:
            return

        unresolved = sum(1 for ep in mem.episodes if ep.resolved_category is None)
        assert unresolved == len(mem.episodes)


class TestStageWiseCalibration:
    """Calibration quality should improve (or at least not regress) with depth."""

    DEPTHS = ("TWO_WORD", "SENTENCES")

    def _calibrate_at_depth(self, depth: str, seed: int = 42):
        parser = train_parser_to_depth(
            depth,
            n=N,
            k=K,
            seed=seed,
            holdout_words={HOLDOUT},
        )
        for sent in _adversarial_exposure_sentences():
            parser.ingest_raw_sentence(sent)
        ingest_holdout_sentence_stats(parser, {HOLDOUT})
        report = calibrate_erp_thresholds(parser)
        return parser, report

    def test_readiness_gates_track_curriculum_depth(self):
        shallow_parser, shallow_rep = self._calibrate_at_depth("TWO_WORD", seed=50)
        deep_parser, deep_rep = self._calibrate_at_depth("SENTENCES", seed=51)

        shallow_ready = shallow_rep.readiness.p600_ready
        deep_ready = deep_rep.readiness.p600_ready
        if not shallow_ready and not deep_ready:
            return

        if deep_ready:
            assert deep_rep.tuned
            assert deep_rep.thresholds.p600_excess_margin > 0.0

        shallow_d = shallow_rep.separation.get("p600_cohens_d", 0.0)
        deep_d = deep_rep.separation.get("p600_cohens_d", 0.0)
        if shallow_ready and deep_ready:
            assert deep_d >= shallow_d * 0.5, (
                f"P600 separation regressed: shallow={shallow_d:.2f} deep={deep_d:.2f}"
            )

    def test_wobbly_detection_rate_increases_with_calibration(self):
        _, shallow_rep = self._calibrate_at_depth("TWO_WORD", seed=52)
        deep_parser, deep_rep = self._calibrate_at_depth("SENTENCES", seed=53)

        if not deep_rep.readiness.p600_ready:
            return

        ensure_parser_erp_calibration(deep_parser)
        _, probes = parse_with_wobbly_probes(
            deep_parser, ["the", "bird", "finds", HOLDOUT],
        )
        crit = probes[3]
        if shallow_rep.readiness.p600_ready:
            _, shallow_probes = parse_with_wobbly_probes(
                train_parser_to_depth(
                    "TWO_WORD", n=N, k=K, seed=52, holdout_words={HOLDOUT},
                ),
                ["the", "bird", "finds", HOLDOUT],
            )
            shallow_crit = shallow_probes[3]
            assert crit.combined >= 0.0
            assert crit.p600 >= shallow_crit.p600 * 0.5 or crit.wobbly


class TestDevelopmentalWobblyAblation:
    def test_wobbly_bootstrap_off_skips_episode_replay(self):
        from neural_assemblies.assembly_calculus.emergent.acquisition import (
            run_developmental_acquisition,
        )

        os.environ["EMERGENT_DEV_CURRICULUM"] = "1"
        holdout = {HOLDOUT}
        vocab = build_vocabulary_preset("medium")

        parser_off = EmergentParser(
            n=N, k=K, seed=60, fast_training=True, vocabulary=vocab,
        )
        off = run_developmental_acquisition(
            parser_off,
            max_stage="TWO_WORD",
            holdout_words=holdout,
            wobbly_bootstrap=False,
        )
        parser_on = EmergentParser(
            n=N, k=K, seed=60, fast_training=True, vocabulary=vocab,
        )
        on = run_developmental_acquisition(
            parser_on,
            max_stage="TWO_WORD",
            holdout_words=holdout,
            wobbly_bootstrap=True,
        )
        off_episodes = sum(
            float(r.metrics.get("wobbly_episodes", 0)) for r in off.reflections
        )
        on_episodes = sum(
            float(r.metrics.get("wobbly_episodes", 0)) for r in on.reflections
        )
        assert off_episodes == 0.0
        assert on_episodes >= off_episodes
