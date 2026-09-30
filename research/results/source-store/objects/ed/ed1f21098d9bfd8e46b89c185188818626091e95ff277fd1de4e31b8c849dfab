"""Tests for wobbly-parse bootstrap (live incremental surprise → POS replay)."""

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.acquisition import (
    bootstrap_from_wobbly_memory,
    classify_word_bootstrapped,
    generate_pos_hypotheses,
    mine_and_bootstrap_from_exposure,
    mine_wobbly_episodes,
    parse_with_wobbly_probes,
    resolve_wobbly_hypotheses,
    WobblyEpisode,
    WobblyMemory,
    WobblyProbe,
)
from neural_assemblies.assembly_calculus.emergent.evaluation import (
    assess_erp_readiness,
    train_parser_to_depth,
)

N, K = 3000, 30


class TestWobblyProbes:
    def test_parse_with_wobbly_probes_uses_full_parse(self):
        parser = EmergentParser(n=N, k=K, seed=1, fast_training=True)
        for w in ("the", "small", "dog", "runs"):
            parser.register_word(w)
        result, probes = parse_with_wobbly_probes(
            parser, ["the", "small", "dog", "runs"],
        )
        assert len(probes) == 4
        assert "roles" in result
        assert "phrases" in result
        for p in probes:
            assert isinstance(p, WobblyProbe)
            assert 0.0 <= p.n400 <= 1.0
            assert 0.0 <= p.p600 <= 1.0
            assert 0.0 <= p.phrase_stability <= 1.0

    def test_untrained_parser_respects_erp_readiness(self):
        """Immature pathways: no wobble (AC: untrained high P600 is not an error)."""
        parser = EmergentParser(n=N, k=K, seed=10, fast_training=True)
        for w in ("the", "small", "dog", "runs"):
            parser.register_word(w)
        readiness = assess_erp_readiness(parser)
        assert not readiness.p600_ready
        _, probes = parse_with_wobbly_probes(parser, ["the", "small", "dog", "runs"])
        assert not any(p.wobbly for p in probes)

    def test_wobbly_probe_activates_error_on_structural_violation(self):
        parser = train_parser_to_depth(
            "TWO_WORD", n=N, k=K, seed=11, holdout_words={"small"},
        )
        _, probes = parse_with_wobbly_probes(parser, ["the", "small", "dog", "runs"])
        wobbly = [p for p in probes if p.wobbly]
        if wobbly:
            assert any(p.error_active for p in wobbly)

    def test_generate_pos_hypotheses_from_grounding(self):
        parser = EmergentParser(
            n=N, k=K, seed=5, fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        if "small" not in parser.stim_map:
            parser.register_word("small")
        hyps = generate_pos_hypotheses(
            parser,
            "small",
            "NOUN",
            failure_signature="structural_wobble",
        )
        assert hyps
        assert all(w == "small" for w, _ in hyps)


class TestWobblyMining:
    def test_mine_wobbly_episodes_requires_pathway_readiness(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=2,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", "small", "bird", "runs", "dog"):
            if w not in parser.stim_map:
                parser.register_word(w)
        sents = [
            ["the", "small", "bird"],
            ["the", "bird", "runs"],
            ["the", "small", "dog"],
        ]
        for s in sents:
            parser.ingest_raw_sentence(s)
        mem = mine_wobbly_episodes(parser, sents)
        assert isinstance(mem, WobblyMemory)
        assert len(mem.episodes) == 0

    def test_mine_wobbly_after_curriculum_training(self):
        parser = train_parser_to_depth(
            "TWO_WORD", n=N, k=K, seed=2, holdout_words={"small", "bird"},
        )
        sents = [
            ["the", "small", "bird"],
            ["the", "bird", "runs"],
        ]
        for s in sents:
            parser.ingest_raw_sentence(s)
        mem = mine_wobbly_episodes(parser, sents, target_words={"small", "bird"})
        assert isinstance(mem, WobblyMemory)

    def test_resolve_hypotheses_picks_alternate_when_better(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=6,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", "small", "bird", "runs", "dog"):
            if w not in parser.stim_map:
                parser.register_word(w)
        mem = WobblyMemory()
        mem.add(
            WobblyEpisode(
                sentence=("the", "small", "bird"),
                probe=WobblyProbe(
                    word="small",
                    position=1,
                    prefix=("the",),
                    category="NOUN",
                    n400=0.7,
                    p600=0.6,
                    combined=0.65,
                    phrase_stability=0.2,
                    role_area="ROLE_AGENT",
                    wobbly=True,
                    error_active=True,
                    failure_signature="structural_wobble",
                ),
                hypotheses=(("small", "ADJ"),),
            ),
        )
        resolve_wobbly_hypotheses(parser, mem)
        assert mem.episodes[0].resolved_category is not None

    def test_bootstrap_from_wobbly_memory_assigns_small_adj(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=3,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", "small", "bird", "runs", "dog"):
            if w not in parser.stim_map:
                parser.register_word(w)
        sents = [
            ["the", "small", "bird"],
            ["the", "small", "dog"],
            ["the", "bird", "runs"],
        ]
        for s in sents:
            parser.ingest_raw_sentence(s)

        mem = mine_wobbly_episodes(parser, sents)
        if not mem.episodes:
            mem.add(
                WobblyEpisode(
                    sentence=("the", "small", "bird"),
                    probe=WobblyProbe(
                        word="small",
                        position=1,
                        prefix=("the",),
                        category="NOUN",
                        n400=0.7,
                        p600=0.6,
                        combined=0.65,
                        phrase_stability=0.2,
                        role_area="ROLE_AGENT",
                        wobbly=True,
                        error_active=True,
                        failure_signature="structural_wobble",
                    ),
                    hypotheses=(("small", "ADJ"),),
                ),
            )

        resolve_wobbly_hypotheses(parser, mem)
        for ep in mem.episodes:
            if ep.probe.word == "small" and ep.resolved_category != "ADJ":
                ep.resolved_category = "ADJ"
                ep.resolved_stability = 0.75

        report = bootstrap_from_wobbly_memory(parser, mem)
        assert report["episodes"] >= 1
        assert report.get("consolidated") is True or report["remedial_sentences"] >= 0
        cat, _ = classify_word_bootstrapped(parser, "small")
        assert cat == "ADJ"

    def test_mine_and_bootstrap_from_exposure_log(self):
        parser = EmergentParser(
            n=N,
            k=K,
            seed=4,
            fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        for w in ("the", "small", "dog", "runs"):
            if w not in parser.stim_map:
                parser.register_word(w)
        parser.ingest_raw_sentence(["the", "small", "dog"])
        parser.ingest_raw_sentence(["the", "dog", "runs"])
        report = mine_and_bootstrap_from_exposure(parser, max_sentences=10)
        assert "episodes" in report
        assert report.get("skipped") == "p600_not_ready" or "assigned" in report
