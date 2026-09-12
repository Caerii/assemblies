"""Tests for emergent training performance optimizations."""

import time

import pytest

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import PREDICTION
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.training.perf import (
    budget_rounds,
    dedupe_grounded_sentences,
    effective_stage_phases,
    resolve_engine,
    sequence_rounds_per_step,
    stage_distributional_reps,
    stage_training_rounds,
)

N, K, ROUNDS = 3000, 30, 6


class TestTrainingPerf:
    def test_holdout_boost_rejects_removed_transition_cache(self):
        from neural_assemblies.assembly_calculus.emergent.curriculum.holdout_bridges import (
            train_holdout_bridge_boost,
        )

        with pytest.raises(TypeError, match="transition_cache"):
            train_holdout_bridge_boost(object(), set(), transition_cache=None)

    def test_resolve_engine_returns_string(self):
        name = resolve_engine("auto", n_hint=10_000)
        assert isinstance(name, str)
        assert name

    def test_resolve_engine_prefers_torch_at_scale(self, monkeypatch):
        monkeypatch.setattr(
            "neural_assemblies.core.backend._detect_torch_cuda",
            lambda: True,
        )
        monkeypatch.setattr(
            "neural_assemblies.core.engine.list_engines",
            lambda: ["numpy_sparse", "torch_sparse"],
        )
        assert resolve_engine("auto", n_hint=1_000_000) == "torch_sparse"
        assert resolve_engine("auto", n_hint=10_000) == "numpy_sparse"

    def test_resolve_engine_force_gpu(self, monkeypatch):
        monkeypatch.setenv("ASSEMBLIES_FORCE_GPU", "1")
        monkeypatch.setattr(
            "neural_assemblies.core.engine.list_engines",
            lambda: ["numpy_sparse", "torch_sparse"],
        )
        # resolve_engine's force-GPU branch verifies the engine actually loads
        # via ensure_engine (which imports torch). torch is not installed in
        # this environment, so ensure_engine("torch_sparse") returns False and
        # selection correctly falls through to numpy_sparse. Mock ensure_engine
        # to simulate a working GPU engine -- this is what "GPU available" means
        # to resolve_engine, not merely being listed in the registry.
        monkeypatch.setattr(
            "neural_assemblies.core.engine.ensure_engine",
            lambda name: name in ("numpy_sparse", "torch_sparse"),
        )
        assert resolve_engine("auto", n_hint=10_000) == "torch_sparse"

    def test_classify_word_cached_skips_full_readout(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        cat1, _ = parser.classify_word_cached("dog")
        cat2, _ = parser.classify_word_cached("dog")
        assert cat1 == cat2 == "NOUN"
        assert "dog" in parser._category_cache

    def test_train_lexicon_skip_known(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        count_before = sum(len(v) for v in parser.core_lexicons.values())
        parser.train_lexicon(skip_known=True)
        count_after = sum(len(v) for v in parser.core_lexicons.values())
        assert count_after == count_before

    def test_prediction_lexicon_cached(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        parser._ensure_prediction_lexicon(["dog", "cat"])
        assert "dog" in parser.prediction_lexicon
        snap = parser.prediction_lexicon["dog"]
        parser._ensure_prediction_lexicon(["dog", "cat"])
        assert parser.prediction_lexicon["dog"] is snap

    def test_parse_incremental_uses_category_cache(self):
        parser = EmergentParser(
            n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)
        words = ["the", "dog", "chases", "the", "cat"]

        parser._category_cache.clear()
        parser.parse_incremental(words, reset=True)
        for w in words:
            assert w in parser._category_cache

        calls: list = []
        original = parser.classify_word

        def counting_classify(word, grounding=None):
            calls.append(word)
            return original(word, grounding=grounding)

        parser.classify_word = counting_classify  # type: ignore[method-assign]
        parser.parse_incremental(words, reset=True)
        assert calls == []

    def test_stage_distributional_reps_fast_mode(self):
        assert stage_distributional_reps("SENTENCES", fast=False) == 3
        assert stage_distributional_reps("SENTENCES", fast=True) == 2
        assert stage_distributional_reps("FIRST_WORDS", fast=True) == 1

    def test_budget_rounds_fast_bridge(self):
        train_r, infer_r, bridge_r = budget_rounds(10, fast=True)
        assert bridge_r == max(2, train_r // 2)
        assert bridge_r <= infer_r or bridge_r == max(2, train_r // 2)

    def test_sequence_rounds_per_step_fast(self):
        assert sequence_rounds_per_step(6, 3, fast=True) == 3
        assert sequence_rounds_per_step(6, 3, fast=False) == 6

    def test_dedupe_grounded_sentences(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        duped = sents + sents[:3]
        out = dedupe_grounded_sentences(duped, parser.stim_map)
        assert len(out) <= len(duped)
        assert len(out) == len(sents)

    def test_direct_context_matches_categories(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        words = ["the", "dog", "chases", "the", "cat"]
        direct = parser.build_context_incremental(words, reset=True, direct=True)
        fiber = parser.build_context_incremental(words, reset=True, direct=False)
        assert direct["categories"] == fiber["categories"]

    def test_stage_training_rounds_fast(self):
        assert stage_training_rounds("DIALOGUE", fast=False) == 7
        assert stage_training_rounds("DIALOGUE", fast=True) == 4

    def test_effective_stage_phases_fast_dialogue(self):
        phases = [
            "lexicon", "distributional", "roles", "phrases",
            "word_order", "mood", "prediction", "dialogue",
        ]
        out = effective_stage_phases("DIALOGUE", phases, fast=True)
        assert "word_order" not in out
        assert "prediction" in out

    def test_ingest_index_stats(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        idx = parser.compile_corpus(sents)
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            ingest_index_stats,
        )
        before = parser.dist_stats.sentences_seen
        ingest_index_stats(parser, idx)
        assert parser.dist_stats.sentences_seen == before + len(idx.sentences)

    def test_distributional_from_index(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        idx = parser.compile_corpus(sents)
        parser.train_distributional_from_index(idx, repetitions=1)
        assert parser.dist_stats.sentences_seen >= len(idx.sentences)

    def test_compile_corpus_builds_transitions(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        idx = parser.compile_corpus(sents)
        assert idx.max_sentence_length >= 2
        assert len(idx.transitions) > 0
        assert len(idx.corpus_vocab) > 0
        total_bridges = sum(
            max(0, len([w for w in s.words if w in parser.stim_map]) - 1)
            for s in sents
        )
        assert len(idx.transitions) <= total_bridges

    def test_adaptive_rounds_high_freq(self):
        from neural_assemblies.assembly_calculus.emergent.training.perf import (
            adaptive_rounds,
        )
        assert adaptive_rounds(5, 25) <= 3
        assert adaptive_rounds(5, 1) == 5

    def test_train_unsupervised_uses_cached_classify(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        from neural_assemblies.assembly_calculus.emergent.core.grounding import GroundingContext
        from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
            GroundedSentence,
            create_training_sentences,
        )
        gs = [
            GroundedSentence(
                words=s.words,
                contexts=[GroundingContext()] * len(s.words),
                roles=[None] * len(s.words),
            )
            for s in create_training_sentences()[:5]
        ]
        for s in gs:
            for w in s.words:
                parser.classify_word_cached(w)
        calls: list = []
        orig = parser.classify_word

        def counting(word, grounding=None):
            calls.append(word)
            return orig(word, grounding=grounding)

        parser.classify_word = counting  # type: ignore[method-assign]
        parser.train_unsupervised(gs, repetitions=1)
        assert calls == []

    def test_train_next_token_avoids_connectome_reset(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        parser._ensure_prediction_lexicon(["dog", "cat"])

        resets = []
        orig = parser.brain._engine.reset_area_connections

        def tracking_reset(area):
            resets.append(area)
            return orig(area)

        parser.brain._engine.reset_area_connections = tracking_reset
        sents = create_training_sentences()[:5]
        parser.train_next_token(sents)
        assert PREDICTION not in resets

    def test_build_context_incremental_light(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        words = ["the", "dog", "chases"]
        light = parser.build_context_incremental(words, reset=True)
        full = parser.parse_incremental(words, reset=True)
        assert light["categories"] == full["categories"]
        assert "roles" not in light or light.get("roles") is None or light.get("roles") == {}

    def test_light_parse_skips_roles(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        words = ["the", "dog", "chases", "the", "cat"]
        result = parser.parse_incremental(words, reset=True, light=True)
        assert result["categories"]["dog"] == "NOUN"
        assert result["roles"] == {}
        assert result["phrases"] == {}

    def test_train_next_token_faster_after_prediction_fix(self):
        parser = EmergentParser(
            n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        t0 = time.perf_counter()
        parser.train_next_token(sents)
        elapsed = time.perf_counter() - t0
        assert elapsed < 25.0, f"train_next_token too slow: {elapsed:.1f}s"

    def test_train_for_agent_still_passes_smoke(self):
        parser = EmergentParser(
            n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True,
        )
        parser.train_for_agent(include_blocks=False)
        preds = parser.predict_next(["the", "dog"])
        assert isinstance(preds, list)

    def test_transition_cache_skips_retrained_bridges(self):
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            TransitionCache,
            compile_corpus,
        )

        parser = EmergentParser(
            n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()[:8]
        idx = compile_corpus(parser, sents)
        cache = TransitionCache()

        parser.train_next_token(sents, corpus_index=idx, transition_cache=cache)
        assert len(cache._trained) == len(idx.transitions)

        import time
        t0 = time.perf_counter()
        parser.train_next_token(sents, corpus_index=idx, transition_cache=cache)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"retrain should be no-op, took {elapsed:.2f}s"

    def test_bridge_vocab_smaller_than_stim_map(self):
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )

        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        idx = compile_corpus(parser, sents)
        assert len(idx.bridge_vocab) <= len(idx.corpus_vocab)
        assert len(idx.bridge_vocab) > 0

    def test_context_ring_reduces_expand_during_bridges(self):
        sents = create_training_sentences()

        def count_expands(parser) -> int:
            expand_n = {"n": 0}
            orig = parser.brain._engine._expand_connectomes

            def counting(*args, **kwargs):
                expand_n["n"] += 1
                return orig(*args, **kwargs)

            parser.brain._engine._expand_connectomes = counting
            parser.train_next_token(sents)
            return expand_n["n"]

        # compiled optimization is active only off norm_init (Brain default disables it)
        base = EmergentParser(
            n=N, k=K, seed=42, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        base.train_lexicon(skip_known=False)
        base._context_ring_capacity_cols = 0
        base._enable_context_ring_mode = lambda _c: None  # type: ignore[method-assign]
        base._enable_prediction_ring_mode = lambda _c: None  # type: ignore[method-assign]
        without_ring = count_expands(base)

        ringed = EmergentParser(
            n=N, k=K, seed=43, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        ringed.train_lexicon(skip_known=False)
        with_ring = count_expands(ringed)

        assert with_ring <= without_ring * 1.15, (
            f"context ring expand variance: {with_ring} vs {without_ring}"
        )
        assert ringed._context_ring_capacity_cols > 0

    def test_prediction_ring_reduces_expand_during_bridges(self):
        sents = create_training_sentences()

        def count_expands(parser, disable_pred_ring=False) -> int:
            expand_n = {"n": 0}
            orig = parser.brain._engine._expand_connectomes

            def counting(*args, **kwargs):
                expand_n["n"] += 1
                return orig(*args, **kwargs)

            parser.brain._engine._expand_connectomes = counting
            if disable_pred_ring:
                parser._enable_prediction_ring_mode = lambda _c: None  # type: ignore[method-assign]
            parser.train_next_token(sents)
            return expand_n["n"]

        # compiled optimization is active only off norm_init (Brain default disables it)
        no_pred_ring = EmergentParser(
            n=N, k=K, seed=44, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        no_pred_ring.train_lexicon(skip_known=False)
        no_pred_ring._enable_prediction_ring_mode = lambda _c: None  # type: ignore[method-assign]
        without = count_expands(no_pred_ring, disable_pred_ring=True)

        with_pred_ring = EmergentParser(
            n=N, k=K, seed=45, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        with_pred_ring.train_lexicon(skip_known=False)
        with_ring = count_expands(with_pred_ring)

        assert with_ring <= without * 1.1, (
            f"prediction ring expand variance: {with_ring} vs {without}"
        )
        assert with_pred_ring._prediction_ring_capacity_cols > 0

    def test_compiled_bridge_reduces_sampling(self):
        """Compiled topology skips truncated-normal winner sampling."""
        from contextlib import nullcontext

        import neural_assemblies.assembly_calculus.emergent.parser_mixins.prediction as pred_mod

        sents = create_training_sentences()

        def count_winner_sampling(parser) -> int:
            sim = parser.brain._engine._sparse_sim
            sample_n = {"n": 0}
            orig = sim.sample_new_winner_inputs

            def counting(*args, **kwargs):
                sample_n["n"] += 1
                return orig(*args, **kwargs)

            sim.sample_new_winner_inputs = counting
            parser.train_next_token(sents)
            return sample_n["n"]

        # compiled optimization is active only off norm_init (Brain default disables it)
        compiled = EmergentParser(
            n=N, k=K, seed=46, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        compiled.train_lexicon(skip_known=False)
        with_compiled = count_winner_sampling(compiled)
        mapping = compiled.brain._engine.get_neuron_id_mapping("CONTEXT")
        assert mapping is None or len(mapping) == len(set(mapping)), (
            "compiled ring allocation must preserve unique stable neuron IDs"
        )

        microscopic = EmergentParser(
            n=N, k=K, seed=46, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        microscopic.train_lexicon(skip_known=False)
        microscopic._compiled_training_enabled = False
        microscopic.brain.projection_fidelity = "exact"
        original_ct = pred_mod.compiled_topology
        pred_mod.compiled_topology = lambda _p, _s: nullcontext()  # type: ignore[misc]
        try:
            without_compiled = count_winner_sampling(microscopic)
        finally:
            pred_mod.compiled_topology = original_ct

        assert with_compiled < without_compiled * 0.8, (
            f"compiled bridge should cut sampling: "
            f"{with_compiled} vs {without_compiled}"
        )

    def test_context_preserve_topology_reduces_bridge_sampling(self):
        """Preserving CONTEXT w during bridge prefixes enables compiled advances."""
        from neural_assemblies.assembly_calculus.emergent.parser_mixins.incremental import (
            IncrementalMixin,
        )

        sents = create_training_sentences()

        def count_winner_sampling(parser) -> int:
            sim = parser.brain._engine._sparse_sim
            sample_n = {"n": 0}
            orig = sim.sample_new_winner_inputs

            def counting(*args, **kwargs):
                sample_n["n"] += 1
                return orig(*args, **kwargs)

            sim.sample_new_winner_inputs = counting
            parser.train_next_token(sents)
            return sample_n["n"]

        # compiled optimization is active only off norm_init (Brain default disables it)
        normal = EmergentParser(
            n=N, k=K, seed=50, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        normal.train_lexicon(skip_known=False)
        baseline = count_winner_sampling(normal)

        broken = EmergentParser(
            n=N, k=K, seed=50, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        broken.train_lexicon(skip_known=False)

        def force_w_reset(**_kw):
            IncrementalMixin._reset_context_for_bridge(
                broken, preserve_topology=False,
            )

        broken._reset_context_for_bridge = force_w_reset  # type: ignore[method-assign]
        worse = count_winner_sampling(broken)

        assert baseline < worse, (
            f"preserve_topology should reduce sampling: {baseline} vs {worse}"
        )

    def test_compiled_bridge_predict_next_parity(self):
        """Fixed-topology bridge training preserves readout rankings."""
        sents = create_training_sentences()
        prefixes = [
            ["the", "dog"],
            ["the", "cat"],
            ["a", "big", "dog"],
        ]

        def train_and_predict(seed: int, disable_compiled: bool):
            from contextlib import nullcontext

            import neural_assemblies.assembly_calculus.emergent.parser_mixins.prediction as pred_mod

            parser = EmergentParser(
                n=N, k=K, seed=seed, rounds=ROUNDS, fast_training=True,
            )
            original_ct = None
            if disable_compiled:
                original_ct = pred_mod.compiled_topology
                pred_mod.compiled_topology = lambda _p, _s: nullcontext()  # type: ignore[misc]
                parser._compiled_training_enabled = False
                parser.brain.projection_fidelity = "exact"
            parser.train_lexicon(skip_known=False)
            parser.train_next_token(sents)
            if disable_compiled and original_ct is not None:
                pred_mod.compiled_topology = original_ct
            out = {}
            for pref in prefixes:
                ranked = parser.predict_next(pref)
                out[tuple(pref)] = ranked[:5]
            return out

        microscopic = train_and_predict(47, disable_compiled=True)
        compiled = train_and_predict(47, disable_compiled=False)

        for key in microscopic:
            micro_top = microscopic[key]
            comp_top = compiled[key]
            assert micro_top, f"no predictions for {key}"
            assert comp_top, f"compiled: no predictions for {key}"
            assert micro_top[0][0] == comp_top[0][0], (
                f"top-1 mismatch on {key}: {micro_top[0]} vs {comp_top[0]}"
            )

    def test_compiled_role_reduces_sampling(self):
        """Compiled topology on active role pathways skips winner sampling.

        The pathway must carry nonzero drive. A zero-drive projection exits
        before either fidelity policy reaches the selector and therefore cannot
        distinguish exact from compiled execution.
        """
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )

        sents = create_training_sentences()
        idx = compile_corpus(
            EmergentParser(n=N, k=K, seed=48, rounds=ROUNDS),
            sents,
        )

        def count_winner_sampling(parser) -> int:
            sim = parser.brain._engine._sparse_sim
            sample_n = {"n": 0}
            orig = sim.sample_new_winner_inputs

            def counting(*args, **kwargs):
                sample_n["n"] += 1
                return orig(*args, **kwargs)

            sim.sample_new_winner_inputs = counting
            parser.train_unsupervised(sents, repetitions=1, corpus_index=idx)
            return sample_n["n"]

        compiled = EmergentParser(
            n=N, k=K, seed=48, rounds=ROUNDS, fast_training=True,
            norm_init=False,  # compiled training is only active off norm_init
        )
        compiled.train_lexicon(skip_known=False)
        with_compiled = count_winner_sampling(compiled)

        microscopic = EmergentParser(
            n=N, k=K, seed=48, rounds=ROUNDS, fast_training=True,
            norm_init=False,  # compiled training is only active off norm_init
        )
        microscopic.train_lexicon(skip_known=False)
        import neural_assemblies.assembly_calculus.emergent.parser_mixins.unsupervised as unsup_mod
        from contextlib import nullcontext

        original_ct = unsup_mod.compiled_topology
        unsup_mod.compiled_topology = lambda _p, _s: nullcontext()  # type: ignore[misc]
        try:
            without_compiled = count_winner_sampling(microscopic)
        finally:
            unsup_mod.compiled_topology = original_ct

        assert with_compiled < without_compiled * 0.8, (
            f"compiled roles should cut sampling: "
            f"{with_compiled} vs {without_compiled}"
        )

    def test_compiled_role_lexicon_parity(self):
        """Compiled role training preserves role assemblies on trained pathways."""
        from neural_assemblies.assembly_calculus.assembly import overlap
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )

        sents = create_training_sentences()
        probe_words = ["dog", "cat", "boy"]

        def train(seed: int, disable_compiled: bool):
            p = EmergentParser(
                n=N, k=K, seed=seed, rounds=ROUNDS, fast_training=True,
            )
            if disable_compiled:
                from contextlib import nullcontext

                import neural_assemblies.assembly_calculus.emergent.parser_mixins.unsupervised as unsup_mod

                original_ct = unsup_mod.compiled_topology
                unsup_mod.compiled_topology = lambda _p, _s: nullcontext()  # type: ignore[misc]
                p._compiled_training_enabled = False
                p.brain.projection_fidelity = "exact"
            p.train_lexicon(skip_known=False)
            idx = compile_corpus(p, sents)
            p.train_unsupervised(sents, repetitions=1, corpus_index=idx)
            if disable_compiled:
                unsup_mod.compiled_topology = original_ct
            return p, idx

        microscopic, idx = train(49, disable_compiled=True)
        compiled, _ = train(49, disable_compiled=False)

        word_roles = {
            u.word: u.role_area
            for u in idx.role_updates
            if u.word in probe_words
        }
        assert word_roles, "expected role updates for probe words"

        overlaps = []
        for word, role_area in word_roles.items():
            micro_asm = microscopic.role_lexicons[role_area].get(word)
            comp_asm = compiled.role_lexicons[role_area].get(word)
            assert micro_asm is not None and comp_asm is not None
            overlaps.append(overlap(micro_asm, comp_asm))

        assert sum(o >= 0.80 for o in overlaps) >= len(overlaps) - 1, (
            f"too many role assembly drifts: {overlaps}"
        )
        assert sum(overlaps) / len(overlaps) >= 0.78, (
            f"mean role assembly overlap too low: {overlaps}"
        )

    def test_projection_fidelity_exact_by_default(self):
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        assert parser.brain.projection_fidelity == "exact"

    def test_projection_fidelity_fuzzy_alias(self):
        from neural_assemblies.core.projection_fidelity import ProjectionFidelity

        assert ProjectionFidelity.normalize("fuzzy") == ProjectionFidelity.COMPILED
        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.brain.projection_fidelity = "fuzzy"
        assert parser.brain.projection_fidelity == "compiled"

    def test_lexicon_and_phon_context_advance_are_nondegenerate(self):
        """Both context-advance methods produce non-degenerate predictions.

        Two ways to advance CONTEXT for a word:
        - lexicon-core: reuse the stored, multimodally-GROUNDED core assembly
          (``apply_lexicon_word`` projects phon + grounding stimuli together),
        - phon-core: re-derive the core from the phonological stimulus ALONE.

        These are legitimately DIFFERENT representations (measured overlap
        between the stored grounded assembly and the phon-only assembly is only
        ~0.3-0.5), so they train different CONTEXT->PREDICTION bridges and give
        different -- but individually valid -- predictions. An earlier version
        of this test asserted the two produce IDENTICAL top-1; that equivalence
        only held under the compiled-training COLLAPSE (both paths degenerated
        to one context-independent attractor). This test instead guards the
        property that actually matters and that the collapse violated: each
        method's predictions VARY with context (are not a single collapsed
        word). See the lexicon-vs-phon divergence investigation for why they are
        not required to agree.
        """
        sents = create_training_sentences()[:12]

        def train(use_lexicon: bool):
            p = EmergentParser(
                n=N, k=K, seed=52, rounds=ROUNDS, fast_training=True,
            )
            p.train_lexicon(skip_known=False)
            if not use_lexicon:
                orig = p._advance_context_direct

                def no_lexicon(word, **kw):
                    kw.pop("use_lexicon_core", None)
                    return orig(word, use_lexicon_core=False, **kw)

                p._advance_context_direct = no_lexicon  # type: ignore[method-assign]
            p.train_next_token(sents)
            return p

        probes = [
            ["the"],
            ["the", "dog"],
            ["the", "dog", "chases"],
        ]
        for use_lexicon in (False, True):
            parser = train(use_lexicon=use_lexicon)
            preds = [parser.predict_next(words)[0][0] for words in probes]
            # Every prediction must be a real vocabulary word...
            for words, pred in zip(probes, preds):
                assert pred and pred != "<NON-WORD>", (
                    f"{'lexicon' if use_lexicon else 'phon'}-core predicted "
                    f"{pred!r} for {words!r}"
                )
            # ...and predictions must VARY with context (not the single
            # context-independent word the compiled collapse produced).
            assert len(set(preds)) > 1, (
                f"{'lexicon' if use_lexicon else 'phon'}-core produced "
                f"degenerate context-independent predictions: {preds}"
            )

    def test_lexicon_context_skips_phon_in_accumulate(self):
        """Bridge context advances inject lexicon cores instead of phon→core."""
        import neural_assemblies.assembly_calculus.emergent.parser_mixins.incremental as inc_mod

        sents = create_training_sentences()
        parser = EmergentParser(
            n=N, k=K, seed=53, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)

        counts = {"lexicon": 0, "phon": 0}
        orig_step = inc_mod.accumulate_context_step

        def spy_step(brain, **kwargs):
            if kwargs.get("core_assembly") is not None:
                counts["lexicon"] += 1
            elif kwargs.get("phon") is not None:
                counts["phon"] += 1
            return orig_step(brain, **kwargs)

        inc_mod.accumulate_context_step = spy_step  # type: ignore[assignment]
        try:
            parser.train_next_token(sents)
        finally:
            inc_mod.accumulate_context_step = orig_step

        assert counts["lexicon"] > 0
        assert counts["phon"] == 0

    def test_prefix_trie_lcp_reduces_context_advances(self):
        """DFS + LCP scheduling advances fewer words than naive prefix rebuild."""
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )

        parser = EmergentParser(
            n=N, k=K, seed=54, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        idx = compile_corpus(parser, sents)

        advance_n = {"n": 0}
        orig = parser._advance_context_direct

        def counting_advance(word, **kwargs):
            advance_n["n"] += 1
            return orig(word, **kwargs)

        parser._advance_context_direct = counting_advance  # type: ignore[method-assign]
        parser.train_next_token(sents, corpus_index=idx)

        naive = sum(len(p) for p in {t.prefix_words for t in idx.transitions})
        assert advance_n["n"] < naive, (
            f"LCP should beat naive rebuild: {advance_n['n']} vs {naive}"
        )

    def test_compile_dialogue_pairs_produces_transitions(self):
        from neural_assemblies.assembly_calculus.emergent.curriculum.dialogue import (
            get_dialogue_pairs,
        )
        from neural_assemblies.assembly_calculus.emergent.training.compiler import (
            compile_dialogue_pairs,
        )

        parser = EmergentParser(n=N, k=K, seed=42, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        pairs = get_dialogue_pairs()
        idx = compile_dialogue_pairs(parser, pairs)
        assert len(idx.transitions) > 0
        assert all(len(t.prefix_words) >= 1 for t in idx.transitions)

    def test_dialogue_uses_shared_bridge_path(self):
        from neural_assemblies.assembly_calculus.emergent.curriculum.dialogue import (
            get_dialogue_pairs,
        )

        parser = EmergentParser(
            n=N, k=K, seed=55, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)
        pairs = get_dialogue_pairs()[:5]
        calls = {"n": 0}
        orig = parser.train_next_token

        def spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        parser.train_next_token = spy  # type: ignore[method-assign]
        parser.train_dialogue(pairs)
        assert calls["n"] == 1

    def test_topology_linker_skips_repeat_link(self):
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )
        from neural_assemblies.assembly_calculus.emergent.training.linker import (
            topology_needs_link,
        )

        # compiled optimization is active only off norm_init (Brain default disables it)
        parser = EmergentParser(
            n=N, k=K, seed=56, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()[:8]
        idx = compile_corpus(parser, sents)
        lex = list(idx.bridge_vocab & set(parser.stim_map.keys()))

        link_calls = {"n": 0}
        import neural_assemblies.assembly_calculus.emergent.training.linker as tl

        orig = tl.link_context_topology

        def counting_ctx(*args, **kwargs):
            link_calls["n"] += 1
            return orig(*args, **kwargs)

        tl.link_context_topology = counting_ctx
        try:
            parser.train_next_token(sents, corpus_index=idx)
            first = link_calls["n"]
            assert parser._bridge_topology_linked
            assert not topology_needs_link(parser, idx, lex)
            parser.train_next_token(sents, corpus_index=idx)
            second = link_calls["n"]
        finally:
            tl.link_context_topology = orig

        assert first >= 1
        assert second == first, "repeat train should skip context relink"

    def test_compiled_training_parity_top1(self):
        """Fast compiled path matches exact path on predict_next top-1 probes."""
        from contextlib import nullcontext

        import neural_assemblies.assembly_calculus.emergent.parser_mixins.prediction as pred_mod
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )
        from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
            collect_transition_probes,
            compare_predict_next_parity,
        )

        sents = create_training_sentences()[:12]
        idx = compile_corpus(
            EmergentParser(n=N, k=K, seed=57, rounds=ROUNDS),
            sents,
        )
        prefixes, _ = collect_transition_probes(idx, max_probes=20)

        def train(seed: int, *, disable_compiled: bool):
            p = EmergentParser(
                n=N, k=K, seed=seed, rounds=ROUNDS, fast_training=True,
            )
            p.train_lexicon(skip_known=False)
            if disable_compiled:
                original = pred_mod.compiled_topology
                pred_mod.compiled_topology = lambda _p, _s: nullcontext()
                p._compiled_training_enabled = False
                p.brain.projection_fidelity = "exact"
            p.train_next_token(sents, corpus_index=idx)
            if disable_compiled:
                pred_mod.compiled_topology = original
            return p

        exact = train(57, disable_compiled=True)
        compiled = train(57, disable_compiled=False)
        parity = compare_predict_next_parity(exact, compiled, prefixes)
        assert parity["top1_agreement"] >= 0.85, (
            f"compiled vs exact top-1 agreement too low: {parity}"
        )

    def test_evaluate_corpus_parity_after_train(self):
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )
        from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
            evaluate_corpus_parity,
        )

        parser = EmergentParser(
            n=N, k=K, seed=58, rounds=ROUNDS, fast_training=True,
        )
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()[:10]
        idx = compile_corpus(parser, sents)
        parser.train_next_token(sents, corpus_index=idx)
        scores = evaluate_corpus_parity(parser, idx, max_probes=15)
        assert scores["total"] > 0
        assert scores["top1"] >= 0.0

    def test_role_topology_linker_skips_repeat_link(self):
        from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
            compile_corpus,
        )
        from neural_assemblies.assembly_calculus.emergent.training.linker import (
            role_topology_needs_link,
        )

        # compiled optimization is active only off norm_init (Brain default disables it)
        parser = EmergentParser(
            n=N, k=K, seed=59, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()[:10]
        idx = compile_corpus(parser, sents)

        pregrow_calls = {"n": 0}
        orig = parser._pregrow_role_pathways

        def counting_pregrow(corpus_index):
            pregrow_calls["n"] += 1
            return orig(corpus_index)

        parser._pregrow_role_pathways = counting_pregrow  # type: ignore[method-assign]
        try:
            parser.train_unsupervised(sents, corpus_index=idx, repetitions=1)
            first = pregrow_calls["n"]
            assert parser._role_topology_linked
            assert not role_topology_needs_link(parser, idx)
            parser.train_unsupervised(sents, corpus_index=idx, repetitions=1)
            second = pregrow_calls["n"]
        finally:
            parser._pregrow_role_pathways = orig  # type: ignore[method-assign]

        assert first >= 1
        assert second == first, "repeat role train should skip pregrow relink"

    def test_dual_metric_learnability_gate(self):
        """Compiled DIALOGUE training preserves learnability vs exact baseline."""
        from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
            run_dual_metric_gate,
        )

        report = run_dual_metric_gate(
            n=N, k=K, seed=60, max_probes=15, qa_subset=10,
        )
        parity = report["optimizer_parity"]
        assert report["dialogue_delta"] >= -0.30, (
            f"compiled dialogue accuracy regressed vs exact: {report}"
        )
        assert report["next_token_top5_delta"] >= -0.30, (
            f"compiled next-token top-5 regressed vs exact: {report}"
        )
        assert parity["total"] >= 1

    def test_lexicon_topology_linker_skips_repeat_pregrow(self):
        import neural_assemblies.assembly_calculus.emergent.training.linker as tl

        # compiled optimization is active only off norm_init (Brain default disables it)
        parser = EmergentParser(
            n=N, k=K, seed=61, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        link_calls = {"n": 0}
        orig = tl.link_lexicon_topology

        def spy(p, plan, **kw):
            link_calls["n"] += 1
            return orig(p, plan, **kw)

        tl.link_lexicon_topology = spy
        try:
            parser.train_lexicon(skip_known=False)
            first = link_calls["n"]
            assert parser._lexicon_topology_linked
            parser.train_lexicon(skip_known=True)
            parser.train_lexicon(skip_known=True)
            second = link_calls["n"]
        finally:
            tl.link_lexicon_topology = orig

        assert first >= 1
        assert second == first

    def test_dialogue_schedule_includes_consolidation_fast(self):
        from neural_assemblies.assembly_calculus.emergent.evaluation.parity import (
            build_dialogue_stage_schedule,
        )

        parser = EmergentParser(
            n=N, k=K, seed=62, rounds=ROUNDS, fast_training=True,
            vocabulary=build_vocabulary_preset("medium"),
        )
        schedule = build_dialogue_stage_schedule(parser, seed=62)
        assert schedule.consolidation_passes >= 1

    def test_compile_lexicon_plan_respects_skip_known(self):
        from neural_assemblies.assembly_calculus.emergent.training.compiler import (
            compile_lexicon_plan,
        )

        parser = EmergentParser(n=N, k=K, seed=63, rounds=ROUNDS)
        parser.train_lexicon(skip_known=False)
        full = compile_lexicon_plan(parser, skip_known=False)
        partial = compile_lexicon_plan(parser, skip_known=True)
        assert len(partial.lexicon_ops) == 0
        assert len(full.lexicon_ops) > 0

    def test_lexicon_delta_skips_pregrow_when_capacity_sufficient(self):
        from neural_assemblies.assembly_calculus.emergent.training.linker import (
            _lexicon_area_needs_pregrow,
            link_lexicon_topology,
        )
        from neural_assemblies.assembly_calculus.emergent.training.compiler import (
            compile_lexicon_plan,
        )

        # compiled optimization is active only off norm_init (Brain default disables it)
        parser = EmergentParser(
            n=N, k=K, seed=64, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        words = list(
            EmergentParser(n=N, k=K, seed=64, rounds=ROUNDS, norm_init=False).stim_map.keys()
        )[:20]
        parser.train_lexicon(skip_known=False, words=set(words[:12]))

        caps = dict(parser._core_ring_capacity_cols)
        linked = dict(parser._lexicon_linked_words_by_core)

        plan = compile_lexicon_plan(parser, skip_known=True, words=set(words[12:16]))
        by_area = {}
        for op in plan.lexicon_ops:
            by_area.setdefault(op.core_area, []).append(op)

        assert plan.lexicon_ops
        for area, ops in by_area.items():
            linked_words = linked.get(area, 0)
            caps[area] = max(
                int(caps.get(area, 0)),
                (linked_words + len(ops)) * parser.k,
            )
        parser._core_ring_capacity_cols = caps

        assert all(
            not _lexicon_area_needs_pregrow(parser, area, ops)
            for area, ops in by_area.items()
        )

        pregrow = {"n": 0}
        import neural_assemblies.assembly_calculus.emergent.training.linker as tl

        orig_batch = __import__(
            "neural_assemblies.assembly_calculus.emergent.training.batch",
            fromlist=["BatchProjector"],
        ).BatchProjector

        class SpyBatch:
            def __init__(self, p):
                self._inner = orig_batch(p)

            def apply_lexicon_word(self, *args, **kwargs):
                pregrow["n"] += 1
                return self._inner.apply_lexicon_word(*args, **kwargs)

        tl.BatchProjector = SpyBatch
        try:
            link_lexicon_topology(parser, plan)
        finally:
            tl.BatchProjector = orig_batch

        assert pregrow["n"] == 0

    def test_lexicon_delta_pregrow_only_new_words(self):
        from neural_assemblies.assembly_calculus.emergent.training.compiler import (
            compile_lexicon_plan,
        )

        # compiled optimization is active only off norm_init (Brain default disables it)
        parser = EmergentParser(
            n=N, k=K, seed=65, rounds=ROUNDS, fast_training=True, norm_init=False,
        )
        all_words = list(parser.stim_map.keys())
        first = set(all_words[:10])
        extra = set(all_words[10:14])
        parser.train_lexicon(skip_known=False, words=first)
        linked_after_first = sum(parser._lexicon_linked_words_by_core.values())

        plan = compile_lexicon_plan(parser, skip_known=True, words=extra)
        assert len(plan.lexicon_ops) == len(extra)

        parser.train_lexicon(skip_known=True, words=extra)
        assert linked_after_first == 10
        assert sum(parser._lexicon_linked_words_by_core.values()) == 14


class TestSweepModePerf:
    def test_sweep_mode_distributional_reps_is_one(self, monkeypatch):
        monkeypatch.setenv("EMERGENT_SWEEP_MODE", "1")
        from neural_assemblies.assembly_calculus.emergent.training import perf as perf_mod

        monkeypatch.setattr(perf_mod, "sweep_mode_enabled", lambda: True)
        assert stage_distributional_reps("SENTENCES", fast=True) == 1

    def test_sweep_mode_skips_sentences_polarity(self, monkeypatch):
        monkeypatch.setenv("EMERGENT_SWEEP_MODE", "1")
        from neural_assemblies.assembly_calculus.emergent.training import perf as perf_mod

        monkeypatch.setattr(perf_mod, "sweep_mode_enabled", lambda: True)
        phases = ["lexicon", "roles", "phrases", "polarity", "mood"]
        out = effective_stage_phases("SENTENCES", phases, fast=True)
        assert "polarity" not in out
        assert "mood" not in out

    def test_delta_mining_single_sentence_in_sweep_mode(self, monkeypatch):
        monkeypatch.setenv("EMERGENT_SWEEP_MODE", "1")
        from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
            delta_mining_sentences,
        )

        sents = delta_mining_sentences("small")
        assert len(sents) == 1
        assert sents[0] == ["the", "bird", "finds", "small"]
