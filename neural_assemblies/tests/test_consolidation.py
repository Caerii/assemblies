"""Tests for consolidation calculus and emergent pathway schedules."""

import numpy as np
import pytest

from neural_assemblies.assembly_calculus import (
    PathwayReplay,
    accumulate_context,
    accumulate_context_step,
    consolidate,
    project,
)
from neural_assemblies.assembly_calculus.contracts import ConsolidationProtocolPlan
from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    CONTEXT,
    NOUN_CORE,
    ROLE_AGENT,
    VP,
)
from neural_assemblies.assembly_calculus.emergent.training.consolidation import (
    build_role_pathway_protocol,
    consolidate_role_pathways,
    consolidate_vp_pathways,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain

N, K, ROUNDS = 10000, 100, 6
P, SEED = 0.05, 42


def test_consolidate_rejects_empty_protocol_or_invalid_passes():
    with pytest.raises(ValueError, match="nonempty step"):
        ConsolidationProtocolPlan(())
    with pytest.raises(ValueError, match="positive integer"):
        ConsolidationProtocolPlan((PathwayReplay("A", "B"),), passes=0)
    with pytest.raises(TypeError, match="PathwayReplay"):
        consolidate(_minimal_brain(), [object()])


def _minimal_brain(seed=SEED):
    brain = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    brain.add_stimulus("phon_dog", K)
    brain.add_area(NOUN_CORE, N, K)
    brain.add_area(ROLE_AGENT, N, K)
    return brain


def _conn_nonzero(brain, src: str, tgt: str) -> int:
    conn = brain.connectomes[src][tgt]
    return int(np.count_nonzero(conn.weights))


class TestConsolidationCalculus:
    def test_consolidate_strengthens_pathway_without_reset(self):
        brain = _minimal_brain()
        project(brain, "phon_dog", NOUN_CORE, rounds=ROUNDS)
        brain.areas[NOUN_CORE].fix_assembly()
        brain.project(
            {},
            {NOUN_CORE: [ROLE_AGENT], ROLE_AGENT: [ROLE_AGENT]},
        )
        episodic_nnz = _conn_nonzero(brain, NOUN_CORE, ROLE_AGENT)
        brain._engine.reset_area_connections(ROLE_AGENT)
        after_reset_nnz = _conn_nonzero(brain, NOUN_CORE, ROLE_AGENT)

        consolidate(
            brain,
            [
                PathwayReplay(
                    source_area=NOUN_CORE,
                    target_area=ROLE_AGENT,
                    stimulus="phon_dog",
                    rounds=ROUNDS,
                )
            ],
            passes=2,
            # This test IS the episodic-reset scenario (connections cleared
            # above), the one case that must opt IN to area preparation.
            prepare_areas=True,
        )
        consolidated_nnz = _conn_nonzero(brain, NOUN_CORE, ROLE_AGENT)

        assert episodic_nnz > 0
        assert after_reset_nnz == 0
        assert consolidated_nnz > after_reset_nnz

    def test_accumulate_context_matches_manual_steps(self):
        brain = _minimal_brain(seed=43)
        brain.add_stimulus("phon_runs", K)
        brain.add_area("VERB_CORE", N, K)
        brain.add_area(CONTEXT, N, K)

        manual = accumulate_context_step(
            brain,
            phon="phon_dog",
            core_area=NOUN_CORE,
            context_area=CONTEXT,
            rounds=ROUNDS,
        )
        brain.inhibit_areas([NOUN_CORE, "VERB_CORE", CONTEXT])
        auto = accumulate_context(
            brain,
            [("phon_dog", NOUN_CORE)],
            context_area=CONTEXT,
            rounds=ROUNDS,
        )
        from neural_assemblies.assembly_calculus.assembly import overlap

        assert overlap(manual, auto) > 0.9

    def test_lexicon_core_context_matches_phon_path(self):
        brain = _minimal_brain(seed=44)
        brain.add_area(CONTEXT, N, K)
        project(brain, "phon_dog", NOUN_CORE, rounds=ROUNDS)
        lexicon_asm = _snap(brain, NOUN_CORE)

        brain.inhibit_areas([NOUN_CORE, CONTEXT])
        phon_ctx = accumulate_context_step(
            brain,
            phon="phon_dog",
            core_area=NOUN_CORE,
            context_area=CONTEXT,
            rounds=ROUNDS,
        )

        brain.inhibit_areas([NOUN_CORE, CONTEXT])
        lex_ctx = accumulate_context_step(
            brain,
            core_area=NOUN_CORE,
            context_area=CONTEXT,
            core_assembly=lexicon_asm,
            rounds=ROUNDS,
        )

        from neural_assemblies.assembly_calculus.assembly import overlap

        assert overlap(phon_ctx, lex_ctx) > 0.85


class TestEmergentConsolidation:
    def test_role_protocol_nonempty_for_annotated_sentences(self):
        parser = EmergentParser(n=3000, k=30, seed=42, rounds=6)
        parser.train_lexicon(skip_known=False)
        sents = create_training_sentences()
        protocol = build_role_pathway_protocol(parser, sents)
        assert len(protocol) > 0
        assert all(isinstance(s, PathwayReplay) for s in protocol)

    def test_consolidate_role_pathways_runs(self):
        parser = EmergentParser(n=3000, k=30, seed=42, rounds=6)
        parser.train_lexicon(skip_known=False)
        parser.train_roles(create_training_sentences())
        edges = consolidate_role_pathways(
            parser, create_training_sentences(), passes=1,
        )
        assert len(edges) > 0
        assert any(src.endswith("CORE") for src, _ in edges)

    def test_consolidate_vp_pathways_runs(self):
        parser = EmergentParser(n=3000, k=30, seed=42, rounds=6)
        parser.train_lexicon(skip_known=False)
        edges = consolidate_vp_pathways(
            parser, create_training_sentences(), passes=1,
        )
        assert any(tgt == VP for _, tgt in edges)


class TestAccumulateContextIntegration:
    def test_advance_context_direct_uses_calculus_op(self):
        parser = EmergentParser(n=3000, k=30, seed=42, rounds=6)
        parser.train_lexicon(skip_known=False)
        parser._reset_context_state()
        parser._advance_context_direct("dog", rounds=6)
        ctx_asm = _snap(parser.brain, CONTEXT)
        assert ctx_asm.area == CONTEXT
        assert len(ctx_asm.winners) == parser.k
