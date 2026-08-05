"""
Topology linker — single link-time pass for bridge training.

Allocates CONTEXT/PREDICTION connectome depth, builds the prediction
lexicon under compiled topology, and preallocates stim vectors so bridge
training never rediscovers matrix growth.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

from neural_assemblies.assembly_calculus.ops import project, _snap

from ..core.areas import CONTEXT, PREDICTION, ROLE_AGENT, ROLE_PATIENT
from ..training.compiled import (
    bridge_topology_spec,
    compiled_topology,
    prediction_topology_spec,
)
from ..training.compiler import link_preallocate_stim_targets

if TYPE_CHECKING:
    from ..parser_mixins.core import CoreParserMixin
    from ..core.corpus_index import CorpusIndex
    from ..training.compiler import CompiledLexiconPlan


def _max_prefix_len(corpus_index: "CorpusIndex") -> int:
    max_len = corpus_index.max_sentence_length
    for trans in corpus_index.transitions:
        max_len = max(max_len, len(trans.prefix_words))
    return max(max_len, 1)


def _missing_lexicon_words(
    parser: "CoreParserMixin",
    lex_targets: Sequence[str],
) -> List[str]:
    lex = getattr(parser, "prediction_lexicon", None) or {}
    return [w for w in lex_targets if w not in lex]


def topology_needs_link(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
    lex_targets: Sequence[str],
) -> bool:
    """True when CONTEXT/PREDICTION capacity or lexicon still needs linking."""
    if not getattr(parser, "_bridge_topology_linked", False):
        return True
    max_len = _max_prefix_len(corpus_index)
    if int(getattr(parser, "_context_ring_capacity", 0)) < max_len:
        return True
    if _missing_lexicon_words(parser, lex_targets):
        return True
    return False


def link_context_topology(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
) -> None:
    """Pregrow CONTEXT ring to corpus max prefix length (once)."""
    max_len = _max_prefix_len(corpus_index)
    if int(getattr(parser, "_context_ring_capacity", 0)) >= max_len:
        return

    sentences = corpus_index.grounded
    if not sentences:
        return

    parser._pregrow_context_capacity(sentences, max_len=max_len)

    longest_prefix: Tuple[str, ...] = ()
    for trans in corpus_index.transitions:
        if len(trans.prefix_words) > len(longest_prefix):
            longest_prefix = trans.prefix_words

    if len(longest_prefix) < 2:
        return

    with parser.brain.frozen():
        engine = parser.brain._engine
        parser._reset_context_winners()
        parser.brain.areas[CONTEXT].w = 0
        if hasattr(engine, "_areas") and CONTEXT in engine._areas:
            engine._areas[CONTEXT].w = 0
        for word in longest_prefix:
            if word in parser.stim_map:
                parser._advance_context_direct(
                    word, rounds=parser.inference_rounds,
                )
        arb_phon = parser.stim_map.get(longest_prefix[-1])
        if arb_phon is not None:
            parser.brain.project(
                {arb_phon: [PREDICTION]},
                {CONTEXT: [PREDICTION]},
            )
            parser.brain.project({}, {PREDICTION: [PREDICTION]})

    if hasattr(engine, "_areas") and CONTEXT in engine._areas:
        parser._context_ring_capacity_cols = max(
            int(getattr(parser, "_context_ring_capacity_cols", 0)),
            int(engine._areas[CONTEXT].w),
        )
    parser._reset_context_for_bridge(preserve_topology=True)


def link_prediction_lexicon(
    parser: "CoreParserMixin",
    lex_targets: Sequence[str],
) -> None:
    """Pregrow + snap prediction lexicon in one compiled session."""
    if not hasattr(parser, "prediction_lexicon"):
        parser.prediction_lexicon = {}

    parser._bootstrap_prediction_connectivity()
    missing = _missing_lexicon_words(parser, lex_targets)
    if not missing:
        return

    with parser.brain.frozen():
        for word in missing:
            phon = parser.stim_map.get(word)
            if phon is None:
                continue
            parser._clear_prediction_activity()
            project(parser.brain, phon, PREDICTION, rounds=1)
            parser._clear_prediction_activity()

    engine = parser.brain._engine
    if hasattr(engine, "_areas") and PREDICTION in engine._areas:
        parser._prediction_ring_capacity_cols = max(
            int(getattr(parser, "_prediction_ring_capacity_cols", 0)),
            int(engine._areas[PREDICTION].w),
        )

    pred_spec = prediction_topology_spec(parser)
    with compiled_topology(parser, pred_spec):
        for word in missing:
            phon = parser.stim_map.get(word)
            if phon is None:
                continue
            parser._clear_prediction_activity()
            project(parser.brain, phon, PREDICTION, rounds=parser.rounds)
            parser.prediction_lexicon[word] = _snap(parser.brain, PREDICTION)
            if hasattr(engine, "_areas") and PREDICTION in engine._areas:
                parser._prediction_ring_capacity_cols = max(
                    int(getattr(parser, "_prediction_ring_capacity_cols", 0)),
                    int(engine._areas[PREDICTION].w),
                )


def link_bridge_topology(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
    lex_targets: Sequence[str],
    *,
    force: bool = False,
) -> None:
    """Single link pass: CONTEXT ring, PREDICTION lexicon, stim prealloc."""
    if not force and not topology_needs_link(parser, corpus_index, lex_targets):
        link_preallocate_stim_targets(parser, (CONTEXT, PREDICTION))
        return

    link_context_topology(parser, corpus_index)
    link_prediction_lexicon(parser, lex_targets)
    link_preallocate_stim_targets(parser, (CONTEXT, PREDICTION))
    parser._bridge_topology_linked = True


def refresh_training_plan_topology(parser: "CoreParserMixin"):
    """Rebuild bridge topology spec after linking."""
    return bridge_topology_spec(parser)


def role_topology_needs_link(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
) -> bool:
    """True when core→role pathways still need linking."""
    if not getattr(parser, "_role_topology_linked", False):
        return True
    if not getattr(parser, "_role_paths_bootstrapped", False):
        return True
    if not corpus_index.role_updates:
        return False
    caps = getattr(parser, "_role_ring_capacity_cols", {}) or {}
    for role_area in (ROLE_AGENT, ROLE_PATIENT):
        if int(caps.get(role_area, 0)) < parser.k:
            return True
    return False


def link_role_topology(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
    *,
    force: bool = False,
) -> None:
    """Single link pass: pregrow core→role pathways and prealloc stim."""
    if not corpus_index.role_updates:
        return

    if not force and not role_topology_needs_link(parser, corpus_index):
        link_preallocate_stim_targets(parser, (ROLE_AGENT, ROLE_PATIENT))
        return

    parser._pregrow_role_pathways(corpus_index)
    # Phrase areas need the SAME treatment and never got it: VP's self-fiber
    # had zero columns on a fully trained parser, so the ERP probe that reads
    # it returned exactly 0.0 (#104). Idempotent and guarded by its own flag.
    parser._pregrow_phrase_pathways()

    core_areas: set = set()
    for update in corpus_index.role_updates:
        if update.word in parser.stim_map:
            core_areas.add(parser._word_core_area(update.word))

    link_preallocate_stim_targets(
        parser, tuple(core_areas) + (ROLE_AGENT, ROLE_PATIENT),
    )
    parser._role_topology_linked = True


def _group_lexicon_ops_by_core(plan: "CompiledLexiconPlan") -> Dict[str, list]:
    from ..training.compiler import group_lexicon_ops_by_core

    return group_lexicon_ops_by_core(plan)


def _lexicon_capacity_for_words(word_count: int, k: int) -> int:
    """Conservative column estimate for *word_count* lexicon assemblies."""
    return max(word_count, 0) * k


def _lexicon_area_needs_pregrow(
    parser: "CoreParserMixin",
    area: str,
    new_ops: Sequence,
    *,
    force: bool = False,
) -> bool:
    """True when *new_ops* require a plasticity-off pregrow pass in *area*."""
    if force or not getattr(parser, "_lexicon_topology_linked", False):
        return bool(new_ops)
    if not new_ops:
        return False

    caps = getattr(parser, "_core_ring_capacity_cols", {}) or {}
    linked = getattr(parser, "_lexicon_linked_words_by_core", {}) or {}
    if int(caps.get(area, 0)) < parser.k:
        return True

    total_words = int(linked.get(area, 0)) + len(new_ops)
    return int(caps.get(area, 0)) < _lexicon_capacity_for_words(total_words, parser.k)


def _record_lexicon_linked_words(
    parser: "CoreParserMixin",
    by_core: Dict[str, list],
) -> None:
    counts = dict(getattr(parser, "_lexicon_linked_words_by_core", {}))
    for area, ops in by_core.items():
        counts[area] = counts.get(area, 0) + len(ops)
    parser._lexicon_linked_words_by_core = counts


def lexicon_topology_needs_link(
    parser: "CoreParserMixin",
    plan: "CompiledLexiconPlan",
) -> bool:
    """True when any pending lexicon op still needs topology linking/pregrow."""
    if not plan.lexicon_ops:
        return False
    if not getattr(parser, "_lexicon_topology_linked", False):
        return True

    by_core = _group_lexicon_ops_by_core(plan)
    return any(
        _lexicon_area_needs_pregrow(parser, area, ops)
        for area, ops in by_core.items()
    )


def link_lexicon_topology(
    parser: "CoreParserMixin",
    plan: "CompiledLexiconPlan",
    *,
    force: bool = False,
) -> None:
    """Pregrow core-area connectome depth for lexicon batch training."""
    from ..training.batch import BatchProjector

    if not plan.lexicon_ops:
        return

    by_core = _group_lexicon_ops_by_core(plan)
    needs_pregrow = force or lexicon_topology_needs_link(parser, plan)

    if not needs_pregrow:
        link_preallocate_stim_targets(parser, plan.core_areas)
        _record_lexicon_linked_words(parser, by_core)
        return

    ops_to_pregrow: Dict[str, list] = {
        area: ops
        for area, ops in by_core.items()
        if _lexicon_area_needs_pregrow(parser, area, ops, force=force)
    }

    if ops_to_pregrow:
        batch = BatchProjector(parser)
        with parser.brain.frozen():
            for ops in ops_to_pregrow.values():
                for op in ops:
                    ctx = parser.word_grounding.get(op.word)
                    if ctx is None:
                        continue
                    batch.apply_lexicon_word(
                        op.word, ctx, op.core_area, rounds=parser.inference_rounds,
                    )

        engine = parser.brain._engine
        caps = dict(getattr(parser, "_core_ring_capacity_cols", {}))
        for core_area in ops_to_pregrow:
            w = parser.k
            if hasattr(engine, "_areas") and core_area in engine._areas:
                w = max(w, int(engine._areas[core_area].w))
            caps[core_area] = max(int(caps.get(core_area, 0)), w)
        parser._core_ring_capacity_cols = caps

    link_preallocate_stim_targets(parser, plan.core_areas)
    _record_lexicon_linked_words(parser, by_core)
    parser._lexicon_topology_linked = True
