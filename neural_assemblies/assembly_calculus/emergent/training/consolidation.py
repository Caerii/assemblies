"""
Emergent-parser consolidation schedules.

Builds declarative ``ConsolidationStep`` protocols from grounded training
sentences and runs them via ``assembly_calculus.consolidation.consolidate``.

Naming follows the research infrastructure module
(``consolidate_role_pathways``, ``consolidate_vp_pathways``) with
``_pathways`` suffix to distinguish schedule builders from the core
``consolidate()`` primitive.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set

from neural_assemblies.assembly_calculus.consolidation import (
    ConsolidationStep,
    MergeReplay,
    MultiProjectReplay,
    PathwayEdge,
    PathwayReplay,
    consolidate,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    GROUNDING_TO_CORE,
    NUMBER,
    ROLE_AGENT,
    ROLE_PATIENT,
    VERB_CORE,
    VP,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    GroundedSentence,
)

_ROLE_ANNOTATION_TO_AREA = {
    "agent": ROLE_AGENT,
    "patient": ROLE_PATIENT,
}

_NUMBER_STIMULI = {"SG": "number_SG", "PL": "number_PL"}


def build_role_pathway_protocol(
    parser,
    sentences: List[GroundedSentence],
) -> List[ConsolidationStep]:
    """Compile role-binding episodes into pathway replay steps."""
    steps: List[ConsolidationStep] = []
    rounds = parser.rounds

    for sent in sentences:
        for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
            if role is None or role == "action":
                continue
            role_area = _ROLE_ANNOTATION_TO_AREA.get(role)
            if role_area is None or word not in parser.stim_map:
                continue
            core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
            steps.append(
                PathwayReplay(
                    source_area=core_area,
                    target_area=role_area,
                    stimulus=parser.stim_map[word],
                    rounds=rounds,
                )
            )
    return steps


def build_vp_pathway_protocol(
    parser,
    sentences: List[GroundedSentence],
) -> List[ConsolidationStep]:
    """Compile phrase-structure episodes into merge replay steps."""
    steps: List[ConsolidationStep] = []
    rounds = parser.rounds

    for sent in sentences:
        subj_word = subj_ctx = verb_word = obj_word = obj_ctx = None

        for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
            if role == "agent":
                subj_word, subj_ctx = word, ctx
            elif role == "action":
                verb_word = word
            elif role == "patient":
                obj_word, obj_ctx = word, ctx

        if not (subj_word and verb_word and subj_word in parser.stim_map):
            continue

        subj_core = GROUNDING_TO_CORE[subj_ctx.dominant_modality]
        steps.append(
            MergeReplay(
                source_a=subj_core,
                source_b=VERB_CORE,
                target=VP,
                stimulus_a=parser.stim_map[subj_word],
                stimulus_b=parser.stim_map[verb_word],
                rounds=rounds,
            )
        )

        if obj_word and obj_word in parser.stim_map:
            obj_core = GROUNDING_TO_CORE[obj_ctx.dominant_modality]
            steps.append(
                PathwayReplay(
                    source_area=obj_core,
                    target_area=VP,
                    stimulus=parser.stim_map[obj_word],
                    rounds=rounds,
                )
            )

    return steps


def build_number_role_pathway_protocol(
    parser,
    sentences: List[GroundedSentence],
) -> List[ConsolidationStep]:
    """Compile NUMBER co-projection role episodes."""
    steps: List[ConsolidationStep] = []
    rounds = parser.rounds

    for sent in sentences:
        for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
            if role is None or role == "action":
                continue
            role_area = _ROLE_ANNOTATION_TO_AREA.get(role)
            if role_area is None or word not in parser.stim_map:
                continue

            core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
            num_stim = _NUMBER_STIMULI[parser.detect_number(word)]
            steps.append(
                MultiProjectReplay(
                    sources=(core_area, NUMBER),
                    target=role_area,
                    stimulus_projections=(
                        (parser.stim_map[word], core_area),
                        (num_stim, NUMBER),
                    ),
                    rounds=rounds,
                )
            )
    return steps


def build_number_vp_pathway_protocol(
    parser,
    sentences: List[GroundedSentence],
) -> List[ConsolidationStep]:
    """Compile NUMBER-aware VP merge episodes."""
    steps: List[ConsolidationStep] = []
    rounds = parser.rounds

    for sent in sentences:
        subj_word = subj_ctx = verb_word = obj_word = obj_ctx = None

        for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
            if role == "agent":
                subj_word, subj_ctx = word, ctx
            elif role == "action":
                verb_word = word
            elif role == "patient":
                obj_word, obj_ctx = word, ctx

        if not (subj_word and verb_word and subj_word in parser.stim_map):
            continue

        subj_core = GROUNDING_TO_CORE[subj_ctx.dominant_modality]
        num_stim = _NUMBER_STIMULI[parser.detect_number(subj_word)]
        steps.append(
            MultiProjectReplay(
                sources=(subj_core, VERB_CORE, NUMBER),
                target=VP,
                stimulus_projections=(
                    (parser.stim_map[subj_word], subj_core),
                    (parser.stim_map[verb_word], VERB_CORE),
                    (num_stim, NUMBER),
                ),
                rounds=rounds,
            )
        )

        if obj_word and obj_word in parser.stim_map:
            obj_core = GROUNDING_TO_CORE[obj_ctx.dominant_modality]
            steps.append(
                PathwayReplay(
                    source_area=obj_core,
                    target_area=VP,
                    stimulus=parser.stim_map[obj_word],
                    rounds=rounds,
                )
            )

    return steps


def consolidate_role_pathways(
    parser,
    training_sentences: List[GroundedSentence],
    *,
    passes: int = 1,
    log_fn: Optional[Callable] = None,
) -> Set[PathwayEdge]:
    """Replay role binding without reset — persistent core→role weights.

    ``prepare_areas=False`` is what makes "without reset" true. The SOURCE of a
    role replay is a CORE area, which holds the stabilized lexicon; preparing
    it rewinds ``w`` and re-issues neuron IDs, invalidating every stored
    assembly of the words being replayed. Measured at ``DIALOGUE``:
    ``NOUN_CORE`` w=2493 -> 976 with 48 of 74 nouns left unmappable, which then
    raised "Assembly neuron N not in area mapping" at parse time. Preparation
    is for replay onto a connectome that was just CLEARED; this replays onto a
    live one.
    """
    protocol = build_role_pathway_protocol(parser, training_sentences)
    if passes <= 0:
        if log_fn:
            log_fn("  Skipping role pathway consolidation (passes=0)")
        return set()

    edges = consolidate(parser.brain, protocol, passes=passes,
                        prepare_areas=False)
    if log_fn:
        passes_str = f" ({passes} pass{'es' if passes != 1 else ''})"
        log_fn(
            f"  Consolidated {len(edges)} role pathways{passes_str}: "
            + ", ".join(f"{c}->{r}" for c, r in sorted(edges))
        )
    return edges


def consolidate_vp_pathways(
    parser,
    training_sentences: List[GroundedSentence],
    *,
    passes: int = 1,
    log_fn: Optional[Callable] = None,
) -> Set[PathwayEdge]:
    """Replay VP merge without reset — persistent phrase-structure weights.

    ``prepare_areas=False`` for the same reason as the role pathways above: the
    merge sources are core areas holding the stabilized lexicon.
    """
    protocol = build_vp_pathway_protocol(parser, training_sentences)
    if passes <= 0:
        if log_fn:
            log_fn("  Skipping VP pathway consolidation (passes=0)")
        return set()

    edges = consolidate(parser.brain, protocol, passes=passes,
                        prepare_areas=False)
    if log_fn:
        passes_str = f" ({passes} pass{'es' if passes != 1 else ''})"
        log_fn(
            f"  Consolidated {len(edges)} VP pathways{passes_str}: "
            + ", ".join(f"{c}->{r}" for c, r in sorted(edges))
        )
    return edges


def consolidate_number_role_pathways(
    parser,
    training_sentences: List[GroundedSentence],
    *,
    passes: int = 1,
    log_fn: Optional[Callable] = None,
) -> Set[PathwayEdge]:
    """Replay NUMBER co-projection role binding without reset.

    ``prepare_areas=False`` -- same core-area sources, same reason.
    """
    protocol = build_number_role_pathway_protocol(parser, training_sentences)
    if passes <= 0:
        if log_fn:
            log_fn("  Skipping number-role pathway consolidation (passes=0)")
        return set()

    edges = consolidate(parser.brain, protocol, passes=passes,
                        prepare_areas=False)
    if log_fn and edges:
        passes_str = f" ({passes} pass{'es' if passes != 1 else ''})"
        log_fn(f"  Number-role consolidated {len(edges)} pathways{passes_str}")
    return edges


def consolidate_number_vp_pathways(
    parser,
    training_sentences: List[GroundedSentence],
    *,
    passes: int = 1,
    log_fn: Optional[Callable] = None,
) -> Set[PathwayEdge]:
    """Replay NUMBER-aware VP merge without reset.

    ``prepare_areas=False`` -- same core-area sources, same reason.
    """
    protocol = build_number_vp_pathway_protocol(parser, training_sentences)
    if passes <= 0:
        if log_fn:
            log_fn("  Skipping number-VP pathway consolidation (passes=0)")
        return set()

    edges = consolidate(parser.brain, protocol, passes=passes,
                        prepare_areas=False)
    if log_fn and edges:
        passes_str = f" ({passes} pass{'es' if passes != 1 else ''})"
        log_fn(f"  Number-VP consolidated {len(edges)} pathways{passes_str}")
    return edges


# Backward-compatible aliases for research infrastructure imports
consolidate_role_connections = consolidate_role_pathways
consolidate_vp_connections = consolidate_vp_pathways
consolidate_number_role_connections = consolidate_number_role_pathways
consolidate_number_vp_connections = consolidate_number_vp_pathways
