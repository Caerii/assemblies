"""
Consolidation Pass — Persistent Hebbian Connections

After parser.train(), the parser calls reset_area_connections() on role
and VP areas, wiping all Hebbian-trained weights. The consolidation pass
replays the same training projections WITHOUT the reset, creating
persistent Hebbian-strengthened connections for trained pathways.

This creates the asymmetry needed for P600 instability to differentiate:
- Trained pathways (e.g., NOUN_CORE->ROLE_AGENT): Hebbian-strengthened
- Untrained pathways (e.g., VERB_CORE->ROLE_*): Random baseline only

IMPORTANT: Bootstrap connectivity must run BEFORE consolidation.
Empty weights trigger the zero-signal early return in _sparse.py,
preventing consolidation projections from running.

References:
  - research/plans/P600_REANALYSIS.md: consolidation rationale
  - src/assembly_calculus/emergent/_parser_core.py:274-317: train_roles()
  - src/assembly_calculus/emergent/_parser_core.py:420-479: train_phrases()
"""

from typing import List, Optional, Callable

from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, VERB_CORE, VP,
    GROUNDING_TO_CORE,
)
from src.assembly_calculus.ops import project, merge


# Role annotation string -> brain area (matches _parser_core._ROLE_MAP)
_ROLE_MAP_LOCAL = {
    "agent": ROLE_AGENT,
    "patient": ROLE_PATIENT,
}


def consolidate_role_connections(
    parser: EmergentParser,
    training_sentences: List[GroundedSentence],
    n_passes: int = 1,
    log_fn: Optional[Callable] = None,
) -> None:
    """Replay role binding without reset, creating persistent core->role weights.

    During train_roles(), the parser calls reset_area_connections(role_area)
    after each word to isolate role assemblies for readout. This wipes all
    Hebbian-trained weights from core->role pathways.

    The consolidation pass replays the same role binding projections WITHOUT
    the reset, so Hebbian-strengthened connections accumulate and persist.

    Args:
        n_passes: Number of times to iterate through the full training set.
            Default 1 preserves original behavior. Use 0 to skip consolidation
            entirely (models L2/unconsolidated grammar). Higher values model
            more developmental experience.

    After consolidation:
    - NOUN_CORE->ROLE_AGENT: Hebbian-strengthened (trained animal nouns)
    - NOUN_CORE->ROLE_PATIENT: Hebbian-strengthened (trained animal nouns)
    - VERB_CORE->ROLE_*: Still empty (verbs don't get role annotations)

    This creates the asymmetry needed for instability to differentiate:
    trained nouns converge faster in role areas than untrained nouns/verbs.
    """
    if n_passes <= 0:
        if log_fn:
            log_fn("  Skipping role consolidation (n_passes=0)")
        return

    brain = parser.brain
    consolidated = set()

    for _pass in range(n_passes):
        for sent in training_sentences:
            for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
                if role is None or role == "action":
                    continue
                role_area = _ROLE_MAP_LOCAL.get(role)
                if role_area is None:
                    continue
                if word not in parser.stim_map:
                    continue

                core_area = GROUNDING_TO_CORE[ctx.dominant_modality]
                phon = parser.stim_map[word]

                # Activate word in core area
                project(brain, phon, core_area, rounds=parser.rounds)
                brain.areas[core_area].fix_assembly()

                # Project core -> role with recurrence (Hebbian learning ON)
                for _ in range(parser.rounds):
                    brain.project(
                        {},
                        {core_area: [role_area], role_area: [role_area]},
                    )

                brain.areas[core_area].unfix_assembly()
                consolidated.add((core_area, role_area))
                # NO reset_area_connections() — connections persist!

    # Clear activations but keep weights
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    if log_fn:
        passes_str = f" ({n_passes} pass{'es' if n_passes != 1 else ''})"
        log_fn(f"  Consolidated {len(consolidated)} role pathways{passes_str}: "
               + ", ".join(f"{c}->{r}" for c, r in sorted(consolidated)))


def consolidate_vp_connections(
    parser: EmergentParser,
    training_sentences: List[GroundedSentence],
    n_passes: int = 1,
    log_fn: Optional[Callable] = None,
) -> None:
    """Replay phrase structure training without reset for VP connections.

    During train_phrases(), the parser calls reset_area_connections(VP)
    after each sentence, wiping NOUN_CORE->VP, VERB_CORE->VP, and VP->VP
    connections. This consolidation pass replays the merge operations
    WITHOUT the reset.

    Args:
        n_passes: Number of times to iterate through the full training set.
            Default 1 preserves original behavior. Use 0 to skip.

    After consolidation:
    - NOUN_CORE->VP: Hebbian-strengthened (subject/object nouns)
    - VERB_CORE->VP: Hebbian-strengthened (verbs)
    - VP->VP: Hebbian-strengthened (self-recurrence from merge)
    """
    if n_passes <= 0:
        if log_fn:
            log_fn("  Skipping VP consolidation (n_passes=0)")
        return

    brain = parser.brain
    consolidated = set()

    for _pass in range(n_passes):
        for sent in training_sentences:
            subj_word = None
            verb_word = None
            obj_word = None
            subj_ctx = None
            obj_ctx = None

            for word, ctx, role in zip(sent.words, sent.contexts, sent.roles):
                if role == "agent":
                    subj_word = word
                    subj_ctx = ctx
                elif role == "action":
                    verb_word = word
                elif role == "patient":
                    obj_word = word
                    obj_ctx = ctx

            if subj_word and verb_word and subj_word in parser.stim_map:
                subj_core = GROUNDING_TO_CORE[subj_ctx.dominant_modality]

                # Activate both source assemblies
                project(
                    brain, parser.stim_map[subj_word],
                    subj_core, rounds=parser.rounds,
                )
                project(
                    brain, parser.stim_map[verb_word],
                    VERB_CORE, rounds=parser.rounds,
                )

                # Merge subject + verb into VP (Hebbian learning ON)
                merge(brain, subj_core, VERB_CORE, VP, rounds=parser.rounds)
                consolidated.add((subj_core, VP))
                consolidated.add((VERB_CORE, VP))

                if obj_word and obj_word in parser.stim_map:
                    obj_core = GROUNDING_TO_CORE[obj_ctx.dominant_modality]
                    project(
                        brain, parser.stim_map[obj_word],
                        obj_core, rounds=parser.rounds,
                    )
                    brain.areas[obj_core].fix_assembly()
                    for _ in range(parser.rounds):
                        brain.project(
                            {},
                            {obj_core: [VP], VP: [VP]},
                        )
                    brain.areas[obj_core].unfix_assembly()
                    consolidated.add((obj_core, VP))

                # NO reset_area_connections(VP) — connections persist!

    # Clear activations but keep weights
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    if log_fn:
        passes_str = f" ({n_passes} pass{'es' if n_passes != 1 else ''})"
        log_fn(f"  Consolidated {len(consolidated)} VP pathways{passes_str}: "
               + ", ".join(f"{c}->{r}" for c, r in sorted(consolidated)))
