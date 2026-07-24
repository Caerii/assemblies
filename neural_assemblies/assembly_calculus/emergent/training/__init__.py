"""Training pipeline: compile → link → run.

WHY TRAINING HAS A COMPILER.  Naively, training a corpus means: for each
sentence, for each word, project.  That is correct and unusably slow, for a
reason specific to the sparse engine rather than to Python.  Every projection
into a not-yet-grown area triggers candidate sampling and connectome
expansion, and the same shared prefix ("the dog ...") is re-walked once per
sentence that starts with it.  The dominant cost is rediscovering structure
that was already there.

The three stages remove that cost without changing the calculus:

    compile  (``compiler.py``)  Turn the corpus index into a minimal
             schedule of operations.  A ``PrefixTrie`` collapses shared
             sentence prefixes so each distinct prefix is built once.  The
             output is a plan -- data, no side effects.
    link     (``linker.py``)  Do all connectome growth up front: pregrow
             CONTEXT/PREDICTION depth, preallocate stimulus vectors, build
             the prediction lexicon.  After linking, the topology needed by
             the plan exists.
    run      (``batch.py`` + ``compiled.py``)  Execute the plan against the
             now-frozen topology, so projections are pure weight updates on
             existing columns.

The names are the analogy and it is exact: compile plans, link resolves
addresses, run executes -- and, as with a real toolchain, the point is that
the observable result must be unchanged.

WHERE THE APPROXIMATION IS.  Only the run stage's "compiled" projection
fidelity departs from the exact dynamics, by skipping candidate sampling and
selecting winners from existing columns.  ``compiled.py`` states the contract:
training may use it, INFERENCE NEVER DOES, and equivalence is defined by
readout/overlap gates rather than step-by-step identity.  If you are measuring
something and the number looks wrong, check which fidelity produced it first.
"""

from .batch import BatchProjector
from .compiled import CompiledTopologySpec, bridge_topology_spec, compiled_topology, lexicon_topology_spec, role_topology_spec
from .compiler import (
    CompiledLexiconPlan,
    CompiledRolePlan,
    CompiledTrainingPlan,
    LexiconOp,
    PrefixTrie,
    RoleOp,
    compile_dialogue_pairs,
    compile_lexicon_plan,
    compile_role_plan,
    compile_training_plan,
    group_lexicon_ops_by_core,
    link_preallocate_stim_targets,
)
from .consolidation import (
    build_role_pathway_protocol,
    build_vp_pathway_protocol,
    consolidate_role_pathways,
    consolidate_vp_pathways,
)
from .linker import (
    lexicon_topology_needs_link,
    link_bridge_topology,
    link_lexicon_topology,
    link_prediction_lexicon,
    link_role_topology,
    role_topology_needs_link,
    topology_needs_link,
)
from .perf import (
    PRESET_VOCAB_SKIP_THRESHOLD,
    STAGE_WORD_ORDER_REPS,
    adaptive_rounds,
    dedupe_grounded_sentences,
    developmental_curriculum_enabled,
    effective_stage_phases,
    sequence_rounds_per_step,
    should_skip_early_curriculum,
    stage_distributional_reps,
    stage_training_rounds,
)
from .schedule import TrainingSchedule, TrainingScheduleExecutor

__all__ = [
    "BatchProjector",
    "CompiledLexiconPlan",
    "CompiledRolePlan",
    "CompiledTopologySpec",
    "CompiledTrainingPlan",
    "LexiconOp",
    "PRESET_VOCAB_SKIP_THRESHOLD",
    "PrefixTrie",
    "RoleOp",
    "STAGE_WORD_ORDER_REPS",
    "TrainingSchedule",
    "TrainingScheduleExecutor",
    "adaptive_rounds",
    "bridge_topology_spec",
    "build_role_pathway_protocol",
    "build_vp_pathway_protocol",
    "compile_dialogue_pairs",
    "compile_lexicon_plan",
    "compile_role_plan",
    "compile_training_plan",
    "compiled_topology",
    "consolidate_role_pathways",
    "consolidate_vp_pathways",
    "dedupe_grounded_sentences",
    "developmental_curriculum_enabled",
    "effective_stage_phases",
    "group_lexicon_ops_by_core",
    "lexicon_topology_needs_link",
    "lexicon_topology_spec",
    "link_bridge_topology",
    "link_lexicon_topology",
    "link_prediction_lexicon",
    "link_preallocate_stim_targets",
    "link_role_topology",
    "role_topology_needs_link",
    "role_topology_spec",
    "sequence_rounds_per_step",
    "should_skip_early_curriculum",
    "stage_distributional_reps",
    "stage_training_rounds",
    "topology_needs_link",
]
