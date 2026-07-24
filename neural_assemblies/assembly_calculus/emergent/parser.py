"""
EmergentParser — 48-area emergent NEMO parser on numpy_sparse.

Thin assembler: composes mixin modules into a single EmergentParser class
via multiple inheritance. Each mixin provides a cohesive feature group.

Cross-mixin calls resolve at runtime via Python's MRO since `self` is
the fully-composed class.  No mixin imports another mixin class.

References:
    Mitropolsky, D. & Papadimitriou, C. H. (2025).
    "Simulated Language Acquisition with Neural Assemblies."
"""

from .core import CorpusIndex, compile_corpus
from .curriculum import CurriculumTrainer, StageResult, _STAGE_CONFIG
from .evaluation import (
    EvaluationSuite,
    collect_transition_probes,
    compare_learnability_parity,
    compare_predict_next_parity,
    evaluate_corpus_parity,
    exact_training_mode,
    run_dual_metric_gate,
    score_next_token_probes,
)
from .parser_mixins import (
    BlocksMixin,
    CoreParserMixin,
    DialogueMixin,
    DistributionalMixin,
    DistributionalStats,
    GenerationMixin,
    IncrementalMixin,
    InstructionMixin,
    MorphosyntaxMixin,
    PlansMixin,
    ConstituentOrderMixin,
    PredictionMixin,
    StatePredictionMixin,
    StructuredMixin,
    UnsupervisedMixin,
)
from .training import (
    BatchProjector,
    CompiledLexiconPlan,
    CompiledRolePlan,
    CompiledTrainingPlan,
    LexiconOp,
    PrefixTrie,
    RoleOp,
    TrainingSchedule,
    TrainingScheduleExecutor,
    compile_dialogue_pairs,
    compile_lexicon_plan,
    compile_role_plan,
    compile_training_plan,
    lexicon_topology_needs_link,
    link_bridge_topology,
    link_lexicon_topology,
    link_role_topology,
    role_topology_needs_link,
    topology_needs_link,
)


class EmergentParser(
    PlansMixin,
    StructuredMixin,
    BlocksMixin,
    DialogueMixin,
    InstructionMixin,
    ConstituentOrderMixin,
    PredictionMixin,
    StatePredictionMixin,
    UnsupervisedMixin,
    GenerationMixin,
    IncrementalMixin,
    MorphosyntaxMixin,
    DistributionalMixin,
    CoreParserMixin,
):
    """48-area emergent NEMO parser composed from feature mixins."""
    pass


__all__ = [
    "EmergentParser",
    "CurriculumTrainer",
    "StageResult",
    "_STAGE_CONFIG",
    "EvaluationSuite",
    "DistributionalStats",
    "CorpusIndex",
    "compile_corpus",
    "TrainingSchedule",
    "TrainingScheduleExecutor",
    "BatchProjector",
    "CompiledTrainingPlan",
    "PrefixTrie",
    "compile_dialogue_pairs",
    "compile_training_plan",
    "compile_lexicon_plan",
    "compile_role_plan",
    "CompiledLexiconPlan",
    "CompiledRolePlan",
    "LexiconOp",
    "RoleOp",
    "collect_transition_probes",
    "compare_learnability_parity",
    "compare_predict_next_parity",
    "evaluate_corpus_parity",
    "exact_training_mode",
    "link_bridge_topology",
    "link_lexicon_topology",
    "link_role_topology",
    "lexicon_topology_needs_link",
    "role_topology_needs_link",
    "run_dual_metric_gate",
    "topology_needs_link",
]
