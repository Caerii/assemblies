"""Inspectable contracts and immutable schedules for calculus operations.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects

Layout: each operation family's PLANS (immutable schedules) sit with their CONTRACTS;
this package re-exports every name, so imports are unchanged.

    schedule       argument checks, ProjectionStep, building and executing a schedule
    contract       OperationContract, ContractedOperation, implements
    projection     activation, readout, fiber materialization, projection,
                   reciprocal projection, convergence
    next_token     lexicon build, next-token training, scoring and prediction
    recovery       recovery, cue replacement, pattern completion
    association    association, merge, separation
    binding        binding, its read, input drive, strength, source binding, recall
    consolidation  consolidation (and its protocol), context accumulation (and a step)
    sequence       sequence memorization, ordered recall
    attention      attention
    registry       OPERATION_CONTRACTS: every contract by operation name
"""

from .schedule import (_require_name, _positive_rounds, _explicit_bool, ProjectionStep, _schedule, _execute_schedule)  # noqa: F401
from .contract import (_P, _R_co, OperationContract, ContractedOperation, implements)  # noqa: F401
from .projection import (ActivationPlan, ReadoutPlan, FiberMaterializationPlan, ProjectionPlan, ReciprocalProjectionPlan, ConvergencePlan, CONVERGENCE_CONTRACT, READOUT_CONTRACT, FIBER_MATERIALIZATION_CONTRACT, ACTIVATION_CONTRACT, PROJECTION_CONTRACT, RECIPROCAL_PROJECTION_CONTRACT)  # noqa: F401
from .next_token import (NextTokenTrainingPlan, NextTokenScorePlan, LexiconBuildPlan, NextTokenPredictionPlan, LEXICON_BUILD_CONTRACT, NEXT_TOKEN_PREDICTION_CONTRACT, NEXT_TOKEN_TRAINING_CONTRACT, NEXT_TOKEN_SCORE_CONTRACT)  # noqa: F401
from .recovery import (RecoveryPlan, CueReplacementPlan, _COMPLETION_OBSERVATION_MODES, PreparedCompletion, CompletionPlan, RECOVERY_CONTRACT, CUE_REPLACEMENT_CONTRACT, COMPLETION_CONTRACT)  # noqa: F401
from .association import (AssociationPlan, _UNSTIMULATED_SOURCE_MODES, MergePlan, SeparationPlan, ASSOCIATION_CONTRACT, MERGE_CONTRACT, SEPARATION_CONTRACT)  # noqa: F401
from .binding import (BindingPlan, BindingReadPlan, InputDrivePlan, BindingStrengthPlan, SourceBindingPlan, BindingRecallPlan, BINDING_CONTRACT, BINDING_READ_CONTRACT, INPUT_DRIVE_CONTRACT, BINDING_STRENGTH_CONTRACT, SOURCE_BINDING_CONTRACT, BINDING_RECALL_CONTRACT)  # noqa: F401
from .consolidation import (ConsolidationPlan, ConsolidationProtocolPlan, ContextAccumulationPlan, ContextAccumulationStepPlan, CONSOLIDATION_CONTRACT, CONSOLIDATION_PROTOCOL_CONTRACT, CONTEXT_ACCUMULATION_CONTRACT, CONTEXT_STEP_CONTRACT)  # noqa: F401
from .sequence import (OrderedRecallPlan, SequenceMemorizePlan, ORDERED_RECALL_CONTRACT, SEQUENCE_MEMORIZE_CONTRACT)  # noqa: F401
from .attention import (AttentionPlan, ATTENTION_CONTRACT)  # noqa: F401
from .registry import (OPERATION_CONTRACTS)  # noqa: F401
