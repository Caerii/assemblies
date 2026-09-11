"""
Assembly Calculus — named operations for neural assembly computation.

Provides first-class functions for the operations defined in:
Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (PNAS 2020)
Dabagia et al. "Computation with Sequences of Assemblies" (Neural Comp 2025)

THE MODEL IN ONE PARAGRAPH.  A brain is a set of areas, each holding ``n``
neurons with random recurrent and inter-area connectivity (Erdos-Renyi with
probability ``p``).  One time step is: every area sums the synaptic input
arriving from currently-firing presynaptic neurons; the top ``k`` neurons by
input fire (winners-take-all, standing in for local inhibition); every synapse
that carried input from a firing neuron to a winner is multiplied by
``(1 + beta)`` (Hebbian plasticity).  An ASSEMBLY is a set of ``k`` neurons
that has become stable under this dynamics -- it re-fires as a unit, so it can
serve as the model's representation of a concept, word, or memory.

That is the entire mechanism.  Everything in this package is a SCHEDULE over
it: which fibers are open on which step.  ``ops.py`` holds the named
schedules of the calculus, ``fiber.py`` makes gating declarative, and the
higher layers (``parser.py``, ``emergent/``, ``fsm.py``, ``pfa.py``) compose
those schedules into language and automaton behaviour.

TWO THINGS THAT CATCH EVERY NEW READER:

* Operations MUTATE the brain.  Plasticity is on by default, so measuring
  changes what is measured and running the same operation twice does not give
  the same answer.  Set ``brain.disable_plasticity = True`` for a pure read.
* An :class:`Assembly` is a SNAPSHOT, not a live handle.  An area holds one
  winner set and the next projection overwrites it; what persists across
  operations is the connectome, not the object.  Comparisons are by
  ``overlap``, never equality, and the baseline for "unrelated" is
  ``chance_overlap`` (k/n), not zero.

Modules with the most expository detail, if you are reading to understand
rather than to call: ``ops.py`` (the calculus), ``epwta.py`` (E%-WTA
formation, Hoff et al. 2026), ``binding.py`` (the four binding failure modes,
each with its measured signature), and
``emergent/parser_mixins/constituent_order.py`` (word order as synapses).

Operations:
    project            Stimulus → Area assembly formation
    reciprocal_project Area → Area assembly copying
    associate          Link two assemblies through a shared target
    merge              Combine two assemblies into a conjunctive representation
    pattern_complete   Recover full assembly from partial activation
    separate           Verify two stimuli create distinct assemblies
    sequence_memorize  Memorize an ordered sequence of stimuli
    ordered_recall     Recall a memorized sequence from a cue (requires LRI)
    consolidate        Systems consolidation replay (no connectome reset)
    activate_assembly    Inject a lexicon assembly snapshot into an area
    accumulate_context   Incremental prefix assembly into a context area

Readout:
    fuzzy_readout      Best-matching word above threshold, or None
    readout_all        All words with overlaps, sorted descending
    build_lexicon      Project each word's stimulus, snapshot the assembly

Structured computation:
    FSMNetwork         Deterministic finite state machine via assemblies
    PFANetwork         Probabilistic finite automaton via assemblies
    RandomChoiceArea   Neural coin-flip for stochastic selection
    ScaffoldNetwork    Main + auxiliary areas for faster sequence memorization

Next-token prediction:
    build_next_token_model  Build vocabulary lexicon for prediction
    train_on_corpus         Train on corpus via sequence memorization
    predict_next_token      Predict next token from context via overlap
    score_corpus            Score prediction accuracy on a corpus

Language parsing:
    NemoParser             Composed parser: category + role + word order
    EmergentParser         48-area emergent NEMO: 7 POS from grounding
                           (8 core areas; CONJ arrives distributionally)

Data:
    Assembly           Immutable snapshot of a neural assembly
    Sequence           Ordered list of assembly snapshots
    Lexicon            Dict mapping word strings to Assembly snapshots
    overlap            Measure overlap between two assemblies
    chance_overlap     Expected random overlap (k/n)

Control:
    FiberCircuit       Declarative gating of projection channels
"""

from .recovery import replace_neurons, observe_recovery, RecoveryObservation
from .assembly import Assembly, overlap, chance_overlap, overlap_from_binary
from .metrics import (
    compute_anchored_instability,
    compute_jaccard_instability,
    mean_jaccard_instability,
    measure_n400,
)
from .sequence import Sequence
from .contracts import (
    ASSOCIATION_CONTRACT, COMPLETION_CONTRACT, MERGE_CONTRACT,
    ORDERED_RECALL_CONTRACT, SEQUENCE_MEMORIZE_CONTRACT, SEPARATION_CONTRACT,
    OPERATION_CONTRACTS, PROJECTION_CONTRACT, RECIPROCAL_PROJECTION_CONTRACT,
    AssociationPlan, CompletionPlan, MergePlan, OperationContract,
    PreparedCompletion, ProjectionPlan, ProjectionStep, ReciprocalProjectionPlan,
    OrderedRecallPlan, SequenceMemorizePlan, SeparationPlan,
)
from .ops import (
    project,
    bind,
    read_binding,
    reciprocal_project,
    associate,
    merge,
    pattern_complete,
    separate,
    learn_assembly,
    learn_assembly_from_pattern,
    consolidate_pair,
    sequence_memorize,
    ordered_recall,
    activate_assembly,
)
from .consolidation import (
    PathwayReplay,
    MergeReplay,
    MultiProjectReplay,
    accumulate_context,
    accumulate_context_step,
    build_context_word_steps,
    consolidate,
    inhibit_all_areas,
    prepare_area_for_replay,
    replay_merge,
    replay_pathway,
)
from .tracing import (
    AssemblyTrace,
    PatternCompletionDiagnostic,
    ProjectionSweepConfig,
    RecallSweepConfig,
    ResponseDiagnostic,
    ResponseTrace,
    TraceStep,
    associate_trace,
    lri_recall_sweep,
    merge_trace,
    ordered_recall_trace,
    pattern_complete_trace,
    project_trace,
    projection_sweep,
    reciprocal_project_trace,
    snapshot_area,
    source_response_traces,
)
from .fiber import FiberCircuit
from .readout import fuzzy_readout, readout_all, build_lexicon, Lexicon
from .attention import AttentionCandidate, AttentionResult, attend
from .fsm import FSMNetwork
from .coin_config import AttractorConfig, SeedMixtureChoice
from .context_choice import ContextAttractorChoice, ContextChoiceProtocol, ContextChoiceObservation
from .pfa import PFANetwork, RandomChoiceArea, FlipMode, SoftmaxContextCoin
from .scaffold import (
    ScaffoldNetwork,
    ScaffoldRecallResult,
    compare_scaffold_vs_simple,
    sequence_memorize_scaffold,
)
from .transitions import Transition, TransitionMap
from .next_token import (
    build_next_token_model, train_on_corpus,
    predict_next_token, score_corpus,
)
from .parser import NemoParser
from .emergent import EmergentParser

__all__ = [
    # Data
    "Assembly", "AssemblyTrace", "PatternCompletionDiagnostic",
    "OperationContract", "ProjectionPlan", "ProjectionStep",
    "AssociationPlan", "CompletionPlan", "PreparedCompletion", "MergePlan",
    "ReciprocalProjectionPlan",
    "OPERATION_CONTRACTS", "PROJECTION_CONTRACT",
    "RECIPROCAL_PROJECTION_CONTRACT", "ASSOCIATION_CONTRACT", "MERGE_CONTRACT",
    "COMPLETION_CONTRACT",
    "ProjectionSweepConfig", "RecallSweepConfig", "ResponseDiagnostic",
    "ResponseTrace", "TraceStep", "Sequence", "Lexicon",
    "overlap", "chance_overlap", "overlap_from_binary", "snapshot_area",
    "compute_anchored_instability", "compute_jaccard_instability",
    "mean_jaccard_instability", "measure_n400",
    # Operations
    "project", "bind", "read_binding", "reciprocal_project",
    "associate", "merge",
    "pattern_complete", "separate", "learn_assembly", "learn_assembly_from_pattern",
    "consolidate_pair",
    "project_trace", "reciprocal_project_trace", "associate_trace",
    "merge_trace", "pattern_complete_trace", "ordered_recall_trace",
    "source_response_traces", "projection_sweep", "lri_recall_sweep",
    "sequence_memorize", "ordered_recall", "activate_assembly",
    "ProjectionPlan", "ReciprocalProjectionPlan", "AssociationPlan", "MergePlan",
    "CompletionPlan", "OrderedRecallPlan", "SequenceMemorizePlan", "SeparationPlan",
    "OperationContract", "OPERATION_CONTRACTS",
    # Consolidation & context
    "PathwayReplay", "MergeReplay", "MultiProjectReplay",
    "consolidate", "replay_pathway", "replay_merge",
    "inhibit_all_areas", "prepare_area_for_replay",
    "accumulate_context", "accumulate_context_step", "build_context_word_steps",
    "replace_neurons", "observe_recovery", "RecoveryObservation",
    # Readout
    "fuzzy_readout", "readout_all", "build_lexicon",
    "AttentionCandidate", "AttentionResult", "attend",
    # Structured computation
    "AttractorConfig", "ContextAttractorChoice", "ContextChoiceProtocol", "ContextChoiceObservation",
    "FSMNetwork", "PFANetwork", "SeedMixtureChoice", "RandomChoiceArea", "FlipMode", "SoftmaxContextCoin",
    "ScaffoldNetwork", "ScaffoldRecallResult", "compare_scaffold_vs_simple",
    "sequence_memorize_scaffold",
    "Transition", "TransitionMap",
    # Control
    "FiberCircuit",
    # Next-token prediction
    "build_next_token_model", "train_on_corpus",
    "predict_next_token", "score_corpus",
    # Language parsing
    "NemoParser",
    "EmergentParser",
]
