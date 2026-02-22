"""
Shared infrastructure for Assembly Calculus experiments.

This library provides composable building blocks for sentence processing
experiments. Each module handles one concern:

  vocabulary  — word lists, category membership, core area mappings
  grammar     — context-free grammar for generating variable-length sentences
  brain_setup — brain creation, area registration, lexicon construction
  training    — prediction bridge and role binding training protocols
  measurement — N400, P600, and composite ERP measurement
  generation  — reverse readout, prediction chain, grammaticality scoring
  subgrammar  — production-rule decomposition, per-subgrammar measurement
  unsupervised — role discovery + training without role labels
  grounding   — sensory feature co-projection for semantic structure
  freeform    — free-form learner: language from scratch, one sentence at a time

Experiments compose these modules rather than duplicating infrastructure.
The architecture supports extension to new word categories, role slots,
and sentence structures by adding vocabulary entries and grammar rules —
no changes to training or measurement logic required.
"""

from research.experiments.lib.vocabulary import (
    Vocabulary,
    DEFAULT_VOCAB,
    DET_VOCAB,
    RECURSIVE_VOCAB,
)
from research.experiments.lib.grammar import SimpleCFG, RecursiveCFG, DetCFG
from research.experiments.lib.brain_setup import (
    create_language_brain,
    build_lexicon,
    build_semantic_structure,
    activate_word,
)
from research.experiments.lib.training import (
    train_prediction_pair,
    train_binding,
    train_sentence,
)
from research.experiments.lib.measurement import (
    measure_n400,
    measure_p600,
    measure_erps_at_position,
    forward_predict_from_context,
    measure_role_leakage,
    generate_test_triples,
    generate_pp_test_triples,
)
from research.experiments.lib.generation import (
    build_core_lexicon,
    readout_from_role,
    generate_from_prediction_chain,
    score_generation,
    check_novelty,
)
from research.experiments.lib.subgrammar import (
    SUBGRAMMAR_DEFS,
    classify_sentence,
    partition_by_subgrammar,
    SubgrammarStats,
)
from research.experiments.lib.unsupervised import (
    UnsupervisedConfig,
    create_unsupervised_brain,
    discover_role_area,
    train_sentence_unsupervised,
    build_role_mapping,
    train_binding_with_feedback,
    train_sentence_integrated,
)
from research.experiments.lib.freeform import (
    FreeFormConfig,
    FreeFormLearner,
)
from research.experiments.lib.grounding import (
    SENSORY_FEATURES,
    SEMANTIC_GROUPS,
    ALL_SENSORY_STIMULI,
    create_grounded_brain,
    ground_word,
    activate_sensory_only,
    train_prediction_pair_grounded,
    build_grounded_lexicon,
    IntegratedConfig,
    create_integrated_brain,
)
