"""
Shared infrastructure for Assembly Calculus experiments.

This library provides composable building blocks for sentence processing
experiments. Each module handles one concern:

  vocabulary  — word lists, category membership, core area mappings
  grammar     — context-free grammar for generating variable-length sentences
  brain_setup — brain creation, area registration, lexicon construction
  training    — prediction bridge and role binding training protocols
  measurement — N400, P600, and composite ERP measurement

Experiments compose these modules rather than duplicating infrastructure.
The architecture supports extension to new word categories, role slots,
and sentence structures by adding vocabulary entries and grammar rules —
no changes to training or measurement logic required.
"""

from research.experiments.lib.vocabulary import (
    Vocabulary,
    DEFAULT_VOCAB,
    RECURSIVE_VOCAB,
)
from research.experiments.lib.grammar import SimpleCFG, RecursiveCFG
from research.experiments.lib.brain_setup import (
    create_language_brain,
    build_lexicon,
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
