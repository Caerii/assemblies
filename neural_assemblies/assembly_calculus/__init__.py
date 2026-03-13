"""
Assembly Calculus — named operations for neural assembly computation.

Provides first-class functions for the operations defined in:
Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (PNAS 2020)
Dabagia et al. "Computation with Sequences of Assemblies" (Neural Comp 2025)

Operations:
    project            Stimulus → Area assembly formation
    reciprocal_project Area → Area assembly copying
    associate          Link two assemblies through a shared target
    merge              Combine two assemblies into a conjunctive representation
    pattern_complete   Recover full assembly from partial activation
    separate           Verify two stimuli create distinct assemblies
    sequence_memorize  Memorize an ordered sequence of stimuli
    ordered_recall     Recall a memorized sequence from a cue (requires LRI)

Readout:
    fuzzy_readout      Best-matching word above threshold, or None
    readout_all        All words with overlaps, sorted descending
    build_lexicon      Project each word's stimulus, snapshot the assembly

Structured computation:
    FSMNetwork         Deterministic finite state machine via assemblies
    PFANetwork         Probabilistic finite automaton via assemblies
    RandomChoiceArea   Neural coin-flip for stochastic selection

Next-token prediction:
    build_next_token_model  Build vocabulary lexicon for prediction
    train_on_corpus         Train on corpus via sequence memorization
    predict_next_token      Predict next token from context via overlap
    score_corpus            Score prediction accuracy on a corpus

Language parsing:
    NemoParser             Composed parser: category + role + word order
    EmergentParser         40-area emergent NEMO: 7+ POS from grounding

Data:
    Assembly           Immutable snapshot of a neural assembly
    Sequence           Ordered list of assembly snapshots
    Lexicon            Dict mapping word strings to Assembly snapshots
    overlap            Measure overlap between two assemblies
    chance_overlap     Expected random overlap (k/n)

Control:
    FiberCircuit       Declarative gating of projection channels
"""

from .assembly import Assembly, overlap, chance_overlap
from .sequence import Sequence
from .ops import (
    project,
    reciprocal_project,
    associate,
    merge,
    pattern_complete,
    separate,
    sequence_memorize,
    ordered_recall,
)
from .fiber import FiberCircuit
from .readout import fuzzy_readout, readout_all, build_lexicon, Lexicon
from .fsm import FSMNetwork
from .pfa import PFANetwork, RandomChoiceArea
from .next_token import (
    build_next_token_model, train_on_corpus,
    predict_next_token, score_corpus,
)
from .parser import NemoParser
from .emergent import EmergentParser

__all__ = [
    # Data
    "Assembly", "Sequence", "Lexicon", "overlap", "chance_overlap",
    # Operations
    "project", "reciprocal_project", "associate", "merge",
    "pattern_complete", "separate",
    "sequence_memorize", "ordered_recall",
    # Readout
    "fuzzy_readout", "readout_all", "build_lexicon",
    # Structured computation
    "FSMNetwork", "PFANetwork", "RandomChoiceArea",
    # Control
    "FiberCircuit",
    # Next-token prediction
    "build_next_token_model", "train_on_corpus",
    "predict_next_token", "score_corpus",
    # Language parsing
    "NemoParser",
    "EmergentParser",
]
