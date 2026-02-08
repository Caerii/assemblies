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

Data:
    Assembly           Immutable snapshot of a neural assembly
    Sequence           Ordered list of assembly snapshots
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

__all__ = [
    # Data
    "Assembly", "Sequence", "overlap", "chance_overlap",
    # Operations
    "project", "reciprocal_project", "associate", "merge",
    "pattern_complete", "separate",
    "sequence_memorize", "ordered_recall",
    # Control
    "FiberCircuit",
]
