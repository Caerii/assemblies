"""Constants and statistics shared by every parser training stage.

Here rather than in ``core.py`` so a stage module can import a tuning constant
without importing the parser class -- which is what made these constants hard
to find, and what let a value drift out of step with the comment justifying it.

Every number here is a MEASURED default. The comments carry the measurements,
because a constant with no recorded justification is indistinguishable from a
constant nobody has ever tested.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from ..core.areas import (
    ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION, ROLE_LABEL_TO_AREA,
)

#: Projection rounds for MERGE into a multi-assembly area.
#: Below `self.rounds` because deep reinforcement builds one strong attractor
#: and merges the rest. On the ISOLATED primitive this is decisive: 94 merges
#: into an n=1000 area give pairwise overlap 0.752 at rounds 10, 0.078 at 3,
#: 0.051 at 1.
#:
#: NOT load-bearing on the parser's own path, and it is not what collapsed VP
#: -- that was `reset_area_connections(VP)`, see `train_phrases`. Swept on the
#: real path after removing the reset (seed 42, 32 constituents), the value
#: barely matters and is not even monotonic:
#:
#:     rounds    1      2      3      5     10
#:     overlap   0.299  0.444  0.562  0.390  0.406
#:     rank-1    0.781  0.781  0.781  0.781  0.781   (subject cue; flat)
#:
#: Kept at 2 for consistency with `_ROLE_BINDING_ROUNDS`, which faced the same
#: shared-area problem. Do not read the spread above as tuning signal: it is
#: one seed, and the differences are within what seed variation covers.
MERGE_ROUNDS = 2


# Modality field names on GroundingContext (order matters for dominant_modality)
_MODALITY_FIELDS = (
    "visual", "motor", "properties", "spatial",
    "social", "temporal", "emotional",
)

# Role annotation string -> brain area
# Weight of the structural (word-order / gating) prior relative to lexical
# binding evidence, which is an overlap in [0, 1]. Above 1.0 structure wins
# whenever lexical evidence is absent or weak, so a filler never seen in any
# role is still assigned systematically; a strong stored binding can still
# overturn a lower-ranked structural preference.
_STRUCTURAL_PRIOR = 1.2

# Additive smoothing applied when the per-role lexical margins are normalised
# into a distribution over the competing roles (see _assign_roles_neural). It
# damps the case where two margins differ only by measurement jitter, which is
# what a corpus with no lexical role preference produces. Measured over 8 seeds
# on a 4-noun corpus (balanced SVO accuracy / object-initial accuracy when the
# corpus makes each noun exclusive to one role):
#     eps = 0.00   0.995 / 1.000
#     eps = 0.05   0.995 / 1.000
#     eps = 0.10   1.000 / 0.812
#     eps = 0.15   0.938 / 0.500
# Above ~0.05 the smoothing starts eating the evidence it is meant to protect,
# so it is kept small and deliberately does not carry the decision.
_LEXICAL_SMOOTHING = 0.05

# Steps used to bind a filler into a role area.
#
# This is a *traversal* of an already-stabilized lexical assembly, not the
# formation of a new one. The two are different constants in the literature:
# Papadimitriou et al. (PNAS 2020) report "a stable assembly is formed after
# about T = 10 steps", and Mitropolsky & Collins & Papadimitriou (TACL 2021)
# converge project* in 10-20 firing epochs -- both for formation. Mitropolsky &
# Papadimitriou (2025) fire a word for tau = 2 steps to traverse a trained
# pathway, and 2 is also what measures best here: binding quality degrades
# monotonically with added recurrence (pairwise overlap between fillers 0.875
# at T=2 rising to 0.997 at T=20, with retrieval falling from 6/6 to 2/6),
# because many fillers share one role area and recurrence merges them.
_ROLE_BINDING_ROUNDS = 2

# Re-exported, not redefined: `core.areas.ROLE_LABEL_TO_AREA` is the one copy.
_ROLE_MAP = ROLE_LABEL_TO_AREA

# Role brain area -> human-readable label
_ROLE_LABEL = {
    ROLE_AGENT: "AGENT",
    ROLE_ACTION: "ACTION",
    ROLE_PATIENT: "PATIENT",
}


def _int_defaultdict() -> "defaultdict":
    """Picklable factory for a nested ``defaultdict(int)``.

    A nested ``lambda: defaultdict(int)`` is stored on the instance as the
    outer defaultdict's ``default_factory``, and local lambdas cannot be
    pickled -- which broke checkpoint/disk-cache round-trips of any parser
    carrying dist_stats. A module-level function pickles by reference.
    """
    return defaultdict(int)


@dataclass
class DistributionalStats:
    """Distributional statistics for category inference from raw text.

    Tracks position distributions, transitions, and co-occurrences to
    infer word categories without grounding information. Ported from
    the distributional tracking pattern in learner.py.
    """
    word_count: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    position_counts: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: defaultdict(_int_defaultdict))
    transitions: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int))
    category_transitions: Dict[Tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int))
    word_cooccurrence: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(_int_defaultdict))
    word_as_pre_verb: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    word_as_post_verb: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    word_as_action: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    # Counts of complete S/V/O surface permutations observed in ROLE-ANNOTATED
    # sentences, keyed by an order label from ``core.word_order.WORD_ORDERS``.
    # This is the only evidence that can separate a subject-initial order from
    # its object-initial twin (SVO/OVS, SOV/OSV, VSO/VOS): those pairs have
    # identical part-of-speech transition statistics. See
    # ``core/word_order.py`` for the identifiability argument.
    role_order_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    sentences_seen: int = 0

