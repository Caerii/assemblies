"""Constituent-order typology: the six basic orders of S, V and O.

WHAT IS BUILT IN AND WHAT IS LEARNED -- read this before quoting any number
produced by code that imports this module.

BUILT IN (architectural given, not a result):

* The LABEL SET.  ``WORD_ORDERS`` enumerates the six permutations of subject,
  verb and object in Python.  The model does not discover that there are six
  possible orders, nor invent the names; it is handed the hypothesis space and
  chooses within it.  Calling the resulting typology "emergent" would be
  overclaiming.  What is emergent is WHICH of the six a given corpus is
  assigned to, and everything downstream of that assignment.
* The mapping from an order label to the slot sequence, to the verb's position
  class, and to a default role ranking.  These are definitional -- "OVS" means
  object, verb, subject -- not empirical.

LEARNED (from the corpus, by ``DistributionalMixin.infer_word_order``):

* The choice among the six labels, from role annotations and category
  transition statistics.
* The per-category surface position profiles, when the corpus supplies enough
  evidence (see ``DistributionalMixin.position_profiles``).

AN IDENTIFIABILITY LIMIT THAT MUST NOT BE PAPERED OVER.  Part-of-speech
transition statistics alone cannot separate the six orders; they can only
separate the verb's position into three classes:

    verb-initial   V N N     -> VSO or VOS
    verb-medial    N V N     -> SVO or OVS
    verb-final     N N V     -> SOV or OSV

Within each pair the two members have IDENTICAL category sequences.  Deciding
between them requires knowing which noun is the subject, i.e. role
annotations (or some other role cue such as case marking, which this model
does not yet have).  ``infer_word_order`` therefore reports its evidence
source, and when only transitions are available it says so and returns the
subject-initial member of the pair with the confidence halved.  A
transitions-only inference of "SVO" is NOT evidence against OVS.
"""

from typing import Dict, List, Tuple

# The six basic constituent orders. BUILT IN -- see module docstring.
WORD_ORDERS: Tuple[str, ...] = ("SVO", "SOV", "VSO", "OSV", "OVS", "VOS")

# Orders in which the object precedes the subject.
OBJECT_INITIAL_ORDERS: Tuple[str, ...] = ("OSV", "OVS", "VOS")

# Verb position class -> the two orders sharing it (see docstring).
VERB_POSITION_CLASSES: Dict[str, Tuple[str, ...]] = {
    "initial": ("VSO", "VOS"),
    "medial": ("SVO", "OVS"),
    "final": ("SOV", "OSV"),
}


def order_slots(order: str) -> Tuple[str, ...]:
    """Return the slot sequence of *order*, e.g. ``"OVS" -> ('O','V','S')``."""
    return tuple(order.upper())


def noun_slots(order: str) -> Tuple[str, ...]:
    """Slot sequence with the verb removed, e.g. ``"OVS" -> ('O','S')``.

    Used to map the nouns of a sentence, in surface order, onto roles.
    """
    return tuple(s for s in order_slots(order) if s != "V")


def verb_position_class(order: str) -> str:
    """Return ``"initial"`` / ``"medial"`` / ``"final"`` for *order*."""
    idx = order_slots(order).index("V")
    return ("initial", "medial", "final")[idx]


def is_object_initial(order: str) -> bool:
    """True when the object precedes the subject in *order*."""
    slots = order_slots(order)
    return slots.index("O") < slots.index("S")


def canonical_for_class(cls: str) -> str:
    """Subject-initial representative of a verb-position class.

    This is the value returned when only transition evidence is available and
    the subject/object ordering is not identifiable. It is a TIE-BREAK, not a
    finding -- callers should consult the reported evidence source.
    """
    return VERB_POSITION_CLASSES[cls][0]


def position_profiles_for_order(order: str) -> Dict[str, Tuple[float, float]]:
    """Typology-derived position prior over normalised sentence position.

    THIS IS A PRIOR, NOT A LEARNED RESULT.  It is derived definitionally from
    the slot sequence: with three slots, slot *i* occupies the normalised
    interval ``[i/3, (i+1)/3]``.  NOUN spans the union of the S and O slots,
    which for the verb-medial orders is deliberately the whole sentence --
    position genuinely carries no information about nounhood in an N V N
    language, and a narrower range would be fabricated precision.

    Modifier categories are placed relative to the noun slots (determiners and
    adjectives immediately before the first noun slot); that placement assumes
    a head-final noun phrase and is itself a prior, not evidence.
    """
    slots = order_slots(order)
    n = len(slots)

    def interval(slot: str) -> Tuple[float, float]:
        i = slots.index(slot)
        return (i / n, (i + 1) / n)

    s_lo, s_hi = interval("S")
    o_lo, o_hi = interval("O")
    v_lo, v_hi = interval("V")

    noun = (min(s_lo, o_lo), max(s_hi, o_hi))
    first_noun_lo = min(s_lo, o_lo)

    return {
        # Determiners / adjectives sit at the left edge of the first noun slot.
        "DET": (first_noun_lo, first_noun_lo + 1.0 / (2 * n)),
        "ADJ": (first_noun_lo, first_noun_lo + 1.0 / n),
        "NOUN": noun,
        "VERB": (v_lo, v_hi),
        "PRON": (s_lo, s_hi),
        # Adverbs and adpositions attach late in the clause regardless of the
        # S/V/O permutation; this is the weakest part of the prior.
        "ADV": ((n - 1) / n, 1.0),
        "PREP": ((n - 1) / n, 1.0),
    }


def order_from_role_sequence(seq: List[str]) -> str:
    """Map an observed sequence of role slots to an order label, or ``""``.

    *seq* is a list drawn from ``{"S", "V", "O"}`` in surface order, with
    repeats already collapsed. Returns ``""`` when the sequence is not one of
    the six full permutations (e.g. an intransitive S V clause).
    """
    label = "".join(seq)
    return label if label in WORD_ORDERS else ""
