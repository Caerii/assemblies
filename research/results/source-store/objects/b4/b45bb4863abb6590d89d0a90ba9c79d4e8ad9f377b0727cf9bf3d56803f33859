"""Rule programs: a LEARNED category selects a NEMO fiber/area control program.

This is the join between what this repo contributes and what NEMO contributes.
Grounding learns that "dog" is a NOUN (this repo's claim); the category then
selects a fiber program (NEMO's mechanism). Nothing here hand-specifies which
word is which -- `program_for_category` is keyed on the category the parser
inferred.

Ported from the rule tables in `.reference/dmitropolsky-assemblies/parser.py`
(`generic_noun`, `generic_trans_verb`, ...), translated into this repo's areas.

AREA MAPPING, and why it is not mechanical
------------------------------------------
The reference has ONE explicit lexical area, LEX, holding every word, with
fibers LEX<->SUBJ and LEX<->OBJ. This repo splits the lexicon across eight CORE
areas by category and binds into THEMATIC roles:

    reference LEX   ->  the word's own CORE area (NOUN_CORE, VERB_CORE, ...)
    reference SUBJ  ->  ROLE_AGENT
    reference OBJ   ->  ROLE_PATIENT
    reference VERB  ->  ROLE_ACTION

Mapping LEX per-category rather than to a single area is the substantive
difference. It also means "which core area is the source" is already decided by
the learned category, so a category error shows up as a fiber that was never
opened -- which is the ELAN-analogue signal discussed in the audit.

HOW WORD ORDER IS ENCODED (the part worth understanding)
--------------------------------------------------------
The reference starts with `initial_areas=[LEX, SUBJ, VERB]`: SUBJ and VERB are
open and **OBJ IS CLOSED**. A noun's PRE rules open fibers to BOTH ROLE_AGENT
and ROLE_PATIENT -- the noun does not choose. Whichever ROLE AREA is currently
open decides where it lands. Then the transitive verb's POST rules do

    DISINHIBIT ROLE_PATIENT     # open the object slot
    INHIBIT    ROLE_AGENT       # close the subject slot

so the *next* noun can only bind as patient. Word order is therefore *when the
slot switch fires*, carried by the intervening word's rules -- not a ranking
consulted at scoring time.

That is what makes this different from `constituent_role_order()`, which reads
`self.word_order_type` (a Python string) while SCORING candidate roles. Here the
order configures the GATING and the neural dynamics do the binding.

ALL SIX ORDERS, VIA SLOT CONSUMPTION (see `SLOT_SEQUENCES`)
------------------------------------------------------------
The rules above are the SVO table, where the VERB carries the slot switch. That
does not generalize, and the reason is worth keeping: for SOV both nouns precede
the verb, so the switch must fire after the FIRST NOUN -- and then the same
category (NOUN) needs two different POST rule sets, which a category->program
map cannot express.

Attaching the advance to SLOT CONSUMPTION rather than to a word dissolves it.
The order becomes a SEQUENCE of slots, exactly one is open at a time, and every
content word takes the next; the identical machinery then runs all six orders,
object-initial ones included. Measured on the reversible items, seed 42:
SVO/SOV/VSO/VOS/OSV/OVS all 1.000, with SVO reproducing the SVO-table result.

HONEST SCOPE, unchanged in the part that matters. This is NOT "word order is
emergent". The order still enters as a learned discrete parameter -- it selects
a slot sequence instead of a rule table, and the gating is a PERFECT POSITIONAL
TEMPLATE for whichever order it is given, exactly as the SVO path was. Scoring
positional items with a positional mechanism is why every order reads 1.000;
irreversible items, where lexical experience must override position, remain 0.000
for all six, because gating still has no lexical route (that is mutual
inhibition's job -- see `nemo_competitive_ab.py`).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

from neural_assemblies.core.inhibition import (
    DISINHIBIT, INHIBIT, Rule, area_rule, fiber_rule,
)

from .core.areas import (
    ADJ_CORE, CORE_AREAS, DET_CORE, NOUN_CORE, PRON_CORE, ROLE_ACTION,
    ROLE_AGENT, ROLE_PATIENT, VERB_CORE,
)

#: Role slots a filler can bind into, in the order SVO consumes them.
SVO_SLOTS = (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT)


class RuleProgram(NamedTuple):
    """Rules applied before and after the word's projection rounds."""

    pre: List[Rule]
    post: List[Rule]


def competitive_verb_program(core_area: str = VERB_CORE) -> RuleProgram:
    """Transitive verb that OPENS the object slot WITHOUT closing the subject.

    The difference from `trans_verb_program` is one missing rule -- it does not
    `INHIBIT ROLE_AGENT` -- and that single omission changes the model. With
    both role slots open, a following noun projects into BOTH, and MUTUAL
    INHIBITION arbitrates on learned weight instead of word order deciding.

    That is the AC-native lexical override: the preference is already in the
    connectome (a noun trained only as a patient has a weak agent pathway), and
    leaving both slots open is what lets it express. Measured in isolation on
    the two TRAINED role areas, MI picks the corpus-taught role 7/7.
    """
    return RuleProgram(
        pre=[
            fiber_rule(DISINHIBIT, core_area, ROLE_ACTION, 0),
            fiber_rule(DISINHIBIT, ROLE_ACTION, ROLE_AGENT, 0),
        ],
        post=[
            fiber_rule(INHIBIT, core_area, ROLE_ACTION, 0),
            area_rule(DISINHIBIT, ROLE_PATIENT, 0),
            # NOTE the absence of `area_rule(INHIBIT, ROLE_AGENT, 0)`.
        ],
    )


def competitive_initial_open_areas() -> List[str]:
    """BOTH role slots open from the start, so lexical preference can decide.

    The SVO program opens only ROLE_AGENT, so the FIRST noun is forced there
    before it can ever compete -- measured: for "ball chases dog" the project
    map at word 1 is [NOUN_CORE, ROLE_AGENT] alone. `ball` never gets the chance
    to prefer PATIENT, whatever the corpus taught. Opening both is the
    precondition for the competition to mean anything.
    """
    return list(CORE_AREAS) + [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]


def initial_open_areas(word_order_type: str = "SVO") -> List[str]:
    """Areas that start disinhibited. Mirrors `initial_areas=[LEX, SUBJ, VERB]`.

    Every CORE area is open (they are the lexical source), plus the first role
    slot and the action slot. ROLE_PATIENT starts CLOSED -- that is what stops
    the first noun binding as patient, and it is the initial condition the whole
    order mechanism rests on.
    """
    if word_order_type not in (None, "SVO"):
        raise NotImplementedError(
            f"only SVO rule programs exist so far, got {word_order_type!r}. "
            f"See the module docstring: the slot switch moves to a different "
            f"word for orders where the object precedes the verb."
        )
    return list(CORE_AREAS) + [ROLE_AGENT, ROLE_ACTION]


def noun_program(core_area: str = NOUN_CORE) -> RuleProgram:
    """A noun offers itself to BOTH role slots and lets area state decide.

    This is the move the categorical lookup replaced. `structural_role_area`
    picks a target from the word's category; here the word opens fibers to
    agent and patient alike, and whichever slot is open receives it. No
    comparison between areas, no winner-take-all, no Python set.
    """
    return RuleProgram(
        pre=[
            fiber_rule(DISINHIBIT, core_area, ROLE_AGENT, 0),
            fiber_rule(DISINHIBIT, core_area, ROLE_PATIENT, 0),
        ],
        post=[
            fiber_rule(INHIBIT, core_area, ROLE_AGENT, 0),
            fiber_rule(INHIBIT, core_area, ROLE_PATIENT, 0),
        ],
    )


def trans_verb_program(core_area: str = VERB_CORE) -> RuleProgram:
    """A transitive verb binds into ROLE_ACTION and ADVANCES THE SLOT.

    The two POST area rules are the word-order mechanism: opening ROLE_PATIENT
    and closing ROLE_AGENT is what makes the following noun an object. Removing
    either one does not degrade the parse, it changes the grammar.
    """
    return RuleProgram(
        pre=[
            fiber_rule(DISINHIBIT, core_area, ROLE_ACTION, 0),
            fiber_rule(DISINHIBIT, ROLE_ACTION, ROLE_AGENT, 0),
        ],
        post=[
            fiber_rule(INHIBIT, core_area, ROLE_ACTION, 0),
            area_rule(DISINHIBIT, ROLE_PATIENT, 0),   # open the object slot
            area_rule(INHIBIT, ROLE_AGENT, 0),        # close the subject slot
        ],
    )


def intrans_verb_program(core_area: str = VERB_CORE) -> RuleProgram:
    """Same as transitive minus the object slot -- it never opens ROLE_PATIENT.

    The reference draws exactly this distinction (`generic_intrans_verb` omits
    `DISINHIBIT OBJ`), and it is why an intransitive sentence cannot acquire a
    spurious patient.
    """
    return RuleProgram(
        pre=[
            fiber_rule(DISINHIBIT, core_area, ROLE_ACTION, 0),
            fiber_rule(DISINHIBIT, ROLE_ACTION, ROLE_AGENT, 0),
        ],
        post=[
            fiber_rule(INHIBIT, core_area, ROLE_ACTION, 0),
            area_rule(INHIBIT, ROLE_AGENT, 0),
        ],
    )


def modifier_program(core_area: str) -> RuleProgram:
    """Determiners and adjectives: bind, then close, without advancing a slot.

    They must not touch area state -- a determiner that advanced the slot would
    make "the dog" put "dog" in the object position.
    """
    return RuleProgram(
        pre=[fiber_rule(DISINHIBIT, core_area, ROLE_AGENT, 0),
             fiber_rule(DISINHIBIT, core_area, ROLE_PATIENT, 0)],
        post=[fiber_rule(INHIBIT, core_area, ROLE_AGENT, 0),
              fiber_rule(INHIBIT, core_area, ROLE_PATIENT, 0)],
    )


#: Learned category -> the CORE area that category's words live in.
CATEGORY_CORE = {
    "NOUN": NOUN_CORE,
    "PRON": PRON_CORE,
    "VERB": VERB_CORE,
    "ADJ": ADJ_CORE,
    "DET": DET_CORE,
}


def program_for_category(category: str, *, transitive: bool = True,
                         core_area: Optional[str] = None) -> Optional[RuleProgram]:
    """Select the fiber program for a LEARNED category.

    Returns None for categories with no program yet, so a caller can fall back
    rather than silently parsing with no gating.

    `transitive` is a property of the verb, which this repo does not currently
    infer; the caller must supply it. Defaulting it to True is a real
    simplification and is why intransitive sentences need checking separately.
    """
    core = core_area or CATEGORY_CORE.get(category)
    if core is None:
        return None
    if category in ("NOUN", "PRON"):
        return noun_program(core)
    if category == "VERB":
        return trans_verb_program(core) if transitive else intrans_verb_program(core)
    if category in ("ADJ", "DET"):
        return modifier_program(core)
    return None


#: The order of role slots each word order CONSUMES them in.
#:
#: This is what generalizes the gating past SVO, and it required abandoning the
#: idea that a WORD carries the slot switch. In SVO the verb advances
#: AGENT->PATIENT because the object follows it; in SOV both nouns precede the
#: verb, so the switch must fire after the FIRST NOUN -- and then the same
#: category (NOUN) would need two different POST rule sets, which a
#: category->program map cannot express. That is the wall the module docstring
#: described.
#:
#: Attach the advance to SLOT CONSUMPTION instead of to a word and it
#: disappears: the order is just a SEQUENCE of slots, exactly one is open at a
#: time, and every content word takes the next. The identical mechanism then
#: runs all six orders, and it is the same winner-closes-slot device the
#: competitive path already uses -- close what was filled, open what is next.
#:
#: Modifiers must NOT advance (a determiner would make "the dog" put `dog` in
#: the object slot), so the sequencer only fires for content categories. That
#: is a decision on the LEARNED category, not on the word.
SLOT_SEQUENCES: Dict[str, tuple] = {
    "SVO": (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT),
    "SOV": (ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION),
    "VSO": (ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT),
    "VOS": (ROLE_ACTION, ROLE_PATIENT, ROLE_AGENT),
    "OSV": (ROLE_PATIENT, ROLE_AGENT, ROLE_ACTION),
    "OVS": (ROLE_PATIENT, ROLE_ACTION, ROLE_AGENT),
}

#: Categories that CONSUME a role slot. Determiners and adjectives bind without
#: consuming, which is why they must not advance the sequence.
CONTENT_CATEGORIES = frozenset({"NOUN", "PRON", "VERB"})


def slot_sequence(word_order_type: str = "SVO") -> tuple:
    """Slots in the order this word order fills them."""
    try:
        return SLOT_SEQUENCES[word_order_type]
    except KeyError:
        raise NotImplementedError(
            f"no slot sequence for {word_order_type!r}; "
            f"known orders are {sorted(SLOT_SEQUENCES)}"
        ) from None


def sequential_initial_open_areas(word_order_type: str = "SVO") -> List[str]:
    """Every CORE area, plus ONLY the first slot of the sequence.

    Stricter than `initial_open_areas`, which opens ROLE_AGENT and ROLE_ACTION
    together because the SVO rules relied on the verb finding ACTION already
    open. Under sequential gating exactly one slot is open at any time and the
    sequencer opens the next, so the initial condition is simply "the first slot
    of this order" -- and it is what makes an object-initial order like OVS
    expressible at all.
    """
    return list(CORE_AREAS) + [slot_sequence(word_order_type)[0]]


def sequential_verb_program(core_area: str = VERB_CORE) -> RuleProgram:
    """A verb that binds its action and does NOT move the slot itself.

    Identical to `trans_verb_program` minus the two POST area rules, because
    under sequential gating the advance belongs to the sequencer rather than to
    the verb. Keeping the verb's own advance as well would double-step the
    sequence.
    """
    return RuleProgram(
        pre=[
            fiber_rule(DISINHIBIT, core_area, ROLE_ACTION, 0),
            fiber_rule(DISINHIBIT, ROLE_ACTION, ROLE_AGENT, 0),
        ],
        post=[fiber_rule(INHIBIT, core_area, ROLE_ACTION, 0)],
    )


#: Role slots a filler can be expected in. ROLE_ACTION is excluded: it is open
#: throughout the SVO program, so counting it would make every verb "expected"
#: everywhere and the mismatch could never fire.
_FILLER_SLOTS = (ROLE_AGENT, ROLE_PATIENT)


def category_addresses_open_slot(state, category: str, *,
                                 transitive: bool = True,
                                 core_area: Optional[str] = None,
                                 slots=_FILLER_SLOTS) -> Optional[bool]:
    """Does this word's LEARNED category address the slot the syntax has open?

    THE ELAN ANALOGUE, and it costs nothing to compute. The audit's finding was
    that "there is no ELAN because expected-vs-actual is not represented
    anywhere". With fiber state it IS: `expected` is whichever role slot the
    rule programs have left disinhibited, and `actual` is the set of slots the
    incoming word's program opens fibers to. A category violation is exactly
    the case where those do not intersect --

        noun in object position   noun_program opens fibers to AGENT and
                                  PATIENT, PATIENT is the open slot   -> True
        verb in object position   trans_verb_program targets ROLE_ACTION
                                  only, so the open patient slot is
                                  never addressed                     -> False
        NOVEL noun                still a noun program, so it addresses the
                                  slot exactly like a trained noun     -> True

    That last row is the point: this responds to word-CATEGORY violations and
    is blind to lexical novelty, which is the dissociation the ERP literature
    reports between ELAN and N400. It is also available at word ONSET, before
    any binding or settling, matching ELAN's early timing.

    HONEST SCOPE. This is a predicate over gating state, NOT a neural energy --
    it reads the rule program rather than measuring dynamics. What keeps it
    from being a category lookup in disguise is that the CATEGORY IS LEARNED
    (that is this repo's contribution); the mismatch is between a learned
    category's program and the syntactic state, and nothing here names which
    word is which. Its virtue is the same one that made the competition the
    best P600 candidate: no ad-hoc energy function.

    Returns None when no filler slot is open at all -- there is nothing being
    expected, so "mismatch" is not defined rather than false.
    """
    open_slots = [s for s in slots if state.area_open(s)]
    if not open_slots:
        return None
    program = program_for_category(
        category, transitive=transitive, core_area=core_area)
    if program is None:
        # A category with no rule program cannot address anything. That is a
        # mismatch, not a missing measurement.
        return False
    # Only fibers leaving the WORD'S OWN CORE count. `trans_verb_program` also
    # opens ROLE_ACTION<->ROLE_AGENT so the verb can reach its subject, and
    # counting that made a sentence-initial verb look like it addressed the
    # agent slot -- i.e. no category violation at all. Caught by
    # `test_subject_position_also_addressed_by_a_noun`.
    core = core_area or CATEGORY_CORE.get(category)
    targets = {rule.a2 for rule in program.pre
               if rule.kind == "fiber" and rule.action == DISINHIBIT
               and rule.a1 == core}
    return any(slot in targets for slot in open_slots)


def all_areas() -> List[str]:
    """Areas the inhibition state machine must know about."""
    return list(CORE_AREAS) + [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
