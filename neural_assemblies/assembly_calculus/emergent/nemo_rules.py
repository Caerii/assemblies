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

HONEST SCOPE: SVO ONLY, FOR NOW
-------------------------------
Only SVO is implemented, matching the reference. Do not read this as "word order
is now emergent" -- it is not. The order still enters as a learned discrete
parameter; what changes is that it selects a gating program instead of being
consulted during scoring.

Generalizing is not a loop over six orders, because WHICH WORD carries the slot
switch depends on the order. For SVO the verb switches AGENT->PATIENT (the
object follows it). For SOV both nouns precede the verb, so the switch must fire
on the FIRST NOUN instead. Getting that right needs the state-dependent advance
that the index channels exist to express, and it should be done after the SVO
path is measured against the current parser -- not speculatively.
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


def all_areas() -> List[str]:
    """Areas the inhibition state machine must know about."""
    return list(CORE_AREAS) + [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
