"""The category->program table, and the SVO gating it produces.

The headline test is `test_svo_sequence_routes_two_nouns_to_different_slots`:
the same noun program runs twice and lands in different roles purely because the
verb moved the open slot. That is the property the Python `inhibited` set is
currently standing in for.
"""

import pytest

from neural_assemblies.core.inhibition import InhibitionState, apply_rule
from neural_assemblies.assembly_calculus.emergent.nemo_rules import (
    CATEGORY_CORE, CONTENT_CATEGORIES, SLOT_SEQUENCES, all_areas,
    category_addresses_open_slot, initial_open_areas, intrans_verb_program,
    noun_program, program_for_category, sequential_initial_open_areas,
    sequential_verb_program, slot_sequence, trans_verb_program,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    NOUN_CORE, ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT, VERB_CORE,
)


class _FakeBrain:
    """Only `areas[x].winners` is consulted by project_map."""

    class _A:
        def __init__(self, active):
            self.winners = [1, 2, 3] if active else []

    def __init__(self, active):
        self.areas = {a: self._A(a in active) for a in all_areas()}


class TestInitialState:
    def test_patient_slot_starts_closed(self):
        """The initial condition the whole order mechanism rests on."""
        s = InhibitionState(all_areas(), initial_open_areas("SVO"))
        assert s.area_open(ROLE_AGENT)
        assert s.area_open(ROLE_ACTION)
        assert not s.area_open(ROLE_PATIENT), (
            "ROLE_PATIENT must start inhibited, or the first noun can bind as "
            "patient and SVO is not encoded at all")

    def test_unimplemented_orders_raise(self):
        """Better than silently parsing SOV with SVO gating."""
        with pytest.raises(NotImplementedError, match="only SVO"):
            initial_open_areas("SOV")


class TestCategorySelectsProgram:
    def test_learned_category_selects_the_core_area(self):
        assert CATEGORY_CORE["NOUN"] == NOUN_CORE
        assert CATEGORY_CORE["VERB"] == VERB_CORE

    def test_unknown_category_returns_none(self):
        """Callers must be able to fall back, not parse ungated."""
        assert program_for_category("CONJ") is None

    def test_transitivity_changes_the_program(self):
        t = program_for_category("VERB", transitive=True)
        i = program_for_category("VERB", transitive=False)
        opens_patient = [r for r in t.post
                         if r.kind == "area" and r.a1 == ROLE_PATIENT]
        assert opens_patient, "transitive verb must open the object slot"
        assert not [r for r in i.post
                    if r.kind == "area" and r.a1 == ROLE_PATIENT], (
            "intransitive verb must NOT open the object slot, or an "
            "intransitive sentence can acquire a spurious patient")


class TestNounOffersBothSlots:
    def test_noun_opens_agent_and_patient(self):
        """The noun does not choose; area state decides."""
        p = noun_program(NOUN_CORE)
        targets = {r.a2 for r in p.pre if r.kind == "fiber"}
        assert targets == {ROLE_AGENT, ROLE_PATIENT}


class TestSvoGating:
    def _state(self):
        return InhibitionState(all_areas(), initial_open_areas("SVO"))

    def test_first_noun_can_only_reach_agent(self):
        s = self._state()
        for r in noun_program(NOUN_CORE).pre:
            apply_rule(s, r)
        b = _FakeBrain({NOUN_CORE})
        targets = s.project_map(b).get(NOUN_CORE, [])
        assert ROLE_AGENT in targets
        assert ROLE_PATIENT not in targets, (
            "patient slot is closed, so the noun must not reach it")

    def test_verb_advances_the_slot(self):
        s = self._state()
        for r in trans_verb_program(VERB_CORE).post:
            apply_rule(s, r)
        assert s.area_open(ROLE_PATIENT), "verb must OPEN the object slot"
        assert not s.area_open(ROLE_AGENT), "verb must CLOSE the subject slot"

    def test_svo_sequence_routes_two_nouns_to_different_slots(self):
        """THE POINT. Identical noun program, different role, no scoring.

        Nothing here compares areas, ranks candidates, or consults a stored
        word-order string. The second noun becomes the patient because the verb
        moved the open slot.
        """
        s = self._state()
        noun = noun_program(NOUN_CORE)
        verb = trans_verb_program(VERB_CORE)
        b_noun = _FakeBrain({NOUN_CORE})

        for r in noun.pre:
            apply_rule(s, r)
        first = s.project_map(b_noun).get(NOUN_CORE, [])
        for r in noun.post:
            apply_rule(s, r)

        for r in verb.pre:
            apply_rule(s, r)
        for r in verb.post:
            apply_rule(s, r)

        for r in noun.pre:
            apply_rule(s, r)
        second = s.project_map(b_noun).get(NOUN_CORE, [])

        assert ROLE_AGENT in first and ROLE_PATIENT not in first
        assert ROLE_PATIENT in second and ROLE_AGENT not in second

    def test_no_war_of_fibers_in_a_correct_sequence(self):
        """A well-formed program never offers a word two slots at once."""
        s = self._state()
        for r in noun_program(NOUN_CORE).pre:
            apply_rule(s, r)
        b = _FakeBrain({NOUN_CORE})
        s.check_war_of_fibers(s.project_map(b), NOUN_CORE)   # must not raise

    def test_intransitive_never_opens_the_object_slot(self):
        s = self._state()
        for r in intrans_verb_program(VERB_CORE).post:
            apply_rule(s, r)
        assert not s.area_open(ROLE_PATIENT)


class TestElanCategoryMismatch:
    """The ELAN analogue: does the word's category address the open slot?

    The load-bearing row is `test_novel_noun_behaves_like_a_trained_noun`. A
    measure that fired for novel words would be a generic anomaly detector, not
    an ELAN -- the ERP literature's whole point is that ELAN tracks word
    CATEGORY while N400 tracks lexical/semantic fit, and they dissociate.
    """

    @staticmethod
    def _object_position_state():
        """State after 'the dog chases': the object slot is the open one."""
        state = InhibitionState(all_areas(), initial_open_areas("SVO"))
        for rule in trans_verb_program(VERB_CORE).post:
            apply_rule(state, rule)
        assert state.area_open(ROLE_PATIENT)
        assert not state.area_open(ROLE_AGENT)
        return state

    def test_noun_addresses_the_open_object_slot(self):
        state = self._object_position_state()
        assert category_addresses_open_slot(state, "NOUN") is True

    def test_verb_in_object_position_is_a_mismatch(self):
        """The category violation: the verb's program never targets the slot."""
        state = self._object_position_state()
        assert category_addresses_open_slot(state, "VERB") is False, (
            "a transitive verb opens VERB_CORE->ROLE_ACTION only, so the open "
            "patient slot is never addressed -- that IS the category violation"
        )

    def test_novel_noun_behaves_like_a_trained_noun(self):
        """Blind to lexical novelty, by construction.

        Novelty lives in the connectome, not the rule table: an unseen noun
        that was CATEGORISED as a noun selects `noun_program` exactly like a
        trained one. So this fires for category violations and not for novel
        words, which is the ELAN/N400 dissociation.
        """
        state = self._object_position_state()
        assert (category_addresses_open_slot(state, "NOUN")
                is category_addresses_open_slot(state, "PRON") is True)

    def test_subject_position_also_addressed_by_a_noun(self):
        """At sentence start the open slot is AGENT, and a noun addresses it."""
        state = InhibitionState(all_areas(), initial_open_areas("SVO"))
        assert category_addresses_open_slot(state, "NOUN") is True
        assert category_addresses_open_slot(state, "VERB") is False

    def test_returns_none_when_no_filler_slot_is_open(self):
        """Nothing expected -> mismatch is undefined, not False."""
        state = InhibitionState(all_areas(), initial_open_areas("SVO"))
        state.inhibit_area(ROLE_AGENT, 0)
        assert category_addresses_open_slot(state, "NOUN") is None

    def test_unknown_category_is_a_mismatch_not_a_missing_value(self):
        state = self._object_position_state()
        assert category_addresses_open_slot(state, "ADVERB") is False


class TestSequentialSlotOrders:
    """One mechanism, six word orders.

    The old rule table hard-coded the verb as the word that advances the slot,
    which is why anything but SVO raised. Attaching the advance to SLOT
    CONSUMPTION instead makes the order just a sequence, and the same machinery
    runs all six -- including the object-initial orders that could not be
    expressed at all before.
    """

    def test_every_order_has_all_three_slots_exactly_once(self):
        for order, seq in SLOT_SEQUENCES.items():
            assert sorted(seq) == sorted(
                (ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT)), order

    def test_sequence_matches_the_name(self):
        """S/O/V in the name must be the order the slots are consumed in."""
        letter = {ROLE_AGENT: "S", ROLE_PATIENT: "O", ROLE_ACTION: "V"}
        for order, seq in SLOT_SEQUENCES.items():
            assert "".join(letter[s] for s in seq) == order

    def test_object_initial_order_opens_the_patient_slot_first(self):
        """OVS was previously inexpressible: the first slot is the OBJECT."""
        opened = sequential_initial_open_areas("OVS")
        assert ROLE_PATIENT in opened
        assert ROLE_AGENT not in opened, (
            "an object-initial order must NOT start with the agent slot open, "
            "or the first noun binds as agent and the order is not encoded"
        )

    def test_svo_still_opens_the_agent_slot_first(self):
        opened = sequential_initial_open_areas("SVO")
        assert ROLE_AGENT in opened
        assert ROLE_PATIENT not in opened

    def test_only_the_first_slot_is_open(self):
        """Exactly one open slot is what makes the readout unambiguous."""
        for order in SLOT_SEQUENCES:
            opened = set(sequential_initial_open_areas(order))
            roles = opened & {ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT}
            assert roles == {slot_sequence(order)[0]}, order

    def test_unknown_order_raises_rather_than_defaulting(self):
        """Silently falling back to SVO would fake support for an order."""
        with pytest.raises(NotImplementedError):
            slot_sequence("XYZ")

    def test_sequential_verb_program_does_not_move_the_slot(self):
        """The sequencer owns the advance; a verb carrying it too double-steps."""
        prog = sequential_verb_program(VERB_CORE)
        assert not [r for r in prog.post if r.kind == "area"], (
            "sequential_verb_program must not contain area rules -- those are "
            "the SVO slot advance, and the sequencer already does it"
        )

    def test_modifiers_are_not_content_and_so_never_advance(self):
        """A determiner that advanced would put 'the dog' in the object slot."""
        assert "DET" not in CONTENT_CATEGORIES
        assert "ADJ" not in CONTENT_CATEGORIES
        assert {"NOUN", "PRON", "VERB"} <= CONTENT_CATEGORIES
