"""The ERP contrast must be AREA-MATCHED, and its frames must be parseable.

Two defect classes, both of which shipped and neither of which any existing test
could see, because both produce plausible numbers rather than errors:

1. OUT-OF-VOCABULARY FRAME WORDS. `verb as object 3` read
   ["she", "hits", "the", "eats"]. `hits` is in no curriculum sentence and no
   holdout -- it existed only in frames.py -- so it categorised UNKNOWN and that
   item's MAIN VERB was unrecognised before its critical word was reached
   (p600 0.0000, phrase_stability 1.0000: the degenerate no-parse reading). A
   third of the category-violation arm was not a category violation.

2. AREA MISMATCH BY CONSTRUCTION. `structural_role_area` dispatches on the
   OBSERVED word's category, and a category violation IS a word whose observed
   category differs from the expected one -- so the violation arm reads a
   different area from its control in every frame set at every seed. Area
   identity alone reproduces the headline AUC with the condition held constant
   (research/notes/p600_is_confounded_with_area_identity.md).

These are STRUCTURAL assertions on purpose. They do not pin a magnitude, so they
survive the metric changing -- which it is expected to, and has, twice.
"""
import pytest

from neural_assemblies.assembly_calculus.emergent.core.grounding import VOCABULARY
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    expected_role_area,
    structural_role_area,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (
    AREA_MATCHED_CALIBRATION_FRAMES,
    DEFAULT_CALIBRATION_FRAMES,
    SWEEP_CALIBRATION_FRAMES,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_AGENT,
    ROLE_PATIENT,
    VP,
)

FRAME_SETS = {
    "DEFAULT": DEFAULT_CALIBRATION_FRAMES,
    "AREA_MATCHED": AREA_MATCHED_CALIBRATION_FRAMES,
    "SWEEP": SWEEP_CALIBRATION_FRAMES,
}


@pytest.mark.parametrize("set_name", sorted(FRAME_SETS))
def test_every_frame_word_is_in_the_vocabulary(set_name):
    """An OOV word in a frame is a silent no-parse, not a violation.

    VOCABULARY is the grounded lexicon the parser is built from; a word outside
    it can never be categorised, so the item measures the parser's failure to
    know a word rather than the contrast the frame was written to express.
    """
    unknown = {
        word
        for _label, _name, words in FRAME_SETS[set_name]
        for word in words
        if word not in VOCABULARY
    }
    assert not unknown, (
        f"{set_name} contains words outside VOCABULARY: {sorted(unknown)}. "
        f"Such an item parses UNKNOWN and reads p600 0.0 / stability 1.0 -- a "
        f"degenerate no-parse that looks like data. Use a trained word, or add "
        f"it to the lexicon deliberately (which moves the whole substrate)."
    )


def test_observed_dispatch_mismatches_the_arms():
    """Documents the confound. If this ever fails, the confound is GONE.

    Not a wish: the grammatical arm's critical word is a NOUN after a verb and
    the violation arm's is a VERB in the same slot, so observed-category
    dispatch necessarily sends them to different areas.
    """
    grammatical = structural_role_area("NOUN", verb_seen=True)
    violation = structural_role_area("VERB", verb_seen=True)
    assert grammatical == ROLE_PATIENT
    assert violation == VP
    assert grammatical != violation


def test_expected_dispatch_matches_the_arms():
    """The whole point: the same slot regardless of the word that arrived."""
    grammatical = expected_role_area(verb_seen_before=True, object_open=True)
    violation = expected_role_area(verb_seen_before=True, object_open=True)
    assert grammatical == violation == ROLE_PATIENT


def test_expected_dispatch_claims_nothing_before_the_verb():
    """Before the verb, a verb AND more subject material are both licensed.

    Returning an area here would invent structure the grammar does not predict,
    so None is the correct answer and the caller falls back.
    """
    assert expected_role_area(verb_seen_before=False, object_open=True) is None
    assert expected_role_area(verb_seen_before=False, object_open=False) is None


def test_expected_dispatch_claims_nothing_when_the_slot_is_closed():
    """An intransitive verb opens no object slot, so nothing is predicted.

    That position is the empty-project detector's business (#24), not the
    P600's -- there is no expected role area to read.
    """
    assert expected_role_area(verb_seen_before=True, object_open=False) is None


def test_pre_consumption_state_is_what_makes_the_verb_probe_vp():
    """Guards the `verb_seen_before` contract by exhibiting the failure.

    Taking `verb_seen` AFTER the word is consumed would make the verb itself
    predict an object slot and probe ROLE_PATIENT -- the same class of error the
    expected-slot dispatch exists to remove. With the correct (pre-consumption)
    state the verb falls back and reads VP.
    """
    # Correct: at the verb, no verb had been seen yet.
    assert expected_role_area(verb_seen_before=False, object_open=True) is None
    assert structural_role_area("VERB", verb_seen=False) == VP

    # Wrong (post-consumption) state would have claimed the object slot.
    assert expected_role_area(verb_seen_before=True, object_open=True) == ROLE_PATIENT


def test_subject_position_is_unchanged_by_the_flag():
    """The fallback must leave the pre-verb arm exactly as it shipped."""
    assert structural_role_area("NOUN", verb_seen=False) == ROLE_AGENT
    assert expected_role_area(verb_seen_before=False, object_open=True) is None
