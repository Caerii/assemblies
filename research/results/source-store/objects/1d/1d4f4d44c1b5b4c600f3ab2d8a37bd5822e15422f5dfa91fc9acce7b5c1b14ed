"""Pin the homograph-safe lexicon index (multi-entry per surface form).

The old ``_build_lexicon_index`` mapped each surface form to the FIRST
entry that claimed it, with NOUNS indexed before VERBS.  Noun-verb
homographs -- "loves", "hates", "fears", "hopes", "answers", "surprises"
(3sg verb forms that are also noun plurals), "lives" (life.plural /
live.3sg), "thought" (noun / think.past) -- therefore resolved to the
noun entry: ``lookup_verb_form`` returned None for them, ``detect_tense``
could not see them as verbs, and ``detect_number``'s teacher labelled
corpus verb tokens as noun plurals.  Found during task #129
(what_variation_buys), where these homographs contaminated the tense and
number test sets.

The index now keeps EVERY claimant per surface form, in category-priority
order, and POS-aware call sites filter with ``lookup_lexicon_entries``.
"""

import pytest

from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (
    lookup_lexicon_entries,
    lookup_lexicon_entry,
    lookup_verb_form,
    verb_surface_form,
)
from neural_assemblies.assembly_calculus.emergent.parser_mixins.morphosyntax import (
    MorphosyntaxMixin,
)


class _Teacher(MorphosyntaxMixin):
    """Minimal host for the pure-Python detect_* teacher methods."""

    def __init__(self):
        self.word_grounding = {}


# The 3sg-verb / noun-plural homographs reported by task #129.
NOUN_VERB_3SG_HOMOGRAPHS = ["loves", "hates", "fears", "hopes",
                            "answers", "surprises"]


class TestHomographVisibility:
    """"loves" must be visible as BOTH a verb 3sg form and a noun plural."""

    def test_loves_has_both_readings(self):
        candidates = lookup_lexicon_entries("loves")
        by_pos = {pos: entry for entry, pos in candidates}
        assert "NOUN" in by_pos and "VERB" in by_pos
        assert by_pos["NOUN"]["forms"]["plural"] == "loves"
        assert by_pos["VERB"]["forms"]["3sg"] == "loves"

    @pytest.mark.parametrize("word", NOUN_VERB_3SG_HOMOGRAPHS)
    def test_3sg_homographs_visible_as_verbs(self, word):
        # Each of these read None before the multi-entry index.
        result = lookup_verb_form(word)
        assert result is not None
        assert result[1] == "PRESENT"
        verb_entry = lookup_lexicon_entries(word, pos="VERB")[0][0]
        assert verb_entry["forms"]["3sg"] == word

    def test_cross_lemma_homographs(self):
        # "lives" is life.plural AND live.3sg -- different lemma strings.
        assert lookup_verb_form("lives") == ("live", "PRESENT")
        lemmas = {e["lemma"] for e, _ in lookup_lexicon_entries("lives")}
        assert lemmas == {"life", "live"}
        # "thought" is a noun AND think.past -- the old index hid the
        # PAST reading from detect_tense entirely.
        assert lookup_verb_form("thought") == ("think", "PAST")

    def test_pos_filter(self):
        nouns = lookup_lexicon_entries("loves", pos="NOUN")
        assert len(nouns) == 1 and nouns[0][1] == "NOUN"
        assert lookup_lexicon_entries("loves", pos="ADJ") == []
        assert lookup_lexicon_entries("not-a-word") == []


class TestBackCompat:
    """lookup_lexicon_entry keeps the old category-priority winner."""

    def test_first_candidate_is_noun(self):
        result = lookup_lexicon_entry("loves")
        assert result is not None
        entry, pos = result
        assert pos == "NOUN" and entry["lemma"] == "love"

    def test_plain_verb_still_resolves(self):
        result = lookup_lexicon_entry("runs")
        assert result is not None
        entry, pos = result
        assert pos == "VERB" and entry["lemma"] == "run"

    def test_unknown_word_is_none(self):
        assert lookup_lexicon_entry("not-a-word") is None


class TestVerbSurfaceForm:
    def test_homograph_verb_gets_3sg(self):
        # Old behavior: noun entry shadowed the verb, returned bare "love",
        # silently breaking subject agreement in generated corpora.
        assert verb_surface_form("love") == "loves"

    def test_inflected_form_is_not_treated_as_lemma(self):
        # "saw" matches see.past under the VERB filter; without the
        # entry-lemma guard this would return "sees".
        assert verb_surface_form("saw") == "saw"

    def test_noun_only_lemma_unchanged(self):
        assert verb_surface_form("dog") == "dog"


class TestTeacherSignals:
    def test_detect_tense_sees_homograph_past(self):
        teacher = _Teacher()
        # think.past was invisible behind the noun "thought".
        assert teacher.detect_tense(["the", "boy", "thought"]) == "PAST"
        assert teacher.detect_tense(["the", "girl", "loves", "the",
                                     "dog"]) == "PRESENT"

    def test_detect_number_declines_ambiguous_forms(self):
        teacher = _Teacher()
        # Unambiguous noun plural.
        assert teacher.detect_number("dogs") == "PL"
        assert teacher.detect_number("dog") == "SG"
        # Homographs are noun plural AND verb 3sg: the context-free
        # teacher must not label corpus verb tokens PL (the old index
        # labelled every one of these PL via the shadowing noun).
        for word in NOUN_VERB_3SG_HOMOGRAPHS + ["lives"]:
            assert teacher.detect_number(word) == "SG"
