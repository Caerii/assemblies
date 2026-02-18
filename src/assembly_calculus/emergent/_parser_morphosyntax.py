"""Morphosyntax mixin: tense, mood, polarity, number detection/training and conjunction handling."""

from typing import Dict, List, Tuple

from src.assembly_calculus.ops import _snap
from .areas import (
    VERB_CORE, CONJ_CORE, TENSE, MOOD, POLARITY, NUMBER,
    GROUNDING_TO_CORE,
)


class MorphosyntaxMixin:
    """Tense, mood, polarity detection and training; conjunction handling."""

    _NEGATION_WORDS = frozenset([
        "not", "n't", "no", "never", "neither", "nor", "nobody",
        "nothing", "nowhere", "don't", "doesn't", "didn't",
        "won't", "wouldn't", "can't", "couldn't", "shouldn't",
    ])

    _FUTURE_MARKERS = frozenset(["will", "shall", "gonna"])
    _PROGRESSIVE_AUX = frozenset(["is", "am", "are", "was", "were"])
    _PERFECT_AUX = frozenset(["has", "have", "had"])
    _WH_WORDS = frozenset([
        "who", "what", "where", "when", "why", "how", "which",
    ])
    _AUX_WORDS = frozenset([
        "do", "does", "did", "is", "am", "are", "was", "were",
        "has", "have", "had", "will", "shall", "can", "could",
        "would", "should", "may", "might", "must",
    ])
    _CONJUNCTION_WORDS = frozenset([
        "and", "but", "or", "nor", "so", "yet", "for",
        "because", "although", "if", "when", "while", "since",
    ])

    def detect_tense(self, words: List[str]) -> str:
        """Detect tense from verb morphology and auxiliaries.

        Args:
            words: List of word tokens.

        Returns:
            One of: "PRESENT", "PAST", "FUTURE", "PROGRESSIVE", "PERFECT".
        """
        from .vocabulary_builder import lookup_verb_form

        for i, word in enumerate(words):
            # Check for future markers
            if word.lower() in self._FUTURE_MARKERS:
                return "FUTURE"

            # Check for progressive: aux + -ing form
            if (word.lower() in self._PROGRESSIVE_AUX and
                    i + 1 < len(words)):
                result = lookup_verb_form(words[i + 1])
                if result and result[1] == "PROGRESSIVE":
                    return "PROGRESSIVE"

            # Check for perfect: has/have + past participle
            if (word.lower() in self._PERFECT_AUX and
                    i + 1 < len(words)):
                result = lookup_verb_form(words[i + 1])
                if result and result[1] == "PERFECT":
                    return "PERFECT"

            # Check verb forms
            result = lookup_verb_form(word)
            if result:
                _, tense = result
                if tense == "PAST":
                    return "PAST"

        return "PRESENT"

    def detect_mood(self, words: List[str]) -> str:
        """Detect sentence mood from surface cues.

        Args:
            words: List of word tokens.

        Returns:
            One of: "DECLARATIVE", "INTERROGATIVE", "IMPERATIVE".
        """
        if not words:
            return "DECLARATIVE"

        first = words[0].lower()

        # Question: starts with wh-word or auxiliary
        if first in self._WH_WORDS:
            return "INTERROGATIVE"
        if first in self._AUX_WORDS:
            return "INTERROGATIVE"

        # Imperative: starts with a verb (no subject)
        grounding = self.word_grounding.get(first)
        if grounding and grounding.dominant_modality == "motor":
            return "IMPERATIVE"

        return "DECLARATIVE"

    def detect_polarity(self, words: List[str]) -> str:
        """Detect sentence polarity from negation markers.

        Args:
            words: List of word tokens.

        Returns:
            One of: "AFFIRMATIVE", "NEGATIVE".
        """
        for word in words:
            if word.lower() in self._NEGATION_WORDS:
                return "NEGATIVE"
        return "AFFIRMATIVE"

    def train_tense(self, sentences: List[List[str]]) -> None:
        """Train TENSE area from verb morphology in sentences.

        Creates tense stimuli and projects verb+tense → TENSE area.

        Args:
            sentences: List of token lists.
        """
        # Register tense stimuli
        tense_stims = {}
        for tense_name in ("PRESENT", "PAST", "FUTURE",
                           "PROGRESSIVE", "PERFECT"):
            stim_name = f"tense_{tense_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            tense_stims[tense_name] = stim_name

        for sent in sentences:
            tense = self.detect_tense(sent)
            tense_stim = tense_stims[tense]

            # Find the verb in this sentence using grounding (fast path)
            for word in sent:
                if word not in self.stim_map:
                    continue
                grounding = self.word_grounding.get(word)
                if grounding and grounding.dominant_modality == "motor":
                    phon = self.stim_map[word]
                    # Project tense + verb → TENSE area
                    self.brain.project(
                        {tense_stim: [TENSE], phon: [VERB_CORE]},
                        {VERB_CORE: [TENSE]},
                    )
                    if self.rounds > 1:
                        self.brain.project_rounds(
                            target=TENSE,
                            areas_by_stim={tense_stim: [TENSE]},
                            dst_areas_by_src_area={
                                VERB_CORE: [TENSE], TENSE: [TENSE],
                            },
                            rounds=self.rounds - 1,
                        )
                    break  # One tense per sentence

    def train_mood(self, sentences: List[List[str]]) -> None:
        """Train MOOD area from sentence mood detection.

        Args:
            sentences: List of token lists.
        """
        mood_stims = {}
        for mood_name in ("DECLARATIVE", "INTERROGATIVE", "IMPERATIVE"):
            stim_name = f"mood_{mood_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            mood_stims[mood_name] = stim_name

        for sent in sentences:
            mood = self.detect_mood(sent)
            mood_stim = mood_stims[mood]
            self.brain.project(
                {mood_stim: [MOOD]},
                {MOOD: [MOOD]},
            )
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=MOOD,
                    areas_by_stim={mood_stim: [MOOD]},
                    dst_areas_by_src_area={MOOD: [MOOD]},
                    rounds=self.rounds - 1,
                )

    def train_polarity(self, sentences: List[List[str]]) -> None:
        """Train POLARITY area from negation detection.

        Args:
            sentences: List of token lists.
        """
        pol_stims = {}
        for pol_name in ("AFFIRMATIVE", "NEGATIVE"):
            stim_name = f"polarity_{pol_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            pol_stims[pol_name] = stim_name

        for sent in sentences:
            polarity = self.detect_polarity(sent)
            pol_stim = pol_stims[polarity]
            self.brain.project(
                {pol_stim: [POLARITY]},
                {POLARITY: [POLARITY]},
            )
            if self.rounds > 1:
                self.brain.project_rounds(
                    target=POLARITY,
                    areas_by_stim={pol_stim: [POLARITY]},
                    dst_areas_by_src_area={POLARITY: [POLARITY]},
                    rounds=self.rounds - 1,
                )

    def train_conjunctions(self, sentences: List[List[str]]) -> None:
        """Train CONJ_CORE area with conjunction words.

        Args:
            sentences: List of token lists.
        """
        for sent in sentences:
            for word in sent:
                if word.lower() not in self._CONJUNCTION_WORDS:
                    continue
                # Register if needed
                self.register_word(word) if hasattr(
                    self, 'register_word') else None
                phon = self.stim_map.get(word)
                if phon is None:
                    continue

                # Project phon → CONJ_CORE
                self.brain._engine.reset_area_connections(CONJ_CORE)
                stim_dict = {phon: [CONJ_CORE]}
                self.brain.project(stim_dict, {})
                if self.rounds > 1:
                    self.brain.project_rounds(
                        target=CONJ_CORE,
                        areas_by_stim=stim_dict,
                        dst_areas_by_src_area={CONJ_CORE: [CONJ_CORE]},
                        rounds=self.rounds - 1,
                    )

                if CONJ_CORE not in self.core_lexicons:
                    self.core_lexicons[CONJ_CORE] = {}
                self.core_lexicons[CONJ_CORE][word] = _snap(
                    self.brain, CONJ_CORE)
                self.brain._engine.reset_area_connections(CONJ_CORE)

    def detect_number(self, word: str) -> str:
        """Detect grammatical number from word grounding features.

        Checks the word's grounding context for explicit SG/PL features.
        Falls back to "SG" (singular) as default.

        Args:
            word: A single word token.

        Returns:
            "SG" or "PL".
        """
        grounding = self.word_grounding.get(word)
        if grounding:
            for mod in ("visual", "motor", "properties", "spatial",
                        "social", "temporal", "emotional"):
                features = getattr(grounding, mod)
                if "SG" in features:
                    return "SG"
                if "PL" in features:
                    return "PL"
        return "SG"

    def train_number(self, sentences: List[List[str]]) -> None:
        """Train NUMBER area from morphological number features.

        For each content word (noun or verb) in each sentence, detects
        its grammatical number and projects number_stim + phon into the
        NUMBER area. This creates separate SG and PL assemblies in
        NUMBER, each associated with core-area word assemblies.

        Follows the same pattern as train_tense().

        Args:
            sentences: List of token lists.
        """
        number_stims = {}
        for num_name in ("SG", "PL"):
            stim_name = f"number_{num_name}"
            if stim_name not in self.brain.stimuli:
                self.brain.add_stimulus(stim_name, self.k)
            number_stims[num_name] = stim_name

        for sent in sentences:
            for word in sent:
                if word not in self.stim_map:
                    continue
                grounding = self.word_grounding.get(word)
                if grounding is None:
                    continue

                # Only train number for content words (nouns and verbs)
                mod = grounding.dominant_modality
                if mod not in ("visual", "motor"):
                    continue

                num = self.detect_number(word)
                num_stim = number_stims[num]
                core_area = GROUNDING_TO_CORE[mod]
                phon = self.stim_map[word]

                # Project number_stim + word phon -> NUMBER area
                self.brain.project(
                    {num_stim: [NUMBER], phon: [core_area]},
                    {core_area: [NUMBER]},
                )
                if self.rounds > 1:
                    self.brain.project_rounds(
                        target=NUMBER,
                        areas_by_stim={num_stim: [NUMBER]},
                        dst_areas_by_src_area={
                            core_area: [NUMBER], NUMBER: [NUMBER],
                        },
                        rounds=self.rounds - 1,
                    )
