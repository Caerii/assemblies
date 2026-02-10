"""Curriculum-based developmental training for EmergentParser.

Progressively trains the parser through stages of increasing complexity,
using the lexicon's frequency and age-of-acquisition data to select
stage-appropriate vocabulary.  Plasticity decreases at later stages.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .areas import CORE_TO_CATEGORY
from .grounding import GroundingContext
from .training_data import GroundedSentence


@dataclass
class StageResult:
    """Metrics from training a single curriculum stage."""
    stage_name: str
    vocab_size: int
    classification_accuracy: float
    beta: float
    sentences_trained: int
    phases_run: List[str] = field(default_factory=list)


# Stage name -> (beta, sentence_complexity, phases)
_STAGE_CONFIG = {
    "FIRST_WORDS": {
        "beta": 0.15,
        "complexity": 1,
        "phases": ["lexicon"],
    },
    "VOCABULARY_SPURT": {
        "beta": 0.12,
        "complexity": 2,
        "phases": ["lexicon", "distributional"],
    },
    "TWO_WORD": {
        "beta": 0.10,
        "complexity": 2,
        "phases": ["lexicon", "distributional", "roles"],
    },
    "SENTENCES": {
        "beta": 0.10,
        "complexity": 4,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "mood", "polarity"],
    },
    "COMPLEX_GRAMMAR": {
        "beta": 0.08,
        "complexity": 6,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "mood", "polarity",
                    "conjunctions"],
    },
}


class CurriculumTrainer:
    """Wraps EmergentParser training with developmental curriculum stages.

    Progressively trains the parser through stages of increasing complexity,
    using the lexicon's frequency and age-of-acquisition data to select
    stage-appropriate vocabulary.  Plasticity decreases at later stages.

    Uses src/lexicon/curriculum.py's Curriculum class for word selection
    and stage management.
    """

    def __init__(self, parser):
        self.parser = parser
        self.stage_results: List[StageResult] = []
        self._lexicon_manager = self._build_lexicon_manager()

    @staticmethod
    def _build_lexicon_manager():
        """Build a LexiconManager from the raw lexicon data files."""
        from src.lexicon.lexicon_manager import (
            LexiconManager, Word, WordCategory, SemanticDomain,
        )
        from src.lexicon.data import (
            NOUNS, VERBS, ADJECTIVES, ADVERBS,
            PREPOSITIONS, PRONOUNS, DETERMINERS, CONJUNCTIONS,
        )

        # Map POS labels -> WordCategory
        _CAT_MAP = {
            "NOUN": WordCategory.NOUN,
            "VERB": WordCategory.VERB,
            "ADJ": WordCategory.ADJECTIVE,
            "ADV": WordCategory.ADVERB,
            "PREP": WordCategory.PREPOSITION,
            "PRON": WordCategory.PRONOUN,
            "DET": WordCategory.DETERMINER,
            "CONJ": WordCategory.CONJUNCTION,
        }

        # Map domain strings -> SemanticDomain (best-effort)
        _DOMAIN_MAP = {}
        for member in SemanticDomain:
            _DOMAIN_MAP[member.name] = member

        lm = LexiconManager()

        categories = [
            (NOUNS, "NOUN"),
            (VERBS, "VERB"),
            (ADJECTIVES, "ADJ"),
            (ADVERBS, "ADV"),
            (PREPOSITIONS, "PREP"),
            (PRONOUNS, "PRON"),
            (DETERMINERS, "DET"),
            (CONJUNCTIONS, "CONJ"),
        ]

        for entries, pos in categories:
            wc = _CAT_MAP[pos]
            for entry in entries:
                domains = []
                for d in entry.get("domains", []):
                    if d in _DOMAIN_MAP:
                        domains.append(_DOMAIN_MAP[d])

                word = Word(
                    lemma=entry["lemma"],
                    category=wc,
                    forms=entry.get("forms", {}),
                    semantic_domains=domains,
                    features=entry.get("features", {}),
                    frequency=entry.get("freq", 0.0),
                    age_of_acquisition=entry.get("aoa", 10.0),
                )
                lm.add_word(word)

        return lm

    def _generate_sentences(self, words: list,
                            complexity: int) -> List[List[str]]:
        """Generate simple training sentences from a word list.

        Produces deterministic sentence patterns appropriate for the given
        complexity level (number of words per sentence).
        """
        from src.lexicon.lexicon_manager import WordCategory
        import random as _rng

        nouns = [w for w in words if w.category == WordCategory.NOUN]
        verbs = [w for w in words if w.category == WordCategory.VERB]
        adjs = [w for w in words if w.category == WordCategory.ADJECTIVE]
        dets = [w for w in words if w.category == WordCategory.DETERMINER]
        preps = [w for w in words if w.category == WordCategory.PREPOSITION]

        sentences: List[List[str]] = []

        if complexity == 1:
            # Single-word stage: just nouns
            for n in nouns[:20]:
                sentences.append([n.lemma])
            for v in verbs[:10]:
                sentences.append([v.lemma])
            return sentences

        if complexity == 2:
            # Two-word: DET+NOUN or NOUN+VERB
            for n in nouns[:15]:
                if dets:
                    sentences.append([dets[0].lemma, n.lemma])
                for v in verbs[:5]:
                    sentences.append([n.lemma, v.lemma])
            return sentences

        # complexity >= 3: full SVO sentences
        _rng_state = _rng.getstate()
        _rng.seed(42)

        det_word = dets[0].lemma if dets else "the"

        for _ in range(min(50, len(nouns) * len(verbs))):
            subj = _rng.choice(nouns) if nouns else None
            verb = _rng.choice(verbs) if verbs else None
            if subj is None or verb is None:
                continue

            sent = [det_word, subj.lemma, verb.lemma]

            if complexity >= 4 and nouns:
                obj = _rng.choice(nouns)
                sent.extend([det_word, obj.lemma])

            if complexity >= 5 and adjs:
                adj = _rng.choice(adjs)
                sent.insert(1, adj.lemma)

            if complexity >= 6 and preps and nouns:
                prep = _rng.choice(preps)
                loc = _rng.choice(nouns)
                sent.extend([prep.lemma, det_word, loc.lemma])

            sentences.append(sent)

        _rng.setstate(_rng_state)
        return sentences

    def _set_global_beta(self, beta: float) -> None:
        """Set plasticity (beta) for all area-to-area connections."""
        brain = self.parser.brain
        for area_name in brain.areas:
            area = brain.areas[area_name]
            for src in area.beta_by_area:
                area.beta_by_area[src] = beta
                brain._engine.set_beta(area_name, src, beta)

    def _evaluate_classification(self, words: list) -> float:
        """Quick classification accuracy on a word list.

        Checks whether known-category words are classified correctly
        by the parser.
        """
        from src.lexicon.lexicon_manager import WordCategory

        _CAT_LABEL = {
            WordCategory.NOUN: "NOUN",
            WordCategory.VERB: "VERB",
            WordCategory.ADJECTIVE: "ADJ",
            WordCategory.ADVERB: "ADV",
            WordCategory.PREPOSITION: "PREP",
            WordCategory.PRONOUN: "PRON",
            WordCategory.DETERMINER: "DET",
            WordCategory.CONJUNCTION: "CONJ",
        }

        correct = 0
        total = 0
        for w in words:
            expected = _CAT_LABEL.get(w.category)
            if expected is None:
                continue
            lemma = w.lemma
            if lemma not in self.parser.stim_map:
                continue
            grounding = self.parser.word_grounding.get(lemma)
            cat, _ = self.parser.classify_word(lemma, grounding=grounding)
            if cat == expected:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _get_stage_words(self, stage_name: str) -> list:
        """Get words appropriate for a curriculum stage.

        Uses AoA and frequency thresholds matching developmental stages.
        """
        _STAGE_THRESHOLDS = {
            "FIRST_WORDS": {"max_aoa": 2.0, "min_freq": 4.0,
                            "target": 50},
            "VOCABULARY_SPURT": {"max_aoa": 2.5, "min_freq": 3.5,
                                 "target": 200},
            "TWO_WORD": {"max_aoa": 3.0, "min_freq": 3.0,
                         "target": 300},
            "SENTENCES": {"max_aoa": 4.0, "min_freq": 2.5,
                          "target": 500},
            "COMPLEX_GRAMMAR": {"max_aoa": 5.0, "min_freq": 2.0,
                                "target": 800},
        }

        thresholds = _STAGE_THRESHOLDS.get(stage_name)
        if thresholds is None:
            return []

        max_aoa = thresholds["max_aoa"]
        min_freq = thresholds["min_freq"]
        target = thresholds["target"]

        candidates = self._lexicon_manager.get_by_aoa(max_aoa)
        candidates = [w for w in candidates
                      if w.frequency >= min_freq]

        # Sort by frequency (desc) then AoA (asc)
        candidates.sort(
            key=lambda w: (-w.frequency, w.age_of_acquisition))

        return candidates[:target]

    def train_stage(self, stage_name: str) -> StageResult:
        """Train the parser for one curriculum stage.

        Args:
            stage_name: One of the keys in _STAGE_CONFIG.

        Returns:
            StageResult with metrics for this stage.
        """
        config = _STAGE_CONFIG[stage_name]
        beta = config["beta"]
        complexity = config["complexity"]
        phases = config["phases"]

        stage_words = self._get_stage_words(stage_name)

        if not stage_words:
            return StageResult(
                stage_name=stage_name,
                vocab_size=0,
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=0,
                phases_run=[],
            )

        # Set plasticity
        self._set_global_beta(beta)

        # Register words in the parser
        for w in stage_words:
            self.parser.register_word(w.lemma)

        # Generate training sentences
        sentences = self._generate_sentences(stage_words, complexity)

        # Run applicable training phases
        run_phases = []

        if "lexicon" in phases:
            # Build GroundedSentence wrappers for lexicon training
            grounded = []
            for sent in sentences:
                contexts = [
                    self.parser.word_grounding.get(w, GroundingContext())
                    for w in sent
                ]
                grounded.append(GroundedSentence(
                    words=sent, contexts=contexts,
                    roles=[None] * len(sent),
                ))
            self.parser.train_lexicon()
            run_phases.append("lexicon")

        if "distributional" in phases:
            self.parser.train_distributional(sentences, repetitions=3)
            run_phases.append("distributional")

        if "roles" in phases:
            grounded_for_roles = []
            for sent in sentences:
                contexts = [
                    self.parser.word_grounding.get(w, GroundingContext())
                    for w in sent
                ]
                grounded_for_roles.append(GroundedSentence(
                    words=sent, contexts=contexts,
                    roles=[None] * len(sent),
                ))
            self.parser.train_unsupervised(
                grounded_for_roles, repetitions=3)
            run_phases.append("roles")

        if "phrases" in phases:
            grounded_for_phrases = []
            for sent in sentences:
                contexts = [
                    self.parser.word_grounding.get(w, GroundingContext())
                    for w in sent
                ]
                grounded_for_phrases.append(GroundedSentence(
                    words=sent, contexts=contexts,
                    roles=[None] * len(sent),
                ))
            self.parser.train_phrases(grounded_for_phrases)
            run_phases.append("phrases")

        if "word_order" in phases:
            self.parser.train_word_order_typological(sentences)
            run_phases.append("word_order")

        if "tense" in phases:
            self.parser.train_tense(sentences)
            run_phases.append("tense")

        if "mood" in phases:
            self.parser.train_mood(sentences)
            run_phases.append("mood")

        if "polarity" in phases:
            self.parser.train_polarity(sentences)
            run_phases.append("polarity")

        if "conjunctions" in phases:
            self.parser.train_conjunctions(sentences)
            run_phases.append("conjunctions")

        # Evaluate
        accuracy = self._evaluate_classification(stage_words)

        result = StageResult(
            stage_name=stage_name,
            vocab_size=len(stage_words),
            classification_accuracy=accuracy,
            beta=beta,
            sentences_trained=len(sentences),
            phases_run=run_phases,
        )
        self.stage_results.append(result)
        return result

    def train_curriculum(
        self,
        max_stage: str = "SENTENCES",
    ) -> List[StageResult]:
        """Train the parser through multiple curriculum stages.

        Runs stages in order from FIRST_WORDS up to (and including)
        max_stage.

        Args:
            max_stage: Last stage name to train (default "SENTENCES").

        Returns:
            List of StageResult for each stage trained.
        """
        stage_order = [
            "FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
            "SENTENCES", "COMPLEX_GRAMMAR",
        ]

        results = []
        for stage_name in stage_order:
            result = self.train_stage(stage_name)
            results.append(result)
            if stage_name == max_stage:
                break

        return results
