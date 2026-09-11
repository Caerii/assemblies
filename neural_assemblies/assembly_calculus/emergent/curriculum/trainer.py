"""Curriculum-based developmental training for EmergentParser.

Progressively trains the parser through stages of increasing complexity,
using the lexicon's frequency and age-of-acquisition data to select
stage-appropriate vocabulary.  Plasticity decreases at later stages.

WHY A CURRICULUM RATHER THAN THE WHOLE CORPUS AT ONCE.  Two independent
reasons converge on the same design.

The developmental one: children do not receive their vocabulary uniformly.
The stage names below track the observed sequence -- babble, first words, the
vocabulary spurt, two-word utterances, full sentences -- and vocabulary is
selected by frequency and age of acquisition, so the model meets words in
roughly the order a child does.  If category structure genuinely falls out of
distributional and grounding regularities, it should fall out under that
ordering too, and testing it is part of the claim.

The mechanical one: assemblies are formed by potentiating a connectome that
is already carrying every earlier item, so training order is not neutral.
Presenting complex multi-clause input before the lexicon has stabilised means
role and phrase training write against assemblies that are still moving.
Staging exists so each phase trains on something the previous phase already
made stable.

THE DECREASING BETA IS THE POINT, not a tuning artifact.  ``_STAGE_CONFIG``
runs plasticity from 0.20 at BABBLE down to 0.06 at CONVERSATION.  High beta
early means each exposure moves the connectome a long way, so a handful of
presentations suffices to carve out a word -- fast acquisition, but easily
overwritten.  Low beta later means new material perturbs established
structure only slightly, so the grammar learned earlier survives continued
input.  This is the stability-plasticity trade-off implemented as a schedule,
and it is why a late stage cannot simply be run with an early stage's beta.

Each stage lists its ``phases``: which of the parser's training routines run
at that stage.  Phases are cumulative in practice -- later stages re-run
lexicon and roles rather than assuming earlier work is untouched -- which is
what keeps earlier structure refreshed as beta falls.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..core.grounding import GroundingContext
from ..core.sentence import SentencePlan
from .data import GroundedSentence
from .generation import SentenceGenerator
from ..core.areas import GROUNDING_TO_CORE, DET_CORE
from ..training.perf import (
    PRESET_VOCAB_SKIP_THRESHOLD,
    effective_stage_phases,
    stage_training_rounds,
)




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
    "BABBLE": {
        "beta": 0.20,
        "complexity": 0,
        "phases": [],
    },
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
        # "number" was schedulable (schedule.py runs it) but listed by NO
        # stage -- the `phrases` dormant-selector shape. It joins the stages
        # that carry "tense": the corpus now varies subject number, and a
        # phase that never runs can't read the variation.
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "number", "mood",
                    "polarity", "prediction"],
    },
    "COMPLEX_GRAMMAR": {
        "beta": 0.08,
        "complexity": 6,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "number", "mood",
                    "polarity", "conjunctions"],
    },
    "INSTRUCTIONS": {
        "beta": 0.10,
        "complexity": 3,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "mood", "prediction"],
    },
    "DIALOGUE": {
        "beta": 0.08,
        "complexity": 4,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "mood", "prediction", "dialogue"],
    },
    "CONVERSATION": {
        "beta": 0.06,
        "complexity": 6,
        "phases": ["lexicon", "distributional", "roles",
                    "phrases", "word_order", "tense", "number", "mood",
                    "polarity", "conjunctions", "prediction", "dialogue",
                    "conversation"],
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

    def __init__(self, parser, *, holdout_words: Optional[Set[str]] = None):
        self.parser = parser
        self.holdout_words: Set[str] = set(holdout_words or ())
        self.parser.lexicon_holdouts = set(self.holdout_words)
        #: The corpus-generation concern, split out whole -- see
        #: `generation.py` for what lives there and why. Shares the
        #: holdout set so bridge-line gating stays consistent.
        self.generation = SentenceGenerator(
            parser, holdout_words=self.holdout_words)
        self.stage_results: List[StageResult] = []
        self._lexicon_manager = self._build_lexicon_manager()
        from ..core.corpus_index import TransitionCache
        self._transition_cache = TransitionCache()

    @staticmethod
    def _build_lexicon_manager():
        """Build a LexiconManager from the raw lexicon data files."""
        from neural_assemblies.lexicon.lexicon_manager import (
            LexiconManager, Word, WordCategory, SemanticDomain,
        )
        from neural_assemblies.lexicon.data import (
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

    def _set_global_beta(self, beta: float) -> None:
        """Stage plasticity schedule -- DELEGATES to the parser's beta
        policy owner. This method used to loop over the engine store
        directly, which silently ERASED per-fiber overlays
        (`role_bind_gain`) at every stage boundary; `set_base_beta`
        re-prices them instead. Kept as a thin wrapper because parity.py
        and the stage schedules call it by this name."""
        self.parser.set_base_beta(beta)

    def _evaluate_classification(self, words: list) -> float:
        """Quick classification accuracy on a word list.

        Checks whether known-category words are classified correctly
        by the parser.
        """
        from neural_assemblies.lexicon.lexicon_manager import WordCategory

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
            cat, _ = self.parser.classify_word_cached(lemma, grounding=grounding)
            if cat == expected:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _get_stage_words(self, stage_name: str) -> list:
        """Get words appropriate for a curriculum stage.

        Uses AoA and frequency thresholds matching developmental stages.
        """
        _STAGE_THRESHOLDS = {
            "BABBLE": {"max_aoa": 0.0, "min_freq": 0.0, "target": 0},
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
            "DIALOGUE": {"max_aoa": 4.5, "min_freq": 2.2,
                         "target": 600},
            "CONVERSATION": {"max_aoa": 6.0, "min_freq": 1.8,
                             "target": 1000},
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

        selected = candidates[:target]
        if self.holdout_words:
            selected = [
                w for w in selected if w.lemma not in self.holdout_words
            ]
        return selected

    def _stage_needs_lexicon(self, stage_words: list) -> bool:
        """True if any stage word is missing from core lexicons."""
        for w in stage_words:
            lemma = w.lemma
            ctx = self.parser.word_grounding.get(lemma, GroundingContext())
            core = GROUNDING_TO_CORE.get(ctx.dominant_modality, DET_CORE)
            if lemma not in self.parser.core_lexicons.get(core, {}):
                return True
        return False

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
        phases = effective_stage_phases(
            stage_name, config["phases"], fast=self.parser.fast_training,
        )

        old_rounds = self.parser.rounds
        stage_rounds = stage_training_rounds(
            stage_name, fast=self.parser.fast_training,
        )
        if stage_rounds is not None:
            self.parser.rounds = stage_rounds

        try:
            return self._train_stage_impl(
                stage_name, config, beta, complexity, phases,
            )
        finally:
            self.parser.rounds = old_rounds

    def _train_stage_impl(
        self,
        stage_name: str,
        config: dict,
        beta: float,
        complexity: int,
        phases: list,
    ) -> StageResult:
        stage_words = self._get_stage_words(stage_name)

        if stage_name == "BABBLE":
            from ..train_progress import current_progress
            prog = current_progress()
            n_babble = len(getattr(self.parser, "babble_forms", []))
            prog.info(f"stage BABBLE: {n_babble} forms registered (pre-lexical)")
            result = StageResult(
                stage_name="BABBLE",
                vocab_size=n_babble,
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=0,
                phases_run=["babble"],
            )
            self.stage_results.append(result)
            return result

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
        sentences = self.generation.generate(
            stage_words, complexity, stage_name=stage_name,
        )
        self.generation.register_surface_forms(sentences, stage_words)

        from ..train_progress import current_progress
        from .data import create_instruction_sentences
        from .holdout_bridges import merge_prediction_corpus
        from ..training.schedule import TrainingScheduleExecutor
        from ..curriculum.conversation import get_conversation_curriculum

        prog = current_progress()
        prog.info(
            f"stage {stage_name}: {len(stage_words)} words, "
            f"{len(sentences)} sentences, beta={beta}",
        )

        if "prediction" in phases:
            if self.holdout_words:
                extra_pred = merge_prediction_corpus(
                    self.parser,
                    self.holdout_words,
                    stage_name=stage_name,
                )
            else:
                extra_pred = create_instruction_sentences()
        else:
            extra_pred = None

        schedule = TrainingScheduleExecutor.build_stage_schedule(
            self.parser, stage_name, sentences, phases,
            extra_prediction=extra_pred,
            conversation_sents=(
                get_conversation_curriculum(self.parser.word_grounding)
                if "conversation" in phases else None
            ),
            transition_cache=self._transition_cache,
        )

        schedule.transition_cache = self._transition_cache

        if "lexicon" in phases and not self._stage_needs_lexicon(stage_words):
            prog.info("skip lexicon (all stage words already trained)")
            schedule.phases = [p for p in schedule.phases if p != "lexicon"]

        if "dialogue" in phases:
            from ..curriculum.dialogue import get_dialogue_pairs
            from ..curriculum.conversation import get_conversation_pairs

            schedule.dialogue_pairs = (
                get_dialogue_pairs()
                + get_conversation_pairs(self.parser.word_grounding)
            )

        if self.holdout_words:
            schedule.holdout_words = self.holdout_words

        executor = TrainingScheduleExecutor(self.parser)
        run_phases = [r.phase for r in executor.run(schedule)]

        # Evaluate (skip in sweep mode — classification pass is for logging only)
        from ..training.perf import sweep_mode_enabled

        if sweep_mode_enabled():
            accuracy = -1.0
        else:
            with prog.phase("evaluate"):
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

    def train_remedial(
        self,
        sentences: List[GroundedSentence],
        *,
        phases: Optional[List[str]] = None,
        label: str = "ADAPTIVE",
        beta: float = 0.10,
        holdout_words: Optional[Set[str]] = None,
    ) -> StageResult:
        """Run a lightweight remedial pass on targeted sentences."""
        if not sentences:
            return StageResult(
                stage_name=label,
                vocab_size=0,
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=0,
                phases_run=[],
            )

        phases = list(phases or ["distributional"])
        old_rounds = self.parser.rounds
        self.parser.rounds = min(self.parser.rounds, 2)

        try:
            self._set_global_beta(beta)
            from ..train_progress import current_progress
            from ..training.schedule import TrainingScheduleExecutor

            prog = current_progress()
            prog.info(
                f"remedial {label}: {len(sentences)} sentences, phases={phases}",
            )

            schedule = TrainingScheduleExecutor.build_stage_schedule(
                self.parser,
                label,
                # Carry the event across rather than dropping to tokens: a
                # remedial pass re-trains the same sentences, and a sentence
                # that loses its scene here would silently switch to positional
                # roles halfway through training.
                [SentencePlan(list(s.words), event=getattr(s, "event", None),
                              mood=getattr(s, "mood", "declarative"))
                 for s in sentences],
                phases,
                transition_cache=self._transition_cache,
            )
            schedule.transition_cache = self._transition_cache
            if holdout_words:
                schedule.holdout_words = set(holdout_words)

            executor = TrainingScheduleExecutor(self.parser)
            run_phases = [r.phase for r in executor.run(schedule)]

            return StageResult(
                stage_name=label,
                vocab_size=len({w for s in sentences for w in s.words}),
                classification_accuracy=0.0,
                beta=beta,
                sentences_trained=len(sentences),
                phases_run=run_phases,
            )
        finally:
            self.parser.rounds = old_rounds

    def train_curriculum(
        self,
        max_stage: str = "SENTENCES",
        *,
        holdout_words: Optional[Set[str]] = None,
    ) -> List[StageResult]:
        """Train the parser through multiple curriculum stages.

        Runs stages in order from FIRST_WORDS up to (and including)
        max_stage.

        Args:
            max_stage: Last stage name to train (default "SENTENCES").
            holdout_words: Optional lexicon holdouts for generalization probes.

        Returns:
            List of StageResult for each stage trained.
        """
        if holdout_words is not None:
            self.holdout_words = set(holdout_words)
            self.parser.lexicon_holdouts = set(self.holdout_words)
        stage_order = [
            "FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
            "SENTENCES", "COMPLEX_GRAMMAR",
        ]

        results = []
        from ..train_progress import current_progress
        prog = current_progress()
        for stage_name in stage_order:
            with prog.section(stage_name):
                result = self.train_stage(stage_name)
            results.append(result)
            acc_label = (
                "skipped (sweep)"
                if result.classification_accuracy < 0
                else f"{result.classification_accuracy:.1%}"
            )
            prog.info(
                f"{stage_name} complete: vocab={result.vocab_size} "
                f"acc={acc_label} phases={result.phases_run}",
            )
            if stage_name == max_stage:
                break

        return results

    def train_conversation_path(
        self,
        max_stage: str = "DIALOGUE",
        *,
        skip_early_if_loaded: bool = True,
        holdout_words: Optional[Set[str]] = None,
    ) -> List[StageResult]:
        """Train developmental stages through naturalistic conversation.

        Runs FIRST_WORDS → … → DIALOGUE (or CONVERSATION).

        When ``skip_early_if_loaded`` and vocabulary is already large
        (preset vocab), jumps directly to DIALOGUE/CONVERSATION stages.

        Args:
            max_stage: Last stage; ``DIALOGUE`` or ``CONVERSATION``.
            skip_early_if_loaded: Skip early developmental stages for presets.
            holdout_words: Optional lexicon holdouts for generalization probes.

        Returns:
            List of StageResult per stage trained.
        """
        if holdout_words is not None:
            self.holdout_words = set(holdout_words)
            self.parser.lexicon_holdouts = set(self.holdout_words)
        full_order = [
            "FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
            "SENTENCES", "COMPLEX_GRAMMAR", "DIALOGUE", "CONVERSATION",
        ]
        if max_stage not in full_order:
            raise ValueError(
                f"max_stage {max_stage!r} not in {full_order}"
            )

        from ..train_progress import current_progress
        from ..training.perf import should_skip_early_curriculum

        prog = current_progress()
        prog.info(f"conversation path target: {max_stage}")

        if should_skip_early_curriculum(len(self.parser.stim_map), max_stage) and skip_early_if_loaded:
            stage_order = full_order[full_order.index("DIALOGUE"):]
            prog.info(
                f"skip early stages (vocab={len(self.parser.stim_map)} "
                f">= {PRESET_VOCAB_SKIP_THRESHOLD})",
            )
        else:
            stage_order = full_order

        results = []
        for stage_name in stage_order:
            with prog.section(stage_name):
                result = self.train_stage(stage_name)
            results.append(result)
            prog.info(
                f"{stage_name}: vocab={result.vocab_size} "
                f"acc={('skipped (sweep)' if result.classification_accuracy < 0 else f'{result.classification_accuracy:.1%}')}",
            )
            if stage_name == max_stage:
                break
        return results
