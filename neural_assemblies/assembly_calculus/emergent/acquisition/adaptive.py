"""Adaptive curriculum: turn stage reflection into targeted remedial training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from ..core.grounding import GroundingContext
from ..core.sentence import GroundedSentence

if TYPE_CHECKING:
    from ..curriculum import CurriculumTrainer
    from ..parser import EmergentParser
    from .orchestrator import StageReflection


def classification_accuracy_usable(accuracy: float) -> bool:
    """False when per-stage classification was skipped (sweep sentinel ``-1``)."""
    return accuracy >= 0.0


MAX_REMEDIAL_TARGETS = 12
REMEDIAL_SENTENCES_PER_TARGET = 4


@dataclass(frozen=True)
class AdaptiveHint:
    """Structured remediation action derived from stage reflection."""
    action: str
    words: Tuple[Tuple[str, str], ...] = ()
    detail: str = ""


@dataclass
class MisclassificationTarget:
    word: str
    expected: str
    predicted: str
    failure_mode: Optional[str] = None


@dataclass
class AdaptivePlan:
    """Concrete training adjustments before the next developmental stage."""
    phases: List[str] = field(default_factory=list)
    targets: List[MisclassificationTarget] = field(default_factory=list)
    ingest_holdout_stats: bool = False
    register_fuzzy: bool = False
    extra_babble_forms: int = 0
    actions_taken: List[str] = field(default_factory=list)

    @property
    def needs_remedial_training(self) -> bool:
        return bool(self.targets and self.phases)

    @property
    def needs_any_action(self) -> bool:
        return (
            self.needs_remedial_training
            or self.ingest_holdout_stats
            or self.register_fuzzy
            or self.extra_babble_forms > 0
        )


@dataclass
class RemediationResult:
    """Outcome of an adaptive remediation pass."""
    stage: str
    sentences_trained: int
    phases_run: List[str]
    targets_addressed: List[str]
    classification_accuracy_after: float = 0.0


_CAT_LABEL = {
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "PREP": "PREP",
    "PRON": "PRON",
    "DET": "DET",
    "CONJ": "CONJ",
}


def _word_category_label(word_obj) -> Optional[str]:
    from neural_assemblies.lexicon.lexicon_manager import WordCategory

    mapping = {
        WordCategory.NOUN: "NOUN",
        WordCategory.VERB: "VERB",
        WordCategory.ADJECTIVE: "ADJ",
        WordCategory.ADVERB: "ADV",
        WordCategory.PREPOSITION: "PREP",
        WordCategory.PRONOUN: "PRON",
        WordCategory.DETERMINER: "DET",
        WordCategory.CONJUNCTION: "CONJ",
    }
    return mapping.get(word_obj.category)


def collect_misclassifications(
    parser: "EmergentParser",
    stage_words: Sequence,
) -> List[MisclassificationTarget]:
    """Words in stage vocabulary classified with wrong POS."""
    targets: List[MisclassificationTarget] = []
    for w in stage_words:
        expected = _word_category_label(w)
        if expected is None:
            continue
        lemma = w.lemma
        if lemma not in parser.stim_map:
            continue
        grounding = parser.word_grounding.get(lemma)
        predicted, _ = parser.classify_word(lemma, grounding=grounding)
        if predicted != expected:
            targets.append(
                MisclassificationTarget(
                    word=lemma,
                    expected=expected,
                    predicted=predicted,
                ),
            )
    return targets


def _vocab_by_category(
    parser: "EmergentParser",
) -> Dict[str, List[str]]:
    """Bucket registered words by dominant-modality POS proxy."""
    buckets: Dict[str, List[str]] = {
        "NOUN": [],
        "VERB": [],
        "ADJ": [],
        "ADV": [],
        "PREP": [],
        "PRON": [],
        "DET": [],
    }
    mod_to_cat = {
        "visual": "NOUN",
        "motor": "VERB",
        "properties": "ADJ",
        "temporal": "ADV",
        "spatial": "PREP",
        "social": "PRON",
        "none": "DET",
    }
    for word, ctx in parser.word_grounding.items():
        if word not in parser.stim_map:
            continue
        cat = mod_to_cat.get(ctx.dominant_modality, "NOUN")
        buckets.setdefault(cat, []).append(word)
    for cat in buckets:
        buckets[cat].sort()
    return buckets


def _pick(rng: random.Random, words: List[str], exclude: Optional[str] = None) -> str:
    pool = [w for w in words if w != exclude]
    if not pool:
        pool = list(words)
    return rng.choice(pool) if pool else "the"


def build_remedial_sentences(
    parser: "EmergentParser",
    targets: Sequence[MisclassificationTarget],
    *,
    sentences_per_target: int = 10,
    seed: int = 0,
) -> List[GroundedSentence]:
    """Generate grounded sentences that reinforce correct POS frames for targets."""
    if not targets:
        return []

    rng = random.Random(seed)
    buckets = _vocab_by_category(parser)
    dets = buckets["DET"] or ["the", "a"]
    nouns = buckets["NOUN"] or ["dog", "cat"]
    verbs = buckets["VERB"] or ["runs", "sees"]
    prons = buckets["PRON"] or ["she"]
    preps = buckets["PREP"] or ["on"]

    sentences: List[GroundedSentence] = []
    seen: Set[Tuple[str, ...]] = set()

    def _add(words: List[str], roles: Optional[List[Optional[str]]] = None) -> None:
        key = tuple(words)
        if key in seen:
            return
        seen.add(key)
        contexts = [
            parser.word_grounding.get(w, GroundingContext())
            for w in words
        ]
        if roles is None:
            roles = [None] * len(words)
        sentences.append(GroundedSentence(words=words, contexts=contexts, roles=roles))

    for target in targets:
        word = target.word
        expected = target.expected
        for _ in range(sentences_per_target):
            det = _pick(rng, dets)
            noun = _pick(rng, nouns, exclude=word)
            verb = _pick(rng, verbs, exclude=word)
            noun2 = _pick(rng, nouns, exclude=word)

            if expected == "ADJ":
                _add([det, word, noun], [None, None, "agent"])
                _add([det, word, noun, verb], [None, None, "agent", "action"])
                _add(["a", word, noun], [None, None, "agent"])
            elif expected == "VERB":
                _add([noun, word], ["agent", "action"])
                _add([det, noun, word], [None, "agent", "action"])
                _add([det, noun, word, det, noun2],
                     [None, "agent", "action", None, "patient"])
                _add([_pick(rng, prons), word, det, noun2],
                     ["agent", "action", None, "patient"])
            elif expected == "NOUN":
                _add([det, word, verb], [None, "agent", "action"])
                _add([det, noun, verb, det, word],
                     [None, "agent", "action", None, "patient"])
                _add([det, word, verb, _pick(rng, preps), det, noun2],
                     [None, "agent", "action", None, None, None])
            elif expected == "ADV":
                _add([det, noun, verb, word], [None, "agent", "action", None])
            elif expected == "DET":
                _add([word, noun, verb], [None, "agent", "action"])
            elif expected == "PRON":
                _add([word, verb, det, noun2], ["agent", "action", None, "patient"])
            elif expected == "PREP":
                _add([det, noun, verb, word, det, noun2],
                     [None, "agent", "action", None, None, None])
            else:
                _add([det, noun, verb])

    return sentences


def build_adaptive_plan(
    reflection: "StageReflection",
    parser: "EmergentParser",
    stage_words: Sequence,
    *,
    holdout_words: Optional[Set[str]] = None,
) -> AdaptivePlan:
    """Translate reflection hints into a concrete adaptive training plan."""
    plan = AdaptivePlan()
    holdout = set(holdout_words or ())

    for hint in reflection.adaptive_hints:
        if hint.action == "babble_more":
            plan.extra_babble_forms = max(plan.extra_babble_forms, 12)
            plan.actions_taken.append("extra_babble")
        elif hint.action == "register_fuzzy":
            plan.register_fuzzy = True
            plan.actions_taken.append("register_fuzzy")
        elif hint.action == "ensure_distributional":
            if "distributional" not in plan.phases:
                plan.phases.append("distributional")
            plan.actions_taken.append("ensure_distributional")
        elif hint.action == "word_order_replay":
            if "word_order" not in plan.phases:
                plan.phases.append("word_order")
            plan.actions_taken.append("word_order_replay")
        elif hint.action == "holdout_remedial":
            plan.ingest_holdout_stats = True
            for word, expected in hint.words:
                plan.targets.append(
                    MisclassificationTarget(
                        word=word,
                        expected=expected,
                        predicted="UNKNOWN",
                        failure_mode=hint.detail or None,
                    ),
                )
            if "distributional" not in plan.phases:
                plan.phases.append("distributional")
            plan.actions_taken.append(f"holdout_remedial:{','.join(w for w, _ in hint.words)}")
        elif hint.action == "remedial_pos":
            plan.actions_taken.append("remedial_pos")

    if any(h.action == "remedial_pos" for h in reflection.adaptive_hints):
        mis = collect_misclassifications(parser, stage_words)
        known = {(t.word, t.expected) for t in plan.targets}
        for m in mis[:MAX_REMEDIAL_TARGETS]:
            if (m.word, m.expected) not in known:
                plan.targets.append(m)
        if plan.targets and "distributional" not in plan.phases:
            plan.phases.append("distributional")
        weak_grounding = any(
            t.failure_mode == "weak_grounding_readout" for t in plan.targets
        )
        if weak_grounding and "lexicon" not in plan.phases:
            plan.phases.insert(0, "lexicon")

    if plan.ingest_holdout_stats and holdout:
        plan.actions_taken.append("ingest_holdout_stats")

    # Deduplicate targets by word
    deduped: Dict[str, MisclassificationTarget] = {}
    for t in plan.targets:
        deduped[t.word] = t
    plan.targets = list(deduped.values())

    return plan


def apply_adaptive_plan(
    parser: "EmergentParser",
    trainer: "CurriculumTrainer",
    reflection: "StageReflection",
    plan: AdaptivePlan,
    *,
    stage_words: Sequence,
    holdout_words: Optional[Set[str]] = None,
    seed: int = 0,
) -> Optional[RemediationResult]:
    """Execute adaptive plan after a developmental stage."""
    if not plan.needs_any_action:
        return None

    from .babble import register_early_fuzzy_variants, train_babble_stage
    from .pos_inference import infer_holdout_categories, ingest_holdout_sentence_stats

    phases_run: List[str] = []
    sentences_trained = 0
    holdout = set(holdout_words or ())

    if plan.extra_babble_forms > 0:
        train_babble_stage(
            parser,
            n_forms=plan.extra_babble_forms,
            n_utterances=plan.extra_babble_forms,
            seed=seed,
        )
        phases_run.append("babble")

    if plan.register_fuzzy and stage_words:
        lemmas = [
            w.lemma for w in stage_words[:12]
            if hasattr(w, "lemma")
        ]
        if lemmas:
            register_early_fuzzy_variants(parser, lemmas)
            phases_run.append("fuzzy_surfaces")

    if plan.ingest_holdout_stats and holdout:
        ingest_holdout_sentence_stats(parser, holdout)
        infer_holdout_categories(parser, holdout)
        phases_run.append("holdout_stats")

    if plan.needs_remedial_training:
        remedial = build_remedial_sentences(
            parser,
            plan.targets,
            sentences_per_target=REMEDIAL_SENTENCES_PER_TARGET,
            seed=seed + 1,
        )
        result = trainer.train_remedial(
            remedial,
            phases=plan.phases,
            label=f"ADAPTIVE_{reflection.stage}",
            holdout_words=holdout or None,
        )
        sentences_trained = result.sentences_trained
        phases_run.extend(result.phases_run)

    accuracy_after = trainer._evaluate_classification(stage_words) if stage_words else 0.0

    return RemediationResult(
        stage=reflection.stage,
        sentences_trained=sentences_trained,
        phases_run=phases_run,
        targets_addressed=[t.word for t in plan.targets],
        classification_accuracy_after=accuracy_after,
    )
