"""POS inference from grounding readout + corpus statistics.

THE PROBLEM.  A held-out word was deliberately never trained: its phon
stimulus exists but no assembly was ever formed for it in any core area.  So
the ordinary route -- project and see which core area recognises it -- has
nothing to recognise.  Categorising it anyway is the generalisation test, and
this module is what runs it.

THE APPROACH.  No single source of evidence is sufficient for an unseen word,
so several weak ones are combined:

    grounding readout  the word's grounding FEATURES were registered even
                       though the word was not trained, and features are
                       shared with trained peers ("bird" shares visual_ANIMAL
                       with "dog"), so they still drive the right core area
    distributional     where the word appears and what surrounds it
    frame              which bigram frames it occurs in, which is what
                       separates function-word sub-types

``emergent_fuse_signals`` combines them, and the design choice worth naming is
that the weights are NOT hand-set per source.  Each signal's weight comes from
``signal_confidence``, computed from how peaked its own score distribution is
and how much exposure backed it.  A source that is confused about this
particular word down-weights ITSELF.  That is why a word with rich grounding
and no exposure, and a word with heavy exposure and no grounding, can both be
categorised by the same code path without a rule for each case.

Scope note: this infers categories for OOV holdouts from multimodal grounding
and distributional exposure.  It is not structural weight materialization (see
``research...materialize_structural_weights``) and not wobbly episode replay
(see ``replay_wobbly_episodes``), which are separate mechanisms that also
touch OOV words.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

from ..core.areas import ADV_CORE, CORE_TO_CATEGORY, GROUNDING_TO_CORE, FUNC_SUBCAT_TO_CORE

if TYPE_CHECKING:
    from ..parser_mixins.core import CoreParserMixin
    from ..core.grounding import GroundingContext

_MODALITY_FIELDS = (
    "visual", "motor", "properties", "spatial",
    "social", "temporal", "emotional",
)

# Emergent modality → core mapping (extends dominant-modality shortcut).
_GROUNDING_MODALITY_TO_CORE = {
    **GROUNDING_TO_CORE,
    "emotional": ADV_CORE,
}

_EXPOSURE_LOG_LIMIT = 4000

ScoreValue = float | str
BootstrapScores = Dict[str, ScoreValue]


def numeric_category_scores(scores: Mapping[str, ScoreValue]) -> Dict[str, float]:
    """Extract numeric category scores from a result carrying provenance."""
    return {
        key: float(value)
        for key, value in scores.items()
        if not key.startswith("_") and isinstance(value, (int, float))
    }


def is_word_in_lexicon(parser: "CoreParserMixin", word: str) -> bool:
    for lex in parser.core_lexicons.values():
        if word in lex:
            return True
    return False


def _clean_category_scores(scores: Dict[str, float]) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in scores.items()
        if k and k != "UNKNOWN" and not str(k).startswith("_")
    }


def signal_confidence(
    scores: Dict[str, float],
    *,
    exposure: int = 0,
) -> float:
    """Emergent weight from top-1 margin and optional exposure depth.

    ``strength = top * (1 + margin)``, so a source is trusted in proportion to
    both how strongly it backs its best category AND how far that category
    leads the runner-up.  A source that scores 0.9 for NOUN and 0.9 for VERB
    contributes almost nothing beyond its raw magnitude; one that scores 0.5
    and 0.0 contributes more.  This is what lets the fusion in
    ``emergent_fuse_signals`` avoid per-source hand-tuned weights: a signal
    that cannot discriminate discounts itself.

    Exposure multiplies by ``1 + 0.35 * log1p(exposure)``, capped at 2.5 --
    diminishing returns, so a word seen a hundred times is trusted more than
    one seen twice but not fifty times more.  The specific constants 0.35 and
    2.5 are not derived from anything documented here; they set how fast
    exposure earns trust and its ceiling.  Treat them as tuned, and re-tune
    together rather than individually.
    """
    clean = _clean_category_scores(scores)
    if not clean:
        return 0.0
    vals = sorted(clean.values(), reverse=True)
    top = vals[0]
    second = vals[1] if len(vals) > 1 else 0.0
    margin = max(0.0, top - second)
    strength = top * (1.0 + margin)
    if exposure > 0:
        strength *= min(2.5, 1.0 + math.log1p(exposure) * 0.35)
    return strength


def grounding_evidence_scores(
    ctx: Optional["GroundingContext"],
) -> Dict[str, float]:
    """Multi-modality grounding evidence — all active modalities contribute."""
    if ctx is None or not ctx.is_grounded:
        return {}

    raw: Dict[str, float] = defaultdict(float)
    for field in _MODALITY_FIELDS:
        features = getattr(ctx, field, None) or []
        if not features:
            continue
        core = _GROUNDING_MODALITY_TO_CORE.get(field)
        if core is None:
            continue
        cat = CORE_TO_CATEGORY.get(core)
        if cat:
            raw[cat] += float(len(features))

    total = sum(raw.values())
    if total <= 0:
        return {}
    return {cat: val / total for cat, val in raw.items()}


def grounding_modality_prior(
    parser: "CoreParserMixin",
    word: str,
    grounding: Optional["GroundingContext"] = None,
) -> Tuple[str, float]:
    """Backward-compatible prior — best category from full grounding evidence."""
    ctx = grounding if grounding is not None else parser.word_grounding.get(word)
    scores = grounding_evidence_scores(ctx)
    if not scores:
        return "UNKNOWN", 0.0
    cat = max(scores, key=lambda key: scores[key])
    return cat, signal_confidence(scores)


def fuse_category_scores(
    *parts: Tuple[Dict[str, float], float],
) -> Tuple[str, Dict[str, float]]:
    """Weighted fusion of category score dicts (explicit weights)."""
    combined: Dict[str, float] = defaultdict(float)
    for scores, weight in parts:
        for cat, val in scores.items():
            if cat and cat != "UNKNOWN" and not str(cat).startswith("_"):
                combined[cat] += float(val) * weight
    if not combined:
        return "UNKNOWN", {}
    best = max(combined, key=lambda key: combined[key])
    return best, dict(combined)


def record_wobbly_resolution(
    parser: "CoreParserMixin",
    word: str,
    category: str,
    *,
    stability: float = 0.0,
    exposure: int = 1,
) -> None:
    """Record POS resolution from wobbly-parse competition (episodic evidence)."""
    resolutions: Dict[str, Dict[str, object]] = getattr(
        parser, "_wobbly_resolutions", {},
    )
    resolutions[word] = {
        "category": category,
        "stability": stability,
        "exposure": exposure,
    }
    parser._wobbly_resolutions = resolutions


def _wobbly_resolution_scores(
    parser: "CoreParserMixin",
    word: str,
) -> Dict[str, float]:
    """Category scores from wobbly-parse hypothesis competition."""
    entry = getattr(parser, "_wobbly_resolutions", {}).get(word)
    if not entry:
        return {}
    cat = entry.get("category")
    if not cat or cat == "UNKNOWN":
        return {}
    stability = float(entry.get("stability", 0.0))
    exposure = int(entry.get("exposure", 1))
    strength = max(0.35, stability) * min(2.0, 1.0 + 0.15 * exposure)
    return {str(cat): strength}


def emergent_fuse_signals(
    signals: List[Tuple[str, Dict[str, float], int]],
) -> Tuple[str, Dict[str, float]]:
    """Fuse evidence sources with confidence weights that emerge from each signal.

    Each signal is ``(name, category_scores, exposure_count)``.
    """
    combined: Dict[str, float] = defaultdict(float)
    meta: Dict[str, float] = {}

    for name, scores, exposure in signals:
        clean = _clean_category_scores(scores)
        weight = signal_confidence(clean, exposure=exposure)
        if weight <= 0:
            continue
        for cat, val in clean.items():
            combined[cat] += val * weight
        meta[f"_weight_{name}"] = round(weight, 4)

    if not combined:
        return "UNKNOWN", meta

    best = max(combined, key=lambda key: combined[key])
    total = sum(combined.values())
    fused = {cat: val / max(total, 1e-9) for cat, val in combined.items()}
    fused.update(meta)
    return best, fused


def _frame_pos_scores(
    parser: "CoreParserMixin",
    word: str,
) -> Dict[str, float]:
    """Map frame classifier output to open-class POS scores."""
    if not hasattr(parser, "classify_by_frame"):
        return {}
    frame_cat, frame_conf = parser.classify_by_frame(word)
    if not frame_cat or frame_conf <= 0:
        return {}

    category = CORE_TO_CATEGORY.get(FUNC_SUBCAT_TO_CORE.get(frame_cat), frame_cat)
    if category in CORE_TO_CATEGORY.values():
        return {category: frame_conf}
    return {}


def classify_word_bootstrapped(
    parser: "CoreParserMixin",
    word: str,
    grounding: Optional["GroundingContext"] = None,
) -> Tuple[str, BootstrapScores]:
    """Classify by competing emergent evidence — no fixed fusion constants.

    Signals (when available):
    - neural lexicon readout with grounding features projected
    - distributional frame / position statistics from raw exposure
    - multi-modality grounding feature density
    - phon-only readout as last resort
    """
    ctx = grounding if grounding is not None else parser.word_grounding.get(word)
    exposure = parser.dist_stats.word_count.get(word, 0)

    if is_word_in_lexicon(parser, word):
        evidence = parser.classify_word_evidence(word, grounding=ctx)
        scores = evidence.category_scores()
        return evidence.category, {
            "_source": "lexicon_readout" if evidence.source == "neural" else evidence.source,
            **scores,
            "_confidence": signal_confidence(scores),
        }

    dist_scores: Dict[str, float] = {}
    if exposure > 0:
        _, dist_scores = parser.classify_distributional(word)

    signals: List[Tuple[str, Dict[str, float], int]] = []

    if dist_scores:
        signals.append(("distributional", dist_scores, exposure))

    if ctx is None or not ctx.is_grounded:
        frame_scores = _frame_pos_scores(parser, word)
        if frame_scores:
            signals.append(("frame", frame_scores, exposure))
        wobbly_scores = _wobbly_resolution_scores(parser, word)
        if wobbly_scores:
            wobbly_exp = int(
                getattr(parser, "_wobbly_resolutions", {})
                .get(word, {})
                .get("exposure", 1),
            )
            signals.append(("wobbly", wobbly_scores, wobbly_exp))
        if signals:
            cat, fused_numeric = emergent_fuse_signals(signals)
            fused: BootstrapScores = dict(fused_numeric)
            fused["_source"] = "distributional+frame"
            return cat, fused
        evidence = parser.classify_word_evidence(word, grounding=None)
        scores = evidence.category_scores()
        source = "phon" if evidence.source == "neural" else evidence.source
        return evidence.category, {"_source": source, **scores,
                                   "_confidence": signal_confidence(scores)}

    evidence = parser.classify_word_evidence(word, grounding=ctx)
    neural_cat = evidence.category if evidence.source == "neural" else "UNKNOWN"
    neural_by_cat = evidence.category_scores() if evidence.source == "neural" else {}
    # Distributional evidence was already added above; a fallback is not a
    # second independent neural signal.
    if neural_by_cat:
        signals.append(("neural", neural_by_cat, exposure))

    ground_scores = grounding_evidence_scores(ctx)
    if ground_scores:
        signals.append(("grounding", ground_scores, max(1, len(ctx.visual) + len(ctx.motor))))

    frame_scores = _frame_pos_scores(parser, word)
    if frame_scores:
        signals.append(("frame", frame_scores, exposure))

    wobbly_scores = _wobbly_resolution_scores(parser, word)
    if wobbly_scores:
        wobbly_exp = int(
            getattr(parser, "_wobbly_resolutions", {})
            .get(word, {})
            .get("exposure", 1),
        )
        signals.append(("wobbly", wobbly_scores, wobbly_exp))

    if not signals:
        return neural_cat, {
            "_source": "neural_only",
            **neural_by_cat,
            "_confidence": signal_confidence(neural_by_cat),
        }

    cat, fused_numeric = emergent_fuse_signals(signals)
    fused: BootstrapScores = dict(fused_numeric)
    if cat == "UNKNOWN":
        cat = neural_cat if neural_cat != "UNKNOWN" else cat
    fused["_source"] = "bootstrap"
    fused["_neural"] = neural_cat
    fused["_grounding"] = (
        max(ground_scores, key=lambda key: ground_scores[key]) if ground_scores else "UNKNOWN"
    )
    if dist_scores:
        fused["_distributional"] = max(dist_scores, key=lambda key: dist_scores[key])
    fused["_confidence"] = signal_confidence(numeric_category_scores(fused))
    return cat, fused


def record_exposure_sentence(parser: "CoreParserMixin", words: List[str]) -> None:
    """Append a raw sentence to the parser exposure log (for emergent bootstrap)."""
    if not words:
        return
    log: List[List[str]] = getattr(parser, "_exposure_log", [])
    if len(log) >= _EXPOSURE_LOG_LIMIT:
        parser._exposure_log = log[-(_EXPOSURE_LOG_LIMIT - 1):] + [list(words)]
    else:
        log.append(list(words))
        parser._exposure_log = log


def sentences_from_exposure_log(
    parser: "CoreParserMixin",
    target_words: Set[str],
) -> List[List[str]]:
    """Return observed sentences that contain any target word."""
    targets = set(target_words)
    if not targets:
        return []
    log = getattr(parser, "_exposure_log", [])
    return [list(s) for s in log if targets & set(s)]


def sentences_from_transition_paths(
    parser: "CoreParserMixin",
    target_words: Set[str],
    *,
    max_per_word: int = 8,
) -> List[List[str]]:
    """Reconstruct short contexts from observed bigrams — no labeled corpus."""
    stats = parser.dist_stats
    targets = set(target_words)
    out: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()

    def _add(tokens: List[str]) -> None:
        key = tuple(tokens)
        if key in seen or not key:
            return
        seen.add(key)
        out.append(tokens)

    # sorted(): `targets` is a SET OF STRINGS, whose iteration order is
    # randomized per process (PEP 456). `out` is built in this order and becomes
    # the training corpus, so an unsorted loop makes the whole curriculum
    # order-dependent on PYTHONHASHSEED. See `infer_holdout_categories`.
    for word in sorted(targets):
        lefts = sorted(
            ((w1, c) for (w1, w2), c in stats.transitions.items() if w2 == word),
            key=lambda x: -x[1],
        )
        rights = sorted(
            ((w2, c) for (w1, w2), c in stats.transitions.items() if w1 == word),
            key=lambda x: -x[1],
        )
        for w1, _ in lefts[:max_per_word]:
            _add([w1, word])
            for w2, _ in rights[:max_per_word]:
                _add([w1, word, w2])
        for w2, _ in rights[:max_per_word]:
            _add([word, w2])

    return out


def _grounded_from_tokens(
    parser: "CoreParserMixin",
    token_lists: List[List[str]],
) -> List:
    from ..core.grounding import GroundingContext
    from ..core.sentence import GroundedSentence

    grounded = []
    for tokens in token_lists:
        grounded.append(
            GroundedSentence(
                words=tokens,
                contexts=[
                    parser.word_grounding.get(w, GroundingContext())
                    for w in tokens
                ],
            ),
        )
    return grounded


def ingest_holdout_sentence_stats(
    parser: "CoreParserMixin",
    holdout_words: Set[str],
    *,
    allow_canonical_fallback: bool = True,
) -> int:
    """Ingest distributional stats from *observed* contexts containing holdouts.

    Priority:
    1. Sentences from the parser exposure log (raw experience).
    2. Bigram paths reconstructed from ``dist_stats.transitions``.
    3. Optional canonical fallback (labeled curriculum) when nothing was observed.
    """
    from ..core.corpus_index import compile_corpus, ingest_index_stats

    holdout = set(holdout_words)
    if not holdout:
        return 0

    token_lists = sentences_from_exposure_log(parser, holdout)
    if not token_lists:
        token_lists = sentences_from_transition_paths(parser, holdout)

    if not token_lists and allow_canonical_fallback:
        from ..curriculum.data import create_training_sentences

        token_lists = [
            list(s.words)
            for s in create_training_sentences()
            if holdout & set(s.words)
        ]

    if not token_lists:
        return 0

    grounded = _grounded_from_tokens(parser, token_lists)
    idx = compile_corpus(parser, grounded)
    ingest_index_stats(parser, idx)
    return len(token_lists)


def pos_inference_confidence(scores: Mapping[str, ScoreValue]) -> float:
    """Confidence of a POS inference classification for cache gating."""
    return signal_confidence(numeric_category_scores(scores))


def infer_holdout_categories(
    parser: "CoreParserMixin",
    holdout_words: Set[str],
    *,
    min_confidence: float = 0.08,
    refine_passes: int = 2,
    min_count: Optional[int] = None,
) -> Dict[str, str]:
    """Infer holdout POS from emergent fusion; cache only confident assignments."""
    _ = min_count  # legacy kwarg; fusion no longer skips low-exposure words
    assigned: Dict[str, str] = {}
    if not hasattr(parser, "_bootstrap_categories"):
        parser._bootstrap_categories = {}

    targets = {w for w in holdout_words if not is_word_in_lexicon(parser, w)}
    if not targets:
        return assigned

    ingest_holdout_sentence_stats(parser, targets)

    for _pass in range(max(1, refine_passes)):
        # sorted(): THIS IS THE LOOP THAT MADE TRAINING IRREPRODUCIBLE ACROSS
        # PROCESSES (#80). `targets` is a set of strings, so its iteration order
        # is randomized per process (PEP 456), and `classify_word_bootstrapped`
        # historically recruited neurons during classification. Isolated neural
        # queries now prevent that mutation; keep deterministic ordering for
        # parser caches and metadata as well.
        #
        # Measured on `train_parser_to_depth("SENTENCES", seed=42)` before the
        # fix: 16564 / 16572 / 16638 materialized neurons in three processes,
        # with Cohen's d on the ERP contrast moving 1.452 -> 1.291. Pinning
        # PYTHONHASHSEED=0 made it 16491 every time, which is what localized it
        # here. Same class as [[pythonhashseed-nondeterminism]] (#21), which
        # fixed `train_lexicon`; the guard written then covers
        # `p.train(create_training_sentences())` and never reached the
        # CURRICULUM path, so this site stayed open behind a green test.
        for word in sorted(targets):
            cat, scores = classify_word_bootstrapped(parser, word)
            conf = pos_inference_confidence(scores)
            if cat == "UNKNOWN" or conf < min_confidence:
                continue
            assigned[word] = cat
            parser._bootstrap_categories[word] = cat
            parser._category_cache[word] = cat
            if not hasattr(parser, "_dist_categories"):
                parser._dist_categories = {}
            parser._dist_categories[word] = cat

    return assigned


def decompose_word_classification(
    parser: "CoreParserMixin",
    word: str,
    expected: str,
) -> Dict[str, object]:
    """Per-word breakdown: grounding, statistics, fusion, failure mode."""
    ctx = parser.word_grounding.get(word)
    in_lex = is_word_in_lexicon(parser, word)
    dist_n = parser.dist_stats.word_count.get(word, 0)

    evidence = parser.classify_word_evidence(word, grounding=ctx) if ctx else None
    neural_cat = evidence.category if evidence and evidence.source == "neural" else "UNKNOWN"
    neural_by_cat = evidence.category_scores() if evidence and evidence.source == "neural" else {}

    dist_cat, dist_scores = (
        parser.classify_distributional(word) if dist_n > 0 else ("UNKNOWN", {})
    )
    ground_scores = grounding_evidence_scores(ctx)
    prior_cat = max(ground_scores, key=lambda key: ground_scores[key]) if ground_scores else "UNKNOWN"
    boot_cat, boot_scores = classify_word_bootstrapped(parser, word, ctx)

    pre = parser.dist_stats.word_as_pre_verb.get(word, 0)
    post = parser.dist_stats.word_as_post_verb.get(word, 0)
    action = parser.dist_stats.word_as_action.get(word, 0)

    grounding_stims = parser._grounding_stim_names(ctx) if ctx else []
    active_stims = [
        s for s in grounding_stims
        if s in parser._grounding_stim_names_set
    ]

    predicted = boot_cat
    correct = predicted == expected

    failure_mode = None
    if not correct:
        if dist_n == 0:
            failure_mode = "no_distributional_exposure"
        elif max(neural_by_cat.values(), default=0.0) < 0.2:
            failure_mode = "weak_grounding_readout"
        elif dist_cat != expected and neural_cat != expected:
            failure_mode = "both_signals_wrong"
        elif dist_cat == expected and boot_cat != expected:
            failure_mode = "fusion_underweighted_statistics"
        elif prior_cat != expected and not ground_scores.get(expected, 0):
            failure_mode = "grounding_evidence_wrong"
        else:
            failure_mode = "ambiguous_competition"

    return {
        "word": word,
        "expected": expected,
        "in_lexicon": in_lex,
        "grounded": bool(ctx and ctx.is_grounded),
        "dominant_modality": ctx.dominant_modality if ctx else "none",
        "grounding_evidence": dict(ground_scores),
        "grounding_features": grounding_stims,
        "active_grounding_stims": active_stims,
        "dist_count": dist_n,
        "verb_relative": {"pre_verb": pre, "post_verb": post, "action": action},
        "classification_source": evidence.source if evidence else "none",
        "neural_readout": neural_cat,
        "neural_by_category": neural_by_cat,
        "distributional": dist_cat,
        "distributional_scores": dict(list(dist_scores.items())[:6]),
        "modality_prior": prior_cat,
        "bootstrapped": boot_cat,
        "bootstrapped_scores": {
            k: round(v, 3)
            for k, v in numeric_category_scores(boot_scores).items()
        },
        "bootstrap_confidence": round(pos_inference_confidence(boot_scores), 4),
        "correct_neural": neural_cat == expected,
        "correct_distributional": dist_cat == expected,
        "correct_bootstrapped": correct,
        "failure_mode": failure_mode,
    }


def decompose_holdout_classification(
    parser: "CoreParserMixin",
    holdout_words: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Decompose holdout POS accuracy by signal source."""
    holdouts = holdout_words or {
        "bird": "NOUN",
        "finds": "VERB",
        "small": "ADJ",
    }
    per_word = {
        word: decompose_word_classification(parser, word, expected)
        for word, expected in holdouts.items()
    }
    n = len(per_word)
    return {
        "per_word": per_word,
        "accuracy_neural": sum(v["correct_neural"] for v in per_word.values()) / n,
        "accuracy_distributional": sum(
            v["correct_distributional"] for v in per_word.values()
        ) / n,
        "accuracy_bootstrapped": sum(
            v["correct_bootstrapped"] for v in per_word.values()
        ) / n,
        "failure_modes": {
            w: v["failure_mode"] for w, v in per_word.items() if not v["correct_bootstrapped"]
        },
    }


def format_holdout_decomposition(decomp: Dict[str, object]) -> str:
    lines = [
        "Holdout classification decomposition",
        f"  neural:         {decomp['accuracy_neural']:.1%}",
        f"  distributional: {decomp['accuracy_distributional']:.1%}",
        f"  bootstrapped:   {decomp['accuracy_bootstrapped']:.1%}",
        "",
    ]
    for word, info in decomp["per_word"].items():  # type: ignore[union-attr]
        lines.append(f"  {word} (expected {info['expected']})")
        lines.append(
            f"    modality={info['dominant_modality']} "
            f"dist_n={info['dist_count']} "
            f"active_stims={len(info['active_grounding_stims'])}/{len(info['grounding_features'])}"
        )
        if info.get("grounding_evidence"):
            ge = info["grounding_evidence"]
            top = max(ge, key=lambda key: ge[key])
            lines.append(f"    grounding_evidence top={top} ({ge[top]:.2f})")
        lines.append(
            f"    neural={info['neural_readout']} dist={info['distributional']} "
            f"prior={info['modality_prior']} -> bootstrap={info['bootstrapped']} "
            f"(conf={info.get('bootstrap_confidence', 0):.2f})"
        )
        if info["failure_mode"]:
            lines.append(f"    failure_mode={info['failure_mode']}")
        if info["verb_relative"]["action"] or info["verb_relative"]["pre_verb"]:
            vr = info["verb_relative"]
            lines.append(
                f"    verb_relative pre={vr['pre_verb']} action={vr['action']} post={vr['post_verb']}"
            )
    return "\n".join(lines)


# Legacy names (deprecated)
bootstrap_holdout_categories = infer_holdout_categories
bootstrap_confidence = pos_inference_confidence
