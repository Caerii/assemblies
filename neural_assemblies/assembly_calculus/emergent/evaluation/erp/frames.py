"""Composed-ERP calibration frames and frame-level sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .adapters import (
    N400_WEIGHT,
    P600_WEIGHT,
    measure_lexical_surprise,
    measure_live_integration,
)
from .gates import (
    ErpBaseline,
    ErpReadiness,
    ErpThresholds,
    ErpViolation,
    assess_erp_readiness,
    classify_erp_violation,
    default_erp_thresholds,
)
from .runner import ErpProbeResult, run_incremental_erp_probes

if TYPE_CHECKING:
    from ...parser import EmergentParser

CalibrationFrame = Tuple[str, str, List[str]]

#: EVERY WORD IN A FRAME MUST BE IN THE TRAINED VOCABULARY.
#:
#: `verb as object 3` read `["she", "hits", "the", "eats"]` until 2026-08-05.
#: `hits` is in no curriculum sentence and no holdout -- it occurred ONLY in
#: this file -- so the parser categorised it UNKNOWN and that item's MAIN VERB
#: was unrecognised before its critical word was ever reached (measured:
#: p600 0.0000, phrase_stability 1.0000, i.e. the degenerate no-parse reading).
#: A third of the category-violation arm was therefore not a category violation.
#: It is now `chases`, which is trained (16 curriculum occurrences) and makes
#: the item a MINIMAL PAIR of its grammatical control `she chases the cat`, as
#: items 1 and 2 already were.
#:
#: The eight trained verbs are runs/sees/eats/chases/plays/sleeps/reads/finds
#: (`core/grounding.py`). Adding a word to the vocabulary is NOT the cheaper fix
#: -- it moves the substrate for every result in the repo; editing the frame is
#: contained.
#:
#: KNOWN, NOT FIXED HERE: `verb as object` uses `finds` as its critical word,
#: and `finds` is a DEFAULT_LEXICON_HOLDOUT. That item is simultaneously a
#: category violation and a novel word, which is exactly the contrast the
#: `novel_noun` arm exists to isolate. Swapping it changes what the violation
#: arm MEANS, so it wants a measurement rather than an edit -- see #108.
DEFAULT_CALIBRATION_FRAMES: List[CalibrationFrame] = [
    ("grammatical", "trained noun object", ["the", "dog", "chases", "cat"]),
    ("grammatical", "trained noun object 2", ["the", "cat", "sees", "dog"]),
    ("grammatical", "trained noun object 3", ["she", "chases", "the", "cat"]),
    ("category_violation", "verb as object", ["the", "dog", "chases", "finds"]),
    ("category_violation", "verb as object 2", ["the", "cat", "sees", "runs"]),
    ("category_violation", "verb as object 3", ["she", "chases", "the", "eats"]),
    ("novel_noun", "holdout noun object", ["the", "dog", "chases", "bird"]),
    ("novel_noun", "holdout noun subject", ["the", "bird", "sees", "the", "cat"]),
    ("novel_noun", "holdout adj attributive", ["the", "small", "dog", "runs"]),
]

#: Every condition puts its critical word in OBJECT position after the verb.
#:
#: `DEFAULT_CALIBRATION_FRAMES` above cannot support a 2x2 dissociation, and the
#: reason is item design rather than metric choice. Measured pathway counts on
#: that set: every grammatical and category-violation item expects ROLE_PATIENT,
#: but 8 of 12 novel items expect ROLE_AGENT -- "the bird sees the cat" puts the
#: novel word in SUBJECT position, and "the small dog runs" probes an attributive
#: adjective before any verb. So d(novel/grammatical) compares DIFFERENT BRAIN
#: AREAS, and `input_drive`'s own docstring warns that cross-area comparison
#: reverses the ranking on size alone. No metric can rescue that.
#:
#: Here the subject and verb vary while the critical word stays in object
#: position, so all three conditions expect the same slot and the contrast is
#: area-matched by construction. `bird` is the only holdout NOUN
#: (`DEFAULT_LEXICON_HOLDOUTS`), which is why it recurs.
#:
#: KEPT SEPARATE, not substituted, deliberately. Threshold calibration consumes
#: `DEFAULT_CALIBRATION_FRAMES`, so swapping it would move calibrated thresholds
#: and every golden that depends on them; and the adjective frame tests
#: attributive generalization, which is a real phenomenon that simply cannot
#: live in an area-matched role contrast. Promote this to the default only with
#: a measurement behind it.
AREA_MATCHED_CALIBRATION_FRAMES: List[CalibrationFrame] = [
    ("grammatical", "trained noun object", ["the", "dog", "chases", "cat"]),
    ("grammatical", "trained noun object 2", ["the", "cat", "sees", "dog"]),
    ("grammatical", "trained noun object 3", ["she", "chases", "the", "cat"]),
    ("category_violation", "verb as object", ["the", "dog", "chases", "finds"]),
    ("category_violation", "verb as object 2", ["the", "cat", "sees", "runs"]),
    ("category_violation", "verb as object 3", ["she", "chases", "the", "eats"]),
    ("novel_noun", "holdout noun object", ["the", "dog", "chases", "bird"]),
    ("novel_noun", "holdout noun object 2", ["the", "cat", "sees", "bird"]),
    ("novel_noun", "holdout noun object 3", ["she", "chases", "the", "bird"]),
]

# Minimal frame set for sweep-mode calibration (2 per label class).
SWEEP_CALIBRATION_FRAMES: List[CalibrationFrame] = [
    ("grammatical", "trained noun object", ["the", "dog", "chases", "cat"]),
    ("grammatical", "trained noun object 2", ["she", "chases", "the", "cat"]),
    ("category_violation", "verb as object", ["the", "dog", "chases", "finds"]),
    ("category_violation", "verb as object 2", ["she", "chases", "the", "eats"]),
    ("novel_noun", "holdout noun object", ["the", "dog", "chases", "bird"]),
    ("novel_noun", "holdout adj attributive", ["the", "small", "dog", "runs"]),
]


@dataclass
class PositionErpSample:
    """ERP readout at one critical word in a calibration frame."""
    label: str
    sentence: Tuple[str, ...]
    position: int
    word: str
    category: str
    n400: float
    p600: float
    phrase_stability: float
    n400_excess: float = 0.0
    p600_excess: float = 0.0
    violation: Optional[ErpViolation] = None


@dataclass
class _WarmParseState:
    """Incremental parse carry-over for prefix warm-start between frames."""
    circuit: object
    verb_seen: bool
    noun_count: int
    subject_core: Optional[str]
    categories: Dict[str, str]


def critical_position_for_frame(
    label: str,
    known: List[str],
    *,
    holdout_words: Optional[Set[str]] = None,
) -> int:
    """Pick probe index: holdout token for novel frames, else final content word."""
    if label == "novel_noun" and holdout_words:
        for i, w in enumerate(known):
            if w in holdout_words:
                return i
    if label == "novel_noun" and "small" in known:
        return known.index("small")
    return len(known) - 1


def _probe_at_critical_position(
    parser: "EmergentParser",
    known: List[str],
    pos: int,
    *,
    readiness: ErpReadiness,
    baseline: ErpBaseline,
    probe_depth: str,
):
    """ERP probe at one index; parse only through *pos*."""
    _, probes = run_incremental_erp_probes(
        parser,
        known,
        apply_calibration=False,
        baseline=baseline,
        readiness=readiness,
        probe_depth=probe_depth,
        stop_at_position=pos,
        probe_positions={pos},
    )
    for p in probes:
        if p.position == pos:
            return p
    return None


def _longest_cached_prefix(
    known: List[str],
    cache: Dict[Tuple[str, ...], _WarmParseState],
) -> int:
    for plen in range(len(known), -1, -1):
        if tuple(known[:plen]) in cache:
            return plen
    return 0


def _ensure_parsed_through(
    parser: "EmergentParser",
    known: List[str],
    end_pos: int,
    cache: Dict[Tuple[str, ...], _WarmParseState],
) -> _WarmParseState:
    """Parse *known* through index *end_pos* (inclusive), reusing cached prefixes."""
    from ...core.areas import CATEGORY_TO_CORE

    plen = _longest_cached_prefix(known, cache)
    if plen == 0:
        parser._reset_context_state()
        circuit = parser._get_incremental_circuit(reset=True)
        state = _WarmParseState(
            circuit=circuit,
            verb_seen=False,
            noun_count=0,
            subject_core=None,
            categories={},
        )
        start = 0
    else:
        state = cache[tuple(known[:plen])]
        start = plen

    with parser.brain.frozen():
        for i in range(start, end_pos + 1):
            word = known[i]
            cat, state.verb_seen, state.noun_count = (
                parser._advance_incremental_word(
                    word,
                    state.circuit,
                    state.verb_seen,
                    state.noun_count,
                )
            )
            state.categories[word] = cat
            if cat in ("NOUN", "PRON") and i <= 1 and not state.verb_seen:
                state.subject_core = CATEGORY_TO_CORE.get(cat)
            elif state.verb_seen and cat in ("NOUN", "PRON") and state.subject_core is None:
                state.subject_core = CATEGORY_TO_CORE.get(cat)
            cache[tuple(known[: i + 1])] = _WarmParseState(
                circuit=state.circuit,
                verb_seen=state.verb_seen,
                noun_count=state.noun_count,
                subject_core=state.subject_core,
                categories=dict(state.categories),
            )

    return cache[tuple(known[: end_pos + 1])]


def _probe_at_critical_position_warm(
    parser: "EmergentParser",
    known: List[str],
    pos: int,
    *,
    readiness: ErpReadiness,
    baseline: ErpBaseline,
    probe_depth: str,
    cache: Dict[Tuple[str, ...], _WarmParseState],
) -> Optional[ErpProbeResult]:
    """ERP probe at *pos* with prefix parse cache reuse."""
    from ...core.areas import CATEGORY_TO_CORE

    if pos >= len(known):
        pos = len(known) - 1
    word = known[pos]
    prefix = known[:pos]

    if pos > 0:
        _ensure_parsed_through(parser, known, pos - 1, cache)
        state = cache[tuple(known[:pos])]
    else:
        parser._reset_context_state()
        state = _WarmParseState(
            circuit=parser._get_incremental_circuit(reset=True),
            verb_seen=False,
            noun_count=0,
            subject_core=None,
            categories={},
        )
        cache[tuple()] = state

    # Same definedness unwrap as `runner`: `Measured` in, legacy float out, so
    # the stored sample keeps the exact value it had before the migration.
    _n400_m = (
        measure_lexical_surprise(parser, tuple(prefix), word, readiness=readiness)
        if prefix else None
    )
    n400 = 0.0 if _n400_m is None else _n400_m.or_else(
        float((_n400_m.detail or {}).get("legacy", 0.0)))

    with parser.brain.frozen():
        cat, verb_seen, noun_count = parser._advance_incremental_word(
            word,
            state.circuit,
            state.verb_seen,
            state.noun_count,
        )
        subject_core = state.subject_core
        if cat in ("NOUN", "PRON") and len(prefix) <= 1 and not state.verb_seen:
            subject_core = CATEGORY_TO_CORE.get(cat)
        elif verb_seen and cat in ("NOUN", "PRON") and subject_core is None:
            subject_core = CATEGORY_TO_CORE.get(cat)

        p600, role_area, stability = measure_live_integration(
            parser,
            word,
            cat,
            verb_seen=verb_seen,
            subject_core=subject_core,
            readiness=readiness,
            probe_depth=probe_depth,
        )

    cache[tuple(known[: pos + 1])] = _WarmParseState(
        circuit=state.circuit,
        verb_seen=verb_seen,
        noun_count=noun_count,
        subject_core=subject_core,
        categories={**state.categories, word: cat},
    )

    combined = N400_WEIGHT * n400 + P600_WEIGHT * p600
    return ErpProbeResult(
        word=word,
        position=pos,
        prefix=tuple(prefix),
        category=cat,
        n400=round(n400, 4),
        p600=round(p600, 4),
        combined=round(combined, 4),
        phrase_stability=round(stability, 4),
        role_area=role_area,
        wobbly=False,
        error_active=False,
        failure_signature="",
        violation=None,
    )


def collect_frame_samples(
    parser: "EmergentParser",
    frames: List[CalibrationFrame],
    *,
    critical_position: Optional[int] = None,
    readiness: Optional[ErpReadiness] = None,
    baseline: Optional[ErpBaseline] = None,
    thresholds: Optional[ErpThresholds] = None,
    holdout_words: Optional[Set[str]] = None,
    probe_depth: str = "calibration",
    warm_start: bool = False,
) -> List[PositionErpSample]:
    """Measure ERP at critical word position for each calibration frame."""
    readiness = readiness or assess_erp_readiness(parser)
    baseline = baseline if baseline is not None else ErpBaseline()
    thresholds = thresholds or default_erp_thresholds()

    ordered = sorted(frames, key=lambda fr: fr[2]) if warm_start else frames
    prefix_cache: Dict[Tuple[str, ...], _WarmParseState] = {}

    samples: List[PositionErpSample] = []
    for label, _desc, words in ordered:
        known = [w for w in words if w in parser.stim_map]
        if len(known) < 2:
            continue
        pos = critical_position
        if pos is None:
            pos = critical_position_for_frame(
                label, known, holdout_words=holdout_words,
            )
        if pos >= len(known):
            pos = len(known) - 1

        if warm_start:
            p = _probe_at_critical_position_warm(
                parser, known, pos,
                readiness=readiness,
                baseline=baseline,
                probe_depth=probe_depth,
                cache=prefix_cache,
            )
        else:
            p = _probe_at_critical_position(
                parser, known, pos,
                readiness=readiness,
                baseline=baseline,
                probe_depth=probe_depth,
            )
        if p is None:
            continue
        n400_ex = baseline.n400_excess(p.n400)
        p600_ex = baseline.p600_excess(p.p600)
        violation = classify_erp_violation(
            p.n400,
            p.p600,
            readiness=readiness,
            baseline=baseline,
            phrase_stability=p.phrase_stability,
            thresholds=thresholds,
        )
        samples.append(
            PositionErpSample(
                label=label,
                sentence=tuple(known),
                position=pos,
                word=p.word,
                category=p.category,
                n400=p.n400,
                p600=p.p600,
                phrase_stability=p.phrase_stability,
                n400_excess=n400_ex,
                p600_excess=p600_ex,
                violation=violation,
            ),
        )
    return samples


collect_position_samples = collect_frame_samples
