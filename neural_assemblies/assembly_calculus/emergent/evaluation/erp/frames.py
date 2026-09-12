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
from .protocol import ErpProtocol
from .runner import ErpProbeResult, run_incremental_erp_probes

if TYPE_CHECKING:
    from ....fiber import FiberCircuit
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

#: Every word is TRAINED at the depth these frames are measured at, and every
#: critical word sits in object position.
#:
#: WHY THIS EXISTS. Both sets above are authored from the vocabulary of
#: `create_training_sentences()` -- the FULL_TRAIN corpus -- while every ERP
#: number in this repo is measured on `get_parser_cache().fork("SENTENCES")`,
#: which `CurriculumTrainer` builds from the CDS corpus. Audited there
#: (`research/experiments/erp_frame_vocabulary_audit.py`), 9 of their 11 words
#: have NO core lexicon entry, and 9 of 9 items contain a word that is neither
#: trained nor a declared holdout. `chases` -- the main verb of five of the nine
#: default items -- occurs ZERO times in that corpus and classifies NOUN, so
#: "the dog chases cat" parses DET NOUN NOUN NOUN: those items contain no verb,
#: and their category violation violates nothing that was parsed. `small`, the
#: entire point of the attributive item, classifies VERB.
#:
#: That is the `hits` defect recorded above, at scale -- and its fix picked
#: `chases` for "16 curriculum occurrences", a count taken from the wrong
#: corpus. Which is why these words were CHOSEN BY MEASURING THE PARSER
#: (`research/experiments/erp_frame_candidate_selection.py`) against three
#: separate predicates:
#:   * TRAINED -- has a core lexicon assembly. This is what `stim_map` was
#:     standing in for and is not: 517 registered stimuli, 123 trained words.
#:   * CLASSIFIED as the category its slot needs, checked separately because
#:     trained and correctly-classified are independent properties (22 of 126
#:     words classify DIFFERENTLY across two seeds; `small` is one of them, so
#:     seed-stability is part of the predicate).
#:   * OCCURRING at least once in the corpus that trained this depth, so the
#:     word carries distributional evidence and not only lexicon exposure.
#:
#: Each triple is a MINIMAL PAIR -- same determiner, same subject, same verb --
#: so the three arms differ in the critical word ALONE, which is the contrast
#: the labels claim and the shipped sets do not deliver. `bird` recurs because
#: it is the only holdout word classified NOUN.
#:
#: The selector also PROPOSED `toy` as a main verb, because the parser
#: classifies it VERB. That is a classifier error, not an item design, and it
#: is the reason this list is hand-checked rather than generated: a measurement
#: can tell you a word is usable, it cannot tell you the measurement was right.
#:
#: NOT THE DEFAULT, AND MEASURED -- IT FAILED ITS OWN BAR (2026-08-06).
#:
#: Cold, 10 seeds, `disk_hits=0 trained_fresh=10`
#: (`research/experiments/erp_trained_frames_study.py`):
#:
#:     area_matched -> trained_area_matched
#:     p600_auc_of_raw   0.7556 +/- 0.1046  ->  1.0000 +/- 0.0000   [FAIL]
#:         CONSTANT across all seeds
#:     untrained_items   9 -> 0
#:
#: **A perfect 1.0000 with ZERO seed variance is not a better measurement.** It
#: is the `afferent_energy` signature inverted -- that candidate was rejected on
#: AUC 0.000 with zero variance, and zero variance is the signature of a
#: CONSTANT either way. The pre-registered `must_vary` criterion caught it,
#: which is the entire reason that criterion is encoded rather than remembered.
#:
#: LEADING HYPOTHESIS, NOT YET TESTED: the area-identity confound one level
#: over. `expected_slot` area-matched the probe's TARGET, but
#: `measure_live_integration` still derives the SOURCE core from the OBSERVED
#: category (`CATEGORY_TO_CORE[category]`), so the violation arm reads
#: VERB_CORE -> ROLE_PATIENT and its control reads NOUN_CORE -> ROLE_PATIENT.
#: With the old items that difference was buried in noise, because five of nine
#: had no main verb. Clean items may simply have exposed it -- in which case
#: this frame set is measuring whether the critical word is a noun or a verb.
#:
#: So these frames are a BETTER ITEM DESIGN AND AN UNUSABLE MEASUREMENT until a
#: degenerate control separates the two readings, exactly as
#: `p600_is_confounded_with_area_identity.md` did for the target area. Do not
#: promote on the strength of the 1.0000; that number is the reason to doubt it.
#: REFRESHED 2026-08-09 (#113 continuation): the original words
#: (food/take/give/bed) trained and classified correctly when selected
#: (ed2b58b, seeds 11+42) but stopped training on neither code nor corpus
#: intent -- `curriculum/generation.py`'s sentence generator draws from
#: ONE internal `_rng.seed(42)` stream (its own fixed seed, unrelated to
#: the outer parser seed), so any edit that adds/removes a draw earlier in
#: the function reshuffles every later draw. That module did not exist at
#: selection time (0d330b5 extracted it from trainer.py immediately after
#: ed2b58b) and four further commits (#135/#136/#142/#149) touched it
#: since -- each an innocent, unrelated change that nonetheless silently
#: reshuffled which low-frequency nouns/verbs land in a generated
#: sentence. Re-run via `erp_frame_candidate_selection.py`, widened from
#: 2 to 8 seeds (SEEDS in that script) precisely because 2-seed agreement
#: had already once been mistaken for stability. See
#: research/notes/the_calibration_frame_set_was_never_wrong.md.
TRAINED_AREA_MATCHED_CALIBRATION_FRAMES: List[CalibrationFrame] = [
    ("grammatical", "trained noun object", ["the", "dog", "want", "baby"]),
    ("category_violation", "verb as object", ["the", "dog", "want", "come"]),
    ("novel_noun", "holdout noun object", ["the", "dog", "want", "bird"]),
    ("grammatical", "trained noun object 2", ["the", "book", "go", "cat"]),
    ("category_violation", "verb as object 2", ["the", "book", "go", "have"]),
    ("novel_noun", "holdout noun object 2", ["the", "book", "go", "bird"]),
    ("grammatical", "trained noun object 3", ["the", "ball", "do", "man"]),
    ("category_violation", "verb as object 3", ["the", "ball", "do", "see"]),
    ("novel_noun", "holdout noun object 3", ["the", "ball", "do", "bird"]),
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


@dataclass(frozen=True)
class FrameWordStatus:
    """Everything that decides whether a frame word MEANS what its slot says.

    Four independent properties, because they fail independently and the
    combinations are what the shipped frames got wrong:

      * `registered` -- has a phon stimulus. Nearly everything is.
      * `trained` -- has a core lexicon assembly. **This is the predicate
        `collect_frame_samples` was reaching for when it tested `stim_map`.**
      * `holdout` -- declared untrained on purpose. An untrained holdout is
        CORRECT; it is what makes the novel arm novel. An untrained
        non-holdout is the defect.
      * `category` -- what the parser actually calls it. `chases` is untrained
        AND classifies NOUN; `small` is a holdout AND classifies VERB. Neither
        implies the other.
    """
    word: str
    registered: bool
    trained: bool
    holdout: bool
    category: str

    @property
    def usable(self) -> bool:
        """Registered, and either trained or a declared holdout."""
        return self.registered and (self.trained or self.holdout)

    def __str__(self) -> str:
        if not self.registered:
            return f"{self.word}[UNREGISTERED]"
        if self.holdout:
            return f"{self.word}[holdout:{self.category}]"
        if not self.trained:
            return f"{self.word}[UNTRAINED:{self.category}]"
        return f"{self.word}:{self.category}"


def frame_word_status(
    parser: "EmergentParser",
    word: str,
    *,
    holdout_words: Optional[Set[str]] = None,
) -> FrameWordStatus:
    """THE one predicate for "does this parser know this frame word".

    Classification is the expensive part, so callers auditing a whole frame set
    should go through `audit_frame_vocabulary`, which asks once per word.
    """
    holdouts = holdout_words or set()
    trained = any(word in lex for lex in parser.core_lexicons.values())
    registered = word in parser.stim_map
    category = ""
    if registered:
        category, _scores = parser.classify_word(word)
    return FrameWordStatus(
        word=word,
        registered=registered,
        trained=trained,
        holdout=word in holdouts,
        category=category,
    )


def audit_frame_vocabulary(
    parser: "EmergentParser",
    frames: List[CalibrationFrame],
    *,
    holdout_words: Optional[Set[str]] = None,
) -> Dict[str, FrameWordStatus]:
    """Status for every distinct word in *frames*, one classification each."""
    words = {w for _lbl, _desc, ws in frames for w in ws}
    return {
        w: frame_word_status(parser, w, holdout_words=holdout_words)
        for w in sorted(words)
    }


def unusable_frame_words(
    parser: "EmergentParser",
    frames: List[CalibrationFrame],
    *,
    holdout_words: Optional[Set[str]] = None,
) -> Dict[str, List[FrameWordStatus]]:
    """Frame description -> the words in it the parser does not know.

    A frame containing one of these does not implement the contrast its label
    claims, and NOTHING ELSE WILL SAY SO: the substrate is totalizing, so an
    untrained word still gets a stimulus, still gets a category, and still
    returns a p600. This is the check that has to be explicit.
    """
    status = audit_frame_vocabulary(
        parser, frames, holdout_words=holdout_words,
    )
    bad: Dict[str, List[FrameWordStatus]] = {}
    for _label, desc, words in frames:
        offenders = [status[w] for w in words if not status[w].usable]
        if offenders:
            bad[desc] = offenders
    return bad


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
    circuit: "FiberCircuit"
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
    protocol: ErpProtocol,
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
        protocol=protocol,
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
    probe_depth: str,
    cache: Dict[Tuple[str, ...], _WarmParseState],
    protocol: ErpProtocol,
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
        measure_lexical_surprise(parser, tuple(prefix), word,
                                 readiness=readiness, protocol=protocol)
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
            protocol=protocol,
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
    require_known_vocabulary: bool = False,
    protocol: Optional[ErpProtocol] = None,
) -> List[PositionErpSample]:
    """Measure ERP at critical word position for each calibration frame.

    `require_known_vocabulary` raises when a frame contains a word this parser
    neither trained nor declared a holdout. OFF by default because the shipped
    default frames DO NOT PASS IT -- see
    `research/notes/language/the_calibration_frames_are_untrained.md` -- and turning it
    on as a side effect of an unrelated change would break every ERP caller
    rather than answering the question. On for the frame sets that claim to be
    clean, and for the guard test that keeps them that way.
    """
    protocol = ErpProtocol.from_environment() if protocol is None else protocol
    from .runner import _check_engine_identity
    _check_engine_identity(parser, protocol)
    readiness = readiness or assess_erp_readiness(parser)
    baseline = baseline if baseline is not None else ErpBaseline()
    thresholds = thresholds or default_erp_thresholds()

    if require_known_vocabulary:
        bad = unusable_frame_words(
            parser, frames, holdout_words=holdout_words,
        )
        if bad:
            detail = "; ".join(
                f"{desc}: {', '.join(str(s) for s in offenders)}"
                for desc, offenders in sorted(bad.items())
            )
            raise ValueError(
                "calibration frames contain words this parser does not know, "
                "so those items do not implement the contrast their labels "
                f"claim -- {detail}",
            )

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
                probe_depth=probe_depth,
                cache=prefix_cache,
                protocol=protocol,
            )
        else:
            p = _probe_at_critical_position(
                parser, known, pos,
                readiness=readiness,
                baseline=baseline,
                probe_depth=probe_depth,
                protocol=protocol,
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
