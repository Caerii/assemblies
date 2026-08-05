"""POS hypothesis generation and forced-category trials.

ERROR SIGNALS AS SUPERVISION.  The ERP measures elsewhere in this package are
built as SCIENCE -- N400 as prediction error, P600 as settling cost, both
intended to be compared against human data.  This module reuses them as a
LEARNING signal, and the move is worth naming because it is the interesting
idea here: if a large P600 means the parser is struggling to integrate a word,
that same number tells the parser something is wrong with how it has
categorised the word.

The procedure is a forced-choice trial.  When a word produces a wobbly parse,
alternative categories are proposed (``_FALLBACK_POS_ALTERNATIVES`` encodes
which confusions are plausible -- NOUN/ADJ, VERB/NOUN, and so on, i.e. the
pairs that genuinely share distributional contexts).  Each alternative is then
tried: the word is forced into that category's core area, the sentence is
re-run, and the combined N400/P600 cost is measured.  The category that makes
the sentence cheapest to process wins.

This is hypothesis testing against a self-generated signal -- no external
label is consulted at any point.  The claim it supports is that a learner
could correct a misassigned category using only its own processing
difficulty, which is a specific and falsifiable version of "learning from
surprise".

Note the weighting: costs are combined via ``N400_WEIGHT`` and ``P600_WEIGHT``
from the ERP layer, so the two components are not equally decisive.  Any
conclusion about which hypothesis wins depends on that ratio.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ...core.areas import CATEGORY_TO_CORE
from ...evaluation.erp import (
    N400_WEIGHT,
    P600_WEIGHT,
    measure_fresh_stimulus_integration,
    measure_lexical_surprise,
    measure_live_integration,
)

if TYPE_CHECKING:
    from ...parser import EmergentParser

_FALLBACK_POS_ALTERNATIVES: Dict[str, Tuple[str, ...]] = {
    "NOUN": ("ADJ", "VERB"),
    "VERB": ("NOUN",),
    "ADJ": ("NOUN", "ADV"),
    "ADV": ("ADJ", "VERB"),
    "DET": ("ADJ", "PRON"),
    "PRON": ("NOUN", "DET"),
    "PREP": ("DET", "ADV"),
}


def _fresh_integration_cache(parser: "EmergentParser") -> Dict[Tuple[str, str], float]:
    """Per-parser memo for ``measure_fresh_stimulus_integration`` (mining pass)."""
    cache = getattr(parser, "_fresh_integration_cache", None)
    if cache is None:
        cache = {}
        parser._fresh_integration_cache = cache
    return cache


def generate_pos_hypotheses(
    parser: "EmergentParser",
    word: str,
    assigned_category: str,
    *,
    failure_signature: str,
) -> Tuple[Tuple[str, str], ...]:
    """Alternate POS from grounding, distributional stats, and merge stability."""
    from ..pos_inference import grounding_evidence_scores

    if failure_signature in ("lexical_surprise", "lexical_novelty"):
        return ()

    ranked: List[Tuple[str, float]] = []
    gscores = grounding_evidence_scores(parser.word_grounding.get(word))
    for cat, score in gscores.items():
        if cat != assigned_category and cat != "UNKNOWN":
            ranked.append((cat, score * 1.2))

    dist_n = parser.dist_stats.word_count.get(word, 0)
    if dist_n > 0:
        _dist_cat, dist_scores = parser.classify_distributional(word)
        for cat, score in dist_scores.items():
            if cat != assigned_category:
                ranked.append((cat, float(score)))

    for cat in _FALLBACK_POS_ALTERNATIVES.get(assigned_category, ()):
        if cat != assigned_category:
            ranked.append((cat, 0.15))

    if failure_signature == "structural_wobble" and assigned_category in ("NOUN", "PRON"):
        ranked.append(("VERB", 0.25))
    if failure_signature == "lexical_and_structural" and assigned_category == "NOUN":
        ranked.append(("ADJ", 0.3))

    merge_scores: Dict[str, float] = {}
    integration_cache = _fresh_integration_cache(parser)
    for cat, _ in ranked:
        if cat in merge_scores:
            continue
        key = (word, cat)
        if key not in integration_cache:
            try:
                inst, _ = measure_fresh_stimulus_integration(parser, word, cat)
            except (RuntimeError, IndexError, ValueError):
                inst = 1.0
            integration_cache[key] = inst
        merge_scores[cat] = 1.0 - integration_cache[key]

    fused: Dict[str, float] = {}
    for cat, weight in ranked:
        fused[cat] = fused.get(cat, 0.0) + weight
    for cat, stab in merge_scores.items():
        fused[cat] = fused.get(cat, 0.0) + stab * 0.5

    ordered = sorted(fused.items(), key=lambda x: -x[1])
    hyps: List[Tuple[str, str]] = []
    for cat, _ in ordered:
        if cat != assigned_category:
            hyps.append((word, cat))
        if len(hyps) >= 3:
            break
    return tuple(hyps)


def parse_prefix(
    parser: "EmergentParser",
    words: List[str],
    end: int,
) -> Tuple[dict, object, bool, int, Optional[str]]:
    """Parse words[:end] with full FiberCircuit; return circuit + counters."""
    parser._reset_context_state()
    circuit = parser._get_incremental_circuit(reset=True)
    categories: Dict[str, str] = {}
    verb_seen = False
    noun_count = 0
    subject_core: Optional[str] = None

    with parser.brain.frozen():
        for i in range(end):
            word = words[i]
            cat, verb_seen, noun_count = parser._advance_incremental_word(
                word, circuit, verb_seen, noun_count,
            )
            categories[word] = cat
            if cat in ("NOUN", "PRON") and not verb_seen:
                subject_core = CATEGORY_TO_CORE.get(cat)

    return {"categories": categories}, circuit, verb_seen, noun_count, subject_core


def trial_category_in_sentence(
    parser: "EmergentParser",
    words: List[str],
    position: int,
    forced_category: str,
) -> Optional[Tuple[float, float, float, float]]:
    """Re-parse with forced POS at *position*; return combined metrics or None."""
    prefix = words[:position]
    word = words[position]

    try:
        _, circuit, verb_seen, noun_count, subject_core = parse_prefix(
            parser, words, position,
        )
    except (RuntimeError, IndexError, ValueError):
        return None

    with parser.brain.frozen():
        try:
            # `measure_lexical_surprise` returns `Measured`; `.or_else` with
            # the branch's own legacy value keeps this arithmetic byte-
            # identical to before the definedness migration.
            _n400_m = (measure_lexical_surprise(parser, tuple(prefix), word)
                       if prefix else None)
            n400 = 0.0 if _n400_m is None else _n400_m.or_else(
                float((_n400_m.detail or {}).get("legacy", 0.0)))
            _cat, verb_seen, noun_count = parser._advance_incremental_word(
                word,
                circuit,
                verb_seen,
                noun_count,
                forced_category=forced_category,
            )
            p600, _, stability = measure_live_integration(
                parser,
                word,
                forced_category,
                verb_seen=verb_seen,
                subject_core=subject_core,
            )
        except (RuntimeError, IndexError, ValueError):
            return None

    combined = N400_WEIGHT * n400 + P600_WEIGHT * p600
    return combined, p600, n400, stability
