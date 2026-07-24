"""ERP gates — readiness, baselines, violation detection (evaluation.erp.gates).

Calibrated against ``research/results/primitives/RESULTS_composed_erp.md``:

  Grammatical null:  N400 ≈ 0.09,  anchored P600 ≈ 0.02–0.12
  Category violation: N400 ≈ 0.35,  anchored P600 ≈ 5.0 (cumulative)

Wobble = excess over parser-specific baseline, gated on pathway readiness.
P600 requires consolidated role/VP pathways; N400 requires prediction bridges.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import EmergentParser

# Reference nulls from composed ERP (n=10000, k=100, multi-seed).
REFERENCE_N400_GRAMMATICAL = 0.088
REFERENCE_N400_CATEGORY_VIOLATION = 0.351
REFERENCE_P600_GRAMMATICAL = 0.12
REFERENCE_P600_CATEGORY_VIOLATION = 5.24

# Empirically tuned on emergent parser (SENTENCES depth, n=3000 k=30, seed=42).
N400_EXCESS_MARGIN = 0.065
P600_EXCESS_MARGIN = 0.152
NOVEL_N400_EXCESS = 0.125
PHRASE_STABILITY_RATIO = 0.45

MIN_PREDICTION_LEXICON = 5
MIN_DISTRIBUTIONAL_SENTENCES = 15


@dataclass(frozen=True)
class ErpThresholds:
    """Tunable ERP violation gates (empirical or reference defaults)."""
    n400_excess_margin: float = N400_EXCESS_MARGIN
    p600_excess_margin: float = P600_EXCESS_MARGIN
    novel_n400_excess: float = NOVEL_N400_EXCESS
    phrase_stability_ratio: float = PHRASE_STABILITY_RATIO
    source: str = "reference"


def default_erp_thresholds() -> ErpThresholds:
    return ErpThresholds()


def parser_erp_thresholds(parser: "EmergentParser") -> ErpThresholds:
    cached = getattr(parser, "_erp_thresholds", None)
    if isinstance(cached, ErpThresholds):
        return cached
    return default_erp_thresholds()


def parser_erp_baseline(parser: "EmergentParser") -> ErpBaseline:
    cached = getattr(parser, "_erp_baseline", None)
    if isinstance(cached, ErpBaseline):
        return cached
    return ErpBaseline()


@dataclass(frozen=True)
class ErpReadiness:
    """Which ERP readouts are meaningful for this parser state."""
    n400_ready: bool
    p600_ready: bool
    prediction_lexicon_size: int = 0
    sentences_seen: int = 0
    role_pathways_linked: bool = False

    @property
    def any_ready(self) -> bool:
        return self.n400_ready or self.p600_ready


@dataclass
class ErpBaseline:
    """Parser-specific null distribution (median over recent grammatical parses)."""
    n400_median: float = REFERENCE_N400_GRAMMATICAL
    p600_median: float = REFERENCE_P600_GRAMMATICAL
    stability_median: float = 0.55
    sample_size: int = 0
    source: str = "reference"

    def n400_excess(self, value: float) -> float:
        return max(0.0, value - self.n400_median)

    def p600_excess(self, value: float) -> float:
        return max(0.0, value - self.p600_median)


@dataclass
class ErpViolation:
    """Classification of a surprise event relative to AC double-dissociation."""
    wobbly: bool
    failure_signature: str = ""
    n400_excess: float = 0.0
    p600_excess: float = 0.0
    primary: str = ""  # "lexical" | "structural" | "none"


def assess_erp_readiness(parser: "EmergentParser") -> ErpReadiness:
    """Gate probes on trained pathways (AC: untrained = high P600 is not an error)."""
    pred_lex = getattr(parser, "prediction_lexicon", {}) or {}
    sentences_seen = getattr(parser.dist_stats, "sentences_seen", 0)
    role_linked = bool(getattr(parser, "_role_paths_bootstrapped", False))

    n400_ready = len(pred_lex) >= MIN_PREDICTION_LEXICON
    p600_ready = role_linked and sentences_seen >= MIN_DISTRIBUTIONAL_SENTENCES

    return ErpReadiness(
        n400_ready=n400_ready,
        p600_ready=p600_ready,
        prediction_lexicon_size=len(pred_lex),
        sentences_seen=sentences_seen,
        role_pathways_linked=role_linked,
    )


def classify_erp_violation(
    n400: float,
    p600: float,
    *,
    readiness: ErpReadiness,
    baseline: ErpBaseline,
    phrase_stability: float = 1.0,
    thresholds: Optional[ErpThresholds] = None,
) -> ErpViolation:
    """Double-dissociation wobble: structural vs lexical surprise.

    Category violation (POS wrong): high N400 + high P600 (RESULTS table).
    Novel object: high N400, P600 ~ grammatical (exposure gap, not POS flip).
    Structural-only: high P600 with moderate N400 when binding pathway wrong.
    """
    if not readiness.any_ready:
        return ErpViolation(wobbly=False, primary="none")

    th = thresholds or default_erp_thresholds()
    n400_ex = baseline.n400_excess(n400)
    p600_ex = baseline.p600_excess(p600)

    structural = (
        readiness.p600_ready
        and p600_ex >= th.p600_excess_margin
    )
    lexical = readiness.n400_ready and n400_ex >= th.n400_excess_margin

    if structural and lexical:
        return ErpViolation(
            wobbly=True,
            failure_signature="lexical_and_structural",
            n400_excess=n400_ex,
            p600_excess=p600_ex,
            primary="structural",
        )

    if structural:
        return ErpViolation(
            wobbly=True,
            failure_signature="structural_wobble",
            n400_excess=n400_ex,
            p600_excess=p600_ex,
            primary="structural",
        )

    if lexical and not readiness.p600_ready:
        if n400_ex >= th.novel_n400_excess:
            return ErpViolation(
                wobbly=True,
                failure_signature="lexical_surprise",
                n400_excess=n400_ex,
                p600_excess=p600_ex,
                primary="lexical",
            )

    if (
        readiness.n400_ready
        and readiness.p600_ready
        and n400_ex >= th.novel_n400_excess
        and p600_ex < th.p600_excess_margin * 0.5
    ):
        return ErpViolation(
            wobbly=True,
            failure_signature="lexical_novelty",
            n400_excess=n400_ex,
            p600_excess=p600_ex,
            primary="lexical",
        )

    if (
        readiness.p600_ready
        and phrase_stability < baseline.stability_median * th.phrase_stability_ratio
        and p600_ex >= th.p600_excess_margin * 0.6
    ):
        return ErpViolation(
            wobbly=True,
            failure_signature="phrase_instability",
            n400_excess=n400_ex,
            p600_excess=p600_ex,
            primary="structural",
        )

    return ErpViolation(wobbly=False, primary="none")


def calibrate_erp_baseline(
    parser: "EmergentParser",
    sample_sentences: List[List[str]],
    *,
    max_sentences: int = 8,
    measure_fn=None,
    critical_position_only: bool = False,
) -> ErpBaseline:
    """Estimate null ERP levels from recent exposure (parser-specific baseline)."""
    if measure_fn is None or not sample_sentences:
        return ErpBaseline(source="reference")

    n400_vals: List[float] = []
    p600_vals: List[float] = []
    stab_vals: List[float] = []

    for sent in sample_sentences[-max_sentences:]:
        known = [w for w in sent if w in parser.stim_map]
        if len(known) < 2:
            continue
        _, probes = measure_fn(parser, known)
        if critical_position_only and probes:
            pos = len(known) - 1
            probes = [p for p in probes if p.position == pos] or [probes[-1]]
        for p in probes:
            n400_vals.append(p.n400)
            p600_vals.append(p.p600)
            stab_vals.append(p.phrase_stability)

    if not n400_vals:
        return ErpBaseline(source="reference")

    return ErpBaseline(
        n400_median=statistics.median(n400_vals),
        p600_median=statistics.median(p600_vals) if p600_vals else REFERENCE_P600_GRAMMATICAL,
        stability_median=statistics.median(stab_vals) if stab_vals else 0.55,
        sample_size=len(n400_vals),
        source="calibrated",
    )


def areas_with_active_assembly(brain, area_names: List[str]) -> List[str]:
    """Phrase areas that currently hold an assembly (skip empty/unmerged slots)."""
    active: List[str] = []
    for area in area_names:
        if area not in brain.areas:
            continue
        winners = brain.areas[area].winners
        if winners is not None and len(winners) > 0:
            active.append(area)
    return active
