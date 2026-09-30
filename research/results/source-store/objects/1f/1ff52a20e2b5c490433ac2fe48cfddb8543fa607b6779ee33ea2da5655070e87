"""Empirical ERP threshold calibration on the emergent parser.

Tunes excess margins from composed-ERP-style frame probes in
``evaluation.erp.frames`` (midpoint between grammatical and violation quantiles).
"""

from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from .gates import (
    ErpBaseline,
    ErpReadiness,
    ErpThresholds,
    assess_erp_readiness,
    calibrate_erp_baseline,
    classify_erp_violation,
    default_erp_thresholds,
)
from .frames import (
    DEFAULT_CALIBRATION_FRAMES,
    AREA_MATCHED_CALIBRATION_FRAMES,
    CalibrationFrame,
    PositionErpSample,
    collect_frame_samples,
)
from .runner import run_incremental_erp_probes
from .protocol import ErpProtocol

# Lazy at call time, not import time: `diagnostics` pulls in the wider package
# and this module is imported during parser construction.
def _separation(hi, lo, label):
    from neural_assemblies.diagnostics import separation
    return separation(hi, lo, label=label)

if TYPE_CHECKING:
    from ...parser import EmergentParser

# Re-export for backward compatibility
__all__ = [
    "DEFAULT_CALIBRATION_FRAMES",
    "AREA_MATCHED_CALIBRATION_FRAMES",
    "ErpCalibrationReport",
    "PositionErpSample",
    "calibrate_erp_thresholds",
    "collect_position_samples",
    "ensure_parser_erp_calibration",
    "tune_thresholds_from_samples",
]


@dataclass(frozen=True)
class ErpQuantities:
    """THE THREE THINGS CALLED "p600", each named for what it actually is.

    They are genuinely different quantities and the shared word has cost real
    time: two of them were compared as if interchangeable, and the apparent
    contradiction was chased for hours before the definitions were re-read.

        deficit_raw          `sample.p600` -- 1 - normalized pre-k-WTA energy
                             into the role area. Bounded [0,1], and SATURATED:
                             it occupies ~0.7% of that range in every condition,
                             so absolute values carry almost nothing.

        excess_over_baseline `sample.p600_excess` -- max(0, deficit_raw -
                             baseline.p600_median). CLIPPED at its own null, so
                             roughly half the mass sits exactly at 0 and it is
                             not an effect size. Cohen's d computed on it is
                             meaningless (see `_cohens_d`'s caller).

        auc_of_raw           AUC ranking violation against grammatical ON
                             `deficit_raw` -- NOT on the excess. A rank
                             statistic, so it is invariant under every monotone
                             transform and survives redefinition of the
                             underlying quantity. THIS IS THE ONE TO READ.

    `auc_of_raw` is the field most often mis-stated: the key in `separation` is
    spelled `p600_auc`, which says nothing about which quantity was ranked, and
    it ranks the RAW deficit (`calibration.py` passes `catv_p600_raw`).

    `span` is reported alongside because a perfect ordering across 0.7% of the
    scale is both a perfect ordering and a saturated metric at once, and either
    fact alone is misleading.
    """

    deficit_raw_grammatical: List[float]
    deficit_raw_violation: List[float]
    excess_over_baseline_grammatical: List[float]
    excess_over_baseline_violation: List[float]
    auc_of_raw: float
    span_of_raw: float

    @property
    def n(self) -> int:
        """Pairs behind `auc_of_raw`. AUC granularity is 1/(n_hi*n_lo).

        With 3 items per condition the statistic moves in steps of 1/9, so
        differences under ~0.11 are not resolvable at all -- worth knowing
        before reading a change as an effect.
        """
        return len(self.deficit_raw_grammatical) * len(self.deficit_raw_violation)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (f"auc_of_raw={self.auc_of_raw:.3f} (n_pairs={self.n}, "
                f"granularity={1.0 / self.n if self.n else float('nan'):.3f}) "
                f"span_of_raw={self.span_of_raw:.4f}")


@dataclass
class ErpCalibrationReport:
    """Outcome of empirical threshold tuning."""
    readiness: ErpReadiness
    baseline: ErpBaseline
    thresholds: ErpThresholds
    samples: List[PositionErpSample] = field(default_factory=list)
    by_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    separation: Dict[str, float] = field(default_factory=dict)
    tuned: bool = False
    fast_requested: bool = False
    engine_name: str = "unknown"

    def p600_quantities(self) -> ErpQuantities:
        """All three "p600" quantities together, so none is picked by accident.

        THE SANCTIONED READER. `separation["p600_auc"]` and `sample.p600` and
        `sample.p600_excess` remain because goldens and callers depend on those
        names, but reaching for one of them in isolation is how they get
        confused. This returns them side by side, labelled.
        """
        gram = [float(s.p600) for s in self.samples if s.label == "grammatical"]
        catv = [float(s.p600) for s in self.samples
                if s.label == "category_violation"]
        gram_ex = [float(s.p600_excess) for s in self.samples
                   if s.label == "grammatical"]
        catv_ex = [float(s.p600_excess) for s in self.samples
                   if s.label == "category_violation"]
        if gram and catv:
            sep = _separation(catv, gram, label="p600")
            auc, span = sep.auc, sep.span
        else:
            auc = span = float("nan")
        return ErpQuantities(
            deficit_raw_grammatical=gram,
            deficit_raw_violation=catv,
            excess_over_baseline_grammatical=gram_ex,
            excess_over_baseline_violation=catv_ex,
            auc_of_raw=auc,
            span_of_raw=span,
        )

    def summary(self) -> str:
        lines = [
            "ERP calibration report",
            f"  engine: {self.engine_name}",
            f"  readiness: n400={self.readiness.n400_ready} "
            f"p600={self.readiness.p600_ready} "
            f"(lex={self.readiness.prediction_lexicon_size}, "
            f"sents={self.readiness.sentences_seen})",
            f"  baseline: n400={self.baseline.n400_median:.3f} "
            f"p600={self.baseline.p600_median:.3f} "
            f"stab={self.baseline.stability_median:.3f} "
            f"({self.baseline.source}, n={self.baseline.sample_size})",
            f"  thresholds: n400_margin={self.thresholds.n400_excess_margin:.3f} "
            f"p600_margin={self.thresholds.p600_excess_margin:.3f} "
            f"novel_n400={self.thresholds.novel_n400_excess:.3f}",
        ]
        for label, stats in sorted(self.by_label.items()):
            lines.append(
                f"  {label}: n400_ex={stats.get('n400_excess_median', 0):.3f} "
                f"p600_ex={stats.get('p600_excess_median', 0):.3f} "
                f"n={int(stats.get('n', 0))}",
            )
        if self.separation:
            lines.append(
                # AUC first, and the span beside it: a perfect ordering across
                # a sliver of the range is both facts at once. Cohen's d is
                # shown last and parenthesised because it is computed on the
                # clipped excess and is not an effect size (see `separation`).
                f"  separation: n400_auc={self.separation.get('n400_auc', float('nan')):.3f}"
                f"(span {self.separation.get('n400_span', float('nan')):.4f}) "
                f"p600_auc={self.separation.get('p600_auc', float('nan')):.3f}"
                f"(span {self.separation.get('p600_span', float('nan')):.4f})"
                f"  [clipped d: n400={self.separation.get('n400_cohens_d', 0):.2f} "
                f"p600={self.separation.get('p600_cohens_d', 0):.2f}]",
            )
        return "\n".join(lines)


def _cohens_d(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    if len(a) < 2 or len(b) < 2:
        spread = max(
            statistics.pstdev(a) if len(a) > 1 else 0.0,
            statistics.pstdev(b) if len(b) > 1 else 0.0,
            abs(mb - ma) * 0.25,
            1e-3,
        )
        return (mb - ma) / spread
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    pooled = ((va + vb) / 2.0) ** 0.5
    if pooled < 1e-3:
        pooled = max(abs(mb - ma) * 0.25, 1e-3)
    return (mb - ma) / pooled


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[idx]


def _label_stats(samples: List[PositionErpSample], label: str) -> Dict[str, float]:
    subset = [s for s in samples if s.label == label]
    if not subset:
        return {"n": 0}
    n400_ex = [s.n400_excess for s in subset]
    p600_ex = [s.p600_excess for s in subset]
    return {
        "n": float(len(subset)),
        "n400_excess_median": statistics.median(n400_ex),
        "p600_excess_median": statistics.median(p600_ex),
        "n400_excess_p75": _quantile(n400_ex, 0.75),
        "p600_excess_p75": _quantile(p600_ex, 0.75),
        "stability_median": statistics.median([s.phrase_stability for s in subset]),
    }


def tune_thresholds_from_samples(
    samples: List[PositionErpSample],
    *,
    fallback: Optional[ErpThresholds] = None,
) -> ErpThresholds:
    """Midpoint between grammatical p75 and category-violation p25 excess."""
    fb = fallback or default_erp_thresholds()
    gram = [s for s in samples if s.label == "grammatical"]
    catv = [s for s in samples if s.label == "category_violation"]
    novel = [s for s in samples if s.label == "novel_noun"]

    if len(gram) < 2 or len(catv) < 2:
        return fb

    g_n400 = _quantile([s.n400_excess for s in gram], 0.75)
    c_n400 = _quantile([s.n400_excess for s in catv], 0.25)
    g_p600 = _quantile([s.p600_excess for s in gram], 0.75)
    c_p600 = _quantile([s.p600_excess for s in catv], 0.25)

    n400_margin = max(fb.n400_excess_margin * 0.5, (g_n400 + c_n400) / 2.0)
    p600_margin = max(fb.p600_excess_margin * 0.5, (g_p600 + c_p600) / 2.0)

    novel_n400 = fb.novel_n400_excess
    if novel:
        novel_n400 = max(
            fb.novel_n400_excess * 0.5,
            _quantile([s.n400_excess for s in novel], 0.25),
        )

    return ErpThresholds(
        n400_excess_margin=round(n400_margin, 4),
        p600_excess_margin=round(p600_margin, 4),
        novel_n400_excess=round(novel_n400, 4),
        phrase_stability_ratio=fb.phrase_stability_ratio,
        source="empirical",
    )


# Backward-compatible alias
collect_position_samples = collect_frame_samples


def _ensure_minimal_prediction_bridges(parser: "EmergentParser") -> None:
    """Train lightweight next-token bridges if N400 readout is not yet valid."""
    from ..sweep import sweep_mode_enabled

    readiness = assess_erp_readiness(parser)
    if readiness.n400_ready:
        return
    if sweep_mode_enabled() and readiness.prediction_lexicon_size > 0:
        return
    try:
        # ``erp`` is nested below ``evaluation``; three dots are required to
        # reach the sibling ``emergent.curriculum`` package.  The former
        # two-dot import always raised ImportError and was silently swallowed,
        # disabling the bridge bootstrap whenever readiness was low.
        from ...curriculum.data import create_training_sentences
        sents = create_training_sentences()[:30]
        if hasattr(parser, "train_next_token") and sents:
            parser.train_next_token(sents, rebuild_lexicon=True)
    except (RuntimeError, ValueError, ImportError) as error:
        warnings.warn(
            "ERP prediction bootstrap could not train its minimal bridges; "
            f"calibration continues without that intervention ({error!r})",
            RuntimeWarning,
            stacklevel=2,
        )


def _relabel_samples(
    samples: List[PositionErpSample],
    *,
    readiness: ErpReadiness,
    baseline: ErpBaseline,
    thresholds: ErpThresholds,
) -> List[PositionErpSample]:
    """Re-classify frame samples under tuned thresholds (no re-parse)."""
    out: List[PositionErpSample] = []
    for s in samples:
        n400_ex = baseline.n400_excess(s.n400)
        p600_ex = baseline.p600_excess(s.p600)
        violation = classify_erp_violation(
            s.n400,
            s.p600,
            readiness=readiness,
            baseline=baseline,
            phrase_stability=s.phrase_stability,
            thresholds=thresholds,
        )
        out.append(
            PositionErpSample(
                label=s.label,
                sentence=s.sentence,
                position=s.position,
                word=s.word,
                category=s.category,
                n400=s.n400,
                p600=s.p600,
                phrase_stability=s.phrase_stability,
                n400_excess=n400_ex,
                p600_excess=p600_ex,
                violation=violation,
            ),
        )
    return out


def calibrate_erp_thresholds(
    parser: "EmergentParser",
    *,
    frames: Optional[List[CalibrationFrame]] = None,
    grammatical_sentences: Optional[List[List[str]]] = None,
    critical_position: Optional[int] = None,
    ensure_prediction: bool = True,
    probe_depth: str = "calibration",
    fast: bool = False,
    protocol: Optional[ErpProtocol] = None,
) -> ErpCalibrationReport:
    """Calibrate from one observation pass, then relabel those observations.

    ``fast`` remains as a compatibility argument, but no longer changes the
    measurement protocol. The former ``fast=False`` path parsed every frame a
    second time after tuning. Parsing can recruit neurons, so the two modes
    measured different model states and could reverse the reported AUC. A
    threshold changes only classification; it cannot change an already
    observed N400/P600 quantity.
    """
    if critical_position is not None and (
        type(critical_position) is not int or critical_position < 0
    ):
        raise ValueError("critical_position must be a nonnegative integer or None")
    if type(fast) is not bool:
        raise ValueError("fast must be a bool")
    if ensure_prediction:
        _ensure_minimal_prediction_bridges(parser)

    protocol = ErpProtocol.from_environment() if protocol is None else protocol
    from .runner import _check_engine_identity
    _check_engine_identity(parser, protocol)
    readiness = assess_erp_readiness(parser)
    from ..sweep import sweep_mode_enabled
    from .probe_util import critical_probe_measure_fn

    sweep = sweep_mode_enabled()
    frames = list(frames or DEFAULT_CALIBRATION_FRAMES)
    if sweep:
        from .frames import SWEEP_CALIBRATION_FRAMES
        frames = list(SWEEP_CALIBRATION_FRAMES)

    gram_sents = grammatical_sentences or [
        list(words) for lbl, _d, words in frames if lbl == "grammatical"
    ]
    measure_fn = critical_probe_measure_fn(probe_depth, protocol=protocol) if sweep else (
        lambda p, w, **_kw: run_incremental_erp_probes(
            p, w, apply_calibration=False, readiness=readiness,
            probe_depth=probe_depth, protocol=protocol,
        )
    )
    baseline = calibrate_erp_baseline(
        parser,
        gram_sents,
        max_sentences=4 if sweep else 8,
        measure_fn=measure_fn,
        critical_position_only=sweep,
    )

    fb = default_erp_thresholds()
    from ..generalization import default_holdout_set

    holdout = default_holdout_set()
    warm = sweep
    raw_samples = collect_frame_samples(
        parser,
        frames,
        critical_position=critical_position,
        readiness=readiness,
        baseline=baseline,
        thresholds=fb,
        holdout_words=holdout,
        probe_depth=probe_depth,
        warm_start=warm,
        protocol=protocol,
    )
    thresholds = tune_thresholds_from_samples(
        raw_samples, fallback=fb,
    )

    # Threshold tuning changes labels, not the neural quantities already
    # observed. A second collection pass would advance a mutable parser and
    # turn calibration mode into an undocumented training schedule.
    tuned_samples = _relabel_samples(
        raw_samples,
        readiness=readiness,
        baseline=baseline,
        thresholds=thresholds,
    )

    by_label = {
        lbl: _label_stats(tuned_samples, lbl)
        for lbl in ("grammatical", "category_violation", "novel_noun")
    }
    gram_n400 = [s.n400_excess for s in tuned_samples if s.label == "grammatical"]
    catv_n400 = [s.n400_excess for s in tuned_samples if s.label == "category_violation"]
    gram_p600 = [s.p600_excess for s in tuned_samples if s.label == "grammatical"]
    catv_p600 = [s.p600_excess for s in tuned_samples if s.label == "category_violation"]
    # RAW, not excess. The excess is clipped against the grammatical median, so
    # the null arm sits on a 0.0 floor -- see the AUC entries below.
    gram_n400_raw = [s.n400 for s in tuned_samples if s.label == "grammatical"]
    catv_n400_raw = [s.n400 for s in tuned_samples if s.label == "category_violation"]
    gram_p600_raw = [s.p600 for s in tuned_samples if s.label == "grammatical"]
    catv_p600_raw = [s.p600 for s in tuned_samples if s.label == "category_violation"]

    # COHEN'S D IS RETAINED FOR CONTINUITY AND SHOULD NOT BE QUOTED AS AN
    # EFFECT SIZE. It is computed on `*_excess`, which is
    # `max(0, v - baseline.p600_median)` where the baseline IS the grammatical
    # median -- so the null arm is clipped against itself onto an exact 0.0
    # floor with almost no variance. Any reduction in measurement noise then
    # inflates d without the effect growing: making the parse read-only moved
    # d from 1.63 to 3.97 while the absolute gap moved 0.0030 to 0.0047.
    #
    # Measured across four encodings of an IDENTICAL ordering, d spanned
    # 2.241 to 24.754 while AUC was 1.000 throughout.
    #
    # `*_auc` is the quantity to read. It is a rank statistic on the RAW value,
    # invariant under every monotone transform -- so it survives clipping,
    # rescaling, and redefinition of the underlying quantity, which is exactly
    # what broke every absolute threshold here when P600 changed from unbounded
    # churn (0.12 vs 5.24) to a bounded energy deficit (0.989 vs 0.995).
    # `*_span` reports how much of the range the raw values occupy, because a
    # perfect ordering across 0.7% of the scale is both of those things at once.
    separation = {
        "n400_cohens_d": _cohens_d(gram_n400, catv_n400),
        "p600_cohens_d": _cohens_d(gram_p600, catv_p600),
    }
    for name, hi, lo in (("n400", catv_n400_raw, gram_n400_raw),
                         ("p600", catv_p600_raw, gram_p600_raw)):
        if hi and lo:
            sep = _separation(hi, lo, label=name)
            separation[f"{name}_auc"] = sep.auc
            separation[f"{name}_span"] = sep.span

    report = ErpCalibrationReport(
        readiness=readiness,
        baseline=baseline,
        thresholds=thresholds,
        samples=tuned_samples,
        by_label=by_label,
        separation=separation,
        tuned=readiness.p600_ready,
        fast_requested=fast,
        engine_name=str(getattr(parser, "engine_name", "unknown")),
    )
    # Cache the complete observation, not only thresholds.  Reconstructing a
    # report from thresholds made the second caller silently lose samples,
    # separation statistics, and by-label counts.
    parser._erp_thresholds = thresholds
    parser._erp_baseline = baseline
    parser._erp_report = report
    return report


def ensure_parser_erp_calibration(
    parser: "EmergentParser",
    *,
    force: bool = False,
    fast: Optional[bool] = None,
) -> ErpCalibrationReport:
    """Calibrate once per parser unless already cached."""
    if fast is None:
        from ..sweep import erp_fast_calibration_enabled
        fast = erp_fast_calibration_enabled()
    if not force and hasattr(parser, "_erp_thresholds"):
        cached = getattr(parser, "_erp_report", None)
        if isinstance(cached, ErpCalibrationReport):
            return cached
        th = parser._erp_thresholds
        if isinstance(th, ErpThresholds) and th.source == "empirical":
            return ErpCalibrationReport(
                readiness=assess_erp_readiness(parser),
                baseline=getattr(parser, "_erp_baseline", ErpBaseline()),
                thresholds=th,
                engine_name=str(getattr(parser, "engine_name", "unknown")),
                tuned=True,
            )
    return calibrate_erp_thresholds(parser, fast=fast)
