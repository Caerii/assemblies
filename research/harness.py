"""One way to run an A/B over seeds: paired, order-controlled, pre-registered.

WHY THIS EXISTS. 141 experiment scripts, 88 of which re-implement the sys.path
preamble, 60 of which hand-roll a mean over seeds while only 17 use
`diagnostics.ensemble`. Every script re-derives its own protocol, so every script
can independently get pairing, ordering or provenance wrong -- and on 2026-08-05
one of mine got two of the three wrong at once:

  * BOTH ARMS IN ONE PROCESS, control arm first. The second arm was therefore
    only ever measured WARM, while the test suite measured it COLD. A cold
    process's first probes read p600 ~0.43 where warm reads ~0.998.
  * THE SUBSTRATE WAS A CACHED PARSER. `get_parser_cache().fork()` returns a
    clone of a disk-cached backbone. The A/B looked good on cached parsers and
    INVERTED on freshly-trained ones -- an A/B built on cached parsers is
    evidence about cached parsers only. That is the finding this module exists
    to make unrepeatable.

WHAT IT DOES NOT DO. It does not replace `diagnostics.ensemble` /
`paired_delta`; it calls them. Statistics live there, protocol lives here, and
adding a second statistics implementation would be the disease this refactor is
treating.

USAGE::

    from research.harness import study, Criteria

    result = study(
        arms={"obs": measure_obs, "exp": measure_exp},   # (seed) -> {metric: value}
        seeds=[11, 12, 13, 42],
        criteria={"p600_auc": Criteria(
            above=0.5, on_every_seed=True, must_vary=True)},
    )
    print(result)
    assert result.passed

PRE-REGISTRATION IS A FIELD, NOT A HABIT. Writing the adoption bar before the
run is what stopped a 0.75 being quietly accepted earlier today; leaving it to
discipline is how post-hoc rationalisation gets in. `Criteria` records the bar
in the result, so the verdict is reproducible from the artefact alone.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from neural_assemblies.diagnostics import Ensemble, ensemble, paired_delta

MeasureFn = Callable[[int], Mapping[str, float]]


@dataclass(frozen=True)
class Criteria:
    """A PRE-REGISTERED adoption bar for one metric.

    Every field defaults to "do not check", so a study states exactly what it
    committed to and nothing is implied by omission.

    Attributes:
        above: the candidate's mean must exceed this (e.g. 0.5 for an AUC null).
        on_every_seed: `above` must hold for EVERY seed, not just the mean. A
            mean hid a seed-42 inversion once; that is what this is for.
        must_vary: the candidate must not be CONSTANT across seeds. Zero
            variance is the signature of a structural artefact rather than a
            sentence-driven effect -- `afferent_energy` scored exactly 0.000 on
            four seeds and looked like a clean negative result.
        delta_excludes_zero: the paired delta's 95% CI must exclude 0.
        allow_decrease: whether a DROP versus the control still passes. Default
            True, deliberately: removing a confound should be expected to shrink
            an inflated effect, and a bar that treats every decrease as failure
            selects for confounded metrics.
    """

    above: Optional[float] = None
    on_every_seed: bool = False
    must_vary: bool = False
    delta_excludes_zero: bool = False
    allow_decrease: bool = True


@dataclass
class Provenance:
    """What substrate the numbers came off. Recorded because it changed a result.

    `cache_disk_hits > 0` means at least one arm ran on a parser deserialized
    from disk rather than trained in this process. Phase 0 of the canonical
    refactor found a dispatch that passed on cached parsers and FAILED on fresh
    ones, so a study that does not record this cannot be audited later.
    """

    source_fingerprint: str = ""
    cache_stats_before: Dict[str, object] = field(default_factory=dict)
    cache_stats_after: Dict[str, object] = field(default_factory=dict)
    backbone_cache_enabled: bool = True

    def _delta(self, key: str) -> int:
        def as_int(d) -> int:
            v = d.get(key, 0)
            return int(v) if isinstance(v, (int, float)) else 0
        return as_int(self.cache_stats_after) - as_int(self.cache_stats_before)

    @property
    def cache_disk_hits(self) -> int:
        """Parsers deserialized from disk during the study. See class docstring."""
        return self._delta("disk_hits")

    @property
    def trained_fresh(self) -> int:
        """Parsers trained in-process during the study."""
        return self._delta("misses")

    def __str__(self) -> str:  # pragma: no cover - display only
        return (f"substrate: disk_hits={self.cache_disk_hits} "
                f"trained_fresh={self.trained_fresh} "
                f"backbone_cache={'on' if self.backbone_cache_enabled else 'OFF'} "
                f"fingerprint={self.source_fingerprint[:12] or '?'}")


@dataclass
class MetricResult:
    metric: str
    control: Ensemble
    candidate: Ensemble
    delta: Ensemble
    criteria: Optional[Criteria] = None
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:  # pragma: no cover - display only
        verdict = "PASS" if self.passed else "FAIL"
        head = (f"{self.metric:12s} {self.control.mean:8.4f}+/-{self.control.ci:<8.4f}"
                f" -> {self.candidate.mean:8.4f}+/-{self.candidate.ci:<8.4f}"
                f"  delta {self.delta.mean:+.4f}+/-{self.delta.ci:.4f}"
                f"  [{verdict}]")
        return head + "".join(f"\n      - {f}" for f in self.failures)


@dataclass
class StudyResult:
    metrics: Dict[str, MetricResult]
    provenance: Provenance
    seeds: List[int]
    order: str

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics.values())

    def __str__(self) -> str:  # pragma: no cover - display only
        lines = [f"study over {len(self.seeds)} seeds {self.seeds} (order={self.order})",
                 f"  {self.provenance}"]
        lines += ["  " + str(m) for m in self.metrics.values()]
        lines.append(f"  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _provenance_snapshot() -> tuple:
    """Best-effort substrate fingerprint. Never fails a study."""
    fingerprint, stats = "", {}
    try:
        from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
            fingerprint_source_files, get_parser_cache,
        )
        files = fingerprint_source_files()
        fingerprint = f"{len(files)}files"
        stats = dict(get_parser_cache().stats())
    except Exception:                                        # noqa: BLE001
        pass
    return fingerprint, stats


def _check(seeds: Sequence[int], res: MetricResult, crit: Criteria) -> List[str]:
    fails = []
    cand = res.candidate
    if crit.above is not None:
        if cand.mean <= crit.above:
            fails.append(f"mean {cand.mean:.4f} <= {crit.above}")
        if crit.on_every_seed:
            # `Ensemble` carries values, not seeds, so pair against the study's
            # own list -- `ensemble()` preserves input order.
            bad = [s for s, v in zip(seeds, cand.values) if v <= crit.above]
            if bad:
                fails.append(
                    f"below {crit.above} on seeds {bad} -- a mean hides a "
                    f"single-seed inversion, which is how the last structural "
                    f"change passed review")
    if crit.must_vary and len(set(cand.values)) == 1:
        fails.append(
            f"CONSTANT across all seeds ({cand.values[0]:.4f}) -- zero variance "
            f"is the signature of a structural artefact, not an effect")
    if crit.delta_excludes_zero:
        lo, hi = res.delta.mean - res.delta.ci, res.delta.mean + res.delta.ci
        if lo <= 0 <= hi:
            fails.append(f"delta CI [{lo:.4f}, {hi:.4f}] includes 0")
    if not crit.allow_decrease and res.delta.mean < 0:
        fails.append(f"decreased by {abs(res.delta.mean):.4f}")
    return fails


def study(
    *,
    arms: Mapping[str, MeasureFn],
    seeds: Sequence[int],
    criteria: Optional[Mapping[str, Criteria]] = None,
    order: str = "counterbalance",
    control: Optional[str] = None,
) -> StudyResult:
    """Run a paired two-arm study over *seeds* and judge it against *criteria*.

    Args:
        arms: exactly two named callables, ``(seed) -> {metric: value}``. The
            first key is the CONTROL unless *control* names one.
        seeds: at least 3 -- fewer cannot support an interval. Include the seed
            that has previously caught a regression; a seed set chosen after
            seeing results is not a seed set.
        criteria: per-metric pre-registered bars. Metrics without an entry are
            REPORTED BUT NOT JUDGED, which keeps "measured" and "committed to"
            visibly separate.
        order: ``"counterbalance"`` alternates which arm runs first per seed,
            so warm-up and ordering effects fall on both arms equally instead of
            landing entirely on the second one. ``"as_given"`` preserves the
            old, confounded behaviour and must be justified.
        control: which arm is the baseline, if not the first key.

    ORDER CONTROL IS NOT COSMETIC. Running control-then-candidate for every seed
    is exactly how an A/B measured its candidate warm while the suite measured
    it cold, and the two disagreed by a full AUC. Counterbalancing does not make
    the process effect vanish -- it makes it symmetric, so the DELTA is still
    interpretable.
    """
    if len(arms) != 2:
        raise ValueError(f"study() compares exactly two arms, got {list(arms)}")
    seeds = list(seeds)
    if len(seeds) < 3:
        raise ValueError(
            f"{len(seeds)} seeds cannot support an interval; use at least 3 "
            f"(see [[report-distributions-not-point-estimates]])")
    if order not in ("counterbalance", "as_given"):
        raise ValueError(f"unknown order {order!r}")

    names = list(arms)
    ctrl_name = control or names[0]
    if ctrl_name not in arms:
        raise ValueError(f"control {ctrl_name!r} is not one of {names}")
    cand_name = [n for n in names if n != ctrl_name][0]

    fp_before, stats_before = _provenance_snapshot()

    # Collect per-seed readings for both arms, controlling the order.
    readings: Dict[str, Dict[int, Mapping[str, float]]] = {n: {} for n in names}
    for i, seed in enumerate(seeds):
        run_order = [ctrl_name, cand_name]
        if order == "counterbalance" and i % 2 == 1:
            run_order.reverse()
        for name in run_order:
            readings[name][seed] = dict(arms[name](seed))

    metric_names = sorted({
        k for per_seed in readings.values()
        for vals in per_seed.values() for k in vals
    })

    fp_after, stats_after = _provenance_snapshot()
    prov = Provenance(
        source_fingerprint=fp_after or fp_before,
        cache_stats_before=stats_before,
        cache_stats_after=stats_after,
        backbone_cache_enabled=os.environ.get(
            "ASSEMBLIES_BACKBONE_CACHE", "").strip() not in ("0", "off", "false"),
    )

    out: Dict[str, MetricResult] = {}
    for metric in metric_names:
        ctrl = ensemble(lambda s, m=metric: float(readings[ctrl_name][s].get(m, 0.0)),
                        seeds, f"{metric}/{ctrl_name}")
        cand = ensemble(lambda s, m=metric: float(readings[cand_name][s].get(m, 0.0)),
                        seeds, f"{metric}/{cand_name}")
        d = paired_delta(cand, ctrl, f"{metric}/delta")
        res = MetricResult(metric=metric, control=ctrl, candidate=cand, delta=d)
        crit = (criteria or {}).get(metric)
        if crit is not None:
            res.criteria = crit
            res.failures = _check(seeds, res, crit)
        out[metric] = res

    return StudyResult(metrics=out, provenance=prov, seeds=seeds, order=order)


def reseed_everything(seed: int) -> None:
    """Pin every global stream this repo is known to draw from.

    `Brain(seed=)` alone is NOT reproducible -- the global numpy/random streams
    leak between constructions and flip borderline results. Call this at the top
    of each trial, not once per process.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:                                        # noqa: BLE001
        pass
