"""Do the ERP effects survive an ISOLATED probe?

#100 / #32. Every ERP number in this repository is read through a probe that
runs under `brain.frozen()`. frozen() stops plasticity but NOT recruitment, so
the measurement grows the brain while reading it -- 214 raises when the fast
suite runs with NEURAL_ASSEMBLIES_STRICT_PROBES=1, and this package is one of
the densest clusters. #32 has named frozen() probe contamination as half the
P600 root cause since it was filed, on suspicion rather than measurement.

THE QUESTION. Under `read_only()` -- plasticity off AND recruitment off AND
winners restored -- does the grammatical/violation separation survive?

    survives   -> the effect is real and #32's probe half closes
    collapses  -> a headline result was partly measurement order

Both outcomes are worth having. Neither is assumed.

PAIRED BY CONSTRUCTION. Each seed trains ONE parser and forks it twice, so the
two arms differ only in the probe context. The flag is read per call rather
than cached at import, which is what makes this possible in one process.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation import (   # noqa: E402
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import ensemble, paired_delta  # noqa: E402

FLAG = "NEURAL_ASSEMBLIES_ISOLATED_PROBES"

#: READ THE AUC ROWS. `*_cohens_d` is retained only so this run can be compared
#: line-for-line against the pre-#80 one; it is computed on `*_excess`, which
#: clips the grammatical arm against its own median onto a 0.0 floor, so it
#: inflates when measurement noise FALLS. `*_auc` is a rank statistic on the raw
#: value and is invariant under that clipping. See
#: research/notes/erp_metric_is_clipped.md.
KEYS = ("p600_auc", "n400_auc", "p600_span",
        "p600_cohens_d", "n400_cohens_d", "p600_gap")


#: One calibration per (depth, seed, arm), reused across the metric rows. The
#: rows are different READINGS of one measurement, not repeated measurements --
#: recomputing them would be six trainings' worth of work for identical numbers
#: (verified by erp_rerun_pairing_check.py, which is what licenses the reuse).
_CACHE = {}


def _calibrate(depth, seed, isolated):
    hit = _CACHE.get((depth, seed, isolated))
    if hit is not None:
        return hit
    prev = os.environ.get(FLAG)
    os.environ[FLAG] = "1" if isolated else "0"
    try:
        parser = get_parser_cache().fork(depth, seed=seed)
        report = calibrate_erp_thresholds(parser)
    finally:
        if prev is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = prev
    _CACHE[(depth, seed, isolated)] = report
    return report


def metric(depth, isolated, key):
    def _run(seed):
        r = _calibrate(depth, seed, isolated)
        if key in r.separation:
            return float(r.separation[key])
        if key == "p600_gap":
            g = r.by_label.get("grammatical", {})
            c = r.by_label.get("category_violation", {})
            return float(c.get("p600_excess_median", 0.0)
                         - g.get("p600_excess_median", 0.0))
        raise KeyError(key)
    return _run


def main(depth="SENTENCES", seeds=(11, 12, 13, 14, 15)):
    print(f"depth={depth}  seeds={list(seeds)}")
    print("+/- is a t-based 95% CI (diagnostics.ensemble), not a standard error.")
    print("A positive gap / d, and an AUC above 0.5, all mean violations score")
    print("HIGHER, which is the direction the metric exists to show.")
    print("AUC granularity is 1/9 per seed (n=3 samples per arm under ERP_FAST).\n")
    print(f"{'metric':14s} {'frozen() [shipped]':26s} "
          f"{'read_only() [isolated]':26s} delta")
    for key in KEYS:
        a = ensemble(metric(depth, False, key), list(seeds), f"{key}/frozen")
        b = ensemble(metric(depth, True, key), list(seeds), f"{key}/isolated")
        d = paired_delta(b, a, f"{key}/delta")
        if a.values == b.values:
            # Not a null result. Two arms agreeing bit-for-bit means the flag
            # never reached the probe -- see diagnostics.compare_arms.
            verdict = "IDENTICAL -- suspect the flag never reached the probe"
        elif (d.mean - d.ci) * (d.mean + d.ci) > 0:
            verdict = "CHANGED"
        else:
            verdict = "no change"
        print(f"{key:14s} {a.mean:9.3f} +/- {a.ci:<12.3f} "
              f"{b.mean:9.3f} +/- {b.ci:<12.3f} "
              f"{d.mean:+7.3f} +/- {d.ci:.3f}  {verdict}")


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(depth, tuple(range(11, 11 + n)))
