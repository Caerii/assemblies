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
METRICS = ("p600_cohens_d", "n400_cohens_d")


def _calibrate(depth, seed, isolated):
    prev = os.environ.get(FLAG)
    os.environ[FLAG] = "1" if isolated else "0"
    try:
        parser = get_parser_cache().fork(depth, seed=seed)
        return calibrate_erp_thresholds(parser)
    finally:
        if prev is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = prev


def metric(depth, isolated, key):
    def _run(seed):
        r = _calibrate(depth, seed, isolated)
        if key in METRICS:
            return float(r.separation.get(key, 0.0))
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
    print("A positive p600 gap / Cohen's d means violations score HIGHER,")
    print("which is the direction the metric exists to show.\n")
    print(f"{'metric':14s} {'frozen() [shipped]':26s} {'read_only() [isolated]':26s} delta")
    for key in ("p600_cohens_d", "n400_cohens_d", "p600_gap"):
        a = ensemble(metric(depth, False, key), list(seeds), f"{key}/frozen")
        b = ensemble(metric(depth, True, key), list(seeds), f"{key}/isolated")
        d = paired_delta(b, a, f"{key}/delta")
        verdict = "CHANGED" if (d.mean - d.ci) * (d.mean + d.ci) > 0 else "no change"
        print(f"{key:14s} {a.mean:9.3f} +/- {a.ci:<12.3f} "
              f"{b.mean:9.3f} +/- {b.ci:<12.3f} "
              f"{d.mean:+7.3f} +/- {d.ci:.3f}  {verdict}")


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(depth, tuple(range(11, 11 + n)))
