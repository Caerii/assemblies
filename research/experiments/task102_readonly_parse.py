"""Does the ERP separation survive a parse that does not grow the parser?

#102 / #32 / #80. Parsing a sentence RECRUITS: one 5-word sentence of known
words on an already-trained parser adds 663 neurons, and repeated parses of the
SAME sentence return different ERP values (first-word P600 0.434 -> 0.988 ->
0.9868 -> 0.9865). Evaluation is not idempotent, so every reported number
depends on read order.

Running the parse under `read_only()` fixes that -- measured, +0 growth and
bit-identical probes from the second read. THE QUESTION HERE is whether the
grammatical/violation separation survives it, because the outcomes are not
symmetric:

    survives   -> adopt. Evaluation becomes reproducible AND the P600 metric
                  stops sitting at the ceiling (0.99 -> 0.43), which is #32's
                  remaining half.
    collapses  -> the separation DEPENDS on the parser growing while being
                  measured, i.e. a headline result is partly an artifact of the
                  protocol. Worth knowing before building further on it.

WHY THIS NEEDS SEEDS. The probe-isolation delta (#100) read "no change" at 5
seeds and excluded zero at 10. And the calibration here has only ~3 samples per
condition under EMERGENT_ERP_FAST, so a single seed's Cohen's d is extremely
noisy -- one arm returned exactly 4.000. Point estimates are not evidence here.

`compare_arms(strict=True)` guards the other failure: if the two arms return
IDENTICAL values on every seed they did not run different computations, which
is a dead pathway rather than a null result.
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
from neural_assemblies.diagnostics import (                             # noqa: E402
    compare_arms, ensemble, paired_delta,
)


def _materialized(brain):
    return sum(int(brain._engine_for(brain.areas[n]).materialized_count(n) or 0)
               for n in brain.areas)


def _run(depth, seed, readonly_parse):
    parser = get_parser_cache().fork(depth, seed=seed)
    brain = parser.brain
    before = _materialized(brain)
    if readonly_parse:
        with brain.read_only():
            report = calibrate_erp_thresholds(parser)
    else:
        report = calibrate_erp_thresholds(parser)
    return report, _materialized(brain) - before


def metric(depth, readonly_parse, key):
    def _fn(seed):
        report, grew = _run(depth, seed, readonly_parse)
        if key == "grew":
            return float(grew)
        if key.endswith("_cohens_d"):
            return float(report.separation.get(key, 0.0))
        if key == "p600_gap":
            g = report.by_label.get("grammatical", {})
            c = report.by_label.get("category_violation", {})
            return float(c.get("p600_excess_median", 0.0)
                         - g.get("p600_excess_median", 0.0))
        raise KeyError(key)
    return _fn


def main(depth="SENTENCES", seeds=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20)):
    print(f"depth={depth}  seeds={list(seeds)}")
    print("+/- is a t-based 95% CI. Ground truth for `grew` is 0: a read that")
    print("grows the parser cannot be repeated and cannot be ordered freely.\n")
    print(f"{'metric':16s} {'parse grows [HEAD]':26s} {'parse read_only':26s} delta")
    for key in ("grew", "p600_cohens_d", "n400_cohens_d", "p600_gap"):
        arms = {
            "grows": metric(depth, False, key),
            "readonly": metric(depth, True, key),
        }
        # strict=False and checked BY HAND below: two arms agreeing on a
        # SATURATED metric is a real possibility here (#28 says N400 is
        # saturated), and that is information, not a crash.
        out = compare_arms(arms, list(seeds), strict=False)
        a, b = out["grows"], out["readonly"]
        d = paired_delta(b, a, f"{key}/delta")
        if a.values == b.values:
            verdict = "IDENTICAL -- arms did not differ, suspect a dead path"
        elif (d.mean - d.ci) * (d.mean + d.ci) > 0:
            verdict = "CHANGED"
        else:
            verdict = "no change"
        print(f"{key:16s} {a.mean:10.4f} +/- {a.ci:<11.4f} "
              f"{b.mean:10.4f} +/- {b.ci:<11.4f} "
              f"{d.mean:+9.4f} +/- {d.ci:.4f}  {verdict}")


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(depth, tuple(range(11, 11 + n)))
