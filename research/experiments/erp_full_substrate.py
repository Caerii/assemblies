"""Where does the P600 gap land when the WHOLE area competes, as the AC says it does?

THE QUESTION THIS ANSWERS, and it is not the one erp_growth_neutrality asked.
That experiment found recruitment is not neutral: the shipped `frozen()` probe
reads p600 gap 0.0022 while a recruitment-suppressed probe reads 0.0046, more
than double. I was about to call that "recruitment leaks". It is subtler, and
the AC says why.

In the calculus an area has a FIXED n neurons, ALL present, wired G(n,p) with
weights initialised to 1. Untrained neurons are not absent -- they are in the
k-WTA competition from the start, and losing to trained incumbents is the
mechanism, not an artifact. So:

    no_recruit   competition restricted to the MATERIALISED subset, which is
                 biased -- it is roughly the set of past winners
    frozen       that subset plus a SAMPLED handful of fresh neurons
    full         every neuron in the area, which is what the AC specifies

NEITHER of the first two is the calculus. `frozen` is closer to it than
`no_recruit`, because adding untrained competitors moves toward the full
population -- and `frozen` has the SMALLER gap. If the trend is monotone in how
much of the area competes, then our headline P600 separation is inflated by
incomplete materialisation, and the true-substrate value is smaller than
anything we have published.

    PREDICTION, recorded before running:  gap(full) < gap(frozen) < gap(no_recruit)
    If instead gap(full) ~ gap(frozen), materialisation is close enough and the
    no_recruit arm is the outlier -- the opposite conclusion, equally useful.

This is [[sampler-is-the-whole-discrepancy]] reaching the ERP result, and
[[sampler-error-changes-sign-across-arms]] is the standing warning that an A/B
is only safe when the manipulation leaves recruitment alone. Both arms of the
shipped probe comparison touch recruitment, so neither was safe.
"""
import os
import sys
from contextlib import contextmanager
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
from neural_assemblies.diagnostics import compare_arms, paired_delta   # noqa: E402

KEYS = ("materialized", "p600_auc", "p600_span", "p600_gap", "n400_auc")
ARMS = ("frozen", "no_recruit", "full")


def _materialized(brain):
    return sum(int(brain._engine_for(brain.areas[n]).materialized_count(n) or 0)
               for n in brain.areas)


@contextmanager
def _no_recruit(brain):
    engines = [e for e in brain._all_engines() if hasattr(e, "_no_recruitment")]
    saved = [e._no_recruitment for e in engines]
    for e in engines:
        e._no_recruitment = True
    try:
        with brain.frozen():
            yield brain
    finally:
        for e, s in zip(engines, saved):
            e._no_recruitment = s


def _materialize_all(brain) -> int:
    """Give every neuron in every area a compact index, i.e. build the AC's area.

    After this the k-WTA selects from the full population and recruitment has
    nothing left to do, so the `no_recruit` guard below is belt-and-braces
    rather than the intervention.
    """
    total = 0
    for name in sorted(brain.areas):
        engine = brain._engine_for(brain.areas[name])
        if not hasattr(engine, "materialize_area"):
            continue
        try:
            total += int(engine.materialize_area(name) or 0)
        except (RuntimeError, ValueError, MemoryError) as exc:
            print(f"  !! materialize_area({name}) failed: {exc}", flush=True)
    return total


_CACHE = {}


def _run(depth, seed, arm):
    hit = _CACHE.get((depth, seed, arm))
    if hit is not None:
        return hit
    prev = os.environ.get("NEURAL_ASSEMBLIES_ISOLATED_PROBES")
    os.environ["NEURAL_ASSEMBLIES_ISOLATED_PROBES"] = "0"
    try:
        parser = get_parser_cache().fork(depth, seed=seed)
        brain = parser.brain
        if arm == "frozen":
            report = calibrate_erp_thresholds(parser)
        elif arm == "no_recruit":
            with _no_recruit(brain):
                report = calibrate_erp_thresholds(parser)
        elif arm == "full":
            _materialize_all(brain)
            with _no_recruit(brain):
                report = calibrate_erp_thresholds(parser)
        else:
            raise KeyError(arm)
        out = (report, _materialized(brain))
    finally:
        if prev is None:
            os.environ.pop("NEURAL_ASSEMBLIES_ISOLATED_PROBES", None)
        else:
            os.environ["NEURAL_ASSEMBLIES_ISOLATED_PROBES"] = prev
    _CACHE[(depth, seed, arm)] = out
    return out


def metric(depth, arm, key):
    def _fn(seed):
        report, mat = _run(depth, seed, arm)
        if key == "materialized":
            return float(mat)
        if key in report.separation:
            return float(report.separation[key])
        if key == "p600_gap":
            g = report.by_label.get("grammatical", {})
            c = report.by_label.get("category_violation", {})
            return float(c.get("p600_excess_median", 0.0)
                         - g.get("p600_excess_median", 0.0))
        raise KeyError(key)
    return _fn


def main(depth="SENTENCES", seeds=(11, 12, 13, 14, 15)):
    print(f"depth={depth}  seeds={list(seeds)}")
    print("full = every neuron materialised, i.e. the substrate the AC specifies.")
    print("`materialized` is the direct check that the arms differ as claimed.\n")
    print(f"{'metric':13s} {'frozen [shipped]':22s} {'no_recruit':22s} "
          f"{'full [AC]':22s} full-frozen")
    for key in KEYS:
        arms = {a: metric(depth, a, key) for a in ARMS}
        out = compare_arms(arms, list(seeds), strict=False)
        a, b, c = out["frozen"], out["no_recruit"], out["full"]
        d = paired_delta(c, a, "full-frozen")
        verdict = ("CHANGED" if (d.mean - d.ci) * (d.mean + d.ci) > 0
                   else "no change")
        print(f"{key:13s} {a.mean:11.4f}+/-{a.ci:<9.4f} "
              f"{b.mean:11.4f}+/-{b.ci:<9.4f} {c.mean:11.4f}+/-{c.ci:<9.4f} "
              f"{d.mean:+10.4f}+/-{d.ci:.4f}  {verdict}")


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(depth, tuple(range(11, 11 + n)))
