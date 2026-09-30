"""Is RECRUITMENT semantically neutral, as the assembly calculus says it must be?

THE THEORY. In the AC (Papadimitriou et al. 2020; Mitropolsky et al. 2021 Sec 3)
each area has a FIXED n neurons wired by a random G(n,p) graph, weights
initialised to 1. Areas do not grow. Our engine materialises columns lazily,
which is a PERFORMANCE DEVICE standing in for that fixed substrate -- and since
#81 made synapse init content-addressed on (row, col), materialising a neuron
must yield the same weights whenever it happens.

If that is true, then whether a probe is allowed to recruit CANNOT change what
it measures, and `frozen()` (plasticity off, recruitment ON) must agree with the
recruitment-suppressed arm to the last bit.

IT DOES NOT. task100_erp_rerun.log reads p600 AUC 0.972 under frozen() and 0.917
under read_only(). But that comparison confounds TWO interventions, because
`read_only()` = frozen() + no recruitment + WINNERS RESTORED ON EXIT. So the
difference could be recruitment (a substrate defect, fixable) or winner
restoration (a protocol choice, correct by design). The shipped comparison
cannot tell them apart, and one of those answers is a bug and the other is not.

THREE ARMS, which is the smallest design that separates them::

    A  frozen         plasticity off, recruit YES, winners kept as the probe left them
    B  no_recruit     plasticity off, recruit NO,  winners kept as the probe left them
    C  read_only      plasticity off, recruit NO,  winners RESTORED   [shipped]

    A vs B  isolates RECRUITMENT      -> a difference here is a substrate defect
    B vs C  isolates WINNER RESTORE   -> a difference here is protocol, expected

PREDICTION, recorded before running. If materialisation is a faithful lazy
substrate, A == B exactly and only B vs C moves. Any A/B difference is a door-7
[[content-addressed-synapse-init]] leak: something in the recruitment path is
still order-dependent.

`grew` is reported per arm as the direct check that the arms are what they claim
-- arm A must grow, B and C must not. An arm that does not do what its name says
is the failure mode `compare_arms` exists to catch.
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

KEYS = ("grew", "p600_auc", "n400_auc", "p600_span", "p600_gap")


def _materialized(brain):
    return sum(int(brain._engine_for(brain.areas[n]).materialized_count(n) or 0)
               for n in brain.areas)


@contextmanager
def _no_recruit(brain):
    """frozen() + recruitment suppressed, WITHOUT restoring winners.

    Deliberately reaches into `engine._no_recruitment` rather than calling
    `read_only()`, because the whole point is to run read_only()'s recruitment
    half without its winner-restore half. Kept here in the experiment and NOT
    promoted to a Brain method: a third probe context on the Brain would be a
    fourth way to say "read", which is the pattern this repo keeps paying for.
    """
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


_CACHE = {}
_ARMS = ("frozen", "no_recruit", "read_only")


def _run(depth, seed, arm):
    hit = _CACHE.get((depth, seed, arm))
    if hit is not None:
        return hit
    # The ERP probes read this flag per call, so pinning it to 0 forces the
    # legacy frozen() path inside the probe and lets THIS harness own the
    # context. Without it arm A would silently be arm C.
    prev = os.environ.get("NEURAL_ASSEMBLIES_ISOLATED_PROBES")
    os.environ["NEURAL_ASSEMBLIES_ISOLATED_PROBES"] = "0"
    try:
        parser = get_parser_cache().fork(depth, seed=seed)
        brain = parser.brain
        before = _materialized(brain)
        if arm == "frozen":
            report = calibrate_erp_thresholds(parser)
        elif arm == "no_recruit":
            with _no_recruit(brain):
                report = calibrate_erp_thresholds(parser)
        elif arm == "read_only":
            with brain.read_only():
                report = calibrate_erp_thresholds(parser)
        else:
            raise KeyError(arm)
        out = (report, _materialized(brain) - before)
    finally:
        if prev is None:
            os.environ.pop("NEURAL_ASSEMBLIES_ISOLATED_PROBES", None)
        else:
            os.environ["NEURAL_ASSEMBLIES_ISOLATED_PROBES"] = prev
    _CACHE[(depth, seed, arm)] = out
    return out


def metric(depth, arm, key):
    def _fn(seed):
        report, grew = _run(depth, seed, arm)
        if key == "grew":
            return float(grew)
        if key in report.separation:
            return float(report.separation[key])
        if key == "p600_gap":
            g = report.by_label.get("grammatical", {})
            c = report.by_label.get("category_violation", {})
            return float(c.get("p600_excess_median", 0.0)
                         - g.get("p600_excess_median", 0.0))
        raise KeyError(key)
    return _fn


def main(depth="SENTENCES", seeds=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20)):
    print(f"depth={depth}  seeds={list(seeds)}")
    print("A=frozen (recruit YES)  B=no_recruit  C=read_only (B + winners restored)")
    print("A vs B isolates RECRUITMENT -- identical is the PREDICTION, not a worry.")
    print("B vs C isolates WINNER RESTORE -- a difference there is protocol.\n")
    print(f"{'metric':11s} {'A frozen':22s} {'B no_recruit':22s} {'C read_only':22s} "
          f"{'A-B (recruit)':22s} B-C (restore)")
    for key in KEYS:
        arms = {a: metric(depth, a, key) for a in _ARMS}
        # strict=False: arms A and B being IDENTICAL is the hypothesis under
        # test, so raising on it would refuse to report the expected result.
        out = compare_arms(arms, list(seeds), strict=False)
        a, b, c = out["frozen"], out["no_recruit"], out["read_only"]
        ab = paired_delta(a, b, "recruit")
        bc = paired_delta(b, c, "restore")
        tag = "A==B" if a.values == b.values else "A!=B RECRUITMENT LEAKS"
        print(f"{key:11s} {a.mean:9.4f}+/-{a.ci:<10.4f} "
              f"{b.mean:9.4f}+/-{b.ci:<10.4f} {c.mean:9.4f}+/-{c.ci:<10.4f} "
              f"{ab.mean:+9.4f}+/-{ab.ci:<10.4f} {bc.mean:+9.4f}+/-{bc.ci:.4f}  {tag}")


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(depth, tuple(range(11, 11 + n)))
