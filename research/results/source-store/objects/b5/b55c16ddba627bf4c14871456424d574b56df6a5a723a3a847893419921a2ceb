"""Does AFFERENT drive separate grammatical from violation, where self-recurrence cannot?

#108 / #104. The P600's violation arm probes VP, and `VP -> VP` is shape (0,0)
with ZERO synapses -- the parser never declares that fiber. So
`_self_recurrent_energy(brain, VP)` returns exactly 0.000000 and `1 - energy`
is a constant 1.0 whatever the sentence was. The grammatical arm probes
ROLE_PATIENT, which HAS a self-fiber (960x960, 24173 synapses) only because
`_pregrow_role_pathways` explicitly opens one. The two arms were never
measuring comparable quantities.

VP is not unbuilt. It is built the other way round::

    VERB_CORE -> VP   (2543, 480)   50032 synapses
    SUBJ      -> VP    (296, 480)    4360
    OBJ       -> VP    (120, 656)    2736

So `afferent_energy` -- drive INTO an area from its live sources -- is defined
for both arms and needs no new structure. THIS SCRIPT ASKS WHETHER IT
SEPARATES. It is not an adoption.

WHY THIS IS RUN AND NOT ASSUMED. The last structural change here
(`_pregrow_phrase_pathways`, 97438ec) built VP's self-fiber, looked right on
every targeted test, and INVERTED seed 42 to p600_auc 0.444 -- below the null.
It was reverted. A metric change that fixes one arm and flips a contrast is not
a fix.

SEED 42 IS IN THE SET, DELIBERATELY. Targeted runs on seed 11 were green while
42 inverted, so any seed list that omits it cannot detect the failure mode that
actually occurred.

    self  [SHIPPED]   phrase_stability -> _self_recurrent_energy
    aff   [CANDIDATE] ERP_AFFERENT_ENERGY=1 -> afferent_energy

Read the AUC rows. Cohen's d is computed on a clipped excess and is not an
effect size here (research/notes/language/erp_metric_is_clipped.md).
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
from neural_assemblies.diagnostics import ensemble, paired_delta        # noqa: E402

KEYS = ("p600_auc", "p600_span", "n400_auc")
ARMS = ("self", "aff")
FLAG = "ERP_AFFERENT_ENERGY"

_CACHE = {}


def _run(depth, seed, arm):
    hit = _CACHE.get((depth, seed, arm))
    if hit is not None:
        return hit
    prev = os.environ.get(FLAG)
    os.environ[FLAG] = "1" if arm == "aff" else "0"
    try:
        parser = get_parser_cache().fork(depth, seed=seed)
        report = calibrate_erp_thresholds(parser)
    finally:
        if prev is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = prev
    _CACHE[(depth, seed, arm)] = report
    return report


def metric(depth, arm, key):
    def _fn(seed):
        return float(_run(depth, seed, arm).separation.get(key, 0.0))
    return _fn


def main(depth="SENTENCES", seeds=(11, 12, 13, 42)):
    print(f"depth={depth}  seeds={list(seeds)}   (42 is the seed that caught "
          f"the last attempt)")
    print("AUC null is 0.5. BELOW it means the contrast is INVERTED, which is")
    print("worse than no effect and is exactly how the previous fix failed.\n")
    print(f"{'metric':10s} {'self [shipped]':24s} {'aff [candidate]':24s} delta")
    for key in KEYS:
        a = ensemble(metric(depth, "self", key), list(seeds), f"{key}/self")
        b = ensemble(metric(depth, "aff", key), list(seeds), f"{key}/aff")
        d = paired_delta(b, a, f"{key}/delta")
        if a.values == b.values:
            verdict = "IDENTICAL -- the flag never reached the probe"
        elif (d.mean - d.ci) * (d.mean + d.ci) > 0:
            verdict = "CHANGED"
        else:
            verdict = "no change"
        print(f"{key:10s} {a.mean:9.4f}+/-{a.ci:<11.4f} "
              f"{b.mean:9.4f}+/-{b.ci:<11.4f} {d.mean:+8.4f}+/-{d.ci:.4f}  {verdict}")

    print("\nPER-SEED p600_auc (a mean hides an inversion on one seed):")
    for arm in ARMS:
        vals = [metric(depth, arm, "p600_auc")(s) for s in seeds]
        bad = [s for s, v in zip(seeds, vals) if v < 0.5]
        print(f"  {arm:5s} " + "  ".join(f"s{s}={v:.3f}" for s, v in zip(seeds, vals))
              + (f"   INVERTED on {bad}" if bad else "   none below chance"))


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    main(depth, (11, 12, 13, 42))
