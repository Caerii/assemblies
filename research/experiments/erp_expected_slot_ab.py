"""Does dispatching the probe area on the EXPECTED slot area-match the arms?

#108 / #23. The category-violation arm probes VP and its grammatical control
probes ROLE_PATIENT, because `structural_role_area` dispatches on the OBSERVED
word's category -- and a category violation IS a word whose observed category
differs from the expected one. The mismatch is therefore structural: it holds in
every frame set, at every seed, under every definition of energy. Measured in
`erp_which_area_per_arm.py`; the confound is written up in
`research/notes/p600_is_confounded_with_area_identity.md`.

`ERP_EXPECTED_SLOT=1` dispatches on the slot the PARSE PREDICTS instead: after a
verb that licenses an object, the next content word is expected in ROLE_PATIENT
whether it turns out to be `cat` or `finds`.

    obs  [SHIPPED]   structural_role_area(observed category)
    exp  [CANDIDATE] expected_role_area(parse state before the word)

WHAT WOULD MAKE THIS AN ADOPTION, stated before the run:

  1. The arms must AREA-MATCH. Verified separately by
     `erp_which_area_per_arm.py` under the flag -- if they do not, nothing
     below is interpretable and the numbers are noise about noise.
  2. p600_auc must stay ABOVE 0.5 ON EVERY SEED. Not "on average": the last
     structural fix (97438ec) held on seeds 11/12/13 and INVERTED seed 42 to
     0.444, and the mean hid it.
  3. The violation arm must become SENTENCE-SENSITIVE. Zero variance across
     seeds is the signature of a constant, which is the defect, not the fix
     (`afferent_energy` scored exactly 0.000 on four seeds).

A DROP IN AUC IS NOT AUTOMATICALLY A FAILURE HERE, and this is the part worth
reading carefully. The shipped AUC ~1.0 is CONFOUNDED: area identity alone
reproduces it with the condition held constant. Removing a confound should be
expected to shrink an inflated effect. What must not happen is inversion (2) or
constancy (3). A smaller, honestly area-matched effect is the better result.

SEED 42 IS IN THE SET DELIBERATELY. It is the only seed that has ever caught a
structural change here.
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
ARMS = ("obs", "exp")
FLAG = "ERP_EXPECTED_SLOT"

_CACHE = {}


def _run(depth, seed, arm):
    hit = _CACHE.get((depth, seed, arm))
    if hit is not None:
        return hit
    prev = os.environ.get(FLAG)
    os.environ[FLAG] = "1" if arm == "exp" else "0"
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
          f"the last structural change)")
    print("AUC null is 0.5. BELOW it means INVERTED, which is the failure mode")
    print("that actually occurred last time. A SMALLER but honest effect is a")
    print("pass; an inverted or constant one is not.\n")
    print(f"{'metric':10s} {'obs [shipped]':24s} {'exp [candidate]':24s} delta")
    for key in KEYS:
        a = ensemble(metric(depth, "obs", key), list(seeds), f"{key}/obs")
        b = ensemble(metric(depth, "exp", key), list(seeds), f"{key}/exp")
        d = paired_delta(b, a, f"{key}/delta")
        if a.values == b.values:
            # For n400_auc this is the CORRECT result, not a plumbing failure:
            # N400 comes from `measure_lexical_surprise`, which the flag does
            # not touch. Only p600_* should move.
            verdict = (
                "IDENTICAL -- expected for n400 (flag does not touch it); "
                "for p600 it would mean the flag never reached the probe"
            )
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
        spread = max(vals) - min(vals)
        note = f"   INVERTED on {bad}" if bad else "   none below chance"
        if spread == 0.0:
            note += "   AND CONSTANT ACROSS SEEDS -- structural, not sentence-driven"
        print(f"  {arm:5s} " + "  ".join(f"s{s}={v:.3f}" for s, v in zip(seeds, vals))
              + note)


if __name__ == "__main__":
    depth = sys.argv[1] if len(sys.argv) > 1 else "SENTENCES"
    # 11/12/13/42 stay FIRST so a wider run still reports the four seeds the
    # earlier result was read on; 42 must never be dropped.
    seeds = (
        tuple(int(s) for s in sys.argv[2].split(","))
        if len(sys.argv) > 2
        else (11, 12, 13, 42)
    )
    main(depth, seeds)
