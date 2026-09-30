"""Dropping undefined stability readings instead of substituting 0.0 for them.

THE CHANGE. `erp/adapters.py` aggregates phrase-stability readings as

    stabilities = [r.or_else(0.0) for r in readings]
    mean        = sum(stabilities)/len(stabilities) if stabilities else 1.0
    instability = 1.0 - mean

An UNDEFINED reading -- almost always `VP` having no self-fiber -- enters that
mean as a hard 0.0, dragging stability down and p600 up. `defined_values` drops
them instead, which is the correct aggregation and has been sitting one line
away, unused, behind a `_ = defined_values` marker since the Measured migration.

WHY IT WAS NOT JUST SWITCHED. Measured (`erp_or_else_impact.py`): the two
aggregations are NOT a level shift.

  * one undefined among several -> substituting drags the mean down slightly;
  * ALL readings undefined -> substituting gives mean 0.0, instability 1.0, the
    MAXIMUM, while dropping gives an empty list, which takes the `else 1.0`
    branch: instability 0.0, the MINIMUM. Same probe, opposite ends.

On the DEFAULT frames, 12 of 18 novel probes have no defined reading at all, so
that arm's entire stability term is the substituted constant: p600 0.9928
shipped against 0.6252 honest, versus -0.0029 on the other two arms.

TWO PRE-REGISTERED CLAIMS, and the second is the one that can actually fail.

1. `p600_auc_of_raw` is a GRAMMATICAL-vs-VIOLATION statistic, and both of those
   arms have a defined reading, so this should barely move (-0.0029 on each,
   applied to both, is close to no rank change at all). The bar is therefore
   that it must not DEGRADE: above chance, on every seed, not constant.

2. `novel_stability_median` MUST INCREASE, DECISIVELY. This is the falsifiable
   part. If undefined readings are entering the mean as 0.0, removing them can
   only raise the mean -- so `allow_decrease=False` and
   `delta_excludes_zero=True`. If this does NOT rise, my account of the
   mechanism is wrong and the -0.3676 attributed to substitution is coming from
   somewhere else.

THE ORDER MATTERS, and it is why this runs after the frames work rather than
before. Repairing the items removes the undefined readings at source: with the
novel word in OBJECT position the expected role area has an active assembly, so
there is nothing left to substitute. Fix the items first and this change costs
-0.0029; do it the other way round and every published magnitude moves to
correct a term that should not have existed.

RUN COLD -- `ASSEMBLIES_BACKBONE_CACHE=0`.
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
from research.harness import Criteria, study                            # noqa: E402

FLAG = "ERP_DROP_UNDEFINED_STABILITY"
SEEDS = [11, 12, 13, 42, 7, 19, 23, 31, 37, 101]


def _arm(enabled: bool):
    def measure(seed: int):
        prev = os.environ.get(FLAG)
        os.environ[FLAG] = "1" if enabled else "0"
        try:
            parser = get_parser_cache().fork("SENTENCES", seed=seed)
            report = calibrate_erp_thresholds(parser)
        finally:
            if prev is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = prev
        q = report.p600_quantities()
        novel = report.by_label.get("novel_noun", {})
        return {
            "p600_auc_of_raw": q.auc_of_raw,
            "p600_span_of_raw": q.span_of_raw,
            # The quantity actually being changed. Reporting only the AUC would
            # hide the whole effect, because the AUC does not look at this arm.
            "novel_stability_median": float(
                novel.get("stability_median", 0.0)),
            "novel_p600_excess_median": float(
                novel.get("p600_excess_median", 0.0)),
        }
    return measure


def main():
    result = study(
        arms={"substitute_zero": _arm(False), "drop_undefined": _arm(True)},
        seeds=SEEDS,
        control="substitute_zero",
        criteria={
            "p600_auc_of_raw": Criteria(
                above=0.5, on_every_seed=True, must_vary=True,
                allow_decrease=True,
            ),
            "novel_stability_median": Criteria(
                allow_decrease=False, delta_excludes_zero=True,
            ),
        },
    )
    print(result)
    print()
    print("`novel_stability_median` is the falsifiable claim. Undefined")
    print("readings enter the mean as 0.0, so removing them can only raise it.")
    print("If it does not rise, the mechanism account is wrong and the -0.3676")
    print("attributed to substitution is coming from somewhere else.")
    print()
    prov = result.provenance
    if prov.cache_disk_hits > 0:
        print(f"CAVEAT: {prov.cache_disk_hits} parser(s) from the DISK CACHE.")
    elif prov.trained_fresh > 0:
        print(f"{prov.trained_fresh} parser(s) trained fresh in-process.")


if __name__ == "__main__":
    main()
