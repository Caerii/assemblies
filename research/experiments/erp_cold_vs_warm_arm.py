"""Is my own A/B harness order-confounded, and do raw and excess AUC disagree?

Two measurements of "the same" seed-11 contrast disagree IN SIGN:

    erp_expected_slot_ab.py, exp arm   report.separation["p600_auc"] = 1.000
    test_erp_metric_range.py           AUC over raw float(s.p600)    = 0.000
        gram [0.9932, 0.9928, 0.9945]   catv [0.9927, 0.9925, 0.9924]

TWO CANDIDATE EXPLANATIONS, and this script separates them.

(1) RAW vs EXCESS. `separation["p600_auc"]` ranks p600_EXCESS; the test ranks
    RAW p600. But `p600_excess(v) = max(0, v - p600_median)` is MONOTONE, so
    clipping can only create ties (AUC -> 0.5), never invert an ordering.
    On this account 0.000 vs 1.000 is IMPOSSIBLE, so it predicts the two AUCs
    agree in sign here and the discrepancy lies elsewhere.

(2) ORDER / WARM-UP -- my harness's fault. `erp_expected_slot_ab.py` runs the
    obs arm FIRST and the exp arm SECOND in the SAME process. The first two
    probes of a COLD process read p600 ~0.43 with phrase_stability 1.0000 (the
    empty-phrase-areas fallback) where the same frame warm reads ~0.998
    (measured, research/notes/language/p600_is_confounded_with_area_identity.md). So the
    exp arm was only ever measured WARM, and the test measures it COLD. On this
    account the A/B never isolated the dispatch at all.

(2) also predicts the obs arm is unaffected, because obs runs first in both.

This script runs ONE arm per process invocation, so neither arm is warmed by
the other, and prints BOTH AUCs side by side.

    python erp_cold_vs_warm_arm.py obs
    python erp_cold_vs_warm_arm.py exp

If cold-exp raw AUC is ~0.000 while the in-process A/B said 1.000, explanation
(2) holds and the 10-seed adoption evidence is void.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

ARM = sys.argv[1] if len(sys.argv) > 1 else "exp"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
# Set BEFORE importing anything that reads it, and before any parse runs.
os.environ["ERP_EXPECTED_SLOT"] = "1" if ARM == "exp" else "0"

from neural_assemblies.assembly_calculus.emergent.evaluation import (   # noqa: E402
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                   # noqa: E402


def main():
    parser = get_parser_cache().fork("SENTENCES", seed=SEED)
    report = calibrate_erp_thresholds(parser)

    def vals(label, attr):
        return [
            float(getattr(s, attr))
            for s in report.samples
            if s.label == label
        ]

    print(f"arm={ARM}  seed={SEED}  ERP_EXPECTED_SLOT={os.environ['ERP_EXPECTED_SLOT']}")
    print("ONE ARM PER PROCESS -- neither arm is warmed by the other.\n")

    for attr in ("p600", "p600_excess"):
        gram = vals("grammatical", attr)
        catv = vals("category_violation", attr)
        if not (gram and catv):
            print(f"  {attr:12s} MISSING samples")
            continue
        auc = separation(catv, gram, attr).auc
        flag = "  <-- INVERTED" if auc < 0.5 else ""
        print(f"  {attr:12s} auc={auc:.3f}{flag}")
        print(f"               gram {[round(v, 4) for v in gram]}")
        print(f"               catv {[round(v, 4) for v in catv]}")

    print(f"\n  report.separation['p600_auc'] = "
          f"{report.separation.get('p600_auc', float('nan')):.3f}")
    print(f"  baseline p600_median          = {report.baseline.p600_median:.4f}")


if __name__ == "__main__":
    main()
