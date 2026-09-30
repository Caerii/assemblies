"""Is an ERP calibration REPEATABLE within one process, on the same seed?

Prerequisite for the #100/#102 re-run, not a result. Those two experiments read
a paired delta: arm A over seeds 11..15, then arm B over the same seeds. That
pairing is only real if `fork(depth, seed=s)` + `calibrate_erp_thresholds` is a
function of `s` alone. If anything in the path draws from a GLOBAL RNG, arm B's
seed 11 starts from a different global state than arm A's did, and the "delta"
absorbs that difference.

Three calibrations of ONE seed, back to back. Identical => paired. Different =>
the deltas need a reseed before each trial and the old numbers were confounded
by measurement ORDER on top of everything else.
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

KEYS = ("p600_auc", "n400_auc", "p600_cohens_d", "n400_cohens_d")


def once(depth, seed):
    parser = get_parser_cache().fork(depth, seed=seed)
    r = calibrate_erp_thresholds(parser)
    return tuple(round(float(r.separation.get(k, float("nan"))), 6) for k in KEYS)


def main(depth="SENTENCES", seed=11, reps=3):
    rows = [once(depth, seed) for _ in range(reps)]
    print(f"depth={depth} seed={seed}  keys={KEYS}")
    for i, r in enumerate(rows):
        print(f"  rep {i}: {r}")
    same = all(r == rows[0] for r in rows)
    print("\nREPEATABLE" if same else "\nNOT REPEATABLE -- pairing is confounded")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main(
        sys.argv[1] if len(sys.argv) > 1 else "SENTENCES",
        int(sys.argv[2]) if len(sys.argv) > 2 else 11,
    ))
