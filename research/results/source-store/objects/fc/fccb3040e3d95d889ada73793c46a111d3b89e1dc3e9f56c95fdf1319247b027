"""Which denominator makes the P600 energy INDEPENDENT of materialisation?

#104. `_self_recurrent_energy` returns `pre_kwta_total / area.w`, and `area.w`
is the MATERIALISED count -- an artifact of lazy instantiation with no
counterpart in the calculus, where an area has a fixed n. Measured across arms
that vary how much of the area competes, w grows 7.8x while the P600 gap falls
12.5x (erp_full_substrate.log). So the metric's SCALE tracks training history,
which is why a threshold set once ended up 11.9x out of range.

THE FIX IS A DIVISOR, AND THIS PICKS IT BY MEASUREMENT. Four candidates, on the
SAME projection, over three materialisation levels:

    w         pre_kwta_total / area.w            [SHIPPED] materialised count
    cand      pre_kwta_total / pre_kwta_count    mean per candidate ACTUALLY summed
    k         pre_kwta_total / k                 drive arrives from k firing neurons
    null      pre_kwta_total / (count * k * p)   the drive an UNTRAINED area would
                                                 deliver, since initial weights are
                                                 1.0. Ratio is ~1.0 untrained and
                                                 >1 trained -- pool-invariant BY
                                                 CONSTRUCTION and derived from model
                                                 parameters only.

PREDICTION, recorded before running: `w` varies most, `null` least. `cand` is
the honest per-candidate mean but should STILL drift, because a larger pool is
mostly untrained neurons and dilutes the average.

THE ACCEPTANCE CRITERION IS THE SPREAD OF THE RATIO ACROSS ARMS, not the value.
A divisor that merely rescales everything has fixed nothing; the question is
whether the number stops moving when materialisation changes. Reported as
max/min across the three arms -- 1.00 is perfect invariance.

=== RESULT: THE HYPOTHESIS IS REFUTED. NO DIVISOR FIXES IT. ===

`w` 40.58, `k` 9.16, and then `cand` 7.78, `null` 7.78, `topk/mean` 7.77,
`max/mean` 7.76. Every scale-free candidate lands on the same number, and two
of them (`topk/mean`, `max/mean`) are ratios of quantities drawn from the SAME
pool, so pool size cancels ALGEBRAICALLY. A statistic whose normalisation
cannot matter still drifts by 7.77x, far outside the seed intervals. **The
drift is not in the normalisation.**

The prediction was wrong twice over. `null` is `cand` divided by `k*p`, a
CONSTANT, so its drift is identical to `cand`'s by construction -- it was
unfalsifiable, not merely mistaken. And replacing `w` does buy a factor of
five, which is exactly big enough to look like a fix and ship.

WHAT IS ACTUALLY WRONG IS THE POOL. `topk/mean` reads 1.23 lazy and 9.57 full:
the trained assembly barely stands out when the area is lazily materialised,
because

    lazy pool = 42 candidates, k = 30   ->   71% OF THE POOL IS THE ASSEMBLY

There is nothing to stand out FROM; the mean is dominated by the neurons the
statistic is trying to distinguish. Identical vacuity to
`parse_errors.Stability.trustworthy` (pool <= k makes the k-cap unable to
move). The shipped ERP probe runs at pool/k ~ 1.4.

A residual survives that explanation: 188 -> 3000 candidates still moves
`topk/mean` 1.75x. That is the sampler changing the DISTRIBUTION, not just the
count. Pool size explains most of 7.8x, not all of it.

Deliberately NOT run through calibrate_erp_thresholds: this measures the raw
quantity on one controlled projection, so nothing downstream (clipping,
baselines, thresholds) can mask or mimic invariance.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np                                                  # noqa: E402

from neural_assemblies.assembly_calculus.ops import project         # noqa: E402
from neural_assemblies.core.brain import Brain                      # noqa: E402
from neural_assemblies.diagnostics import ensemble                  # noqa: E402

N, K, P = 3000, 30, 0.05
ARMS = ("lazy", "half", "full")


def _build(seed):
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    b.add_stimulus("s", K)
    for a in ("A", "B"):
        b.add_area(a, N, K, 0.1)
    project(b, "s", "A", rounds=10)
    for _ in range(10):                      # train A -> B
        b.project({}, {"A": ["B"]})
    return b


def _materialise(b, arm):
    """Vary how much of B competes, which is the whole independent variable."""
    if arm == "lazy":
        return
    eng = b._engine_for(b.areas["B"])
    if arm == "full":
        eng.materialize_area("B")
        return
    # "half": drive B from fresh noise so it recruits, without training it.
    for i in range(6):
        b.add_stimulus(f"n{i}", K)
        with b.frozen():
            b.project({f"n{i}": ["B"]}, {})


def _energies(b):
    """One probed projection A -> B; every candidate divisor off the SAME sum.

    Calls the engine directly so `pre_kwta_inputs` (the full candidate vector)
    is available. The Brain propagates only the total and the count, and the
    CONCENTRATION measures below need the distribution -- deliberately measured
    before deciding whether that is worth plumbing through.
    """
    eng = b._engine_for(b.areas["B"])
    with b.read_only():
        res = eng.project_into(
            "B", from_stimuli=[], from_areas=["A"],
            plasticity_enabled=False, record_activation=True,
        )
    total = float(res.pre_kwta_total or 0.0)
    count = int(res.pre_kwta_count or 0)
    w = max(int(b.areas["B"].w), 1)
    k = int(b.areas["B"].k)

    vals = np.asarray(res.pre_kwta_inputs, dtype=np.float64)
    mean = float(vals.mean()) if vals.size else 0.0
    topk = np.sort(vals)[-k:] if vals.size >= k else vals
    out = {
        "w": total / w,
        "cand": total / max(count, 1),
        "k": total / max(k, 1),
        # null = cand / (k*p). A CONSTANT factor, so its drift is IDENTICAL to
        # cand's by construction -- kept to make that visible rather than
        # quietly dropping a candidate I had argued was the favourite.
        "null": total / max(count * k * P, 1e-9),
        # CONCENTRATION: how much more drive the best candidates get than the
        # pool average. A ratio of two quantities from the SAME pool, so pool
        # size cancels -- which is what "is this pathway trained" actually asks.
        "topk/mean": (float(topk.mean()) / mean) if mean > 0 else 0.0,
        "max/mean": (float(vals.max()) / mean) if mean > 0 and vals.size else 0.0,
    }
    return out, w, count


def main(seeds=(1, 2, 3, 4, 5)):
    print(f"n={N} k={K} p={P}  seeds={list(seeds)}")
    print("Reported per divisor: mean over seeds at each materialisation level,")
    print("then max/min across levels. 1.00 = the number stopped moving.\n")

    rows = {arm: [] for arm in ARMS}
    meta = {arm: [] for arm in ARMS}
    for seed in seeds:
        for arm in ARMS:
            b = _build(seed)
            _materialise(b, arm)
            e, w, count = _energies(b)
            rows[arm].append(e)
            meta[arm].append((w, count))

    print(f"{'arm':6s} {'w':>9s} {'candidates':>11s}")
    for arm in ARMS:
        ws = [m[0] for m in meta[arm]]
        cs = [m[1] for m in meta[arm]]
        print(f"{arm:6s} {np.mean(ws):9.0f} {np.mean(cs):11.0f}")

    # Each cell is a `diagnostics.ensemble` over seeds, not a bare mean. The
    # conclusion here is that four statistics drift by the SAME factor, and
    # "the same" is a claim about intervals -- a point estimate cannot carry it.
    print(f"\n{'divisor':10s} " + " ".join(f"{a:>19s}" for a in ARMS)
          + f" {'max/min':>9s}  verdict")
    best, best_drift = None, None
    for key in ("w", "cand", "k", "null", "topk/mean", "max/mean"):
        cells = [ensemble(lambda i, a=arm, kk=key: rows[a][i][kk],
                          list(range(len(seeds))), f"{key}/{arm}")
                 for arm in ARMS]
        means = [c.mean for c in cells]
        lo, hi = min(means), max(means)
        drift = hi / lo if lo > 0 else float("inf")
        verdict = ("INVARIANT" if drift < 1.05 else
                   "drifts" if drift < 2 else "SCALE TRACKS MATERIALISATION")
        print(f"{key:10s} "
              + " ".join(f"{c.mean:10.5f}+/-{c.ci:<7.4f}" for c in cells)
              + f" {drift:9.2f}  {verdict}")
        if best_drift is None or drift < best_drift:
            best, best_drift = key, drift
    print(f"\nLOWEST DRIFT: {best!r} at max/min {best_drift:.3f} -- and that is"
          " NOT a recommendation.")
    print("Four scale-free statistics, two of them ratios WITHIN one pool where")
    print("pool size cancels algebraically, drift by the same factor. A quantity")
    print("whose normalisation cannot matter still moves, so the drift is not in")
    print("the normalisation. See the pool/k ratios in the table above.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(tuple(range(1, 1 + n)))
