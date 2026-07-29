"""Does the beta cliff sit at (1+beta)^T = 2? (task #46)

WHAT IS BEING EXPLAINED. Depth 5 works at beta=0.35 and dies at beta=0.40 --
accuracy 0.938 -> 0.164 -- and DOUBLING THE CONNECTION DENSITY DOES NOT MOVE
IT (0.164 at p=0.05, 0.172 at p=0.10), even though density nearly doubles the
margin inside the working regime (4.65 -> 8.11 at beta=0.35). So the cliff is
not a property of the competition statistics, which p controls; it is a
property of the potentiation dynamics.

THE CANDIDATE. With the ladder's merge rounds T=2:

    beta 0.35 -> (1+beta)^T = 1.82     works
    beta 0.40 -> (1+beta)^T = 1.96     dies

The cliff sits almost exactly where ONE MERGE DOUBLES THE WEIGHT. This project
has already found (1+beta)^T to be the single governing variable for
recurrence persistence, so it is a natural suspect here too.

PRE-REGISTERED PREDICTION. If the cliff is at (1+beta)^T = 2 then it MOVES with
T, to beta = 2^(1/T) - 1:

    T = 1  ->  cliff near beta 1.00
    T = 2  ->  cliff near beta 0.41   (already observed between 0.35 and 0.40)
    T = 3  ->  cliff near beta 0.26
    T = 4  ->  cliff near beta 0.19

WHAT WOULD REFUTE IT. The cliff staying near beta 0.4 regardless of T -- which
would mean beta alone matters and the exponent is irrelevant, pointing instead
at something keyed on the per-event increment, e.g. w_max saturation (w_max is
20.0 here) or a fixed threshold in the plasticity clamp.

Either answer is worth having. A cliff at a known analytic location is a design
rule; a cliff at a fixed beta is a bug or a clamp.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

N, M, DEPTH = 4000, 64, 5
SEEDS = [42, 43]

#: T -> betas to scan, chosen to straddle the predicted cliff 2^(1/T) - 1.
SCANS = {
    1: [0.60, 0.80, 1.00, 1.20],
    2: [0.30, 0.35, 0.40, 0.45],
    3: [0.18, 0.22, 0.26, 0.32],
}


def run(T, beta):
    ladder.MERGE_ROUNDS = T
    ladder.BETA = beta
    margs, accs, sprs = [], [], []
    for s in SEEDS:
        full, total, margins, spreads = ladder.trial(N, M, DEPTH, s)
        margs.append(margins[DEPTH])
        accs.append(full[DEPTH] / total)
        sprs.append(spreads[DEPTH])
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    return mean(margs), mean(accs), mean(sprs)


if __name__ == "__main__":
    floor = 50 / N
    print(f"\n  BETA CLIFF LOCATION vs MERGE ROUNDS   n={N} M={M} "
          f"depth={DEPTH} p={ladder.P_}")
    print("  prediction: cliff at (1+beta)^T = 2, i.e. beta = 2^(1/T) - 1\n")

    for T, betas in SCANS.items():
        pred = 2 ** (1.0 / T) - 1
        print(f"  T={T}   predicted cliff at beta {pred:.3f}")
        print(f"  {'beta':>7} {'(1+b)^T':>9} {'marg':>7} {'acc':>7} "
              f"{'spr':>8}")
        for beta in betas:
            t0 = time.time()
            marg, acc, spr = run(T, beta)
            flag = "OK" if acc >= 0.90 else (
                f"DEAD(spr={spr / floor:.0f}x)" if spr > 3 * floor
                else "DEAD(starved)")
            print(f"  {beta:>7.2f} {(1 + beta) ** T:>9.3f} {marg:>7.2f} "
                  f"{acc:>7.3f} {spr:>8.4f}  {flag:<16} "
                  f"[{time.time() - t0:.0f}s]")
        print()
