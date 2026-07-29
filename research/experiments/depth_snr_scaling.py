"""Are beta and p interchangeable through ONE signal-to-noise group? (#46)

Established: depth is bought with beta at low load, and the same move destroys
depth under load. That is a two-parameter description. This tests whether the
underlying quantity is actually ONE number.

THE ARGUMENT. Recruitment at each level is an extreme-value competition. A
candidate neuron's afferent count from a firing assembly of size k is
Binomial(k, p) -- mean kp, sd sqrt(kp(1-p)). The winners are the top k of n
such candidates, so the bar to clear sits roughly

    kp + sqrt(2 ln n) * sqrt(kp(1-p))

above which a neuron wins. A neuron on the trained path carries fibers
potentiated to (1+beta)^T, so its excess drive over an untrained candidate is
about kp * ((1+beta)^T - 1). Dividing the signal by the noise:

    SNR = sqrt(kp / (1-p)) * ((1+beta)^T - 1) / sqrt(2 ln n)

PRE-REGISTERED PREDICTION. If this is the governing quantity then beta and p
are INTERCHANGEABLE: cells with equal SNR should show equal depth performance
even when beta differs 3.5x and p differs 10x. Concretely, margin should be a
function of SNR alone, and cells should sort cleanly by it.

WHAT WOULD REFUTE IT. Margin varying systematically with beta AT FIXED SNR --
which would mean plasticity does something density cannot buy (plausible:
potentiation is CONCENTRATED on the trained path, while raising p adds
afferents everywhere including to competitors, so the two are not obviously
substitutable). Recording the prediction here so the refutation counts.

This also probes a regime the project has never entered: k*p < 1, where most
candidates receive NO input from the assembly at all and selection should
starve regardless of beta.
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

N = int(os.environ.get("SNR_N", "4000"))
M = int(os.environ.get("SNR_M", "64"))
DEPTH = int(os.environ.get("SNR_DEPTH", "3"))
T_MERGE = 2  # the ladder's merge rounds; (1+beta)^T uses this
PS = [float(x) for x in os.environ.get("SNR_PS", "0.01,0.02,0.05,0.10").split(",")]
BETAS = [float(x) for x in os.environ.get("SNR_BETAS", "0.10,0.20,0.35").split(",")]
SEEDS = [int(x) for x in os.environ.get("SNR_SEEDS", "42,43").split(",")]


def snr(p, beta, k, n):
    return (math.sqrt(k * p / (1.0 - p)) * ((1.0 + beta) ** T_MERGE - 1.0)
            / math.sqrt(2.0 * math.log(n)))


def run_cell(p, beta):
    ladder.P_ = p
    ladder.BETA = beta
    margs, sprs, dists = [], [], []
    for s in SEEDS:
        distinct, total, margins, spreads = ladder.trial(N, M, DEPTH, s)
        margs.append(margins[DEPTH])
        sprs.append(spreads[DEPTH])
        dists.append(distinct[DEPTH] / total if total else float("nan"))
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    return mean(margs), mean(sprs), mean(dists)


if __name__ == "__main__":
    k = ladder.K_
    print(f"\n  DEPTH vs SNR   n={N} M={M} k={k} depth={DEPTH} "
          f"T={T_MERGE} seeds={len(SEEDS)}")
    print(f"  SNR = sqrt(kp/(1-p)) * ((1+beta)^T - 1) / sqrt(2 ln n)\n")
    print(f"  {'p':>6} {'beta':>6} {'k*p':>6} {'SNR':>7} "
          f"{'marg':>7} {'spr':>8} {'distinct':>9}")

    rows = []
    for p in PS:
        for beta in BETAS:
            t0 = time.time()
            try:
                marg, spr, dist = run_cell(p, beta)
            except Exception as exc:  # noqa: BLE001
                print(f"  {p:>6.3f} {beta:>6.2f}       "
                      f"FAILED: {type(exc).__name__}: {exc}")
                continue
            s = snr(p, beta, k, N)
            rows.append((s, p, beta, marg, spr, dist))
            print(f"  {p:>6.3f} {beta:>6.2f} {k * p:>6.2f} {s:>7.3f} "
                  f"{marg:>7.2f} {spr:>8.4f} {dist:>9.3f}  "
                  f"[{time.time() - t0:.0f}s]")
        print()

    print("  --- sorted by SNR; if the group governs, margin rises with it "
          "monotonically ---")
    print(f"  {'SNR':>7} {'p':>6} {'beta':>6} {'marg':>7} {'distinct':>9}")
    for s, p, beta, marg, spr, dist in sorted(rows):
        print(f"  {s:>7.3f} {p:>6.3f} {beta:>6.2f} {marg:>7.2f} {dist:>9.3f}")

    # The sharpest form of the test: cells of similar SNR reached from very
    # different (p, beta). If margin still tracks beta there, the group is
    # incomplete and the prediction above is refuted.
    print("\n  --- nearest-SNR pairs reached from DIFFERENT (p, beta) ---")
    srt = sorted(rows)
    for a, b in zip(srt, srt[1:]):
        if a[2] != b[2] and abs(a[0] - b[0]) / max(a[0], 1e-9) < 0.25:
            print(f"  SNR {a[0]:.3f} vs {b[0]:.3f}   "
                  f"(p={a[1]},b={a[2]}) marg {a[3]:.2f}  |  "
                  f"(p={b[1]},b={b[2]}) marg {b[3]:.2f}")
