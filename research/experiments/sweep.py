"""One resumable sweep driver, one tidy schema. (task #46)

WHY THIS REPLACES THE ONE-OFFS. Three separate harnesses now exist -- the gain
scan, the capacity ladder, the beta/depth grid -- each with its own output
format, its own aggregation, and its own idea of what a "cell" is. Every new
question has therefore needed new plumbing, and cross-question comparison has
meant re-deriving the same columns by hand. Before the grammar work can be
scaled, the substrate needs a MAP, and a map wants one coordinate system.

So: every measurement is a point in the same space

    (n, k, p, M, gain, depth, seed)

and every run appends rows in the same schema, one row per LEVEL so depth
profiles come free. Aggregation belongs in analysis, never in measurement.

RESUMABLE, which matters more than it sounds. These sweeps run for hours and
get interrupted -- by a machine reboot, by a better idea, by a bug found
halfway. Completed points are keyed by their full configuration and skipped on
re-run, so a campaign can be extended, interrupted, and resumed without
recomputing or (worse) silently duplicating rows with different code.

CUTS are named presets, because the useful questions are specific
two-dimensional slices rather than a full grid, which would be unaffordable:

  gain_x_n       gain vs system size at fixed load     -- how the boundary moves
  gain_x_alpha   gain vs load at fixed size            -- THE PHASE DIAGRAM
  gain_x_depth   gain vs composition depth             -- does the optimum fall?
  gain_x_kp      gain vs afferent count                -- is g_c really p-free?
  capacity_rel   M ladder at fixed RELATIVE gain       -- capacity done properly

`gain_x_alpha` is the one that most directly serves the grammar work: it places
every ladder cell on a map instead of leaving each as an isolated pass/fail.

`capacity_rel` exists because measuring capacity at a fixed ABSOLUTE gain was
shown to be a confounded protocol -- g_c moves with n, so a fixed gain sits at
a different distance from each size's boundary, and the measured exponent ran
1.03 then 1.95 across two intervals of one sweep. Holding g/g_c(n) fixed is the
comparison that means something.
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("SWEEP_OUT", "sweep.csv"))
T = 2

FIELDS = ["cut", "n", "k", "p", "kp", "M", "alpha", "T", "beta", "gain",
          "depth", "level", "seed", "acc", "margin", "spread", "floor", "Q"]

#: Measured critical gains, for the fixed-RELATIVE-gain cut. From
#: fss_collapse.py at alpha=0.8, depth 5, k=50, p=0.05.
GC_AT_ALPHA_08 = {2000: 1.715, 4000: 1.915, 8000: 2.176}

GAINS_FINE = [1.50, 1.60, 1.70, 1.78, 1.84, 1.90, 1.96, 2.02, 2.10, 2.20, 2.35]


def cut_gain_x_alpha():
    """THE PHASE DIAGRAM: gain against load, at one system size."""
    for M, g in itertools.product([32, 64, 128, 256], GAINS_FINE):
        yield dict(n=4000, k=50, p=0.05, M=M, gain=g, depth=5)


def cut_gain_x_depth():
    """Does the usable gain fall as chains get deeper?"""
    for D, g in itertools.product([2, 3, 4, 5, 6],
                                  [1.50, 1.70, 1.84, 1.96, 2.10]):
        yield dict(n=4000, k=50, p=0.05, M=64, gain=g, depth=D)


def cut_gain_x_kp():
    """g_c was measured p-free at ONE pair of densities. Test it properly."""
    for p, g in itertools.product([0.02, 0.05, 0.10, 0.20],
                                  [1.50, 1.70, 1.84, 1.96, 2.10, 2.35]):
        yield dict(n=4000, k=50, p=p, M=64, gain=g, depth=5)


def cut_capacity_rel():
    """Capacity at fixed RELATIVE gain g = 0.88 * g_c(n), finer M ladder."""
    for n, gc in GC_AT_ALPHA_08.items():
        for M in (24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
            if M * 50 / n > 4.0:      # far past any measured boundary
                continue
            yield dict(n=n, k=50, p=0.05, M=M, gain=0.88 * gc, depth=5)


CUTS = {
    "gain_x_alpha": cut_gain_x_alpha,
    "gain_x_depth": cut_gain_x_depth,
    "gain_x_kp": cut_gain_x_kp,
    "capacity_rel": cut_capacity_rel,
}


def key_of(cut, c, seed):
    return (cut, c["n"], c["k"], f"{c['p']:.5f}", c["M"],
            f"{c['gain']:.4f}", c["depth"], seed)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.add((r["cut"], int(r["n"]), int(r["k"]), f"{float(r['p']):.5f}",
                     int(r["M"]), f"{float(r['gain']):.4f}", int(r["depth"]),
                     int(r["seed"])))
    return out


def main():
    cut = os.environ.get("SWEEP_CUT", "gain_x_alpha")
    seeds = [int(x) for x in os.environ.get("SWEEP_SEEDS", "42,43").split(",")]
    if cut not in CUTS:
        sys.exit(f"unknown cut {cut!r}; have {sorted(CUTS)}")

    configs = list(CUTS[cut]())
    done = done_keys(OUT)
    todo = [(c, s) for c in configs for s in seeds
            if key_of(cut, c, s) not in done]

    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(FIELDS)
        fh.flush()

    print(f"\n  SWEEP {cut}   {len(configs)} configs x {len(seeds)} seeds")
    print(f"  {len(todo)} to run, {len(configs) * len(seeds) - len(todo)} "
          f"already done -> {OUT}\n")

    t_start = time.time()
    for i, (c, seed) in enumerate(todo, 1):
        ladder.K_, ladder.P_ = c["k"], c["p"]
        ladder.BETA = c["gain"] ** (1.0 / T) - 1.0
        ladder.MERGE_ROUNDS = T
        floor = c["k"] / c["n"]
        t0 = time.time()
        full, total, margins, spreads = ladder.trial(
            c["n"], c["M"], c["depth"], seed)
        for L in range(1, c["depth"] + 1):
            q = spreads[L]
            w.writerow([cut, c["n"], c["k"], f"{c['p']:.5f}",
                        f"{c['k'] * c['p']:.3f}", c["M"],
                        f"{c['M'] * c['k'] / c['n']:.4f}", T,
                        f"{ladder.BETA:.6f}", f"{c['gain']:.4f}", c["depth"],
                        L, seed, f"{full[L] / total:.6f}",
                        f"{margins[L]:.6f}", f"{q:.6f}", f"{floor:.6f}",
                        f"{(q - floor) / (1 - floor):.6f}"])
        fh.flush()
        D = c["depth"]
        el = time.time() - t_start
        eta = el / i * (len(todo) - i)
        print(f"  [{i:>3}/{len(todo)}] n={c['n']} M={c['M']} p={c['p']:g} "
              f"D={D} g={c['gain']:.2f} s={seed}  acc={full[D] / total:.3f} "
              f"marg={margins[D]:6.2f}  [{time.time() - t0:.0f}s, "
              f"eta {eta / 60:.0f}m]")
    fh.close()
    print(f"\n  done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
