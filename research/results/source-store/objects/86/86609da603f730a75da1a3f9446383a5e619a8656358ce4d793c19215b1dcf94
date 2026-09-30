"""Shared analysis for every sweep CSV. (task #46)

ONE implementation of load / aggregate / confidence-interval / wedge, imported
by the plotting scripts and the notebook alike. The alternative -- and what was
happening -- is three copies of the same aggregation drifting apart, which is
how two figures of the same data end up disagreeing and nobody can say which is
right.

CONFIDENCE INTERVALS, because the campaign ran on two seeds and every number
reported so far has been a bare mean. Two seeds cannot support the claims that
were being made from them: three separate results died today on the difference
between "two points agree" and "there is an effect" -- superlinear capacity, an
invariance in k, and a module-loader theory of a harness bug. Bootstrap
intervals over seeds are the minimum honest presentation, and where the seed
count is too small for one the code says so rather than drawing a band.
"""

from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

#: Every results file, so a single call sees the whole campaign.
ALL_CSVS = ["sweep.csv", "critical_point_scan.csv", "capacity_direct.csv",
            "capacity_k.csv", "confound_n4000_M32.csv",
            "confound_n2000_M64.csv", "zipf_gain.csv", "zipf_scaled.csv",
            "bridge_parser.csv"]

NUM = ("p", "kp", "alpha", "beta", "gain", "acc", "margin", "spread",
       "floor", "Q", "acc_head", "acc_tail", "zipf_s", "distinct_frac",
       "g_step", "g_eff")
INT = ("n", "k", "M", "T", "depth", "level", "seed", "rounds", "reset",
       "n_items", "collapsed")


def load(paths=None, deepest_only=True):
    """Rows from every CSV that exists, coerced, optionally deepest level only."""
    rows = []
    for name in (paths if paths is not None else ALL_CSVS):
        full = name if os.path.isabs(name) else os.path.join(HERE, name)
        if not os.path.exists(full):
            continue
        with open(full, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out = {"source": os.path.basename(full)}
                for key, val in r.items():
                    if key in INT:
                        out[key] = int(float(val)) if val != "" else None
                    elif key in NUM:
                        out[key] = float(val) if val != "" else float("nan")
                    else:
                        out[key] = val
                if deepest_only and out.get("level") is not None:
                    if out["level"] != out.get("depth"):
                        continue
                rows.append(out)
    return rows


def boot_ci(values, reps=2000, level=0.95, seed=0):
    """(mean, lo, hi) by bootstrap over the sample. NaN bounds if n < 3.

    Deliberately refuses to produce an interval from two points. A band drawn
    from two seeds implies a precision the data does not have, and this project
    has already over-read exactly that.
    """
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) < 3:
        return mean, float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(s) / len(s))
    means.sort()
    lo = means[int((1 - level) / 2 * reps)]
    hi = means[min(reps - 1, int((1 + level) / 2 * reps))]
    return mean, lo, hi


def group(rows, keys, field):
    """{key tuple: [values]} for a field, grouped by the given columns."""
    out = defaultdict(list)
    for r in rows:
        if r.get(field) is None or r.get(field) != r.get(field):
            continue
        out[tuple(r.get(k) for k in keys)].append(r[field])
    return out


def curve(rows, x, y, keys=(), reps=1000):
    """{key: [(x, mean, lo, hi, n_seeds)]} sorted by x, with bootstrap CIs."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get(y) is None or r.get(y) != r.get(y) or r.get(x) is None:
            continue
        buckets[tuple(r.get(k) for k in keys)][r[x]].append(r[y])
    out = {}
    for key, per_x in buckets.items():
        pts = []
        for xv in sorted(per_x):
            m, lo, hi = boot_ci(per_x[xv], reps=reps)
            pts.append((xv, m, lo, hi, len(per_x[xv])))
        out[key] = pts
    return out


def cross_up(pts, lvl=0.5):
    for i in range(len(pts) - 1):
        if pts[i][1] < lvl <= pts[i + 1][1]:
            t = (lvl - pts[i][1]) / max(pts[i + 1][1] - pts[i][1], 1e-12)
            return pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
    return float("nan")


def cross_down(pts, lvl=0.5):
    out = float("nan")
    for i in range(len(pts) - 1):
        if pts[i][1] >= lvl > pts[i + 1][1]:
            t = (pts[i][1] - lvl) / max(pts[i][1] - pts[i + 1][1], 1e-12)
            out = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
    return out


def wedge(pts, lvl=0.5):
    """(g_lo, g_hi, width, peak, g_at_peak, bracketed, note) for a gain curve.

    `bracketed` is false when a wall lies outside the sampled range, in which
    case the width is NOT a measurement. The first campaign grid started at
    g=1.50 and so missed the starvation wall entirely at high load -- the
    lowest gain sampled was already the best one -- and reported nothing amiss.
    """
    if not pts:
        return (float("nan"),) * 5 + (False, "no data")
    lo, hi = cross_up(pts, lvl), cross_down(pts, lvl)
    peak = max(p[1] for p in pts)
    gpeak = next(p[0] for p in pts if p[1] == peak)
    if peak < lvl:
        return lo, hi, float("nan"), peak, gpeak, True, "CLOSED"
    notes = []
    if lo != lo and pts[0][1] >= lvl:
        notes.append(f"lower wall below grid (g>={pts[0][0]:g} already "
                     f"R={pts[0][1]:.2f})")
    if hi != hi and pts[-1][1] >= lvl:
        notes.append(f"upper wall above grid (g<={pts[-1][0]:g} still "
                     f"R={pts[-1][1]:.2f})")
    width = (hi - lo) if (lo == lo and hi == hi) else float("nan")
    return lo, hi, width, peak, gpeak, not notes, "; ".join(notes)


def seed_counts(rows):
    """How many seeds back each configuration -- the first thing to check."""
    per = defaultdict(set)
    for r in rows:
        key = (r.get("cut", r.get("source")), r.get("n"), r.get("k"),
               r.get("M"), r.get("gain"), r.get("depth"))
        per[key].add(r.get("seed"))
    hist = defaultdict(int)
    for key, seeds in per.items():
        hist[len(seeds)] += 1
    return dict(sorted(hist.items()))


def fit_loglinear(xs, ys):
    """y = a + b ln x, least squares. Returns (a, b, residual, r2)."""
    xs = [math.log(x) for x in xs]
    n = len(xs)
    if n < 2:
        return (float("nan"),) * 4
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (float("nan"),) * 4
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = sum((a + b * x - y) ** 2 for x, y in zip(xs, ys))
    tot = sum((y - my) ** 2 for y in ys)
    return a, b, res, (1 - res / tot if tot > 0 else float("nan"))


if __name__ == "__main__":
    rows = load()
    print(f"\n  {len(rows)} deepest-level rows from "
          f"{sorted({r['source'] for r in rows})}")
    print(f"\n  seeds per configuration: {seed_counts(rows)}")
    print("  (configurations backed by 2 seeds cannot carry a confidence "
          "interval)")
