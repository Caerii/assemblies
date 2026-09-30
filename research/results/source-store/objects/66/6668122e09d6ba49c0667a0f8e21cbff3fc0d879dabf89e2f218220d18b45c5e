"""Extract BOTH walls of the retrieval region, not just the upper one. (#46)

WHY THIS EXISTS. Every analysis so far located a single boundary -- the gain
above which assemblies merge -- by looking for where retrieval falls through
one half. That silently assumes retrieval is a decreasing function of gain. It
is not. There is also a LOWER wall: below some gain the chain cannot carry the
signal at all and retrieval starves with assemblies still perfectly distinct.

So the working region is a WEDGE in (load, gain), bounded above by crowding and
below by starvation, and the quantity that actually matters is its WIDTH. A
single-boundary analysis reports a system as healthy right up to the point
where the wedge closes, and then reports the closure as "the upper boundary
moved", which is the wrong mechanism and suggests the wrong fix.

The closure is the real capacity limit, and it is a property of the substrate
rather than of whatever gain someone happened to choose. "M_max at beta=0.3" is
a fact about the tuning; "the load at which no gain works" is a fact about the
system.

WHAT IT GUARDS AGAINST. A grid that does not bracket the wedge produces a
confident and wrong answer, which is exactly what happened on the first
gain_x_alpha run: the grid was anchored on the alpha=0.8 boundary and started
at g=1.50, so at alpha=1.6 the lowest gain sampled was already the best one and
the lower wall lay outside the scan entirely. Every cell therefore reports
whether its wedge is BRACKETED, and an unbracketed edge is never silently
interpolated.
"""

from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSVS = [x for x in os.environ.get("WEDGE_CSVS", "sweep.csv").split(",") if x]
LEVEL = 0.5


def load(paths):
    rows = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(full):
            continue
        with open(full, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(r["level"]) != int(r["depth"]):
                    continue
                rows.append(r)
    return rows


def cross_up(xs, ys, lvl=LEVEL):
    """First x where y rises through lvl (the starvation wall)."""
    for i in range(len(xs) - 1):
        if ys[i] < lvl <= ys[i + 1]:
            t = (lvl - ys[i]) / max(ys[i + 1] - ys[i], 1e-12)
            return xs[i] + t * (xs[i + 1] - xs[i])
    return float("nan")


def cross_down(xs, ys, lvl=LEVEL):
    """Last x where y falls through lvl (the crowding wall)."""
    out = float("nan")
    for i in range(len(xs) - 1):
        if ys[i] >= lvl > ys[i + 1]:
            t = (ys[i] - lvl) / max(ys[i] - ys[i + 1], 1e-12)
            out = xs[i] + t * (xs[i + 1] - xs[i])
    return out


def analyse(rows, group_field):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[float(r[group_field])][float(r["gain"])].append(float(r["acc"]))

    print(f"\n  {'group':>7} {'g_lo':>7} {'g_hi':>7} {'width':>7} "
          f"{'peak R':>7} {'at g':>6}  bracketed")
    out = []
    for key in sorted(by):
        gs = sorted(by[key])
        ys = [statistics.mean(by[key][g]) for g in gs]
        lo, hi = cross_up(gs, ys), cross_down(gs, ys)
        peak = max(ys)
        gpeak = gs[ys.index(peak)]

        # A wall is only real if the grid actually spans it. Below the lowest
        # sampled gain the curve may still be rising; above the highest it may
        # still be falling. Either way an edge value is a bound, not a
        # measurement, and saying so is the whole point of this column.
        lo_ok = lo == lo or ys[0] < LEVEL
        hi_ok = hi == hi or ys[-1] < LEVEL
        note = []
        if not lo_ok:
            note.append(f"lower wall BELOW grid (g>={gs[0]:g} already "
                        f"R={ys[0]:.2f})")
        if not hi_ok:
            note.append(f"upper wall ABOVE grid (g<={gs[-1]:g} still "
                        f"R={ys[-1]:.2f})")
        if peak < LEVEL:
            note.append("WEDGE CLOSED -- no gain reaches R=1/2")

        width = (hi - lo) if (lo == lo and hi == hi) else float("nan")
        out.append((key, lo, hi, width, peak))
        print(f"  {key:>7g} {lo:>7.3f} {hi:>7.3f} {width:>7.3f} "
              f"{peak:>7.3f} {gpeak:>6.2f}  "
              f"{'yes' if not note else '; '.join(note)}")
    return out


if __name__ == "__main__":
    rows = load(CSVS)
    cuts = sorted({r["cut"] for r in rows})
    print(f"\n  WEDGE ANALYSIS   {len(rows)} deepest-level rows   cuts={cuts}")
    for cut in cuts:
        sub = [r for r in rows if r["cut"] == cut]
        field = {"gain_x_alpha": "alpha", "gain_x_depth": "depth",
                 "gain_x_kp": "kp"}.get(cut)
        if not field:
            continue
        print(f"\n  === {cut}  (grouped by {field}) ===")
        res = analyse(sub, field)
        closed = [k for k, _lo, _hi, _w, peak in res if peak < LEVEL]
        if closed:
            print(f"\n    wedge CLOSED at {field} in {closed}")
        widths = [(k, w) for k, _l, _h, w, _p in res if w == w]
        if len(widths) >= 2:
            print(f"    width trend: " + ", ".join(
                f"{k:g}->{w:.2f}" for k, w in widths))
