"""Figures from the tidy sweep. (task #46)

Dispatches on the `cut` column so one command produces whatever has been
measured so far -- these campaigns run for hours and partial data should still
plot.

The headline figure is the PHASE DIAGRAM (`gain_x_alpha`): retrieval over
(load, gain), with the boundary traced through it. That is the artefact the
grammar work needs, because it turns every ladder cell from an isolated
pass/fail into a POINT ON A MAP, and makes the question "will this grammar fit"
a matter of reading coordinates rather than running the ladder again.

Plotting choices worth stating: a diverging map centred on the half-crossing so
the boundary is visible as a colour change rather than inferred from a legend;
the boundary drawn explicitly on top; and the overlap floor marked wherever
overlap is on an axis, since "at the floor" is the reading that matters most
and an unmarked axis hides it.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, os.environ.get("SWEEP_OUT", "sweep.csv"))
OUTDIR = os.path.join(HERE, "figures")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 4.5,
})


def load():
    rows = []
    with open(CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["level"]) != int(r["depth"]):
                continue                      # deepest level only
            rows.append({
                "cut": r["cut"], "n": int(r["n"]), "k": int(r["k"]),
                "p": float(r["p"]), "kp": float(r["kp"]), "M": int(r["M"]),
                "alpha": float(r["alpha"]), "gain": float(r["gain"]),
                "depth": int(r["depth"]), "acc": float(r["acc"]),
                "margin": float(r["margin"]), "Q": float(r["Q"]),
            })
    return rows


def mean_by(rows, keyf, field):
    acc = defaultdict(list)
    for r in rows:
        acc[keyf(r)].append(r[field])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def crossing(xs, ys, level=0.5):
    for i in range(len(xs) - 1):
        if ys[i] >= level > ys[i + 1]:
            t = (ys[i] - level) / max(ys[i] - ys[i + 1], 1e-12)
            return xs[i] + t * (xs[i + 1] - xs[i])
    return float("nan")


def phase_diagram(rows):
    rows = [r for r in rows if r["cut"] == "gain_x_alpha"]
    if not rows:
        return
    alphas = sorted({r["alpha"] for r in rows})
    gains = sorted({r["gain"] for r in rows})
    grid = mean_by(rows, lambda r: (r["alpha"], r["gain"]), "acc")
    Z = np.full((len(gains), len(alphas)), np.nan)
    for i, g in enumerate(gains):
        for j, a in enumerate(alphas):
            if (a, g) in grid:
                Z[i, j] = grid[(a, g)]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    im = ax.pcolormesh(np.arange(len(alphas) + 1), np.arange(len(gains) + 1),
                       Z, cmap="RdYlBu", vmin=0, vmax=1, shading="flat")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("retrieval accuracy $R$")

    # Trace the boundary: per load, the gain where R falls through 1/2.
    bx, by = [], []
    for j, a in enumerate(alphas):
        col = Z[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        gc = crossing([gains[i] for i in np.where(ok)[0]], col[ok])
        if gc == gc:
            bx.append(j + 0.5)
            by.append(np.interp(gc, gains, np.arange(len(gains)) + 0.5))
    if bx:
        ax.plot(bx, by, "k-o", lw=2.0, ms=5, label=r"boundary ($R=1/2$)")
        ax.legend(loc="lower left", labelcolor="k")

    ax.set_xticks(np.arange(len(alphas)) + 0.5)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_yticks(np.arange(len(gains)) + 0.5)
    ax.set_yticklabels([f"{g:g}" for g in gains])
    ax.set_xlabel(r"load  $\alpha = Mk/n$")
    ax.set_ylabel(r"gain  $g = (1+\beta)^T$")
    ax.set_title(f"Phase diagram, depth {rows[0]['depth']} "
                 f"(n = {rows[0]['n']:,}, $kp$ = {rows[0]['kp']:g})")
    ax.grid(False)
    fig.savefig(os.path.join(OUTDIR, "fig4_phase_diagram.png"))
    plt.close(fig)
    print("    fig4_phase_diagram.png")


def gc_versus(rows, cut, xfield, xlabel, fname, title, logx=False):
    rows = [r for r in rows if r["cut"] == cut]
    if not rows:
        return
    xs = sorted({r[xfield] for r in rows})
    grid = mean_by(rows, lambda r: (r[xfield], r["gain"]), "acc")
    gains = sorted({r["gain"] for r in rows})
    pts = []
    for x in xs:
        col = [grid.get((x, g), np.nan) for g in gains]
        ok = [i for i, v in enumerate(col) if v == v]
        if len(ok) < 2:
            continue
        gc = crossing([gains[i] for i in ok], [col[i] for i in ok])
        if gc == gc:
            pts.append((x, gc))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#B4436C")
    for x, gc in pts:
        ax.annotate(f"{gc:.2f}", (x, gc), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color="#B4436C")
    if logx:
        ax.set_xscale("log")
        ax.set_xticks([p[0] for p in pts])
        ax.set_xticklabels([f"{p[0]:g}" for p in pts])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"critical gain  $g_c$  ($R=1/2$)")
    ax.set_title(title)
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)
    print(f"    {fname}")


def capacity_rel(rows):
    rows = [r for r in rows if r["cut"] == "capacity_rel"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    sizes = sorted({r["n"] for r in rows})
    cmap = plt.get_cmap("viridis")
    mm = []
    for i, n in enumerate(sizes):
        sub = [r for r in rows if r["n"] == n]
        ms = sorted({r["M"] for r in sub})
        a = mean_by(sub, lambda r: r["M"], "acc")
        ys = [a[m] for m in ms]
        c = cmap(i / max(len(sizes) - 1, 1) * 0.85)
        ax.plot(ms, ys, "o-", color=c, label=f"n = {n:,}")
        x = crossing(ms, ys)
        if x == x:
            mm.append((n, x))
            ax.axvline(x, color=c, ls=":", lw=1.0)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("items stored  $M$")
    ax.set_ylabel("retrieval accuracy  $R$")
    ax.set_title(r"Capacity at fixed RELATIVE gain $g = 0.88\,g_c(n)$")
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "fig7_capacity_relative.png"))
    plt.close(fig)
    print("    fig7_capacity_relative.png")
    import math
    for i in range(len(mm) - 1):
        (n0, m0), (n1, m1) = mm[i], mm[i + 1]
        print(f"      n {n0}->{n1}: M_max {m0:.0f}->{m1:.0f}  "
              f"exponent {math.log(m1 / m0) / math.log(n1 / n0):.2f}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load()
    print(f"\n  {len(rows)} deepest-level rows; cuts present: "
          f"{sorted({r['cut'] for r in rows})}\n")
    phase_diagram(rows)
    gc_versus(rows, "gain_x_depth", "depth", "composition depth $D$",
              "fig5_gc_vs_depth.png", "Does depth narrow the usable gain?")
    gc_versus(rows, "gain_x_kp", "kp", r"afferent count  $kp$",
              "fig6_gc_vs_kp.png", r"Is $g_c$ really independent of density?",
              logx=True)
    capacity_rel(rows)
