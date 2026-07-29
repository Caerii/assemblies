"""Figures for the gain-driven transition. (task #46)

Reads the tidy CSV from critical_point_scan.py and produces:

  1. ORDER PARAMETER vs GAIN, one curve per system size, with seed spread as a
     band. If the transition is a genuine critical point the curves cross at a
     single g_c; if the threshold merely drifts with n they will not.
  2. RETRIEVAL vs GAIN, the same cut in the other macrostate. Shown beside (1)
     because the starved and collapsed phases are BOTH low-retrieval and are
     distinguished only by the order parameter.
  3. g_c vs n, extracted per size by interpolating where retrieval crosses
     one half. This is the quantity that decides whether "g_c ~ 1.9" is a
     constant of the substrate or an artifact of one system size.
  4. DEPTH PROFILE: order parameter against level, per gain. Composition decays
     along a chain, and this shows whether the decay is graded or a cliff.

Design choices are deliberate: perceptually ordered colours for the size
sequence, direct labelling instead of a legend where it fits, no chartjunk, and
axis limits that do not hide the floor. The overlap floor k/n is drawn
explicitly on every overlap axis, because "at the floor" is the single most
important reading and an unmarked axis makes it invisible.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, os.environ.get("CPS_OUT", "critical_point_scan.csv"))
OUTDIR = os.path.join(HERE, "figures")

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
})


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for key in ("n", "k", "M", "T", "depth", "level", "seed"):
                r[key] = int(r[key])
            for key in ("p", "alpha", "beta", "gain", "acc", "margin",
                        "spread", "floor", "Q"):
                r[key] = float(r[key])
            rows.append(r)
    return rows


def agg(rows, level_is_deepest=True):
    """(n, gain) -> mean/std over seeds, at the deepest level by default."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if level_is_deepest and r["level"] != r["depth"]:
            continue
        out[(r["n"], r["gain"])]["acc"].append(r["acc"])
        out[(r["n"], r["gain"])]["Q"].append(r["Q"])
        out[(r["n"], r["gain"])]["margin"].append(r["margin"])
        out[(r["n"], r["gain"])]["floor"].append(r["floor"])
    return out


def series(a, n, field):
    gs = sorted(g for (nn, g) in a if nn == n)
    mu = np.array([np.mean(a[(n, g)][field]) for g in gs])
    sd = np.array([np.std(a[(n, g)][field]) for g in gs])
    return np.array(gs), mu, sd


def crossing(gs, ys, level=0.5):
    """Gain where a decreasing curve first crosses `level`, by linear interp."""
    for i in range(len(gs) - 1):
        if ys[i] >= level > ys[i + 1]:
            t = (ys[i] - level) / max(ys[i] - ys[i + 1], 1e-12)
            return gs[i] + t * (gs[i + 1] - gs[i])
    return float("nan")


def main():
    if not os.path.exists(CSV):
        sys.exit(f"no data at {CSV}; run critical_point_scan.py first")
    rows = load(CSV)
    os.makedirs(OUTDIR, exist_ok=True)
    a = agg(rows)
    sizes = sorted({n for (n, _) in a})
    cmap = plt.get_cmap("viridis")
    colours = {n: cmap(i / max(len(sizes) - 1, 1) * 0.85)
               for i, n in enumerate(sizes)}

    # ---- Figure 1: the transition, both macrostates ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for n in sizes:
        gs, q, qsd = series(a, n, "Q")
        gs, r, rsd = series(a, n, "acc")
        c = colours[n]
        axes[0].plot(gs, q, "o-", color=c, label=f"n = {n:,}")
        axes[0].fill_between(gs, q - qsd, q + qsd, color=c, alpha=0.15, lw=0)
        axes[1].plot(gs, r, "o-", color=c, label=f"n = {n:,}")
        axes[1].fill_between(gs, r - rsd, r + rsd, color=c, alpha=0.15, lw=0)

    axes[0].set_xlabel(r"gain  $g=(1+\beta)^T$")
    axes[0].set_ylabel(r"order parameter  $Q=(\bar q - q_0)/(1-q_0)$")
    axes[0].set_title("Assemblies merge above a critical gain")
    axes[0].set_ylim(-0.03, 1.0)
    axes[0].legend(loc="upper left")

    axes[1].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[1].set_xlabel(r"gain  $g=(1+\beta)^T$")
    axes[1].set_ylabel("retrieval accuracy  $R$")
    axes[1].set_title("Retrieval fails on the same boundary")
    axes[1].set_ylim(-0.03, 1.05)
    fig.suptitle("Gain-driven transition in compositional retrieval "
                 f"(depth {rows[0]['depth']}, "
                 rf"$\alpha$ = {rows[0]['alpha']:g}, $kp$ = "
                 f"{rows[0]['k'] * rows[0]['p']:g})", y=1.04)
    fig.savefig(os.path.join(OUTDIR, "fig1_transition.png"))
    plt.close(fig)

    # ---- Figure 2: does g_c move with system size? -----------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ns, gcs = [], []
    for n in sizes:
        gs, r, _ = series(a, n, "acc")
        gc = crossing(gs, r, 0.5)
        if gc == gc:
            ns.append(n)
            gcs.append(gc)
    ax.plot(ns, gcs, "o-", color="#B4436C")
    for n, gc in zip(ns, gcs):
        ax.annotate(f"{gc:.2f}", (n, gc), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color="#B4436C")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_xlabel("system size  $n$")
    ax.set_ylabel(r"critical gain  $g_c$   ($R = 1/2$)")
    ax.set_title("Is $g_c$ a constant of the substrate?")
    fig.savefig(os.path.join(OUTDIR, "fig2_gc_vs_n.png"))
    plt.close(fig)

    # ---- Figure 3: decay along the composition chain ---------------------
    per = defaultdict(lambda: defaultdict(list))
    biggest = sizes[-1]
    for r in rows:
        if r["n"] != biggest:
            continue
        per[r["gain"]][r["level"]].append(r["Q"])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    gains = sorted(per)
    cm = plt.get_cmap("magma")
    for i, g in enumerate(gains):
        lv = sorted(per[g])
        ys = [np.mean(per[g][L]) for L in lv]
        ax.plot(lv, ys, "o-", color=cm(0.12 + 0.72 * i / max(len(gains) - 1, 1)),
                label=f"g = {g:g}" if i % 2 == 0 else None)
    ax.set_xlabel("composition level")
    ax.set_ylabel(r"order parameter  $Q$")
    ax.set_title(f"Overlap accumulates along the chain (n = {biggest:,})")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(OUTDIR, "fig3_depth_profile.png"))
    plt.close(fig)

    print(f"  wrote 3 figures to {OUTDIR}")
    for n, gc in zip(ns, gcs):
        print(f"    n={n:>6,}   g_c = {gc:.3f}")


if __name__ == "__main__":
    main()
