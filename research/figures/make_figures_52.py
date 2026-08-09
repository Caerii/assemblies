"""Publication figures for the #52 recipe-propagation 2x2.

Reads ONLY the committed artifact role_recipe_2x2_results.json
(role_recipe_2x2.py seeds 42-51 @ n=3e3 / 42-46 @ n=1e5, extended to
42-51 @ n=1e5 by role_recipe_2x2_ext.py per the pre-stated sequential
plan), writes PNG (200 dpi) + PDF beside this script.

Figures:
  fig_52_interaction   retrieval accuracy per arm (per-seed dots, paired
                       G4-G1 deltas) -- the registered P-SCALE interaction
                       and the decision figure
  fig_52_regressions   parse accuracy (gain-insensitive at ceiling,
                       P-GAIN@3e3) and occupant gap (the crowding cost)
  fig_52_exposure      per-word retrieval vs realized training exposure,
                       by arm -- where the errors live

Run: python make_figures_52.py
"""
from __future__ import annotations

import json
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")

# Colorblind-safe pair (Okabe-Ito).
C_G1 = "#0072B2"   # blue: gain 1 (baseline recipe)
C_G4 = "#D55E00"   # vermillion: gain 4 (the morphology margin lever)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})

NS = (3000, 100000)


def load():
    with open(os.path.join(EXP, "role_recipe_2x2_results.json"),
              encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def jitter(n, width=0.05, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


def arm_values(cells, n, gain, key):
    vals = {}
    for cell_key, c in cells.items():
        cn, g, seed = cell_key.split("-")
        if int(cn) == n and g == f"G{int(gain)}" and c.get(key) is not None:
            vals[int(seed)] = c[key]
    return vals


# ---------------------------------------------------------------- fig 1
def fig_interaction(data):
    cells, analysis = data["cells"], data["analysis"]
    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(8.6, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})

    xs_n = {3000: 0, 100000: 1}
    for n in NS:
        for gain, c, dx in ((1.0, C_G1, -0.13), (4.0, C_G4, 0.13)):
            vals = arm_values(cells, n, gain, "ret_acc")
            ys = [vals[s] for s in sorted(vals)]
            x = xs_n[n] + dx
            ax.scatter(x + jitter(len(ys), seed=int(n) + int(gain)), ys,
                       s=40, color=c, alpha=0.8, zorder=3,
                       edgecolor="white", linewidth=0.6)
            m = st.mean(ys)
            ax.hlines(m, x - 0.1, x + 0.1, color=c, linewidth=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["n = 3,000", "n = 100,000"])
    ax.set_ylabel("role retrieval accuracy (top-1)")
    ax.set_title("gain 4 CROWDS at small n and buys nothing at scale")
    ax.scatter([], [], color=C_G1, label="gain 1 (recipe baseline)")
    ax.scatter([], [], color=C_G4, label="gain 4 (morphology lever)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # Paired per-seed deltas -- the registered interaction.
    for n in NS:
        g1 = arm_values(cells, n, 1.0, "ret_acc")
        g4 = arm_values(cells, n, 4.0, "ret_acc")
        seeds = sorted(set(g1) & set(g4))
        ds = [g4[s] - g1[s] for s in seeds]
        x = xs_n[n]
        axd.scatter(x + jitter(len(ds), seed=7 + int(n)), ds, s=40,
                    color="#555555", alpha=0.8, zorder=3,
                    edgecolor="white", linewidth=0.6)
        key = f"interaction_ret_delta_n{n}"
        m, ci = analysis[key]["mean"], analysis[key]["ci"]
        axd.errorbar([x], [m], yerr=[ci], color="black", capsize=4,
                     linewidth=2, zorder=4)
        axd.annotate(f"{m:+.3f}\n±{ci:.3f}", (x + 0.14, m), va="center",
                     fontsize=9)
    axd.axhline(0, color="#999999", linewidth=1, linestyle="--")
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["n = 3,000", "n = 100,000"])
    axd.set_xlim(-0.4, 1.6)
    axd.set_ylabel("paired (gain 4 − gain 1) delta")
    axd.set_title("the registered interaction")
    fig.suptitle("#52 recipe propagation: role_bind_gain × n on role retrieval",
                 fontsize=12, fontweight="bold", y=1.03)
    save(fig, "fig_52_interaction")


# ---------------------------------------------------------------- fig 2
def fig_regressions(data):
    cells = data["cells"]
    fig, (axp, axg) = plt.subplots(1, 2, figsize=(8.6, 3.6))

    xs_n = {3000: 0, 100000: 1}
    for ax, key, ylab, title in (
            (axp, "parse_acc", "parse accuracy (8 frames, both voices)",
             "parse is gain-INSENSITIVE at ceiling (P-GAIN)"),
            (axg, "gap_mean", "mean occupant gap",
             "the crowding cost is visible in the gap")):
        for n in NS:
            for gain, c, dx in ((1.0, C_G1, -0.13), (4.0, C_G4, 0.13)):
                vals = arm_values(cells, n, gain, key)
                ys = [vals[s] for s in sorted(vals)]
                x = xs_n[n] + dx
                ax.scatter(x + jitter(len(ys), seed=int(n) + int(gain)),
                           ys, s=40, color=c, alpha=0.8, zorder=3,
                           edgecolor="white", linewidth=0.6)
                ax.hlines(st.mean(ys), x - 0.1, x + 0.1, color=c,
                          linewidth=2.5, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["n = 3,000", "n = 100,000"])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    axp.set_ylim(0.0, 1.05)
    axp.scatter([], [], color=C_G1, label="gain 1")
    axp.scatter([], [], color=C_G4, label="gain 4")
    axp.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig_52_regressions")


# ---------------------------------------------------------------- fig 3
def fig_exposure(data):
    rows_by_cell = data["ret_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)

    for ax, n in zip(axes, NS):
        for gain, c in ((1.0, C_G1), (4.0, C_G4)):
            acc = defaultdict(list)
            for cell_key, rows in rows_by_cell.items():
                cn, g, _seed = cell_key.split("-")
                if int(cn) != n or g != f"G{int(gain)}":
                    continue
                for r in rows:
                    acc[r["freq"]].append(1.0 if r["ok"] else 0.0)
            fs = sorted(acc)
            ys = [st.mean(acc[f]) for f in fs]
            ns = [len(acc[f]) for f in fs]
            ax.plot(fs, ys, "-o", color=c, markersize=5,
                    label=f"gain {int(gain)}")
            for f, y, cnt in zip(fs, ys, ns):
                if y < 0.995:
                    ax.annotate(str(cnt), (f, y), textcoords="offset points",
                                xytext=(0, -12), fontsize=7, ha="center",
                                color=c)
        ax.set_title(f"n = {n:,}", fontsize=10)
        ax.set_xlabel("realized training exposure (word count)")
        ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("retrieval accuracy")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("where the errors live: retrieval by exposure stratum "
                 "(counts under sub-ceiling points)",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig_52_exposure")


if __name__ == "__main__":
    data = load()
    fig_interaction(data)
    fig_regressions(data)
    fig_exposure(data)
