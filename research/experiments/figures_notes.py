"""Figures for the research notes, from committed results.

    python research/experiments/figures_notes.py

Writes four PNGs to research/notes/figures/:

  memory_recall_vs_M.png        half-cue recall against stored items, Hebbian
                                control vs refracted vs refracted + gated
                                (n = 4000, k = 60, 20 brains)
  memory_ceiling_vs_nk.png      the ceiling M* against n/k for both arms, with
                                the 0.40 (n/k)^2 line
  organ_soft_rate.png           soft-transition rate and derailments against
                                presentations, and against refraction strength
                                (S5, 100 organs per point)
  organ_arc_drift.png           overlap of the test-time arc with the arc at
                                each training presentation, 15 vs 30
                                presentations (Z60, 4 brains)

Numbers for the ceiling figure are the resolved M* values recorded in
PREREG_refraction_memory.md (Result, Amendments 2 and 4); every other panel
reads a results file.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _results import results_path  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notes", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _load(line, name):
    with open(results_path(line, name)) as fh:
        return json.load(fh)


def _curve(res, key="B/4000"):
    cells = res[key]
    ms = sorted(int(m) for m in cells)
    mean = [float(np.mean(cells[str(m)]["rank1"])) for m in ms]
    lo = [float(np.percentile(cells[str(m)]["rank1"], 10)) for m in ms]
    hi = [float(np.percentile(cells[str(m)]["rank1"], 90)) for m in ms]
    return ms, mean, lo, hi


def memory_recall_vs_M():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, label, color in (
            ("capacity_scaling_results_figure_ctl.json", "Hebbian control", "#888888"),
            ("capacity_scaling_results_figure_ref.json", "refracted, 0.5 beta, masked readout", "#1f77b4"),
            ("capacity_scaling_results_amend5_ref_gated_refine.json", "refracted + convergence-gated write", "#d62728")):
        try:
            res = _load("memory", name)
        except FileNotFoundError:
            continue
        ms, mean, lo, hi = _curve(res)
        ax.plot(ms, mean, "-o", ms=3, color=color, label=label)
        ax.fill_between(ms, lo, hi, color=color, alpha=0.15, lw=0)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("stored assemblies M")
    ax.set_ylabel("half-cue recall, rank-1 (20 brains, 10th to 90th pct)")
    ax.set_title("n = 4000, k = 60, p = 0.5: recall against load")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "memory_recall_vs_M.png"), dpi=150)
    plt.close(fig)


def memory_ceiling_vs_nk():
    # resolved M* from PREREG_refraction_memory.md; (n/k, cell, M*)
    ref = [(33, "(2000, 60)", 431), (33, "(4000, 120)", 383),
           (67, "(4000, 60)", 1978), (67, "(2000, 30)", 1589), (67, "(8000, 120)", 2230),
           (133, "(8000, 60)", 6995), (133, "(4000, 30)", 4749)]
    ctl = [(33, "(2000, 60)", 11.3), (33, "(4000, 120)", 11.3),
           (67, "(4000, 60)", 83.4), (67, "(2000, 30)", 64.3), (67, "(8000, 120)", 88.9),
           (133, "(8000, 60)", 306.8), (133, "(4000, 30)", 262.8)]
    out_of_regime = {"(2000, 30)", "(4000, 30)"}
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.array([20, 200])
    ax.plot(x, 0.40 * x ** 2, "-", color="#1f77b4", lw=1, alpha=0.6, label="0.40 (n/k)²")
    ax.plot(x, 0.017 * x ** 2, "-", color="#888888", lw=1, alpha=0.6, label="0.017 (n/k)²")
    for pts, color, label in ((ref, "#1f77b4", "refracted"), (ctl, "#888888", "Hebbian control")):
        for nk, cell, m in pts:
            filled = cell not in out_of_regime
            ax.plot(nk, m, "o", color=color, mfc=color if filled else "white", ms=6)
        ax.plot([], [], "o", color=color, label=label)
    ax.plot([], [], "o", color="k", mfc="white", label="out of regime (k p < 3 ln n)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n / k")
    ax.set_ylabel("ceiling M* (stored assemblies)")
    ax.set_title("capacity against n/k, 20 brains per cell")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "memory_ceiling_vs_nk.png"), dpi=150)
    plt.close(fig)


def _census_stats(name):
    rows = _load("sequence", name)
    bad = sum(r["n_soft"] + r["n_hard"] for r in rows)
    pairs = sum(r["n_pairs"] for r in rows)
    derail = sum(1 for r in rows if r["first_bad"] < 500)
    return 100.0 * bad / pairs, derail, len(rows)


def organ_soft_rate():
    pres = [(8, "amend5_p8"), (15, "amend5_p15"), (20, "amend7_p20"), (24, "amend7_p24"),
            (28, "amend8_p28"), (30, "amend5_p30")]
    stren = [(0.0, "amend6_s0.0"), (0.05, "amend6_s0.05"), (0.08, "amend6_s0.08"), (0.10, "amend5_p15")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, series, xlabel, title in (
            (axes[0], pres, "presentations per transition (strength = beta)", "the presentation window"),
            (axes[1], stren, "refraction strength (15 presentations)", "strength")):
        xs, rate, der = [], [], []
        for xval, tag in series:
            try:
                r, d, n = _census_stats(f"seq_s5_soft_census_results_hashed_{tag}.json")
            except FileNotFoundError:
                continue
            xs.append(xval); rate.append(max(r, 1e-3)); der.append(d)
            if r == 0.0:
                ax.annotate("0", (xval, 1e-3), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8, color="#1f77b4")
        ax.plot(xs, rate, "-o", color="#1f77b4", label="soft pairs, % of transitions")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 200)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("soft transitions (%; 0 marked)", color="#1f77b4")
        ax2 = ax.twinx()
        ax2.bar(xs, der, width=(0.6 if ax is axes[0] else 0.006), color="#d62728", alpha=0.35, label="words derailing / 100")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("words derailing of 100", color="#d62728")
        ax2.spines["top"].set_visible(False)
        ax.set_title(f"S5, 100 organs per point: {title}")
        if ax is axes[1]:
            ax.axvline(0.1, color="k", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "organ_soft_rate.png"), dpi=150)
    plt.close(fig)


def organ_arc_drift():
    try:
        d = _load("sequence", "seq_s5_arc_drift.json")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for P, color in (("15", "#1f77b4"), ("30", "#d62728")):
        if P not in d:
            continue
        ov = d[P]["overlap_by_presentation"]
        ax.plot(range(1, len(ov) + 1), ov, "-o", ms=3, color=color,
                label=f"{P} presentations: weakest block drive min {d[P]['weak_min']:.0f}, "
                      f"best outsider max {d[P]['best_max']:.0f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training presentation")
    ax.set_ylabel("overlap with the test-time arc")
    ax.set_title("Z60, 4 brains: the arc relocates past the clip")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "organ_arc_drift.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    memory_recall_vs_M()
    memory_ceiling_vs_nk()
    organ_soft_rate()
    organ_arc_drift()
    print("wrote", FIG)
