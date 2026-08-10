"""Publication figures for #28: the N400's saturation, measured precisely.

Reads ONLY the committed artifact n400_landing_2x2_results.json
(n400_landing_2x2.py, 10 seeds, 4 corpus-derived arms, both channels
paired per parser), writes PNG (200 dpi) + PDF beside this script.

Figures:
  fig_28_aucs        per-seed AUC on the three contrasts, energy vs
                     landing channel, chance line -- the verdict figure
  fig_28_saturation  per-item surprise distributions by arm and channel
                     -- where "saturated" becomes a visible fact

Run: python make_figures_28.py
"""
from __future__ import annotations

import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")

# Colorblind-safe (Okabe-Ito).
C_ENERGY = "#CC79A7"   # magenta: energy channel (HOW MUCH)
C_LAND = "#009E73"     # green: landing channel (WHERE)
ARM_COLORS = {"attested": "#0072B2", "unattested_noun": "#E69F00",
              "verb_violation": "#D55E00", "novel": "#555555"}
ARM_LABELS = {"attested": "attested\ncontinuation",
              "unattested_noun": "unattested\nnoun",
              "verb_violation": "verb in\nnoun slot",
              "novel": "novel\n(holdout)"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})

CONTRASTS = ("sem", "syn", "nov")
CONTRAST_LABELS = {
    "sem": "SEMANTIC\n(unattested vs\nattested)",
    "syn": "SYNTACTIC\n(verb vs\nattested)",
    "nov": "NOVELTY\n(novel vs\nattested)",
}


def load():
    with open(os.path.join(EXP, "n400_landing_2x2_results.json"),
              encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def jitter(n, width=0.06, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


def fig_aucs(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = {c: i for i, c in enumerate(CONTRASTS)}
    for label, key, c, dx in (("energy (shipped N400)", "energy",
                               C_ENERGY, -0.14),
                              ("landing (WHERE)", "landing", C_LAND, 0.14)):
        for contrast in CONTRASTS:
            a = data["analysis"].get(f"{contrast}_{key}")
            if not a:
                continue
            ys = a["per_seed"]
            x = xs[contrast] + dx
            ax.scatter(x + jitter(len(ys), seed=hash(key) % 97), ys, s=40,
                       color=c, alpha=0.8, zorder=3, edgecolor="white",
                       linewidth=0.6)
            ax.hlines(a["mean"], x - 0.1, x + 0.1, color=c, linewidth=2.5,
                      zorder=4)
        ax.scatter([], [], color=c, label=label)
    ax.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
    ax.annotate("chance", (2.42, 0.505), fontsize=8, color="#777777")
    ax.axhline(0.75, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.annotate("bar 0.75", (2.42, 0.755), fontsize=8, color="#999999")
    ax.set_xticks(range(len(CONTRASTS)))
    ax.set_xticklabels([CONTRAST_LABELS[c] for c in CONTRASTS], fontsize=9)
    ax.set_ylabel("AUC (violation arm reads higher surprise)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("#28: neither N400 channel sees expectancy "
                 "(10 seeds, paired on the same parsers)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig_28_aucs")


def fig_saturation(data):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    arms = ("attested", "unattested_noun", "verb_violation", "novel")
    for ax, ch, cname in ((axes[0], "energy", "energy channel"),
                          (axes[1], "landing", "landing channel")):
        for i, arm in enumerate(arms):
            vals = [row[ch] for rec in data["seeds"].values()
                    for row in rec["rows"]
                    if row["arm"] == arm and ch in row]
            if not vals:
                continue
            c = ARM_COLORS[arm]
            ax.scatter(i + jitter(len(vals), width=0.16, seed=i), vals,
                       s=12, color=c, alpha=0.4, zorder=3)
            m = st.mean(vals)
            ax.hlines(m, i - 0.25, i + 0.25, color=c, linewidth=2.5,
                      zorder=4)
            ax.annotate(f"{m:.3f}", (i, min(vals) - 0.04), ha="center",
                        fontsize=8, color=c)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=8)
        ax.set_title(cname, fontsize=10)
        ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("N400 surprise (1 = maximum)")
    fig.suptitle("the saturation, per item: every arm reads near-maximum "
                 "surprise on both channels", fontsize=11,
                 fontweight="bold", y=1.03)
    save(fig, "fig_28_saturation")


if __name__ == "__main__":
    data = load()
    fig_aucs(data)
    fig_saturation(data)
