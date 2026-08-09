"""Publication figures for #121: the WHERE-channel ERP role readout.

Reads ONLY the committed artifact erp_pathway_vs_area_binding_results.json
(erp_pathway_vs_area_binding.py, 10 seeds, both channels paired per
parser), writes PNG (200 dpi) + PDF beside this script.

Figures:
  fig_121_dissociation   per-seed AUC on the three contrasts, binding vs
                         drive channel -- the decision figure
  fig_121_deficits       per-item binding-deficit distributions by arm

Run: python make_figures_121.py
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

# Colorblind-safe pair (Okabe-Ito).
C_BIND = "#009E73"   # green: binding channel (WHERE)
C_DRIVE = "#CC79A7"  # magenta: drive channel (HOW MUCH)
ARM_COLORS = {"patient_trained": "#0072B2", "agent_only": "#E69F00",
              "verb_object": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})

CONTRASTS = ("pathway", "shipped", "area")


def load():
    with open(os.path.join(EXP, "erp_pathway_vs_area_binding_results.json"),
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


def fig_dissociation(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = {c: i for i, c in enumerate(CONTRASTS)}
    for label, key, c, dx in (("binding (WHERE)", "BINDING", C_BIND, -0.14),
                              ("drive (HOW MUCH)", "drive", C_DRIVE, 0.14)):
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
    ax.set_xticklabels([
        "PATHWAY only\n(agent_only vs\npatient_trained)",
        "SHIPPED\n(verb_object vs\npatient_trained)",
        "AREA only\n(verb_object vs\nagent_only)",
    ], fontsize=9)
    ax.set_ylabel("AUC (violation arm reads higher deficit)")
    ax.set_ylim(0.25, 1.05)
    ax.set_title("#121: what each ERP channel can see (10 seeds, paired "
                 "on the same parsers)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "fig_121_dissociation")


def fig_deficits(data):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    arms = ("patient_trained", "agent_only", "verb_object")
    for i, arm in enumerate(arms):
        vals = []
        for rec in data["seeds"].values():
            for row in rec.get("binding_rows", {}).get(arm, []):
                if "deficit" in row:
                    vals.append(row["deficit"])
        c = ARM_COLORS[arm]
        ax.scatter(i + jitter(len(vals), width=0.16, seed=i), vals, s=14,
                   color=c, alpha=0.45, zorder=3)
        m = st.mean(vals)
        ax.hlines(m, i - 0.25, i + 0.25, color=c, linewidth=2.5, zorder=4)
        ax.annotate(f"{m:.3f}", (i + 0.28, m), va="center", fontsize=9,
                    color=c)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["patient-trained\nnoun", "agent-only\nnoun",
                        "verb in\nobject slot"], fontsize=9)
    ax.set_ylabel("binding deficit  (1 − best landing overlap)")
    ax.set_title("per-item binding deficits: all seeds pooled "
                 "(100 items/arm)")
    save(fig, "fig_121_deficits")


if __name__ == "__main__":
    data = load()
    fig_dissociation(data)
    fig_deficits(data)
