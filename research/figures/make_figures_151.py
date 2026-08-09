"""Publication figures for the #151 arc: dead fiber -> fix -> ceiling.

Reads ONLY committed artifacts (each panel names its source JSON and the
commit that produced it), writes PNG (200 dpi) + PDF beside this script.

Figures:
  fig_151_gate_before_after   per-seed min(SG,PL) on the E>=2 exam,
                              blocked gate (cd52c1f) vs passing gate
                              (145288c), bar at 0.75
  fig_151_exposure_law        PL accuracy vs training exposure,
                              pre-fix (1288fca, 12 seeds) vs post-fix
                              (childes_phase1_recipe, 10 seeds)
  fig_151_dead_fiber          seed 45 forensics: per-word own-drive
                              pre/post + fiber geometry vs source

Run: python make_figures_151.py
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
C_PRE = "#D55E00"    # vermillion: pre-fix / defect
C_POST = "#0072B2"   # blue: post-fix / healed
C_BAR = "#555555"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load(name):
    with open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def jitter(n, width=0.07, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


# ---------------------------------------------------------------- fig 1
def fig_gate():
    blocked = load("adoption_gate_blocked_results.json")["analysis"]
    passing = load("adoption_gate_results.json")["analysis"]
    pre = blocked["R1_per_seed_min_e2p"]
    post = passing["R1_per_seed_min_e2p"]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for x, ys, c, lbl in ((0, pre, C_PRE, "pre-fix engine\n(gate BLOCKED)"),
                          (1, post, C_POST, "fixed engine\n(gate PASSES)")):
        xs = x + jitter(len(ys), seed=3 + x)
        ax.scatter(xs, ys, s=42, color=c, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.6)
        m = st.mean(ys)
        ax.hlines(m, x - 0.16, x + 0.16, color=c, linewidth=2.5, zorder=4)
        ax.annotate(f"mean {m:.3f}", (x + 0.2, m), va="center",
                    fontsize=9, color=c)
    # The seed the distribution caught.
    i45 = int(np.argmin(pre))
    ax.annotate("seed 45: dead fiber\n(own-drive 0.0 on 45/46 words)",
                (0 + 0.02, pre[i45]), xytext=(0.28, 0.16), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=C_PRE),
                color=C_PRE)
    ax.axhline(0.75, color=C_BAR, linestyle="--", linewidth=1)
    ax.annotate("registered bar 0.75", (1.32, 0.755), fontsize=8.5,
                color=C_BAR)
    ax.set_xlim(-0.45, 1.9)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre-fix engine\n(gate blocked, cd52c1f)",
                        "fixed engine\n(gate passes, 145288c)"], fontsize=9)
    ax.set_ylabel("min(SG, PL) accuracy, E≥2 exam")
    ax.set_title("Adoption gate, per seed (n=10): Brown corpus, n=10⁵\n"
                 "one engine fix takes every seed to ceiling")
    save(fig, "fig_151_gate_before_after")


# ---------------------------------------------------------------- fig 2
def fig_exposure():
    pre = load("per_seed_pl_attribution_results.json")["cells"]
    bins = [("1", lambda e: e == 1), ("2", lambda e: e == 2),
            ("3–5", lambda e: 3 <= e <= 5), ("≥6", lambda e: e >= 6)]

    def per_seed_binned(rows_by_seed):
        out = {b: [] for b, _ in bins}
        for rows in rows_by_seed:
            for b, f in bins:
                xs = [r["ok"] for r in rows if f(r["exposure"] or 0)]
                if xs:
                    out[b].append(st.mean(xs))
        return out

    pre_rows = [c["pl_rows"] for c in pre.values()]
    curves = [("pre-fix engine (12 seeds, 1288fca)", C_PRE,
               per_seed_binned(pre_rows))]
    try:
        post = load("childes_phase1_recipe_results.json")["cells"]
        post_rows = [[r for r in c["rows"] if r["label"] == "PL"]
                     for c in post.values()]
        curves.append(("fixed engine (10 seeds)", C_POST,
                       per_seed_binned(post_rows)))
    except FileNotFoundError:
        print("  (phase1 recipe results absent -- pre-fix curve only)")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(len(bins))
    for lbl, c, data in curves:
        means = [st.mean(data[b]) for b, _ in bins]
        cis = [(2.2 * st.stdev(data[b]) / len(data[b]) ** 0.5
                if len(data[b]) > 2 else 0) for b, _ in bins]
        ax.errorbar(x, means, yerr=cis, color=c, marker="o", capsize=3,
                    linewidth=1.8, label=lbl)
    ax.axhline(0.5, color=C_BAR, linestyle=":", linewidth=1)
    ax.annotate("chance", (len(bins) - 0.85, 0.515), fontsize=8.5,
                color=C_BAR)
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _ in bins])
    ax.set_xlabel("training exposures (episodes) per plural form")
    ax.set_ylabel("PL recall accuracy")
    ax.set_ylim(0.35, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("The exposure law, re-priced by the fix\n"
                 "the E=2 step was substantially dropped Hebbian writes")
    save(fig, "fig_151_exposure_law")


# ---------------------------------------------------------------- fig 3
def fig_dead_fiber():
    pre = load("dead_fiber_hunt_prefix_results.json")["cells"]["45"]
    post = load("dead_fiber_hunt_healed_results.json")["cells"]["45"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    # (a) per-word own-drive, words sorted by pre-fix value.
    words = sorted((r["form"] for r in pre["words"]),
                   key=lambda w: next(r["own"] or 0 for r in pre["words"]
                                      if r["form"] == w))
    pre_by = {r["form"]: r["own"] or 0 for r in pre["words"]}
    post_by = {r["form"]: r["own"] or 0 for r in post["words"]}
    xs = np.arange(len(words))
    ax1.scatter(xs, [pre_by[w] for w in words], s=26, color=C_PRE,
                label="pre-fix: 45/46 EXACTLY 0.0", zorder=3)
    ax1.scatter(xs, [post_by[w] for w in words], s=26, color=C_POST,
                marker="^", label="fixed engine", zorder=3)
    served = max(pre_by, key=pre_by.get)
    ax1.annotate(f"'{served}'\n(first PL word trained)",
                 (words.index(served), pre_by[served]),
                 xytext=(len(words) * 0.45, pre_by[served] * 0.82),
                 fontsize=8.5, color=C_PRE,
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=C_PRE))
    ax1.set_xlabel("the 46 trained plural forms (sorted)")
    ax1.set_ylabel("own-drive at recall (NUMBER_PL)")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax1.set_title("(a) seed 45: the dead fiber, word by word")

    # (b) fiber geometry, log scale.
    cats = ["fiber rows", "fiber extent\n(columns)"]
    pre_v = [pre["pl_fiber"]["shape"][0], pre["pl_fiber"]["extent"]]
    post_v = [post["pl_fiber"]["shape"][0], post["pl_fiber"]["extent"]]
    src = pre["pl_fiber"]["materialized_src"]
    x = np.arange(len(cats))
    ax2.bar(x - 0.18, pre_v, width=0.34, color=C_PRE, label="pre-fix")
    ax2.bar(x + 0.18, post_v, width=0.34, color=C_POST, label="fixed")
    ax2.axhline(src, color=C_BAR, linestyle="--", linewidth=1)
    ax2.annotate(f"NOUN_CORE materialized = {src:,}", (1.4, src * 0.42),
                 ha="right", fontsize=8.5, color=C_BAR)
    for xi, v in zip(x - 0.18, pre_v):
        ax2.annotate(f"{v:,}", (xi, v * 1.3), ha="center", fontsize=8.5,
                     color=C_PRE)
    for xi, v in zip(x + 0.18, post_v):
        ax2.annotate(f"{v:,}", (xi, v * 1.3), ha="center", fontsize=8.5,
                     color=C_POST)
    ax2.set_yscale("log")
    ax2.set_ylim(10, src * 8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("size (log scale)")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2.set_title("(b) the growth ratchet: frozen at episode 3")

    fig.suptitle("NOUN_CORE→NUMBER_PL, seed 45, Brown n=10⁵ "
                 "(dead_fiber_hunt, registered 4ce17d6)", y=1.04,
                 fontsize=10, fontweight="bold")
    save(fig, "fig_151_dead_fiber")


# ---------------------------------------------------------------- fig 4
def fig_residual():
    """P3: what still fails at ceiling -- E=1 x collision load."""
    a = load("childes_phase1_recipe_results.json")["analysis"]
    f, p = a["P3_fail_shared_weighted"], a["P3_pass_shared_weighted"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    rho = a["P2_rho"]
    per = [r for r in a["P2_rho_per_seed"] if r is not None]
    ax1.scatter(jitter(len(per), 0.05, seed=7), per, s=42, color=C_POST,
                alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    ax1.hlines(rho["mean"], -0.16, 0.16, color=C_POST, linewidth=2.5)
    ax1.errorbar([0], [rho["mean"]], yerr=[rho["ci"]], color=C_POST,
                 capsize=4, linewidth=1.4)
    ax1.axhline(0, color=C_BAR, linestyle=":", linewidth=1)
    ax1.annotate(f"mean {rho['mean']:.2f} ± {rho['ci']:.2f}\n"
                 f"(pre-fix read 0.61)", (0.22, rho["mean"]), va="center",
                 fontsize=9, color=C_POST)
    ax1.annotate("undefined for the one\nALL-CORRECT seed (49)",
                 (-0.42, 0.03), fontsize=8, color=C_BAR)
    ax1.set_xlim(-0.5, 0.9)
    ax1.set_xticks([])
    ax1.set_ylabel("Spearman ρ(exposure, correct), PL forms")
    ax1.set_title("(a) exposure law at ceiling:\ntransfers, attenuated")

    for x, d, c, lbl in ((0, p, C_POST, f"passing probes\n(n={p['n']})"),
                         (1, f, C_PRE, f"failing probes\n(n={f['n']})")):
        ax2.bar(x, d["mean"], width=0.55, color=c)
        ax2.errorbar([x], [d["mean"]], yerr=[d["ci"]], color="black",
                     capsize=4, linewidth=1.2)
        ax2.annotate(f"{d['mean']:.1f} ± {d['ci']:.1f}", (x, d["mean"] + 1),
                     ha="center", fontsize=9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"passing probes\n(n={p['n']})",
                         f"failing probes\n(n={f['n']})"], fontsize=9)
    ax2.set_ylabel("SG-shared core rows (weighted)")
    ax2.set_ylim(0, max(f["mean"], p["mean"]) * 1.35)
    ax2.set_title("(b) every failure is E=1, and failures\ncarry 64% more"
                  " collision load")
    fig.suptitle("The residual at ceiling (childes_phase1_recipe, "
                 "10 seeds, Brown n=10⁵)", y=1.06, fontsize=10,
                 fontweight="bold")
    save(fig, "fig_151_residual")


if __name__ == "__main__":
    fig_gate()
    fig_exposure()
    fig_dead_fiber()
    fig_residual()
