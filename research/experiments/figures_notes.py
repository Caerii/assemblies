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


def organ_order10_window():
    """Order-10 prediction against presentation, three arms, with the clip
    edge c* = ln(w_max) / ln(1 + beta)."""
    try:
        d = _load("sequence", "seq_tm_high_order_results.json")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    styles = {("induced", 0.0): ("#1f77b4", "induced state (registered transducer)"),
              ("copy", 0.0): ("#2ca02c", "state = previous arc"),
              ("copy", 4.0): ("#d62728", "state = previous arc, predicted neurons win (g = 4)")}
    for row in d["rows"]:
        if row["set"] != "III":
            continue
        key = (row["mode"], float(row["gain"]))
        if key not in styles:
            continue
        curve = [float(np.mean(pp["acc_ambiguous"])) for pp in row["per_presentation"]]
        ax.plot(range(1, len(curve) + 1), curve, "-o", ms=3, color=styles[key][0], label=styles[key][1])
    cstar = np.log(20.0) / np.log(1.1)
    ax.axvline(cstar, color="k", lw=0.8, ls=":")
    ax.text(cstar + 0.3, 0.08, f"clip edge c* = {cstar:.1f}", fontsize=8)
    ax.axhline(0.5, color="#888888", lw=0.6, ls="--")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("training presentation of the two sequences")
    ax.set_ylabel("rank-1 at the order-10 positions (20 brains)")
    ax.set_title("two 12-word sequences sharing their middle ten: exact until the clip")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "organ_order10_window.png"), dpi=150)
    plt.close(fig)


def organ_strength_pinned():
    """Across-symbol and across-state arc overlap against refraction
    strength: two collapses, one on each side of beta."""
    pts = []
    for st, tag in ((0.0, "amend6_s0.0"), (0.05, "amend6_s0.05"), (0.08, "amend6_s0.08"),
                    (0.10, "amend6_s0.1")):
        try:
            rows = _load("sequence", f"seq_s5_soft_census_results_hashed_{tag}.json")
        except FileNotFoundError:
            continue
        pts.append((st, float(np.mean([r["across_symbol"] for r in rows])),
                    float(np.mean([r["across_state"] for r in rows])),
                    100.0 * sum(r["n_soft"] + r["n_hard"] for r in rows) / sum(r["n_pairs"] for r in rows)))
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = [q[0] for q in pts]
    ax.plot(xs, [q[1] for q in pts], "-o", color="#d62728", label="across-symbol overlap (same state, other generator)")
    ax.plot(xs, [q[2] for q in pts], "-s", color="#1f77b4", label="across-state overlap (same generator, other state)")
    ax.axhline(0.15, color="#888888", lw=0.6, ls="--")
    ax.text(0.001, 0.17, "P-CONJ bar 0.15", fontsize=7, color="#888888")
    ax.axvline(0.10, color="k", lw=0.8, ls=":")
    ax.text(0.091, 0.9, "beta", fontsize=8)
    ax.set_xlabel("refraction strength on the arc (beta = 0.1)")
    ax.set_ylabel("mean arc overlap (S5, 100 organs)")
    ax.set_title("the conjunction collapses on either side of beta")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "organ_strength_pinned.png"), dpi=150)
    plt.close(fig)


def sampler_artifacts():
    """Sampled numpy engine against materialized / hashed: the horizon,
    the soft-transition rate, the load window's lower edge."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    # (a) horizon: first error per seed at p = 0.3
    ax = axes[0]
    try:
        npy = _load("sequence", "seq_a1_horizon_results.json")
        mat = _load("sequence", "seq_a1_horizon_materialized_check.json")
        hsh = _load("sequence", "seq_a1_horizon_results_hashed.json")["rows"]
    except FileNotFoundError:
        npy = mat = hsh = []
    def fe(rows, p=0.3, key="materialized", want=None):
        out = []
        for r in rows:
            if r.get("p") != p:
                continue
            if want is not None and r.get(key) != want:
                continue
            out.append(2001 if r["first_error"] is None else r["first_error"])
        return out
    groups = [("numpy,\nsampled", fe(npy) if npy else []),
              ("numpy,\nmaterialized", fe(mat, want=True) if mat else []),
              ("hashed,\n20 brains", fe(hsh) if hsh else [])]
    for i, (lab, vals) in enumerate(groups):
        jit = np.linspace(-0.15, 0.15, max(len(vals), 1))
        ax.scatter(i + jit, vals, s=14, color=("#d62728" if i == 0 else "#1f77b4"))
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("first error (2001 = none in 2000 digits)")
    ax.set_title("A1 horizon, p = 0.3", fontsize=9)
    ax.set_ylim(0, 2100)
    # (b) soft rate and derailments, E7 sampled vs Addendum 3
    ax = axes[1]
    try:
        e7 = _load("sequence", "seq_s5_soft_census_results.json")
        a3 = _load("sequence", "seq_s5_soft_census_results_hashed.json")
        pairs_e7 = {"Z60": 120, "A4xZ5": 120, "A5": 120, "S5": 240}
        rate_e7 = 100.0 * sum(r["n_soft"] + r["n_hard"] for r in e7) / sum(pairs_e7[r["group"]] for r in e7)
        rate_a3 = 100.0 * sum(r["n_soft"] + r["n_hard"] for r in a3) / sum(r["n_pairs"] for r in a3)
        der_e7 = sum(1 for r in e7 if r["first_bad"] < 500); der_a3 = sum(1 for r in a3 if r["first_bad"] < 500)
        ax.bar([0, 1], [rate_e7, rate_a3], color=["#d62728", "#1f77b4"], width=0.5)
        for i, (v, dd) in enumerate(((rate_e7, der_e7), (rate_a3, der_a3))):
            ax.text(i, v + 0.01, f"{v:.3f}%\n{dd}/40 words derail", ha="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["sampled", "explicit"], fontsize=8)
        ax.set_ylabel("soft transitions (%)")
        ax.set_ylim(0, max(rate_e7, rate_a3) * 1.6)
        ax.set_title("S5 census, 40 organs", fontsize=9)
    except FileNotFoundError:
        pass
    # (c) the load window
    ax = axes[2]
    try:
        sam = _load("sequence", "seq_a2_refraction_load_results.json")
        mat = _load("sequence", "seq_a2_refraction_load_results_materialized.json")
        for rows, color, lab in ((sam, "#d62728", "sampled"), (mat, "#1f77b4", "materialized")):
            r1 = [r for r in rows if r["arm"].startswith("single")]
            ax.plot([r["load"] for r in r1], [r["correct"] for r in r1], "-o", color=color, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("arc load M k / n")
        ax.set_ylabel("correct of 10 seeds")
        ax.set_title("A2, single-mood arm", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    except FileNotFoundError:
        pass
    fig.suptitle("what the sampler added: a false horizon, a seven-fold soft rate, a false lower edge", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sampler_artifacts.png"), dpi=150)
    plt.close(fig)


def memory_gate_U():
    """Recall and the fraction of items converging inside the round budget,
    against stored items: the ceiling sits where items stop converging."""
    try:
        res = _load("memory", "capacity_scaling_results_amend5_ref_gated.json")
    except FileNotFoundError:
        return
    cells = res["B/4000"]
    ms = sorted(int(m) for m in cells)
    rank1 = [float(np.mean(cells[str(m)]["rank1"])) for m in ms]
    conv = [float(np.mean(cells[str(m)]["converged"])) for m in ms]
    rounds = [float(np.mean(cells[str(m)]["rounds_used"])) for m in ms]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ms, rank1, "-o", ms=3, color="#d62728", label="half-cue recall, rank-1")
    ax.plot(ms, conv, "-s", ms=3, color="#1f77b4", label="fraction of items converging inside 8 rounds")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("stored assemblies M (n = 4000, k = 60, gated write)")
    ax.set_ylabel("fraction")
    ax2 = ax.twinx()
    ax2.plot(ms, rounds, "--", color="#888888", lw=1, label="rounds used per item")
    ax2.set_ylabel("rounds used per item", color="#888888")
    ax2.set_ylim(4, 8.5)
    ax2.spines["top"].set_visible(False)
    ax.set_title("the ceiling is where items stop converging")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "memory_gate_U.png"), dpi=150)
    plt.close(fig)


def throughput():
    """Brain-steps per second, numpy engine against the hashed substrate,
    from the run logs named in the note."""
    # A1 horizon: numpy materialized 2000 steps in ~3.5 s per brain
    # (seq_a1_horizon_materialized_check.json 'secs'); hashed 20 brains x
    # 2000 steps in 4-5 s per p (GATE-3 log).
    # Capacity: numpy mirror ~14 s per brain for 512 items at n = 2000
    # (refraction_memory_numpy log); hashed 20 brains x 16,384 items x 2
    # cells in 299 s at n = 4000 / 8000 (Amendment 4 log).
    try:
        mat = _load("sequence", "seq_a1_horizon_materialized_check.json")
        secs = np.mean([r["secs"] for r in mat if r["materialized"]])
    except FileNotFoundError:
        secs = 3.5
    pairs = [("A1 horizon\n(brain-steps / s)", 2000.0 / secs, 20 * 2000 / 4.5),
             ("capacity grid\n(brain-items / s)", 512.0 / 14.0, 2 * 20 * 16384 / 299.0)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(pairs))
    ax.bar(x - 0.18, [p[1] for p in pairs], width=0.36, color="#888888", label="numpy engine, one brain")
    ax.bar(x + 0.18, [p[2] for p in pairs], width=0.36, color="#1f77b4", label="hashed substrate, 20 brains per launch")
    for i, p_ in enumerate(pairs):
        ax.text(i - 0.18, p_[1] * 1.15, f"{p_[1]:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.18, p_[2] * 1.15, f"{p_[2]:.0f}  ({p_[2] / p_[1]:.0f}x)", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs], fontsize=9)
    ax.set_ylabel("rate (log)")
    ax.set_title("one RTX 3080: the width that makes distributions cheap")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "throughput.png"), dpi=150)
    plt.close(fig)


def organ_clip_relocation():
    """Lost vs kept arc neurons at 30 presentations by stimulus base, with
    the clip threshold; from seq_s5_arc_clip.json (a one-minute GPU run)."""
    try:
        d = _load("sequence", "seq_s5_arc_clip.json")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bins = np.arange(min(d["kept_base"] + d["lost_base"]) - 0.5, max(d["kept_base"] + d["lost_base"]) + 1.5, 1)
    ax.hist(d["kept_base"], bins=bins, color="#1f77b4", alpha=0.6, label=f"kept ({len(d['kept_base'])})")
    ax.hist(d["lost_base"], bins=bins, color="#d62728", alpha=0.6, label=f"lost ({len(d['lost_base'])})")
    ax.axvline(d["clip_base"], color="k", lw=0.8, ls=":")
    ax.text(d["clip_base"] + 0.2, ax.get_ylim()[1] * 0.9, f"clip at 30 presentations\n(base >= {d['clip_base']:.1f})", fontsize=8)
    ax.set_xlabel("stimulus base: present rows from the symbol (of 70)")
    ax.set_ylabel("arc neurons (Z60, 4 brains, all pairs)")
    ax.set_title("relocation is the clip: the best-connected neurons leave first")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "organ_clip_relocation.png"), dpi=150)
    plt.close(fig)


def parity_gates():
    """Worst relative drive error per parity gate, from the tests' dump."""
    try:
        with open(os.path.join(os.path.dirname(results_path("substrate", "x")), "parity_errors.json")) as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return
    names = list(d.keys()); vals = [max(d[n], 1e-9) for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(range(len(names)), vals, color="#1f77b4")
    ax.axvline(5e-6, color="k", lw=0.8, ls=":"); ax.text(5e-6 * 1.2, len(names) - 0.6, "gate 5e-6", fontsize=8)
    ax.set_xscale("log"); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("worst relative error against the numpy engine's drive")
    ax.set_title("every hashed unit reproduces the numpy engine's drive")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parity_gates.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    memory_recall_vs_M()
    memory_ceiling_vs_nk()
    organ_soft_rate()
    organ_arc_drift()
    organ_order10_window()
    organ_strength_pinned()
    sampler_artifacts()
    memory_gate_U()
    throughput()
    organ_clip_relocation()
    parity_gates()
    print("wrote", FIG)
