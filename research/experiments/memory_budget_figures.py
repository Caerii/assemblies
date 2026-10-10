"""Figures and interval estimates for the sequence-area budget (paper P8), built
from committed run records only.

Every registered crossing rho_q of Amendments 37 to 43 gets a percentile
bootstrap interval over brains: the brains of a record are resampled with
replacement, the reliability at every ladder point is recomputed from the
resampled brains, and rho_q is re-read (log-interpolated, as the
registrations read it). Exploratory probes (the tau sweep, the links between
areas) are drawn from their committed logs under
research/notes/memory/probes/2026-10-08/ and marked as exploratory in every
figure.

    python research/experiments/memory_budget_figures.py

Writes research/papers/drafts/sequence_budget/figures/*.png, crossings.json and
crossings_table.tex there. Deterministic (fixed bootstrap seed).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

RUNS = os.path.join(ROOT, "research", "results", "runs")
PROBES = os.path.join(ROOT, "research", "notes", "memory", "probes", "2026-10-08")
OUT = os.path.join(ROOT, "research", "papers", "drafts", "sequence_budget", "figures")
RECORDS = {
    "A37": "memory.load-law/load-law-20261008",
    "A38": "memory.load-drift/load-drift-20261008",
    "A39": "memory.load-tau/load-tau-20261008",
    "A40": "memory.load-many/load-many-20261008",
    "A41": "memory.load-peak/load-peak-20261008",
    "A42": "memory.reuse-noise/reuse-noise-20261008",
    "A43": "memory.load-hazard/load-hazard-20261008",
}
BOOT, SEED = 2000, 20261008


# --------------------------------------------------------------------- records
def load(tag):
    path = os.path.join(RUNS, RECORDS[tag], "results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["observations"]


def per_brain(row, L):
    """A ladder point's per-brain reliability: whole (0/1) for one sequence
    (steps >= L - 1), or the brain's fraction of sequences whole."""
    if "whole" in row:
        return list(row["whole"])
    return [1.0 if s >= int(L) - 1 else 0.0 for s in row["steps"]]


def curves():
    """Every registered reliability curve: {(amendment, cell, arm): [(rho, [per-brain])]}."""
    out = {}
    for tag in ("A37", "A38"):
        obs = load(tag)
        for key, cell in obs["cells"].items():
            out[(tag, key, "tau 64")] = [(r["rho"], per_brain(r, L)) for L, r in cell["ladder"].items()]
    obs = load("A39")
    for key, cell in obs["cells"].items():
        for tau, arm in cell["tau"].items():
            out[("A39", key, f"tau {tau}")] = [(r["rho"], per_brain(r, L)) for L, r in arm["ladder"].items()]
    for tag in ("A40", "A43"):
        obs = load(tag)
        if obs is None:
            continue
        for key, cell in obs["cells"].items():
            for name, arm in cell["arms"].items():
                label = "one sequence" if name == "single" else f"l = {name}"
                out[(tag, key, label)] = [(r["rho"], per_brain(r, L)) for L, r in arm["ladder"].items()]
    obs = load("A41")
    for key, cell in obs["cells"].items():
        for f, arm in cell["arms"].items():
            out[("A41", key, f"tau {arm['tau']} ({f} n/k)")] = [(r["rho"], per_brain(r, L))
                                                               for L, r in arm["ladder"].items()]
    return {k: sorted(v) for k, v in out.items()}


# ----------------------------------------------------------------- statistics
def crossing(points, q, bracket=False):
    for (r0, v0), (r1, v1) in zip(points, points[1:]):
        if v0 >= q > v1:
            t = (v0 - q) / (v0 - v1)
            c = math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
            return (c, r0, r1) if bracket else c
    return (None, None, None) if bracket else None


def bootstrap(curve, q, rng):
    """(point estimate, 2.5%, 97.5%, share of resamples with a crossing, ladder
    bracket low, ladder bracket high). The bootstrap sees the brains, not the
    ladder: where reliability jumps between two ladder points on every brain,
    every resample reads the same crossing and the interval collapses to a
    point. The two ladder points that bracket the crossing are the resolution
    limit and are reported beside it."""
    import numpy as np
    rhos = [r for r, _ in curve]
    brains = np.array([b for _, b in curve], dtype=float)          # [points, brains]
    point, b0, b1 = crossing([(r, float(b.mean())) for r, b in zip(rhos, brains)], q, bracket=True)
    B = brains.shape[1]
    vals = []
    for _ in range(BOOT):
        idx = rng.integers(0, B, B)
        m = brains[:, idx].mean(axis=1)
        c = crossing(list(zip(rhos, m.tolist())), q)
        if c is not None:
            vals.append(c)
    if not vals:
        return point, None, None, 0.0, b0, b1
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), len(vals) / BOOT, b0, b1


def intervals(cv):
    import numpy as np
    rng = np.random.default_rng(SEED)
    out = {}
    for (tag, key, arm), curve in sorted(cv.items()):
        out[f"{tag} {key} {arm}"] = {q: bootstrap(curve, q, rng) for q in (0.9, 0.5, 0.1)}
    return out


def write_table(iv):
    lines = [r"\begin{tabular}{llllll}", r"\toprule",
             r"Amendment & Cell $(n,k,p)$ & Arm & $\rho_{90}$ & $\rho_{50}$ [bootstrap] \{ladder bracket\} & $\rho_{10}$ \\",
             r"\midrule"]
    for name, q in iv.items():
        tag, key, arm = name.split(" ", 2)
        n, k, p = key.split("/")

        def f(x):
            return "--" if x is None else f"{x:.3f}"
        p50, lo, hi, share, b0, b1 = q[0.5]
        ci = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]" + ("" if share > 0.99 else f"$^{{{share:.2f}}}$")
        ci += "" if b0 is None else " \\{" + f"{b0:.3f}, {b1:.3f}" + "\\}"
        lines.append(f"{tag} & ({n}, {k}, {p}) & {arm} & {f(q[0.9][0])} & {f(p50)}{ci} & {f(q[0.1][0])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "crossings_table.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# ------------------------------------------------------------- probe parsing
def parse(name, pattern):
    path = os.path.join(PROBES, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [m.groups() for m in (re.search(pattern, line) for line in fh) if m]


# -------------------------------------------------------------------- figures
def figures(cv, iv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def mean(curve):
        return [r for r, _ in curve], [float(np.mean(b)) for _, b in curve]

    # 1. the load law and its domain
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for (tag, key, arm), curve in cv.items():
        if tag == "A37" or (tag == "A41" and "(0.5 n/k)" in arm) or (tag == "A40" and arm == "one sequence"):
            r, m = mean(curve)
            n, k, p = (float(x) for x in key.split("/"))
            if k * p < 3 * math.log(n):                                 # below the regime floor
                ax[0].plot(r, m, ":", color="0.5", lw=1.2, label=f"{tag} {key} (k p < 3 ln n)")
            else:
                ax[0].plot(r, m, "-o", ms=3, lw=1, label=f"{tag} {key}")
        if tag in ("A38",) or (tag == "A39"):
            r, m = mean(curve)
            style = "--s" if "tau 64" in arm else "-o"
            ax[1].plot(r, m, style, ms=3, lw=1, label=f"{tag} {key} {arm}")
    for a, title in zip(ax, ("recovery time by the rule (or n/k <= 150 at tau 64)",
                             "n/k = 200-300: tau = 64 (dashed) against the rule")):
        a.axvline(0.09, color="grey", ls=":", lw=1)
        a.set_xscale("log")
        a.set_xlabel(r"load $\rho = L k \ln n / (n^2 p)$")
        a.set_title(title, fontsize=9)
        a.legend(fontsize=6, loc="lower left")
    ax[0].set_ylabel("fraction of brains replaying the whole sequence")
    fig.suptitle("Fig. 1. One sequence fails all or none at a critical load (dotted: the safe rule, 0.09)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_load_law.png"), dpi=150)
    plt.close(fig)

    # 2. the recovery time against the interval between a neuron's uses
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    reg = []
    for name, q in iv.items():
        if name.startswith("A41"):
            key = name.split(" ")[1]
            n, k, _ = (float(x) for x in key.split("/"))
            f = float(re.search(r"\((\S+) n/k\)", name).group(1))
            p50, lo, hi, _, b0, b1 = q[0.5]
            reg.append((n / k, f, p50, min(lo, b0), max(hi, b1)))
    colors = {20: "C0", 50: "C1", 100: "C2", 200: "C3"}
    for nk, f, p50, lo, hi in reg:
        ax.errorbar([f], [p50], yerr=[[p50 - lo], [hi - p50]],
                    fmt="o", color=colors.get(int(round(nk)), "k"), capsize=3)
    for nk in sorted({int(round(x[0])) for x in reg}):
        pts = sorted((f, p) for x, f, p, _, _ in reg if int(round(x)) == nk)
        ax.plot([a for a, _ in pts], [b for _, b in pts], "-", color=colors.get(nk, "k"), label=f"n/k = {nk} (A41; bars span bootstrap and ladder bracket)")
    rows = parse("probe_taunk.log", r"\((\d+), (\d+), ([\d.]+)\) n/k=(\d+) tau=\d+ tau/\(n/k\)=([\d.]+): rho_50 ([\d.]+)")
    rows += parse("probe_peak.log", r"\((\d+), (\d+), ([\d.]+)\) n/k=(\d+) tau=\d+ tau/\(n/k\)=([\d.]+): rho_50 ([\d.]+)")
    by = {}
    for n, k, p, nk, f, r in rows:
        by.setdefault((n, k, p), []).append((float(f), float(r)))
    for i, (cell, pts) in enumerate(sorted(by.items())):
        pts.sort()
        ax.plot([a for a, _ in pts], [b for _, b in pts], ":x", color="0.4", ms=4, lw=0.8,
                label="exploratory probes" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel(r"recovery time $\tau$ / (n/k)")
    ax.set_ylabel(r"$\rho_{50}$")
    ax.axhline(0.09, color="grey", ls=":", lw=1)
    ax.legend(fontsize=7)
    ax.set_title("Fig. 2. The recovery time matched to reuse: a peak near n/k in small areas", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_recovery_time.png"), dpi=150)
    plt.close(fig)

    # 3. many sequences: graded failure, and the capture-hazard decomposition
    from research.experiments import memory_load_hazard as mh
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag in ("A40", "A43"):
        for (t, key, arm), curve in cv.items():
            if t != tag:
                continue
            r, m = mean(curve)
            ax[0].plot(r, m, "-o", ms=3, lw=1, label=f"{tag} {key} {arm}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"total load $\rho$")
    ax[0].set_ylabel("fraction of sequences whole (mean over brains)")
    ax[0].legend(fontsize=6)
    for tag in ("A40", "A43"):
        obs = load(tag)
        if obs is None:
            continue
        for key, cell in obs["cells"].items():
            a16 = list(cell["arms"]["16"]["ladder"].values())
            a64 = list(cell["arms"]["64"]["ladder"].values())
            pts = []
            for r in a16:
                # the arms' totals are rounded to their own lengths: pair by the nearest load
                near = min(a64, key=lambda x: abs(math.log(x["rho"] / r["rho"])))
                if abs(math.log(near["rho"] / r["rho"])) < 0.03:
                    c, h = mh.fit(r["full"], near["full"])
                    if 0 < h < 0.5:
                        pts.append((r["rho"], h, c))
            pts.sort()
            if pts:
                ax[1].semilogy([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, label=f"h, {tag} {key}")
    ax[1].set_xlabel(r"total load $\rho$")
    ax[1].set_ylabel("per-step hazard h (from l = 16 and 64)")
    ax[1].legend(fontsize=6)
    fig.suptitle("Fig. 3. Short sequences fail one by one; a per-step hazard rising ~3 decades sets the long cliff", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_hazard.png"), dpi=150)
    plt.close(fig)

    # 4. tokens with a type trace
    obs = load("A42")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = 0
    ticks = []
    for key, cell in obs["cells"].items():
        for rho, arms in cell["reuse"].items():
            for u, m in arms.items():
                if u == "None":
                    continue
                same = [v for v in m["same"] if v is not None]
                diff = [v for v in m["different"] if v is not None]
                ax[0].scatter([x] * len(same), same, s=8, color="C0")
                ax[0].scatter([x + 0.3] * len(diff), diff, s=8, color="C3")
                ticks.append((x + 0.15, f"{key}\nrho {rho}, {u}/word"))
                x += 1
    ax[0].set_xticks([t for t, _ in ticks])
    ax[0].set_xticklabels([s for _, s in ticks], fontsize=5, rotation=60)
    ax[0].set_ylabel("overlap of two occurrences (blue: same word; red: different words)")
    for i, (key, cell) in enumerate(obs["cells"].items()):
        for rho, arms in cell["noise"].items():
            nus = sorted(arms, key=float)
            ax[1].plot([float(v) for v in nus], [float(np.mean(arms[v]["whole"])) for v in nus], "-o", ms=3,
                       color=f"C{i}", alpha=0.4 + 0.2 * list(cell["noise"]).index(rho), label=f"{key}, rho {rho}")
    ax[1].set_xlabel("replay noise: share of winners replaced per step")
    ax[1].set_ylabel("fraction of sequences whole")
    ax[1].legend(fontsize=6)
    fig.suptitle("Fig. 4. Recurring words are coded as tokens with a type trace; noise and load multiply (A42)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_tokens_noise.png"), dpi=150)
    plt.close(fig)

    # 5. links between areas (exploratory)
    rows = parse("probe_links.log", r"S=(\d+) C=(\d+) chunks=(\d+) plans=\d+ A=(\d+) .*start hit ([\d.]+)")
    rows += parse("probe_links2.log", r"S=(\d+) C=(\d+) chunks=(\d+) plans=\d+ A=(\d+) .*start hit ([\d.]+)")
    by = {}
    for s, c, j, a, h in rows:
        by.setdefault((int(s), int(c), int(j)), []).append((int(a), float(h)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (s, c, j), pts in sorted(by.items()):
        pts.sort()
        ax.plot([a for a, _ in pts], [h for _, h in pts], ("-" if j == 16 else "--"),
                color={2000: "C0", 4000: "C3"}[c], lw=1 + (s == 8000),
                label=f"n_S {s}, n_C {c}, J {j}")
    ax.set_xscale("log")
    ax.set_xlabel("associations A on the C -> S fiber")
    ax.set_ylabel("fraction of chunk starts evoked")
    ax.legend(fontsize=6)
    ax.set_title("Fig. 5 (exploratory). The link budget follows the source area n_C, not n_S", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_links.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    cv = curves()
    iv = intervals(cv)
    from pathlib import Path
    from research.json_documents import write_derived_document
    write_derived_document(Path(OUT) / "crossings.json",
                           {k: {str(q): v for q, v in d.items()} for k, d in iv.items()})
    write_table(iv)
    figures(cv, iv)
    for name, q in iv.items():
        p50, lo, hi, share, b0, b1 = q[0.5]
        print(f"{name}: rho_50 {p50 and round(p50, 4)} boot [{lo and round(lo, 4)}, {hi and round(hi, 4)}] "
              f"({share:.2f}) ladder ({b0 and round(b0, 4)}, {b1 and round(b1, 4)})")


if __name__ == "__main__":
    main()
