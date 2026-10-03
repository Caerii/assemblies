"""The refracted memory's phase diagram in the (write, load) plane, from records.

No new runs: every point is read from committed records.

    Amendment 23 (memory.recognition, protocol 2): seven cells, each swept from
        beta = 0.0015 to 0.30 theta, with BOTH load windows per rate --
        recognition (rank-1 > 0.5, read from M >= 32) and recall (distinct
        completion > 0.5).
    Amendment 18 (memory.onset): ten cells on a twelve-per-octave grid around
        recall's onset, recall windows only.

A rate's window is the load interval [lower, upper] over which the metric
exceeds one half (lower absent = from the first checkpoint read). In the
plane of x = beta / theta and y = M stored items, four regions appear:

    NO MEMORY         below recognition's onset: rank-1 at chance
    RECOGNITION ONLY  inside the rank-1 window, outside the recall window
    RECALL            inside the distinct-completion window
    LOST              above both windows' upper edges (load past capacity)

and two onset lines, recognition's and recall's, with recall's load-assisted
lower edge (a window that opens only after items are stored).

    python research/experiments/memory_phase_diagram.py   (writes the figure and table)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

RUNS = os.path.join(ROOT, "research", "results", "runs")
A23 = os.path.join(RUNS, "memory.recognition", "recognition-20261003-a23", "results.json")
A18 = os.path.join(RUNS, "memory.onset", "onset-20261002-r3", "results.json")
FIGURE = os.path.join(HERE, "figures", "memory_phase_diagram.png")
TABLE = os.path.join(ROOT, "research", "results", "memory", "phase_diagram_20261003.json")
ONSET_ITEMS = 32


def _window(w):
    """(lower, upper) of a window; lower None -> first checkpoint (2 items)."""
    if w["never"]:
        return None
    upper = w["last_above"] if w["upper_censored"] else w["upper"]
    return (w["lower"] if w["lower"] is not None else 2.0, upper)


def cell_table(cell):
    rows = []
    for s in sorted(cell["sweep"].values(), key=lambda s: s["beta"]):
        win = s["windows"]
        rows.append({"beta": s["beta"], "x": s["beta"] / cell["theta"],
                     "recognition": _window(win["rank1"]) if "rank1" in win else None,
                     "recall": _window(win["complete_distinct"])})
    return rows


def onsets(rows):
    """The weakest rate whose window reaches ONSET_ITEMS, per metric."""
    out = {}
    for metric in ("recognition", "recall"):
        hit = next((r for r in rows if r[metric] and r[metric][1] >= ONSET_ITEMS), None)
        out[metric] = None if hit is None else {"beta": hit["beta"], "x": hit["x"],
                                                "lower": hit[metric][0], "upper": hit[metric][1]}
    return out


def build():
    a23 = json.load(open(A23))["observations"]["cells"]
    a18 = json.load(open(A18))["observations"]["cells"]
    table = {"source": {"recognition_and_recall": os.path.relpath(A23, ROOT).replace("\\", "/"),
                        "recall_fine_grid": os.path.relpath(A18, ROOT).replace("\\", "/")},
             "cells": {}}
    for key, cell in a23.items():
        rows = cell_table(cell)
        table["cells"][key] = {"n": cell["n"], "k": cell["k"], "p": cell["p"],
                               "theta": cell["theta"], "rows": rows, "onsets": onsets(rows)}
    table["recall_onsets_fine"] = {
        key: onsets(cell_table(cell))["recall"] for key, cell in a18.items()}
    return table


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, NullFormatter
    order = sorted(table["cells"].values(), key=lambda c: (c["n"] != 4000, c["n"], c["k"]))
    fig, grid = plt.subplots(2, 5, figsize=(15, 6.4), sharey=True)
    axes = [grid[0][i] for i in range(5)] + [grid[1][i] for i in range(2)]
    for ax in [grid[1][i] for i in range(2, 5)]:
        ax.axis("off")
    for ax, cell in zip(axes, order):
        for r in cell["rows"]:
            x = r["x"]
            if r["recognition"]:
                lo, hi = r["recognition"]
                ax.plot([x, x], [max(lo, ONSET_ITEMS), hi], color="#7aa6d6", lw=6,
                        solid_capstyle="butt")
            if r["recall"]:
                lo, hi = r["recall"]
                ax.plot([x, x], [lo, hi], color="#c0392b", lw=2.5, solid_capstyle="butt")
        on = cell["onsets"]
        for metric, color in (("recognition", "#2e6da4"), ("recall", "#922b21")):
            if on[metric]:
                ax.axvline(on[metric]["x"], color=color, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"n={cell['n']}, k={cell['k']}", fontsize=9)
        ax.set_xlabel(r"$\beta/\theta$", fontsize=9)
        ax.tick_params(labelsize=7)
    grid[0][0].set_ylabel("items stored M", fontsize=9)
    grid[1][0].set_ylabel("items stored M", fontsize=9)
    axes[0].plot([], [], color="#7aa6d6", lw=6, label="recognition (rank-1 > 1/2)")
    axes[0].plot([], [], color="#c0392b", lw=2.5, label="recall (distinct completion > 1/2)")
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("Refracted memory, p = 0.5: load windows by write strength "
                 "(Amendment 23 record)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, dpi=150)


def main():
    table = build()
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=1)
        f.write("\n")
    plot(table)
    for key, cell in table["cells"].items():
        on = cell["onsets"]
        rec, cal = on["recognition"], on["recall"]
        print(f"{key:13s} recognition onset {rec['x'] if rec else None!s:>6.6} (opens at "
              f"{rec['lower'] if rec else '-'!s:.6})  recall onset {cal['x'] if cal else None!s:>6.6} "
              f"(opens at {cal['lower'] if cal else '-'!s:.6})  ratio "
              f"{(cal['x'] / rec['x']) if rec and cal else float('nan'):.2f}")


if __name__ == "__main__":
    main()
