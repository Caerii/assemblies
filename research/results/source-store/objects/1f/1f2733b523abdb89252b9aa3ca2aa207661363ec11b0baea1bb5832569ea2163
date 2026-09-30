"""Figures for task #90 -- what the sampler was doing to every collapse result.

READS THE COMMITTED LOGS, NOT RE-RUNS AND NOT LITERALS. The `.log` files are
the recorded artifacts of the runs that produced the claims; parsing them means
a figure cannot drift from the result it illustrates, and a stale figure fails
loudly (missing file) rather than quietly showing last week's numbers. The cost
is brittle parsers, which is the right trade here because the formats are ours
and they are pinned by the same commit.

Figures produced (research/notes/figures/):

  task90_sampler_sign.png    THE headline. The sampler's error on `spread`
      against M, one line per arm. With norm_init on it reports slightly MORE
      distinctness than the substrate has; with norm_init off it reports
      dramatically LESS. The sign REVERSES, which is why no constant
      correction exists and why an A/B across those arms cannot be trusted.
      The feed-forward arm is drawn as the control: near zero everywhere.

  task90_ceiling.png         Capacity ceiling. acc vs M per n, both engines,
      with chance drawn and censored points marked open. The 0.90 line is the
      ceiling definition, so it is drawn rather than described.

  task90_context.png         Study II: cross-prefix CONTEXT overlap by prefix
      length, both engines, chance floor drawn. Second panel: distinct CONTEXT
      neurons recruited, which is the mechanism -- the sampler confines CONTEXT
      to a smaller pool and a smaller pool is what makes prefixes merge.

  task90_many_areas.png      The compute/capacity trade. Speedup and items
      retained per 1000 neurons against area count, both arms, two panels
      because they are different scales (see `_figstyle`: no dual axes).

  task90_load_triage.png     Load gap per A/B with the flag threshold and the
      control drawn, so "4 of 4 flagged" is visibly a screen and not a verdict.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _figstyle as fs  # noqa: E402

plt = fs.apply()
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "notes", "figures")
ENGINES = ("numpy_sparse", "numpy_exact")


def _log(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {name} -- run the experiment that writes it first; this "
            f"module deliberately does not fabricate or re-derive its inputs")
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, os.path.dirname(HERE))}")


# -- 1. the sign reversal -----------------------------------------------------

ARM_ROW = re.compile(
    r"^\s*(\d+)\s+(rec norm|rec RAW|ff  norm)\s\|"
    r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s\|"
    r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s\|", re.M)


def fig_sampler_sign():
    rows = ARM_ROW.findall(_log("task90_recurrence.log"))
    if not rows:
        raise SystemExit("task90_recurrence.log has no arm rows -- format drift")
    by_arm = defaultdict(list)
    for m, arm, s_acc, _si, s_spr, e_acc, _ei, e_spr in rows:
        by_arm[arm.strip()].append((int(m), float(s_spr), float(e_spr),
                                    float(s_acc), float(e_acc)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    style = {"rec norm": ("#2C6E9B", "o", "recurrent, norm_init ON"),
             "rec RAW": ("#D1495B", "s", "recurrent, norm_init OFF"),
             "ff  norm": ("0.45", "^", "feed-forward (control)")}
    for arm, pts in by_arm.items():
        c, mk, lab = style[arm]
        ms = [p[0] for p in pts]
        gap = [p[2] - p[1] for p in pts]      # exact - sparse
        ax.plot(ms, gap, marker=mk, color=c, label=lab)
    ax.axhline(0, color="0.2", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("items sharing the area, M")
    ax.set_ylabel("spread:  exact  −  sampled")
    ax.set_title("The sampler's error changes SIGN between arms")
    ax.legend(loc="lower left", fontsize=7.5)
    fs.note(ax, "above 0: sampler called it MORE distinct than it is\n"
                "below 0: sampler exaggerated the collapse", "upper left")

    for arm, pts in by_arm.items():
        c, mk, lab = style[arm]
        ax2.plot([p[0] for p in pts], [p[2] for p in pts], marker=mk, color=c,
                 label=lab)
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=c,
                 ls=":", alpha=0.55)
    fs.floor_line(ax2, 50 / 1000, "overlap floor k/n")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("items sharing the area, M")
    ax2.set_ylabel("mean pairwise overlap (spread)")
    ax2.set_title("solid = exact drive,  dotted = sampled")
    fig.suptitle("Task #90: an A/B is only safe when both arms share the "
                 "instrument's error", fontsize=10.5, y=1.02)
    fs.caption(fig, "n=1000, k=50, beta=0.1, 6 seeds. The feed-forward arm is "
                    "the control: little recruits there, and the two engines "
                    "agree to within 0.023 at every M.")
    _save(fig, "task90_sampler_sign.png")


# -- 2. the capacity ceiling --------------------------------------------------

SCALE_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s\|\s*([\d.]+) spr ([\d.]+)"
    r"\s\|\s*([\d.]+) spr ([\d.]+)", re.M)


def fig_ceiling():
    text = _log("task90_n_scaling.log")
    rec = text.split("--- rec ---")[1].split("--- ff ---")[0]
    ff = text.split("--- ff ---")[1].split("READING")[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    for ax, block, title in ((axes[0], rec, "recurrent build"),
                             (axes[1], ff, "feed-forward build (control)")):
        data = defaultdict(list)
        for n, m, _cov, s_acc, _ss, e_acc, _es in SCALE_ROW.findall(block):
            data[int(n)].append((int(m), float(s_acc), float(e_acc)))
        for i, (n, pts) in enumerate(sorted(data.items())):
            col = fs.SEQ[i]
            ms = [p[0] for p in pts]
            ax.plot(ms, [p[2] for p in pts], marker="o", color=col,
                    label=f"n={n:,}")
            ax.plot(ms, [p[1] for p in pts], marker="o", ms=3, color=col,
                    ls=":", alpha=0.55)
            top = max(ms)
            if pts[-1][2] > 0.90:               # censored: never crossed
                ax.plot([top], [pts[-1][2]], marker="o", ms=10, mfc="none",
                        mec=col, mew=1.4)
        ax.axhline(0.90, color="0.2", lw=1.0, ls=(0, (5, 3)))
        ax.text(0.99, 0.905, " ceiling defined here (acc > 0.90)",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7.2, color="0.25")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("items in the area, M")
        ax.set_title(title)
    axes[0].set_ylabel("rank-1 identity (accuracy)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Capacity ceiling vs area size — the scaling is real, "
                 "the exponent is not read off this grid", fontsize=10.5,
                 y=1.02)
    fs.caption(fig, "solid = exact drive,  dotted = sampled,  open ring = "
                    "CENSORED (accuracy never crossed 0.90 here, so that "
                    "ceiling is a LOWER BOUND).\nThe factor-2 M grid brackets "
                    "each true ceiling in [M, 2M), which is why no exponent is "
                    "quoted from this figure — see task90_ceiling_fine_grid.")
    _save(fig, "task90_ceiling.png")


# -- 3. study II, CONTEXT -----------------------------------------------------

CTX_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s\|\s*([\d.]+)\s+(\d+)\s+(\d+)\s\|"
    r"\s*([\d.]+)\s+(\d+)\s+(\d+)\s*$", re.M)


def fig_context():
    rows = CTX_ROW.findall(_log("task90_context.log"))
    if not rows:
        raise SystemExit("task90_context.log has no rows -- format drift")
    L = [int(r[0]) for r in rows]
    s_ov, e_ov = [float(r[2]) for r in rows], [float(r[5]) for r in rows]
    s_nr, e_nr = [int(r[4]) for r in rows], [int(r[7]) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax.plot(L, s_ov, marker="o", color=fs.ENGINE_COLOR["numpy_sparse"],
            label=fs.ENGINE_LABEL["numpy_sparse"])
    ax.plot(L, e_ov, marker="o", color=fs.ENGINE_COLOR["numpy_exact"],
            label=fs.ENGINE_LABEL["numpy_exact"])
    fs.floor_line(ax, 0.05, "chance k/n")
    ax.set_xlabel("prefix length")
    ax.set_ylabel("mean cross-prefix CONTEXT overlap")
    ax.set_title("Study II: the collapse is real, and doubled")
    ax.legend(loc="upper right", fontsize=7.5)

    ax2.plot(L, s_nr, marker="s", color=fs.ENGINE_COLOR["numpy_sparse"])
    ax2.plot(L, e_nr, marker="s", color=fs.ENGINE_COLOR["numpy_exact"])
    ax2.set_xlabel("prefix length")
    ax2.set_ylabel("distinct CONTEXT neurons recruited")
    ax2.set_title("the mechanism: a smaller pool merges more")
    fs.caption(fig, "The published study-II figure (0.7566) came from a "
                    "DIFFERENT harness, so this is not that number re-measured "
                    "— only the ratio and direction transfer.")
    _save(fig, "task90_context.png")


# -- 4. many areas ------------------------------------------------------------

AREA_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*$", re.M)


def fig_many_areas():
    text = _log("many_areas.log")
    # Split on the "--- ARM" banners, NOT on the bare arm name. Splitting on
    # "SPLIT:" ran to the next "READING" and swallowed the GROW table too, so
    # the SPLIT series silently carried eight points and drew as two lines. A
    # parser that over-matches produces a plausible-looking plot, which is
    # worse than one that crashes.
    blocks = {}
    for key in ("SPLIT", "GROW"):
        if f"--- {key}" not in text:
            raise SystemExit(f"many_areas.log has no '--- {key}' banner")
        after = text.split(f"--- {key}")[1]
        # stop at the next banner or at READING, whichever comes first
        for stop in ("\n  --- ", "\n  READING"):
            if stop in after:
                after = after.split(stop)[0]
        blocks[key] = after
    arms = {}
    for key, block in blocks.items():
        rows = AREA_ROW.findall(block)
        if len(rows) != 4:
            raise SystemExit(f"{key} block parsed {len(rows)} rows, expected 4 "
                             f"-- format drift, refusing to plot a guess")
        arms[key] = [(int(r[0]), float(r[5]), float(r[8]), float(r[9]),
                      int(r[3]), int(r[2])) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for i, (key, pts) in enumerate(arms.items()):
        col = ("#2C6E9B", "#22A699")[i]
        A = [p[0] for p in pts]
        ax.plot(A, [p[3] for p in pts], marker="o", color=col,
                label=f"{key} arm")
        cap = [p[1] * p[5] * p[0] / p[4] * 1000 for p in pts]
        ax2.plot(A, cap, marker="s", color=col, label=f"{key} arm")
        # An area loaded PAST its ceiling has collapsed, so its
        # items-retained is low for a reason that is not "capacity per
        # neuron" -- mark those rather than letting the curve imply one
        # smooth quantity. GROW at A=1,2 puts 128 and 64 items in n=1000,
        # whose measured ceiling is 16.
        for (a, acc, *_), c in zip(pts, cap):
            if acc < 0.90:
                ax2.plot([a], [c], marker="x", ms=9, color=col, mew=1.6)
    ax.plot(sorted({p[0] for pts in arms.values() for p in pts}),
            sorted({p[0] for pts in arms.values() for p in pts}),
            color="0.5", ls=(0, (4, 3)), lw=1.0, label="ideal 1/A")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of areas, A")
    ax.set_ylabel("speedup vs A=1")
    ax.set_title("compute: falls roughly 1/A")
    ax.legend(loc="upper left", fontsize=7.5)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("number of areas, A")
    ax2.set_ylabel("items retained per 1000 neurons")
    ax2.set_title("capacity: partitioning a FIXED budget loses")
    ax2.legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Many areas: a compute win paid for in capacity out of a "
                 "fixed neuron budget", fontsize=10.5, y=1.02)
    fs.caption(fig, "x = the area was loaded PAST its ceiling and collapsed, "
                    "so its low retention is not a capacity-per-neuron reading "
                    "(GROW at A=1,2 puts 128 and 64 items in n=1000, whose "
                    "ceiling is 16).\nRouting is handed over free — each "
                    "item's area is known — so chance rises from 1/M to A/M "
                    "and these are an UPPER bound on what modularity buys.",
               y=-0.06)
    _save(fig, "task90_many_areas.png")


# -- 5. the load triage -------------------------------------------------------

TRIAGE_ROW = re.compile(r"^\s*\[(CONFOUNDED|ok\s*)\] gap ([\d.]+)\s+(.+?)\s*$",
                        re.M)


def fig_load_triage():
    rows = TRIAGE_ROW.findall(_log("task90_ab_triage.log"))
    if not rows:
        raise SystemExit("task90_ab_triage.log has no rows -- format drift")
    labels, gaps, flags = [], [], []
    for tag, gap, label in rows:
        labels.append(label.split("  [")[0].split("(")[0].strip())
        gaps.append(float(gap))
        flags.append(tag.strip() == "CONFOUNDED")

    order = np.argsort(gaps)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    cols = ["#D1495B" if flags[i] else "#22A699" for i in order]
    ax.barh([labels[i] for i in order], [gaps[i] for i in order], color=cols,
            height=0.62)
    ax.axvline(0.05, color="0.2", lw=1.2, ls=(0, (5, 3)))
    ax.text(0.05, 1.01, " flag threshold 0.05", fontsize=7.4, color="0.25",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom")
    ax.set_xlabel("area-load gap between the two arms  (w/n)")
    ax.set_title("Which standing A/Bs did not share the sampler's error?")
    ax.grid(axis="y", visible=False)
    fs.caption(fig, "A flag is a SCREEN, not a verdict. Of the two since "
                    "re-derived on exact drive, one conclusion CHANGED "
                    "(norm_init: 8.0x -> 1.0x) and one HELD (rec vs ff).\n"
                    "Sensitivity is UNMEASURED — no unflagged A/B has been "
                    "re-derived, so there is no evidence yet that passing the "
                    "screen clears anything.", y=-0.10)
    _save(fig, "task90_load_triage.png")


# -- 6. the gain confound ------------------------------------------------------

GAIN_ROW = re.compile(
    r"^\s*beta=([\d.]+)\s+gain=\s*([\d.]+)\s+M\*\(\d+\)=\s*([\d.]+)"
    r"\s+M\*\(\d+\)=\s*([\d.]+)\s+a=([\d.]+)", re.M)


def fig_gain_confound():
    rows = GAIN_ROW.findall(_log("task90_gain_confound.log"))
    if not rows:
        raise SystemExit("task90_gain_confound.log has no rows -- format drift")
    beta = [float(r[0]) for r in rows]
    gain = [float(r[1]) for r in rows]
    lo = [float(r[2]) for r in rows]
    hi = [float(r[3]) for r in rows]
    a = [float(r[4]) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax.plot(gain, a, marker="o", color="#D1495B", zorder=3)
    # The measured grid-noise floor of the estimator, drawn as the error it is.
    # Without it a reader cannot tell whether the spread is signal.
    for g, av in zip(gain, a):
        ax.plot([g, g], [av - 0.10, av + 0.10], color="#D1495B", lw=1.2,
                alpha=0.5, zorder=2)
    ax.axhline(1.0, color="0.2", lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(0.99, 1.02, "extensive, a = 1 (the standing alpha* law) ",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7.2, color="0.25")
    for g, av, b in zip(gain, a, beta):
        ax.annotate(f"  beta={b}", (g, av), fontsize=7.4, color="0.35",
                    va="center")
    ax.set_xlabel("absolute gain  (1+beta)^T")
    ax.set_ylabel("fitted exponent  a  in  M_max ~ n^a")
    ax.set_title("The exponent MOVES with gain")

    w = 0.35
    idx = np.arange(len(rows))
    ax2.bar(idx - w / 2, lo, w, color="#4C6EF5", label="M* at n=1000")
    ax2.bar(idx + w / 2, hi, w, color="#22A699", label="M* at n=2000")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"beta={b}" for b in beta])
    ax2.set_yscale("log", base=2)
    ax2.set_ylabel("resolved ceiling  M*")
    ax2.set_title("gain sets the ceiling outright")
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Task #90: the capacity exponent is a fixed-gain artifact, "
                 "not a law", fontsize=10.5, y=1.02)
    fs.caption(fig, "Whiskers are the estimator's MEASURED grid-noise floor "
                    "(0.20 wide); the spread across gains is 0.78, nearly 4x "
                    "that.\nAt beta=0.20 the exponent is 0.87 — SUB-linear — "
                    "so super-linearity does not merely fail to replicate, it "
                    "REVERSES. a = 1 sits inside the range.", y=-0.06)
    _save(fig, "task90_gain_confound.png")


def main():
    print("\n  task #90 figures")
    for fn in (fig_sampler_sign, fig_ceiling, fig_context, fig_many_areas,
               fig_load_triage, fig_gain_confound):
        try:
            fn()
        except SystemExit as e:
            print(f"  SKIP {fn.__name__}: {e}")
    print()


if __name__ == "__main__":
    main()
