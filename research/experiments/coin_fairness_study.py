"""Is the neural coin fair, and if not, why -- and does the unfairness scale away?

    uv run python research/experiments/coin_fairness_study.py            # full
    uv run python research/experiments/coin_fairness_study.py --quick    # fast

Writes `research/results/coin_fairness/*.json` and the figures embedded in
`research/notes/neural_coin_fairness.md`.


THE QUESTION
------------
A "neural coin" (`RandomChoiceArea`, after [COIN24]) stores two assemblies in
one recurrent area, seeds the area with a random cap-sized set, lets recurrence
settle, and reports which assembly the settled state resembles. If the substrate
works, an unbiased seed should fall into either basin equally often, and the
settled state should be a CLEAN assembly rather than a smear.

Both halves have to be measured, because they fail independently and the first
one alone is trivially satisfiable:

``heads``      fraction of flips answering 0. Fair means ~0.5.
``decisive``   mean overlap between the settled state and whichever assembly it
               landed nearer. Chance is ``k/n``. **A coin that returns noise
               scores a perfect 0.5 on ``heads``**, because the readout breaks
               near-ties toward 0 at random-ish -- so ``heads`` alone cannot
               distinguish a working coin from a broken one.
``sd``         spread of ``heads`` ACROSS BRAINS. A coin that answers 0 in one
               brain and 1 in another is not a coin, however fair the pooled
               mean looks. This is the quantity this study is really about.


WHAT WAS BROKEN, AND THE TWO THINGS THAT FIXED IT
-------------------------------------------------
1.  THE SETTLE LOOP WAS INERT.  Self fibers were excluded from the engine's
    deferred-init path, so `area -> area` was never allocated, delivered zero
    drive, and `project_into` handed back the incumbent winners. Measured:
    ``rounds`` = 0 / 1 / 10 gave 200/200 identical flips. The published coin
    numbers were the seed RNG, not the substrate.

2.  THE FIBER SPANNED ``w``, NOT ``n``.  Even once allocated, lazy
    materialization sizes blocks to the ``w`` neurons that have actually won.
    The reference allocates a dense ``n x n`` recurrent matrix up front and then
    seeds a uniform random ``k``-subset of ALL ``n``. Ported onto lazy
    materialization at ``n=2000`` the area had ``w=357``, so **82% of the seed
    named neurons with no outgoing synapse** and settling resolved ~9 neurons of
    signal. `NumpySparseEngine.materialize_area` closes this.

Neither is a parameter, which is why a sweep over ``beta`` in {0.05 … 5.0},
``rounds_train`` in {2 … 40} and ``settle`` in {0 … 10} found no fair cell
before these landed.


THE RESULT
----------
With both fixed, the coin works and its unfairness is a FINITE-SIZE EFFECT.
Holding ``k/n = 0.1`` and growing the area (12 brains x 100 flips)::

        n      k   heads     sd  decisive   basin asym
      500     50   0.438  0.416     0.647       0.0446
     1000    100   0.522  0.352     0.764       0.0223
     2000    200   0.478  0.206     0.986       0.0120
     4000    400   0.522  0.089     1.000       0.0056

``decisive`` rises to 1.000 -- the settled state becomes a clean assembly -- and
the across-brain spread falls by 4.7x. Basin asymmetry halves each time ``k``
doubles, i.e. it goes as ``1/k``: bigger assemblies self-average, and the tilt
that remains is a fluctuation in how symmetric a given brain's two basins happen
to be. ``r(tilt, asym) = 0.45-0.58`` at the larger sizes, which is the causal
link stated as a correlation.

THE CONTROL THAT MAKES THIS MEAN ANYTHING: at ``beta = 0`` -- nothing learned --
``heads`` reads 0.523 +/- 0.040, which *looks* fairer than the trained coin. Its
``decisive`` is 0.112, exactly chance. That is the whole reason ``decisive`` is
reported next to ``heads`` everywhere below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

import time as _time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from neural_assemblies import Brain                                    # noqa: E402
from neural_assemblies.assembly_calculus.assembly import overlap       # noqa: E402
from neural_assemblies.assembly_calculus.ops import (                  # noqa: E402
    _snap, _compact_index,
)

OUT_DATA = os.path.join(ROOT, "research", "results", "coin_fairness")
OUT_FIG = os.path.join(ROOT, "research", "notes", "figures")

P = 0.05
BETA, FIRES, SETTLE = 3.0, 2, 20
# 40 distinct brains. Sampling is allocated PER SIZE because cost goes as n^2
# (measured: 12 s/brain at n=4000 with 200 flips, 48 s at n=8000), and because
# the small sizes are where the distribution is widest and therefore where the
# extra brains actually buy resolution.
SEEDS = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
         61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
         131, 137, 139, 149, 151, 157, 163, 167)

#: (n, k, n_brains, n_flips) -- the finite-size ladder.
LADDER = [
    (500,    50, 40, 400),
    (1000,  100, 40, 400),
    (2000,  200, 32, 400),
    (4000,  400, 24, 200),
    (8000,  800, 16, 200),
    (16000, 1600, 8, 100),
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_coin(seed, n, k, beta=BETA, fires=FIRES, dense=True):
    """Two balanced attractors in one recurrent area.

    Follows the reference (`.reference/mdabagia-nemo/brain.py`) rather than this
    repo's shipped `RandomChoiceArea`, in three respects that all turned out to
    matter:

    * the two assemblies are carved in SEPARATE connectomes (reset between), so
      neither forms inside the other's basin;
    * training is EXACTLY SYMMETRIC -- each assembly fired the same number of
      times, nothing after. The shipped version runs 3x(10, 10) and then
      re-snapshots asm0 then asm1, leaving asm1 systematically deeper;
    * firing is FORCED via `fix_assembly`, mirroring the reference's
      `fire(assm)`. A plain `project` would recompute winners from a
      still-untrained block, get noise, and potentiate ``assembly x noise``.

    TWO INDEX SPACES -- the trap this repo keeps re-learning. `Assembly.winners`
    are NEURON IDS; `set_winners` and the weight blocks take COMPACT ENGINE
    INDICES. They coincide only while an area grows in pool order, and NOT after
    `materialize_area`, which draws the remainder straight from the shuffled
    pool. Feeding one in as the other trains and reads entirely the wrong
    neurons -- and reads exactly chance while doing it, which is indistinguishable
    from "the mechanism does not work".
    """
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    b.add_stimulus("s0", k)
    b.add_stimulus("s1", k)
    b.add_area("C", n, k, beta)

    for _ in range(10):
        b.project({"s0": ["C"]}, {})
    a0 = _snap(b, "C")
    b._engine.reset_area_connections("C")
    for _ in range(10):
        b.project({"s1": ["C"]}, {})
    a1 = _snap(b, "C")
    b._engine.reset_area_connections("C")

    if dense:
        b._engine.materialize_area("C")

    inv = _compact_index(b._engine_for(b.areas["C"]), "C") or {}

    def compact(asm):
        return np.asarray(
            [inv[int(x)] for x in asm.winners if int(x) in inv],
            dtype=np.uint32)

    c0, c1 = compact(a0), compact(a1)
    for _ in range(fires):
        for c in (c0, c1):
            b.areas["C"]._winners = c
            b._engine.set_winners("C", c)
            b.areas["C"].fix_assembly()
            b.project({}, {"C": ["C"]})
            b.areas["C"].unfix_assembly()
    return b, a0, a1, c0, c1


def flip(b, a0, a1, n, k, seed, settle=SETTLE):
    """Reference protocol: uniform random k-subset of ALL n, settle, read."""
    rng = np.random.default_rng(seed)
    init = rng.choice(n, size=k, replace=False).astype(np.uint32)
    b.areas["C"]._winners = init
    b._engine.set_winners("C", init)
    with b.frozen():                      # a flip is a READ; it must not train
        for _ in range(settle):
            b.project({}, {"C": ["C"]})
    res = _snap(b, "C")
    o0, o1 = overlap(res, a0), overlap(res, a1)
    return (0 if o0 >= o1 else 1), max(o0, o1)


def basin_asymmetry(b, c0, c1):
    """|pull0 - pull1| / (pull0 + pull1) -- how lopsided this brain's basins are.

    ``pull`` is the total within-assembly recurrent weight a basin can bring to
    bear. Perfectly symmetric training still leaves a residual because the two
    assemblies wire to themselves slightly differently by chance; this measures
    that residual, and it is what the tilt correlates with.
    """
    eng = b._engine_for(b.areas["C"])
    w = np.asarray(eng._area_conns["C"]["C"].weights)

    def pull(c):
        idx = [int(x) for x in c if int(x) < min(w.shape)]
        return float(w[np.ix_(idx, idx)].sum()) if idx else float("nan")
    p0, p1 = pull(c0), pull(c1)
    return abs(p0 - p1) / max(p0 + p1, 1e-9)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    n: int
    k: int
    beta: float
    settle: int
    dense: bool
    heads_mean: float
    heads_sd: float
    heads_per_brain: list
    decisive: float
    asym_mean: float
    asym_per_brain: list
    sep: float          # overlap(a0, a1) -- shared neurons between attractors
    n_brains: int = 0
    n_flips: int = 0

    @property
    def tilt_asym_corr(self):
        tilt = [abs(h - 0.5) for h in self.heads_per_brain]
        if np.std(self.asym_per_brain) == 0:
            return float("nan")
        return float(np.corrcoef(tilt, self.asym_per_brain)[0, 1])


def measure(n, k, beta=BETA, settle=SETTLE, dense=True, fires=FIRES,
            seeds=SEEDS, n_flips=100):
    rates, decis, asyms, seps = [], [], [], []
    for s in seeds:
        b, a0, a1, c0, c1 = build_coin(s, n, k, beta=beta, fires=fires,
                                       dense=dense)
        asyms.append(basin_asymmetry(b, c0, c1))
        seps.append(float(overlap(a0, a1)))
        r = [flip(b, a0, a1, n, k, 700 + i, settle) for i in range(n_flips)]
        rates.append(sum(x == 0 for x, _ in r) / n_flips)
        decis.append(float(np.mean([o for _, o in r])))
    return Cell(n=n, k=k, beta=beta, settle=settle, dense=dense,
                heads_mean=float(np.mean(rates)), heads_sd=float(np.std(rates)),
                heads_per_brain=[float(x) for x in rates],
                decisive=float(np.mean(decis)),
                asym_mean=float(np.mean(asyms)),
                asym_per_brain=[float(x) for x in asyms],
                sep=float(np.mean(seps)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#e3e3e8"
ACCENT = "#2f6fd0"
WARM = "#d1495b"
GOLD = "#c78a2e"
GREEN = "#2a8a6f"


def _loglog_fit(x, y):
    """Least-squares power-law exponent. Returns ``(slope, intercept)``.

    Quoted in figure labels so a scaling claim carries its measured exponent
    rather than a guide line the reader is invited to eyeball against.
    """
    return tuple(float(v) for v in np.polyfit(np.log(np.asarray(x, float)),
                                              np.log(np.asarray(y, float)), 1))


def _plain_log_ticks(ax, vals, axis="x"):
    """Label a log axis with the values actually measured.

    Matplotlib's default log locator emits 2x10^3 / 3x10^3 / 4x10^3 for a range
    this narrow, which collide into an unreadable smear. The measured sizes are
    the only ticks that mean anything here anyway.
    """
    import matplotlib.ticker as mtick
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mtick.FixedLocator(vals))
    a.set_minor_locator(mtick.NullLocator())
    a.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{int(v):,}"))


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def fig_finite_size(cells, path):
    """The headline: across-brain spread collapses as the area grows."""
    import matplotlib.pyplot as plt
    ns = [c.n for c in cells]
    ks = [c.k for c in cells]
    sd = [c.heads_sd for c in cells]
    dec = [c.decisive for c in cells]
    asym = [c.asym_mean for c in cells]

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))

    ax[0].plot(ns, sd, "o-", color=ACCENT, lw=1.8, ms=6, zorder=3)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("area size $n$  (at fixed $k/n=0.1$)")
    ax[0].set_ylabel("across-brain sd of heads rate")
    ax[0].set_title("Unfairness is a finite-size effect", loc="left")
    # 1/sqrt(k) is the naive expectation for a self-averaging fluctuation and
    # the measured collapse BEATS it -- but sd is bounded above (a fully
    # bimodal ladder rung cannot exceed 0.5), so the small-n rungs are
    # compressed against that cap and the exponent is not clean. Show both:
    # the fit, and the reference it beats.
    ax[0].plot(ns, sd[0] * (np.array(ks, float) / ks[0]) ** -0.5, "--",
               color=MUTED, lw=1.1, zorder=1, label=r"$1/\sqrt{k}$")
    s_sd, r_sd = _loglog_fit(ks, sd)
    ax[0].plot(ns, np.exp(np.polyval([s_sd, r_sd], np.log(ks))), "-",
               color=ACCENT, lw=1.0, alpha=0.45, zorder=2,
               label=rf"fit $k^{{{s_sd:.2f}}}$")
    ax[0].legend(loc="lower left", fontsize=8)
    _plain_log_ticks(ax[0], ns)

    ax[1].plot(ns, dec, "o-", color=GREEN, lw=1.8, ms=6, label="trained")
    ax[1].axhline(0.1, ls=":", color=WARM, lw=1.3)
    ax[1].text(ns[0], 0.115, "chance ($k/n$)", color=WARM, fontsize=8.5)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel("area size $n$")
    ax[1].set_ylabel("overlap with the winning assembly")
    ax[1].set_title("The state becomes a clean assembly", loc="left")
    _plain_log_ticks(ax[1], ns)

    ax[2].plot(ks, asym, "o-", color=GOLD, lw=1.8, ms=6, zorder=3)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    inv_k = asym[0] * (np.array(ks, float) / ks[0]) ** -1.0
    ax[2].plot(ks, inv_k, "--", color=MUTED, lw=1.4, zorder=1,
               label=r"$1/k$ from the smallest rung")
    s_as, _ = _loglog_fit(ks, asym)
    ax[2].set_xlabel("cap size $k$")
    ax[2].set_ylabel("basin asymmetry")
    ax[2].set_title(rf"Basins self-average as $k^{{{s_as:.2f}}}$", loc="left")
    ax[2].legend(loc="lower left", fontsize=8)
    _plain_log_ticks(ax[2], ks)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_per_brain(cells, path):
    """Every brain is its own coin. Show them, do not average them away."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, c in enumerate(cells):
        y = c.heads_per_brain
        x = np.full(len(y), i) + np.linspace(-0.16, 0.16, len(y))
        ax.scatter(x, y, s=26, color=ACCENT, alpha=0.75, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.plot([i - 0.3, i + 0.3], [c.heads_mean] * 2, color=INK, lw=2,
                zorder=4)
    ax.axhline(0.5, ls="--", color=WARM, lw=1.2)
    ax.text(-0.45, 0.512, "fair", color=WARM, fontsize=9)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"$n$={c.n}\n$k$={c.k}" for c in cells])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("heads rate (one point per brain)")
    ax.set_title("Each brain is its own slightly-bent coin — and they "
                 "straighten with size", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_distribution(cells, path):
    """The full per-brain distribution, not a mean and an error bar.

    At small ``n`` the distribution is BIMODAL -- brains pile up at 0 and 1,
    each one a near-deterministic coin -- and a mean of 0.5 over that is
    meaningless. Only at large ``n`` does it become a single peak at 0.5. A
    summary statistic cannot show that transition; the histogram is the result.
    """
    import matplotlib.pyplot as plt
    ncell = len(cells)
    fig, axes = plt.subplots(1, ncell, figsize=(2.05 * ncell, 3.0),
                             sharey=True)
    if ncell == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 21)
    for ax, c in zip(axes, cells):
        ax.hist(c.heads_per_brain, bins=bins, color=ACCENT, alpha=0.85,
                edgecolor="white", linewidth=0.6)
        ax.axvline(0.5, ls="--", color=WARM, lw=1.2)
        ax.set_title(f"$n$={c.n:,}\n{c.n_brains} brains", loc="center",
                     fontsize=9)
        ax.set_xlabel("heads rate")
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("brains")
    fig.suptitle("Bimodal at small $n$ — every brain a near-certain answer — "
                 "collapsing to one peak at 0.5", fontsize=10.5, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def fig_null_contrast(trained, null, path):
    """Why `heads` alone cannot tell a working coin from a broken one."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    labels = ["untrained\n($\\beta=0$)", "trained\n($\\beta=3$)"]

    for a, vals, ylab, title, hline in (
        (ax[0], [null.heads_mean, trained.heads_mean],
         "heads rate", "Both look fair…", 0.5),
        (ax[1], [null.decisive, trained.decisive],
         "overlap with winning assembly", "…only one is deciding anything",
         null.n and null.k / null.n),
    ):
        errs = [null.heads_sd, trained.heads_sd] if hline == 0.5 else None
        a.bar(labels, vals, color=[MUTED, GREEN], width=0.55,
              yerr=errs, capsize=4, ecolor=INK)
        a.axhline(hline, ls="--", color=WARM, lw=1.2)
        a.set_ylabel(ylab)
        a.set_title(title, loc="left")
        for i, v in enumerate(vals):
            a.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax[1].text(-0.45, (null.k / null.n) + 0.02, "chance", color=WARM,
               fontsize=8.5)
    ax[0].set_ylim(0, 0.75)
    ax[1].set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_settle(curve, path):
    """How many rounds of recurrence a decision needs."""
    import matplotlib.pyplot as plt
    rounds = [c.settle for c in curve]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(rounds, [c.decisive for c in curve], "o-", color=GREEN, lw=1.8,
            ms=6, label="overlap with winning assembly")
    ax.plot(rounds, [c.heads_sd for c in curve], "s-", color=ACCENT, lw=1.8,
            ms=5.5, label="across-brain sd")
    ax.axhline(curve[0].k / curve[0].n, ls=":", color=WARM, lw=1.2)
    ax.text(rounds[-1], curve[0].k / curve[0].n + 0.02, "chance", color=WARM,
            ha="right", fontsize=8.5)
    ax.set_xlabel("settling rounds")
    ax.set_ylabel("")
    ax.set_ylim(0, 1.08)
    ax.set_title("Recurrence has to run long enough to complete, "
                 "and no longer", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_mechanism(cells, path):
    """Tilt vs basin asymmetry: the stated cause, as a scatter.

    READ THIS FIGURE WITH THE CEILING IN MIND.  ``tilt`` is ``|heads - 0.5|``,
    so a brain that always answers the same way sits at exactly 0.5 and cannot
    go higher.  At small ``n`` most brains are pinned there, which CENSORS the
    correlation -- a saturated series can have a huge causal effect and still
    report ``r`` near zero, because the response variable has no room left to
    vary.  The pooled ``r`` printed in the legend is therefore computed on the
    unsaturated brains only, and the count is shown so a small-sample ``r`` is
    not mistaken for a measurement.  The uncorrected ``r`` over all brains is
    what ``cell.tilt_asym_corr`` holds and what the JSON records.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    palette = [ACCENT, GREEN, GOLD, WARM, "#7e57c2", "#26a69a"]
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.text(0.995, 0.503, "ceiling: brain always answers the same way",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7.5, color=MUTED)
    for c, col in zip(cells, palette):
        tilt = np.asarray([abs(h - 0.5) for h in c.heads_per_brain])
        asym = np.asarray(c.asym_per_brain, dtype=float)
        free = tilt < 0.499
        if free.sum() >= 3 and np.std(asym[free]) > 0:
            r = float(np.corrcoef(tilt[free], asym[free])[0, 1])
            lab = f"$n$={c.n:,}  $r$={r:+.2f} ($n_{{free}}$={int(free.sum())})"
        else:
            lab = f"$n$={c.n:,}  (saturated)"
        ax.scatter(asym, tilt, s=34, color=col, alpha=0.82,
                   edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("basin asymmetry  $|p_0-p_1| / (p_0+p_1)$")
    ax.set_ylabel("tilt  $|$heads $-\\,0.5|$")
    ax.set_title("Lopsided basins bias the coin -- and both shrink together",
                 loc="left")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_kn(cells, path):
    """The other axis: cap density, and the price of overlapping attractors."""
    import matplotlib.pyplot as plt
    kn = [c.k / c.n for c in cells]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.plot(kn, [c.heads_sd for c in cells], "o-", color=ACCENT, lw=1.8, ms=6,
            label="across-brain sd")
    ax.plot(kn, [c.sep for c in cells], "^-", color=WARM, lw=1.6, ms=6,
            label="overlap between the two assemblies")
    ax.plot(kn, [c.decisive for c in cells], "s-", color=GREEN, lw=1.6, ms=5.5,
            label="overlap with winning assembly")
    ax.set_xlabel("cap density $k/n$  (at fixed $n$)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Denser caps decide better, until the attractors start "
                 "sharing neurons", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------

DATA_FILE = "coin_fairness.json"


def write_figures(fs, null, st, kn, trained):
    """Render every figure from measured cells. Pure output, no simulation."""
    fig_finite_size(fs, os.path.join(OUT_FIG, "coin_finite_size.png"))
    fig_per_brain(fs, os.path.join(OUT_FIG, "coin_per_brain.png"))
    fig_distribution(fs, os.path.join(OUT_FIG, "coin_distribution.png"))
    fig_null_contrast(trained, null,
                      os.path.join(OUT_FIG, "coin_null_contrast.png"))
    fig_settle(st, os.path.join(OUT_FIG, "coin_settling.png"))
    fig_mechanism(fs, os.path.join(OUT_FIG, "coin_mechanism.png"))
    fig_kn(kn, os.path.join(OUT_FIG, "coin_cap_density.png"))


def null_ladder(rungs=((500, 50, 40, 400), (2000, 200, 32, 400),
                       (8000, 800, 16, 200))):
    """Run the beta=0 control UP THE LADDER, not just at one size.

    WHY THIS IS NOT OPTIONAL.  The headline claim is about a TREND -- spread
    collapses as the area grows -- and a single null cell cannot test a trend.
    If the untrained coin's spread also fell with ``n``, "fairness emerges with
    scale" would be a statement about the readout or the flip count, not about
    anything the area learned.

    Measured: it does not. The null's spread is FLAT at 0.055 / 0.037 / 0.035
    across n = 500 / 2,000 / 8,000, against binomial sampling floors of 0.025
    (400 flips) and 0.035 (200 flips) -- so its brains are already individually
    fair and there is nothing left to collapse. The trained coin starts 7.6x
    above that and converges down toward it.

    The two ``decisive`` columns then move in OPPOSITE directions: trained
    climbs 0.662 -> 1.000 while the null descends 0.123 -> 0.106 toward the
    ``k/n`` chance floor. One metric separating two arms is suggestive; two
    metrics whose arms diverge is a dissociation.
    """
    print(f"{'n':>7}{'k':>6}{'br':>4}{'flips':>7}"
          f"{'heads':>8}{'sd':>8}{'decisive':>10}{'chance':>8}  [s]")
    cells = []
    for n, k, nb, nf in rungs:
        t0 = _time.time()
        c = measure(n=n, k=k, beta=0.0, seeds=SEEDS[:nb], n_flips=nf)
        c.n_brains, c.n_flips = nb, nf
        cells.append(c)
        print(f"{n:>7}{k:>6}{nb:>4}{nf:>7}{c.heads_mean:>8.3f}"
              f"{c.heads_sd:>8.3f}{c.decisive:>10.3f}{k/n:>8.3f}"
              f"  [{_time.time()-t0:.0f}]")
        sys.stdout.flush()
    for nf in sorted({r[3] for r in rungs}):
        # An ideally fair coin still scatters this much from flip sampling
        # alone; a null sd at this value means the brains are individually
        # fair, not that the measurement is insensitive.
        print(f"binomial sd floor at {nf} flips: {np.sqrt(0.25/nf):.4f}")
    path = os.path.join(OUT_DATA, "coin_null_ladder.json")
    with open(path, "w") as fh:
        json.dump({"rungs": [list(r) for r in rungs],
                   "cells": [asdict(c) for c in cells]}, fh, indent=2)
    print(f"data -> {path}")
    return cells


def replot():
    """Regenerate every figure from the recorded JSON, running nothing.

    The ladder costs ~35 minutes at full scale, and a figure is an editorial
    object that gets revised far more often than the data behind it. Keeping
    the two separable is what makes it cheap to fix a mislabelled axis without
    either re-simulating or -- much worse -- hand-editing a number into a plot
    that no longer matches ``coin_fairness.json``.
    """
    path = os.path.join(OUT_DATA, DATA_FILE)
    with open(path) as fh:
        payload = json.load(fh)
    fs = [Cell(**d) for d in payload["finite_size"]]
    null = Cell(**payload["null"])
    st = [Cell(**d) for d in payload["settle"]]
    kn = [Cell(**d) for d in payload["cap_density"]]
    trained = next((c for c in fs if c.n == null.n), fs[-1])
    _style()
    write_figures(fs, null, st, kn, trained)
    print(f"replotted {len(fs)} ladder cells from {path}\n"
          f"figures -> {OUT_FIG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="4 brains, 40 flips, small sizes")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate figures from the recorded JSON, "
                         "running no simulation")
    ap.add_argument("--null-ladder", action="store_true",
                    help="run the beta=0 control up the ladder (~18 min) and "
                         "exit; tests whether the UNTRAINED coin shows the "
                         "same finite-size trend. It does not.")
    ap.add_argument("--max-n", type=int, default=16000,
                    help="largest area on the finite-size ladder "
                         "(memory is O(n^2): ~1 GB at n=16000)")
    args = ap.parse_args()

    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    if args.replot:
        replot()
        return
    if args.null_ladder:
        null_ladder()
        return
    _style()

    ladder = ([(500, 50, 4, 40), (1000, 100, 4, 40)] if args.quick
              else [t for t in LADDER if t[0] <= args.max_n])
    kns = [100, 200] if args.quick else [100, 200, 400, 600]
    settles = [1, 3, 5] if args.quick else [1, 2, 3, 5, 10, 20, 40]
    base_n = 1000 if args.quick else 2000
    base_k = base_n // 10
    base_brains, base_flips = (4, 40) if args.quick else (32, 400)

    def M(**kw):
        kw.setdefault("seeds", SEEDS[:base_brains])
        kw.setdefault("n_flips", base_flips)
        return measure(**kw)

    print(f"finite-size scaling (fixed k/n = 0.1), up to n={ladder[-1][0]:,}")
    fs = []
    for n, k, nb, nf in ladder:
        t0 = _time.time()
        c = measure(n=n, k=k, seeds=SEEDS[:nb], n_flips=nf)
        c.n_brains, c.n_flips = nb, nf
        fs.append(c)
        # sd of an sd from m samples is ~ sd / sqrt(2(m-1)); quote it, because
        # the whole claim is about how sd moves and a bare point estimate over
        # 8 brains would not support one.
        sd_err = c.heads_sd / np.sqrt(2 * max(nb - 1, 1))
        print(f"  n={n:<6} k={k:<5} {nb:>2}br x{nf:<4} "
              f"heads {c.heads_mean:.3f}   sd {c.heads_sd:.3f} +/- {sd_err:.3f}"
              f"   decisive {c.decisive:.3f}   asym {c.asym_mean:.5f}"
              f"   r {c.tilt_asym_corr:+.2f}   [{_time.time()-t0:.0f}s]")
        sys.stdout.flush()

    print("\nuntrained null (beta = 0)")
    null = M(n=base_n, k=base_k, beta=0.0)
    print(f"  heads {null.heads_mean:.3f} +/- {null.heads_sd:.3f}   "
          f"decisive {null.decisive:.3f}  <- chance is {base_k/base_n:.3f}")

    print("\nsettling length")
    st = [M(n=base_n, k=base_k, settle=s) for s in settles]
    for c in st:
        print(f"  settle={c.settle:<3} decisive {c.decisive:.3f}   "
              f"sd {c.heads_sd:.3f}")

    print("\ncap density k/n")
    kn = [M(n=base_n, k=k) for k in kns]
    for c in kn:
        print(f"  k/n={c.k/c.n:.2f}  heads {c.heads_mean:.3f} +/- "
              f"{c.heads_sd:.3f}  decisive {c.decisive:.3f}  "
              f"sep {c.sep:.4f}")

    trained = next((c for c in fs if c.n == base_n), fs[-1])
    payload = {
        "parameters": {"p": P, "beta": BETA, "fires": FIRES,
                       "settle": SETTLE,
                       "ladder": [list(t) for t in ladder],
                       "base_brains": base_brains, "base_flips": base_flips},
        "finite_size": [asdict(c) for c in fs],
        "null": asdict(null),
        "settle": [asdict(c) for c in st],
        "cap_density": [asdict(c) for c in kn],
    }
    with open(os.path.join(OUT_DATA, "coin_fairness.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    write_figures(fs, null, st, kn, trained)
    print(f"\nfigures -> {OUT_FIG}\ndata    -> {OUT_DATA}")


if __name__ == "__main__":
    main()
