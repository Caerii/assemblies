"""One figure style for the whole repo, and the conventions that go with it.

`plot_critical_point.py` and `plot_sweep.py` each carried their own copy of the
same rcParams, already drifting (grid.alpha 0.25 vs 0.22). A third copy would
have made "our figures look like this" unenforceable, so it lives here.

THE CONVENTIONS ARE NOT COSMETIC. Each one exists because a figure in this
project once hid something:

* THE FLOOR IS ALWAYS DRAWN. Overlap axes get an explicit `k/n` line. "At the
  floor" is the single most important reading on any distinctness plot and an
  unmarked axis makes it invisible -- the collapse findings were originally
  read off plots without one.
* CHANCE IS ALWAYS DRAWN on any accuracy axis, for the same reason.
* CENSORED AND UNSUPPORTED POINTS ARE MARKED, not silently plotted as if
  measured. A ceiling at the top of a sweep is a bound; an interpolated
  crossing with no interior point is a guess. Both get open markers and a note.
* NO DUAL Y-AXES. Two scales in one frame invite a comparison the data does not
  license; use two panels.
* SEED SPREAD IS SHOWN where it exists, because the difference between an
  effect and seed noise is the thing most often mistaken here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RC = {
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
}

#: Perceptually ordered, for a size/scale sequence (small -> large).
SEQ = ("#4C6EF5", "#22A699", "#F2A65A", "#D1495B", "#7B3FA0")

#: The two engines, used consistently everywhere so a reader learns them once.
ENGINE_COLOR = {"numpy_sparse": "#D1495B", "numpy_exact": "#2C6E9B"}
ENGINE_LABEL = {"numpy_sparse": "sampled (numpy_sparse)",
                "numpy_exact": "exact drive (numpy_exact)"}


def apply():
    plt.rcParams.update(RC)
    return plt


def floor_line(ax, value, label="chance / floor k/n", x=0.99, ha="right"):
    """Draw the chance or overlap floor. Call this on EVERY such axis.

    The label carries an opaque background box. Without one it lands ON the
    curves -- these plots put the interesting data NEAR the floor by
    construction, so the one annotation that must stay readable is exactly the
    one most likely to be overplotted.
    """
    ax.axhline(value, color="0.35", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(x, value, f"{label} = {value:.4f} ", ha=ha, va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7.2, color="0.30",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.82))


def caption(fig, text, y=-0.02):
    """A caveat under the whole figure, where nothing can overplot it.

    Prefer this to `note` whenever the axes are busy. In-axes annotation kept
    colliding with legends and with the near-floor data these plots are ABOUT,
    and a caveat that lands on top of a curve is worse than no caveat: it
    obscures the reading and looks like an error.
    """
    fig.text(0.5, y, text, ha="center", va="top", fontsize=7.6, color="0.30",
             transform=fig.transFigure)


def note(ax, text, loc="lower left"):
    """A short caveat printed inside the axes, where it cannot be cropped off."""
    xy = {"lower left": (0.02, 0.03), "upper left": (0.02, 0.97),
          "lower right": (0.98, 0.03), "upper right": (0.98, 0.97)}[loc]
    ax.text(*xy, text, transform=ax.transAxes, fontsize=7.2, color="0.30",
            ha="right" if "right" in loc else "left",
            va="top" if "upper" in loc else "bottom")
