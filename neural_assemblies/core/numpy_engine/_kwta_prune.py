"""k-WTA is a SELECTION problem, so most of the drive never has to be computed.

The drive read is O(k*n): every column is gathered and summed, and then the
top-k is taken. But the top-k does not need the drive vector -- it needs to
know which k columns are largest, and for that an exact BOUND on a column is
as good as its value whenever the bound already loses.

The drive splits into three terms with completely different cost profiles:

    drive[c] = stim[c]        contiguous 1-D, cheap   -> compute EXACTLY
             + base[c]        expensive strided gather -> BOUND, never compute
             + corr[c]        sparse, O(nnz)           -> compute EXACTLY

`base` is the original Bernoulli connectivity, which is **0/1**, so

    base[c] = sum over active rows of b[i,c]  <=  |S|

with |S| the number of active presynaptic neurons, summed over fibers. That
bound is exact, not statistical. Since plasticity here is purely multiplicative
(``w = b * (1+beta)^t``), a cell is potentiated exactly when ``w > 1.0``, and

    drive[c] = base[c] + sum_{i in S, w>1} (w[i,c] - 1.0)
             = base[c] + corr[c]

so ``U[c] = stim[c] + |S| + corr[c] >= drive[c]``, exact except for one bounded
term. Any column with ``U[c] < tau`` -- tau the k-th largest EXACT drive among
the columns we did evaluate -- provably cannot enter the top-k, and is never
gathered.

**It is sublinear in area size.** The evaluated set is the union of potentiated
supports over the active rows, which is set by the ORGAN'S STRUCTURE (roughly
transitions-per-state * k) and does not grow with n. Measured on the Z60 organ
at k=70: 70 columns evaluated out of 19,999, and the advantage grows with n --
which is exactly the barrier to scaling to large brains, removable exactly.

THREE THINGS THAT TOOK THREE WRONG MEASUREMENTS TO GET RIGHT, and which every
change here has to keep:

1. **"Potentiated" is ``> 1.0``, NOT ``!= 1.0``.** Base cells are 0/1, so
   ``!= 1.0`` counts every structural ZERO as touched -- 19,994 candidates
   instead of 200, i.e. no prune at all. It is also WRONG rather than merely
   slow: a zero cell would contribute ``0 - 1 = -1`` to the correction and
   push the bound BELOW the true drive, which silently drops real winners.
2. **Bound in float64, values in float32.** In f32 the correction sum drifts
   ~1e-6 on drives of ~120 and produces spurious bound violations (f64: zero).
   The VALUES stay f32 and stay bit-identical, because gathering a column
   subset never reorders any column's row sum.
3. **Keep the drive vector FULL LENGTH with ``-inf`` in pruned slots.**
   Shrinking it changes the selector's tie-break, and this project has already
   had a tie-break change flip a verdict with no direction to it
   ([[exact-tables-are-tie-fragile]]). A pruned column has
   ``drive <= U < tau``, so it cannot tie with a winner, and retained columns
   keep their exact values at their own indices.

WHEN IT APPLIES, derived rather than tuned. The prune fires only when the
k-th largest evaluated drive `tau` clears the bound on everything else. For an
area fiber with no stimulus term that bound is `|S| = k`, and the k-th trained
column carries roughly `k*p * (1+beta)^T`, so the condition is

    p * (1 + beta)^T  >  1

i.e. the potentiated signal must exceed the base one. That is not a heuristic:
below it the bound genuinely cannot separate, and DECLINING is the correct
answer rather than a missed opportunity. It also reproduces the measured
fallback pattern on the Z60 organ (organ_p=0.5, k=70, beta=0.1) exactly:

    T= 8   0.5 * 1.1^8  = 1.07   marginal -- fallback / 8x
    T=12   0.5 * 1.1^12 = 1.57   35x
    T=16   0.5 * 1.1^16 = 2.29   60x / 208x
    T=24   0.5 * 1.1^24 = 4.93   60x / 286x

Note this is the SAME quantity that governs pattern completion -- the trained
gain has to beat the extreme value of the untrained pool. A brain deep enough
to complete a half cue is deep enough to prune, which is a pleasant place for
an optimisation's precondition to sit.

WHAT IS PRESERVED, EXACTLY, AND WHAT IS NOT. The winner SET and the resulting
CONNECTOME are bit-identical -- measured over 20 rounds x 3 stimuli at three
seeds, comparing the weight matrix itself rather than a summary. The ORDER of
winners that are EXACTLY TIED is not preserved.

The reason is that `heapq_select_top_k` uses `argpartition` followed by
`argsort`, both unstable, so the position a tied index lands in depends on
values ELSEWHERE in the array -- including the slots the prune left alone. Two
columns measured at 738.114563 each came back in the opposite order, and no
choice of filler value fixes that: an unstable partition reads the whole array.

This is not a defect in the bound, and it does not change which neurons fire.
It is stated here rather than buried because winner order reaches the
connectome through the ROW ORDER of the next gather, so a future change could
in principle turn a tie-order difference into a last-ulp value difference. It
has not here. Making the selector's tie-break canonical would fix it properly
and is a SCIENCE-AFFECTING change -- a tie-break change has already moved
sixteen cells of an exact table in this project with no direction to it
([[exact-tables-are-tie-fragile]]) -- so it needs its own registration and must
not be smuggled in as an optimisation.

GUARDS -- fall back, never guess. Each of these breaks the bound, so the
caller must not prune when any holds:

* ``input_noise_std > 0``  additive noise can lift ANY column, including one
  whose bound had already lost.
* ``norm_init``  the per-column ``1/d_j`` read-time divisor can scale an
  untouched column UP, so ``base <= |S|`` no longer bounds the contribution.
* ``synaptic_scaling``  rescales whole columns by an arbitrary factor, so
  ``w > 1.0`` stops meaning "potentiated" and the b/(1+beta)^t decomposition
  the bound rests on no longer holds.
* anything that reads the FULL drive vector -- ``record_activation``'s
  pre-kWTA snapshot. (``total_activation`` sums WINNERS only and is fine.)

Refraction and LRI are SUBTRACTIVE, so they can only lower a drive and the
bound survives them.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: A cell is potentiated iff its weight is STRICTLY above the unit base. See
#: note 1 above -- `!= 1.0` is both slower and wrong.
POTENTIATED = 1.0


class PotentiatedSupport:
    """Row -> columns that plasticity has ever touched, for one fiber.

    WHY THIS EXISTS RATHER THAN SCANNING THE WEIGHTS. The prune needs the
    SUPPORT of the potentiated cells, and a dense block stores their values but
    not their index -- finding them costs exactly the O(k*n) scan the prune is
    trying to avoid. Plasticity, on the other hand, knows precisely which cells
    it touched: `_apply_plasticity` multiplies the full cross product
    `rows x cols`. Recording that costs O(k^2) against a drive read of O(k*n).

    STORED CSR-STYLE, NOT AS A DICT OF ARRAYS. The first version kept a dict
    and looped over active rows, doing one small fancy-index gather each --
    which measured 6-10x SLOWER than the dense read it was meant to replace,
    because k separate gathers plus k dict lookups cost more than one
    contiguous pass. The row-pointer layout turns the whole read into a single
    vectorised gather, which is the only shape that can win.

    The index is a SUPERSET of the potentiated set: a touched cell whose base
    was 0 stays 0. That is why `correction` re-checks `> POTENTIATED` on the
    gathered VALUES rather than trusting membership.
    """

    __slots__ = ("_rows", "_dirty", "_ptr", "_cols", "_order")

    def __init__(self) -> None:
        self._rows: Dict[int, np.ndarray] = {}
        self._dirty: Dict[int, List[np.ndarray]] = {}
        self._ptr: Optional[np.ndarray] = None      # compiled CSR
        self._cols: Optional[np.ndarray] = None
        self._order: Optional[np.ndarray] = None    # row id -> CSR slot

    def __len__(self) -> int:
        return len(set(self._rows) | set(self._dirty))

    def note(self, rows: Iterable[int], cols: np.ndarray) -> None:
        """Record that every (row, col) pair in the cross product was touched.

        Deferred: the per-row union is materialised only when READ. A training
        loop notes far more often than it prunes, and doing the sort/unique per
        event made maintenance the dominant cost.
        """
        cols = np.asarray(cols, dtype=np.int64)
        if cols.size == 0:
            return
        for r in rows:
            self._dirty.setdefault(int(r), []).append(cols)
        self._ptr = None                     # invalidate the compiled form

    def clear(self) -> None:
        """Drop everything. The caller MUST do this whenever the index space it
        is keyed on is rebuilt -- consolidation renumbers compact indices
        ([[consolidation-resets-the-index-space]]), and a stale row->col map
        would then point at other neurons' columns. Dropping the index only
        costs the prune; keeping a wrong one costs the science."""
        self._rows.clear()
        self._dirty.clear()
        self._ptr = self._cols = self._order = None

    def _compile(self) -> None:
        """Fold pending events into per-row unions, then lay them out CSR."""
        for r, pend in self._dirty.items():
            have = self._rows.get(r)
            parts = pend if have is None else [have] + pend
            self._rows[r] = np.unique(np.concatenate(parts))
        self._dirty.clear()
        if not self._rows:
            self._ptr = np.zeros(1, dtype=np.int64)
            self._cols = np.empty(0, dtype=np.int64)
            self._order = np.empty(0, dtype=np.int64)
            return
        ids = np.fromiter(sorted(self._rows), dtype=np.int64,
                          count=len(self._rows))
        lens = np.fromiter((len(self._rows[int(r)]) for r in ids),
                           dtype=np.int64, count=len(ids))
        self._ptr = np.concatenate([[0], np.cumsum(lens)])
        self._cols = np.concatenate([self._rows[int(r)] for r in ids])
        self._order = ids

    def correction(self, rows: Sequence[int], weights, n_cols: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """``(corr, touched)`` for the active rows, in FLOAT64.

        `corr[c]` is the exact excess of column c above its unit base over the
        active rows; `touched` is the sorted set of columns with any
        potentiated cell there -- the columns the caller must evaluate.

        ONE vectorised gather, not one per row. float64 for the reason in the
        module docstring: the values stay float32, but summing the correction
        in f32 drifts enough to fire spurious bound violations, which turns a
        fast exact path into a slow one at random.
        """
        empty = (np.zeros(n_cols, dtype=np.float64),
                 np.empty(0, dtype=np.int64))
        if n_cols <= 0:
            return empty
        if self._ptr is None:
            self._compile()
        if self._order is None or len(self._order) == 0:
            return empty

        want = np.asarray(rows, dtype=np.int64)
        slot = np.searchsorted(self._order, want)
        slot = slot[slot < len(self._order)]
        if len(slot) == 0:
            return empty
        keep = self._order[slot] == want[:len(slot)]
        slot = slot[keep]
        if len(slot) == 0:
            return empty

        lens = self._ptr[slot + 1] - self._ptr[slot]
        total = int(lens.sum())
        if total == 0:
            return empty
        # Flat index into `_cols` for every (active row, touched col) pair,
        # built without a Python loop: repeat each row's start, then add the
        # within-row offset recovered from the running length total.
        starts = np.repeat(self._ptr[slot], lens)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(lens) - lens, lens)
        cols = self._cols[starts + within]
        rrep = np.repeat(self._order[slot], lens)

        live = cols < n_cols
        if not live.all():
            cols, rrep = cols[live], rrep[live]
            if len(cols) == 0:
                return empty
        vals = np.asarray(weights[rrep, cols], dtype=np.float64)
        pot = vals > POTENTIATED
        if not pot.any():
            return empty
        cols, vals = cols[pot], vals[pot] - POTENTIATED
        corr = np.bincount(cols, weights=vals, minlength=n_cols)
        return corr[:n_cols], np.unique(cols)


def evaluate_set(touched: np.ndarray, stim: Optional[np.ndarray], k: int,
                 n_cols: int) -> np.ndarray:
    """Columns that must be evaluated exactly: the potentiated support, plus
    the top-k of the stimulus term.

    THE STIMULUS TERM CANNOT BE BOUNDED THE WAY THE BASE CAN. It is stored
    pre-summed per column and is not 0/1, so a column with no potentiation at
    all can still win on stimulus drive alone. Including its top-k costs one
    `argpartition` over a contiguous 1-D array -- cheap, and skipping it is a
    silent correctness bug rather than a slow path.

    Returns a sorted, unique index array.
    """
    parts = [np.asarray(touched, dtype=np.int64)]
    if stim is not None and len(stim) and k > 0:
        m = min(int(k), len(stim))
        top = np.argpartition(np.asarray(stim, dtype=np.float64), -m)[-m:]
        parts.append(top.astype(np.int64))
    out = np.unique(np.concatenate(parts)) if len(parts) > 1 else parts[0]
    return out[out < n_cols]


def bound_outside(stim: Optional[np.ndarray], evaluated: np.ndarray,
                  n_cols: int, active_total: int) -> float:
    """Largest possible drive of a column that was NOT evaluated.

    ``max_{c not evaluated} stim[c] + |S|``. With no stimulus term this is
    just ``|S|``, because an unevaluated column has no potentiation and its
    base contribution is at most one per active row.
    """
    base = float(active_total)
    if stim is None or len(stim) == 0:
        return base
    s = np.asarray(stim, dtype=np.float64)[:n_cols]
    if len(evaluated) == 0:
        return base + float(s.max()) if len(s) else base
    mask = np.ones(len(s), dtype=bool)
    ev = evaluated[evaluated < len(s)]
    mask[ev] = False
    if not mask.any():
        return base                      # everything was evaluated
    return base + float(s[mask].max())


def tau_of(drives: np.ndarray, evaluated: np.ndarray, k: int) -> Optional[float]:
    """k-th largest EXACT drive among the evaluated columns, or None if fewer
    than k were evaluated (in which case the prune cannot decide anything)."""
    if len(evaluated) < k or k <= 0:
        return None
    vals = np.asarray(drives, dtype=np.float64)[evaluated]
    return float(np.partition(vals, -k)[-k])


def masked_drive(drives: np.ndarray, evaluated: np.ndarray) -> np.ndarray:
    """The drive vector at FULL LENGTH with `-inf` outside the evaluated set.

    Full length and `-inf` for the reason in note 3: shrinking the vector
    renumbers the columns and changes the selector's tie-break, and this
    project has had a tie-break change move sixteen table cells with no
    direction to it. A pruned column has `drive <= U < tau`, so `-inf` can
    never make it lose a tie it would otherwise have won -- it had already
    lost outright.
    """
    out = np.full(len(drives), -np.inf, dtype=drives.dtype)
    ev = evaluated[evaluated < len(drives)]
    out[ev] = drives[ev]
    return out
