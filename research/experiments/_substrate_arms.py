"""One area, M stored assemblies, the normalization SUBSTRATE ARMS.

Companion to `_substrate.py`, which is the shared harness (readout, probe,
distinctness invariants, statistics, ceiling estimator). This module owns only
the thing that harness has no opinion about: which normalization arm is being
built. Keep the split -- the harness is where the sharp edges are removed, and
adding arm-specific knobs to it would put a study's variable inside the tool
that is supposed to be neutral about studies.

Extracted from `seq_scaling_merger_forensics.py` so a second study can sweep
(n, k, p) instead of being nailed to that one's module-level constants. The
extraction is verified, not asserted: replaying the committed results JSON
through this module reproduces all 12 cells exactly. A shared helper that
quietly disagrees with the evidence it was extracted from is worse than the
duplication it removed.

THE THREE SUBSTRATES, which differ only in what divides the drive:

    B  norm_init         drive_j / d_j       d_j = COUNT of present synapses
                                             potentiation-INVARIANT, EVERY column
    C  synaptic_scaling  w[:,j] *= S/mass_j  mass_j = SUM of current weights
                                             potentiation-DEPENDENT, WINNERS only
    G  both              they cancel different things and compose

G exists because B and C were treated as rival substrates for months and are
not ([[norm-init-and-scaling-are-complementary]]).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import (
    _compact_index, _snap, activate_assembly,
)
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import assembly_overlap

from _substrate import check_distinct as _check_distinct

AREA = "A"


@dataclass(frozen=True)
class Cfg:
    """One operating point. Frozen so a cell cannot mutate the sweep."""
    n: int
    k: int
    p: float
    beta: float
    T: int
    M: int

    @property
    def chance(self) -> float:
        """Expected overlap of two independent k-subsets, as a fraction."""
        return self.k / self.n

    @property
    def kp(self) -> float:
        return self.k * self.p

    @property
    def floor(self) -> float:
        """The `k*p >= 3 ln n` connectivity floor [[SEQ-REGIME]]."""
        return 3.0 * math.log(self.n)

    @property
    def in_regime(self) -> bool:
        return self.kp >= self.floor


def build(cfg: Cfg, arm: str, seed: int) -> Brain:
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=cfg.p, seed=seed, engine="numpy_sparse",
                  recurrent_projection=True,
                  norm_init=(arm in ("B", "G")),
                  synaptic_scaling=(arm in ("C", "D", "E", "G")))
    brain.add_area(AREA, cfg.n, cfg.k, cfg.beta)
    eng = brain._engine_for(brain.areas[AREA])
    if arm == "D":
        eng.synaptic_scaling_setpoint = "degree"
    if arm == "E":
        eng.synaptic_scaling_scope = "all"
    return brain


def train(cfg: Cfg, arm: str, seed: int):
    """Store M assemblies in ONE area, recurrence on, one stimulus each."""
    brain = build(cfg, arm, seed)
    stims = []
    for i in range(cfg.M):
        s = f"s{i}"
        brain.add_stimulus(s, cfg.k)
        stims.append(s)
    stored = []
    for s in stims:
        brain.inhibit_areas([AREA])
        for _ in range(cfg.T):
            brain.project({s: [AREA]}, {AREA: [AREA]})
        stored.append(_snap(brain, AREA))
    return brain, stims, stored


def retrieve(brain, cfg: Cfg, stim=None, half_of=None):
    """One retrieval inside a probe; stimulus-cued or half-assembly-cued.

    `brain.probe()` because RECRUITMENT, not plasticity, is the channel by
    which a readout changes what it reads, and the snap is taken INSIDE the
    block ([[probe-isolation-required]]).
    """
    with brain.probe():
        brain.inhibit_areas([AREA])
        if half_of is not None:
            ids = np.asarray(half_of.winners)[: cfg.k // 2]
            activate_assembly(brain, Assembly(AREA, NeuronIds(ids)))
            for _ in range(cfg.T):
                brain.project({}, {AREA: [AREA]})       # recurrence alone
        else:
            for _ in range(cfg.T):
                brain.project({stim: [AREA]}, {AREA: [AREA]})
        return _snap(brain, AREA)


def column_stats(brain):
    """Per-column in-degree, mass, and the accumulated per-entry scalar.

    The scalar is the MEDIAN nonzero entry, not the mean: the ~k*p potentiated
    entries sit far above the rest, and what matters is the baseline every
    OTHER synapse of that neuron was multiplied by -- the part that carries
    over to unrelated cues.

    `materialized_count`, not `.w`: `.w` means neurons materialized on the
    sparse engine but len(winners) == k on the explicit one, and every
    statistic here is over the materialized population.
    """
    eng = brain._engine_for(brain.areas[AREA])
    conn = eng._area_conns[AREA][AREA]
    w = np.asarray(conn.weights)
    rows = int(eng.materialized_count(AREA) or 0)
    cols = min(rows, w.shape[1])
    sub = np.asarray(w[:rows, :cols], dtype=np.float64)

    nz = sub != 0.0
    nnz = nz.sum(axis=0).astype(np.float64)
    mass = sub.sum(axis=0)
    # Vectorized nanmedian rather than a per-column Python loop: at the large-n
    # end of the sweep `cols` is in the thousands and the loop dominated the
    # cell. Columns with no synapses at all are all-NaN, hence the guard.
    masked = np.where(nz, sub, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(masked, axis=0)
    med = np.where(np.isfinite(med), med, 1.0)
    return nnz, mass, med, rows, cols


def multiplicity(brain, stored, cols):
    """How many of the M stored assemblies contain each compact column.

    Compact index is also RECRUITMENT ORDER, which is why callers plot against
    it: substrate C's multiply-shared elite lives at the very front of it.
    """
    inv = _compact_index(brain._engine_for(brain.areas[AREA]), AREA) or {}
    mult = np.zeros(cols, dtype=np.int32)
    for st in stored:
        c = [inv[int(x)] for x in np.asarray(st.winners)
             if int(x) in inv and inv[int(x)] < cols]
        mult[c] += 1
    return mult


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def score(cfg: Cfg, arm: str, seed: int) -> dict:
    """Train one cell and read every headline off it.

    RANK-1 FROM THE FULL CUE IS THE WEAK READOUT and is reported, never gated
    on: substrate C scores 1.000 on it while its assemblies are merged at 7x
    chance, because the stimulus does the discriminating. Only HALF-cue
    retrieval and PAIRWISE OVERLAP separate an attractor from a lookup table
    ([[completion-works-in-regime-under-norm-init]]).
    """
    brain, stims, stored = train(cfg, arm, seed)
    nnz, mass, med, rows, cols = column_stats(brain)
    mult = multiplicity(brain, stored, cols)
    seen = mult > 0

    full_hits = half_hits = 0
    for i, s in enumerate(stims):
        live = retrieve(brain, cfg, stim=s)
        ov = [assembly_overlap(np.asarray(live.winners), np.asarray(st.winners))
              for st in stored]
        full_hits += int(np.argmax(ov) == i)
        liveh = retrieve(brain, cfg, half_of=stored[i])
        ovh = [assembly_overlap(np.asarray(liveh.winners),
                                np.asarray(st.winners)) for st in stored]
        half_hits += int(np.argmax(ovh) == i)

    pair = [assembly_overlap(np.asarray(stored[i].winners),
                             np.asarray(stored[j].winners))
            for i in range(cfg.M) for j in range(i + 1, cfg.M)]

    def _m(x, mask):
        return float(np.mean(x[mask])) if mask.any() else float("nan")

    pairwise = float(np.mean(pair)) if pair else 0.0
    return {
        "arm": arm, "seed": seed, "rows": rows, "cols": cols,
        "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
        "T": cfg.T, "M": cfg.M, "chance": cfg.chance,
        "rank1_full": full_hits / cfg.M,
        "rank1_half": half_hits / cfg.M,
        "pairwise": pairwise,
        "pairwise_x_chance": pairwise / cfg.chance,
        "union": int(seen.sum()),
        "max_mult": int(mult.max()) if cols else 0,
        "frac_shared": float((mult >= 2).sum() / max(1, seen.sum())),
        # MEAN PAIRWISE OVERLAP IS NEARLY BLIND TO PARTIAL COLLAPSE: 256 items
        # landing on ~58 distinct assemblies moves the mean by almost nothing,
        # because only ~1.3% of PAIRS are identical. Duplicates are what
        # actually destroys a read -- two items with the same assembly are
        # unrecoverable however well separated everything else is. So report
        # the distinct fraction directly, alongside the harness's own verdict.
        # See `_substrate.check_distinct` and [[mean-spread-is-a-degree-statistic]].
        "distinct_frac": float(
            len({tuple(sorted(int(x) for x in np.asarray(st.winners)))
                 for st in stored}) / max(1, len(stored))),
        "distinct_note": _check_distinct(
            [np.asarray(st.winners) for st in stored], cfg.n, cfg.k)[1],
        "deg_pop": float(np.mean(nnz)) if cols else float("nan"),
        # MEASURED degree heterogeneity of the candidate pool. The committed
        # mechanism says the merger is uncancelled candidate in-degree, so this
        # is the quantity it should track across operating points -- the one
        # derived, falsifiable prediction available off a single point.
        "deg_cv": (float(np.std(nnz) / np.mean(nnz))
                   if cols and np.mean(nnz) > 0 else float("nan")),
        "deg_private": _m(nnz, mult == 1),
        "deg_shared": _m(nnz, mult >= 2),
        # rho(in-degree, multiplicity) is UNDEFINED when no neuron is in more
        # than one assembly, because there is no multiplicity to correlate
        # with. That is not missing data -- it is the BEST possible outcome,
        # perfect distinctness, and it happens preferentially on the arms that
        # work. Dropping those seeds would bias the statistic toward the arms
        # that fail ([[undefinedness-correlates-with-outcome]]), and leaving
        # NaN makes `ensemble_from_values` refuse the whole run. Read it as
        # 0.0: no degree-multiplicity association, and say so here rather than
        # let a downstream nan_to_num do it silently.
        "rho_deg_mult": (0.0 if int(mult.max() if cols else 0) <= 1
                         else spearman(nnz[seen], mult[seen].astype(float))),
        "med_private": _m(med, mult == 1),
        "med_shared": _m(med, mult >= 2),
    }


def worker(n, k, p, beta, T, M, arm, seed):
    """Top-level, all-scalar signature: Windows SPAWNS the pool, so a cell must
    carry its whole configuration rather than read module globals the parent
    mutated -- that silent-no-op shape is how a sweep comes back flat."""
    return score(Cfg(n=n, k=k, p=p, beta=beta, T=T, M=M), arm, seed)
