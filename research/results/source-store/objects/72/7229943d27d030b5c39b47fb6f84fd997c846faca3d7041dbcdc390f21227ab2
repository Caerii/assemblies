"""Multi-assembly storage in ONE area on a genuinely SHARED substrate.

``epwta.form_assembly`` builds (or accepts) a stimulus matrix whose rows ALL
fire.  Storing several items by handing each one its own ``stim_weights`` --
or its own disjoint slice of rows -- makes the stimulus->area weights private
to that item, so later items cannot overwrite earlier ones through the
feedforward path.  That is the documented trap that makes forgetting invisible
(``research/experiments/capacity/REPORT.md``, finding 6).

Here every item is a fixed ``k_s``-subset of ONE shared stimulus pool of
``S`` rows, and all items drive one shared recurrent connectome.  Subsets
overlap by chance, so consolidating item i does perturb the afferents of
item j.

The selection and potentiation primitives are imported from
``neural_assemblies.assembly_calculus.epwta`` so that the dynamics are exactly
the library's, not a re-implementation.  ``neural_assemblies/`` is read-only.
"""

from __future__ import annotations

import sys
from neural_assemblies.assembly_calculus.metrics import cosine_similarity
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_assemblies.assembly_calculus.epwta import (  # noqa: E402
    DEFAULT_D_MS, DEFAULT_MIN_SIZE, DEFAULT_SIGMA_C, DEFAULT_TAU_M_MS,
    _potentiate, _select, assembly_density,
)

EPS = DEFAULT_D_MS / DEFAULT_TAU_M_MS   # 0.1


def select(h: np.ndarray, *, policy: str, k: int, sigma_c: float) -> np.ndarray:
    """E%-WTA window (library ``_select``) or fixed-k top-k."""
    if policy == "topk":
        if h.size == 0:
            return np.array([], dtype=int)
        kk = min(k, h.size)
        idx = np.argpartition(-h, kk - 1)[:kk]
        return np.sort(idx)
    return _select(h, mode=policy, eps=EPS, sigma_c=sigma_c)


@dataclass
class Substrate:
    n: int
    p_s: float
    p_i: float
    w_inh: float
    recurrent: np.ndarray
    adjacency: np.ndarray
    stim: np.ndarray            # (S, n)
    item_rows: List[np.ndarray]  # k_s row indices per item


def build_substrate(*, n: int, pool: int, k_s: int, n_items: int,
                    p_s: float, p_i: float, w_inh: float,
                    seed: int) -> Substrate:
    rng = np.random.default_rng(seed)
    present = rng.random((n, n)) < p_s
    recurrent = present.astype(np.float64)
    inh = present & (rng.random((n, n)) < p_i)
    recurrent[inh] = w_inh
    np.fill_diagonal(recurrent, 0.0)
    adjacency = present.copy()
    np.fill_diagonal(adjacency, False)

    present_s = rng.random((pool, n)) < p_s
    stim = present_s.astype(np.float64)
    inh_s = present_s & (rng.random((pool, n)) < p_i)
    stim[inh_s] = w_inh

    rows = [np.sort(rng.choice(pool, size=k_s, replace=False))
            for _ in range(n_items)]
    return Substrate(n=n, p_s=p_s, p_i=p_i, w_inh=w_inh, recurrent=recurrent,
                     adjacency=adjacency, stim=stim, item_rows=rows)


def store(sub: Substrate, item: int, *, policy: str, k: int,
          beta: float, metaplasticity: float, sigma_c: float = DEFAULT_SIGMA_C,
          max_iters: int = 100, min_size: int = DEFAULT_MIN_SIZE) -> dict:
    """Run the Eq. 8-11 formation loop for one item, with plasticity ON.

    Mirrors ``epwta.form_assembly`` step for step; the only change is that the
    stimulus drive comes from this item's rows of the SHARED pool and the
    Hebbian update touches only those rows.
    """
    rows = sub.item_rows[item]
    ever = np.zeros(sub.n, dtype=bool)
    prev = np.array([], dtype=int)
    stationary = synchronized = False
    iterations = 0
    for t in range(1, max_iters + 1):
        iterations = t
        h = sub.stim[rows].sum(axis=0)
        if len(prev):
            h = h + sub.recurrent[prev].sum(axis=0)
        cur = select(h, policy=policy, k=k, sigma_c=sigma_c)
        if len(cur):
            _potentiate(sub.stim, rows, cur, beta, metaplasticity)
            if len(prev):
                _potentiate(sub.recurrent, prev, cur, beta, metaplasticity)
        newly = cur[~ever[cur]] if len(cur) else cur
        ever[cur] = True
        stationary = len(prev) == len(cur)
        synchronized = len(newly) == 0
        prev = cur
        if stationary and synchronized and len(cur) > 0:
            break
    density = assembly_density(sub.adjacency, prev)
    big = len(prev) >= min_size
    dense = density > sub.p_s
    return {"winners": prev, "size": int(len(prev)), "iterations": iterations,
            "formed": bool(stationary and synchronized and dense and big),
            "density": float(density), "big_enough": bool(big),
            "dense_enough": bool(dense),
            "converged": bool(stationary and synchronized)}


def retrieve(sub: Substrate, item: int, *, policy: str, k: int,
             sigma_c: float = DEFAULT_SIGMA_C, rounds: int = 3) -> np.ndarray:
    """Replay the item's stimulus with plasticity OFF (as ``recover_assembly``)."""
    rows = sub.item_rows[item]
    prev = np.array([], dtype=int)
    for _ in range(rounds):
        h = sub.stim[rows].sum(axis=0)
        if len(prev):
            h = h + sub.recurrent[prev].sum(axis=0)
        prev = select(h, policy=policy, k=k, sigma_c=sigma_c)
    return prev


def cosine_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Compatibility name for the canonical set cosine metric."""
    return cosine_similarity(a.tolist(), b.tolist())
