"""[COLT22] Theorem 6 (Learning Linear Thresholds) -- numpy port of the paper's code.

Port of ``.reference/mdabagia-learning-with-assemblies/Halfspace.ipynb``, the
notebook accompanying Dabagia, Papadimitriou & Vempala, "Learning with
Assemblies of Neurons" (COLT 2022).  It exists so the theorem has a GOLDEN
recorded from the authors' own protocol before anything is asserted about this
package's engine -- see ``research/literature/parity/PROTOCOLS.md``.

THE CLAIM (Theorem 6).  Let ``v`` be a nonnegative unit vector.  Define ``D+``
with coordinates ``Bernoulli(k/n + delta * v_i)`` and ``D-`` with coordinates
``Bernoulli(k/n)``.  Then presenting ``Omega(log k)`` samples from ``D+`` forms
an assembly ``A*`` such that, with probability ``1 - o(1)``, a fresh ``D+``
sample produces a cap overlapping **at least 3k/4** of ``A*`` and a ``D-``
sample one overlapping **at most k/4**.

FOUR DETAILS A HAND-BUILT VERSION GETS WRONG, each of which alone breaks the
reproduction:

* ``beta = 1.0``.  Not the 0.1 used elsewhere in this repo.
* Weights are column-normalized at init **and again after training, before
  evaluation**.  This is the theorem's stated assumption that "all of the
  incoming weights to a neuron are normalized to sum to 1" *between training
  and evaluation* -- ongoing homeostasis, not the one-time ``norm_init`` that
  ``.reference/mdabagia-nemo/brain.py`` applies from ``reset()`` alone.
* Evaluation is **recurrent**: ``k_cap(out @ W + x @ A)``.  The recurrent fiber
  is load-bearing, not decorative.
* The classes are a **halfspace with margin**, not two disjoint supports.
  Positives carry ``n_on_p`` of the first ``n/8`` coordinates, negatives
  ``n_on_n`` (which is <= 0 at the default margin, i.e. none), and both draw
  ``Bernoulli(k/n)`` over the remaining coordinates.  The supports therefore
  OVERLAP, which is what makes the claim non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

__all__ = ["ColtHalfspaceResult", "run_colt_halfspace"]


@dataclass
class ColtHalfspaceResult:
    """Outcome of one Theorem 6 reproduction."""

    pos_overlap: float
    neg_overlap: float
    per_seed_pos: List[float]
    per_seed_neg: List[float]
    n_on_pos: int
    n_on_neg: int
    parameters: Dict[str, float] = field(default_factory=dict)

    @property
    def separates(self) -> bool:
        """Whether both halves of the theorem's bound hold on the means."""
        k = float(self.parameters["cap_size"])
        return self.pos_overlap >= 0.75 * k and self.neg_overlap <= 0.25 * k


def k_cap(x: np.ndarray, cap_size: int) -> np.ndarray:
    """Notebook's ``k_cap``: indicator of the ``cap_size`` largest entries."""
    out = np.zeros_like(x)
    if x.ndim == 1:
        out[np.argsort(x)[-cap_size:]] = 1
    else:
        np.put_along_axis(out, np.argsort(x, axis=-1)[:, -cap_size:], 1, axis=-1)
    return out


def _make_samples(rng, n_samples, n_in, cap_size, margin):
    """Halfspace-with-margin classes (notebook cell 4)."""
    halfspace = np.zeros(n_in)
    halfspace[:cap_size] = 1 / np.sqrt(cap_size)          # ||v||_2 == 1
    mean = halfspace.sum() * cap_size / n_in
    tail = n_in // 8
    pos = np.zeros((n_samples, n_in))
    neg = np.zeros((n_samples, n_in))
    pos[:, tail:] = rng.random((n_samples, n_in - tail)) < cap_size / n_in
    neg[:, tail:] = rng.random((n_samples, n_in - tail)) < cap_size / n_in
    n_on_p = int(np.ceil((mean + margin) * np.sqrt(n_in / 8)))
    n_on_n = int(np.floor((mean - margin) * np.sqrt(n_in / 8)))
    for i in range(n_samples):
        pos[i, rng.choice(tail, size=n_on_p, replace=False)] = 1.0
        if n_on_n > 0:
            neg[i, rng.choice(tail, size=n_on_n, replace=False)] = 1.0
    return pos, neg, n_on_p, n_on_n


def _one_seed(seed, n_in, n_neurons, cap_size, sparsity, n_rounds, beta, margin):
    rng = np.random.default_rng(seed)
    mask_w = (rng.random((n_neurons, n_neurons)) < sparsity) & ~np.eye(
        n_neurons, dtype=bool)
    mask_a = rng.random((n_in, n_neurons)) < sparsity
    pos, neg, n_on_p, n_on_n = _make_samples(
        rng, n_neurons, n_in, cap_size, margin)

    W = np.ones((n_neurons, n_neurons)) * mask_w
    A = np.ones((n_in, n_neurons)) * mask_a
    W /= W.sum(axis=0)
    A /= A.sum(axis=0)

    # Training: Omega(log k) sequential positive samples.
    act = np.zeros(n_neurons)
    for j in range(n_rounds):
        x = pos[j]
        new = k_cap(act @ W + x @ A, cap_size)
        A[(x > 0)[:, None] & (new > 0)[None, :]] *= 1 + beta
        W[(act > 0)[:, None] & (new > 0)[None, :]] *= 1 + beta
        act = new
    a_star = act.copy()

    # Homeostasis between training and evaluation -- Theorem 6's assumption.
    A /= A.sum(axis=0)
    W /= W.sum(axis=0)

    # Evaluation: recurrent, from silence, all samples in parallel.
    out_p = np.zeros((n_neurons, n_neurons))
    out_n = np.zeros((n_neurons, n_neurons))
    for _ in range(n_rounds):
        out_p = k_cap(out_p @ W + pos @ A, cap_size)
        out_n = k_cap(out_n @ W + neg @ A, cap_size)

    return (float((out_p * a_star).sum(axis=1).mean()),
            float((out_n * a_star).sum(axis=1).mean()),
            n_on_p, n_on_n)


def run_colt_halfspace(
    seeds=(1, 2, 3, 4, 5),
    n_in: int = 1000,
    n_neurons: int = 1000,
    cap_size: int = 100,
    sparsity: float = 0.1,
    n_rounds: int = 5,
    beta: float = 1.0,
    margin: int = 3,
) -> ColtHalfspaceResult:
    """Run the notebook's Theorem 6 protocol over *seeds* and average."""
    rows = [_one_seed(s, n_in, n_neurons, cap_size, sparsity,
                      n_rounds, beta, margin) for s in seeds]
    pos = [r[0] for r in rows]
    neg = [r[1] for r in rows]
    return ColtHalfspaceResult(
        pos_overlap=float(np.mean(pos)),
        neg_overlap=float(np.mean(neg)),
        per_seed_pos=[float(x) for x in pos],
        per_seed_neg=[float(x) for x in neg],
        n_on_pos=rows[0][2],
        n_on_neg=max(rows[0][3], 0),
        parameters=dict(
            n_in=n_in, n_neurons=n_neurons, cap_size=cap_size,
            sparsity=sparsity, n_rounds=n_rounds, beta=beta, margin=margin,
            seeds=len(list(seeds)),
        ),
    )
