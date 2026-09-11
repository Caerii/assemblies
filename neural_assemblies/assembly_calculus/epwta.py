"""E%-WTA assembly formation (Hoff et al. 2026).

Implements the paper's formation procedure, which differs from the original
Assembly Calculus in both *how* neurons are selected and *when* an assembly is
considered to exist.

Selection (Eq. 4-6) is E%-winners-take-all: the most-stimulated neuron fires
first and recruits interneurons, so only neurons within ``eps = d / tau_m`` of
the peak fire before inhibition arrives. Assembly size is emergent rather than
fixed at ``k``.

Formation (Eq. 8-11) is a stationary state of the dynamics, not merely "no new
winners". The original criterion (``N_t == 0``) does not guarantee the same
neurons keep firing -- the paper's Fig. 2b shows the firing set still churning
after new winners stop appearing. Three conditions are required instead:

    i)   Stationary pattern   |F_t| == |F_t+1|                        (Eq. 8)
    ii)  Synchronization      X_t == {} (no neuron fires for the
         first time this step)                                       (Eq. 9)
    iii) Higher synaptic density than the host area, D_A > D_M = p_s  (Eq. 10)

with density measured on the directed graph (Eq. 11)::

    D = |S| / (|N| (|N| - 1))

plus a floor of ``min_size`` neurons, since conditions (i) and (ii) are
trivially satisfied by groups of one or two neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


__all__ = [
    "FormationResult", "assembly_density", "form_assembly", "recover_assembly",
]

# Paper Table 1 defaults.
DEFAULT_D_MS = 3.0
DEFAULT_TAU_M_MS = 30.0
DEFAULT_MIN_SIZE = 6
# Window width in standard deviations for selection="sigma" (see _select).
DEFAULT_SIGMA_C = 1.7


def _select(h: np.ndarray, *, mode: str, eps: float, sigma_c: float) -> np.ndarray:
    """Which neurons fire this cycle.

    ``mode="epsilon"`` is the paper's rule (Eq. 5): fire if
    ``h_j >= (1 - eps) * h_max``. The window is a fixed FRACTION OF THE MAX,
    i.e. measured from zero.

    ``mode="sigma"`` fires if ``h_j >= h_max - sigma_c * std(h)`` -- the same
    idea with the window measured in spreads of the input distribution.

    Why the second exists: writing ``h ~ mu + sigma*z``, the paper's rule puts
    its lower edge at ``z_low = (1-eps) z_max - eps * mu/sigma``. Since
    ``mu/sigma ~ sqrt(k_s p_s)``, shrinking connectivity or stimulus size drives
    ``z_low -> (1-eps) z_max`` and the window collapses onto a handful of
    neurons, which then fail the size and density conditions. That is the
    mechanism behind the paper's Fig. 4a-b failures, and it is a sensitivity to
    the SHAPE of the input distribution, not its scale. Measuring the window in
    units of sigma removes the mu/sigma term and is scale- and shape-invariant.

    Measured success rate (beta=0.01, 40 seeds):

        regime                 epsilon    sigma
        p_s=.5  k_s=200          0.93      0.95   <- paper's own regime
        p_s=.1  k_s=200          0.35      0.97
        p_s=.05 k_s=200          0.20      0.97
        p_s=.05 k_s=60           0.03      0.75

    Biologically this is divisive normalization by population activity
    (Carandini & Heeger): the inhibitory feedback pool is driven by the summed
    drive, so the effective window tracks the spread of the input distribution
    rather than a fixed proportion of its peak.
    """
    if h.size == 0:
        return np.array([], dtype=int)
    m = float(h.max())
    if m <= 0.0:
        # Eq. 5 is empty for h_max < 0, and F_0 = {} for h_max == 0 (footnote 2).
        return np.array([], dtype=int)
    if mode == "sigma":
        s = float(h.std())
        if s <= 0.0:
            return np.array([], dtype=int)
        return np.where(h >= m - sigma_c * s)[0]
    return np.where(h >= (1.0 - eps) * m)[0]


def _potentiate(mat: np.ndarray, rows, cols, beta: float, meta: float) -> None:
    """Hebbian update (Eq. 3), optionally with metaplasticity.

    ``meta > 0`` scales the effective rate as ``beta / (1 + meta*|w|)``, so a
    synapse becomes harder to change the stronger it already is (Fusi-style
    cascade). Uniform plasticity is what lets a later assembly overwrite an
    earlier one: a synapse belonging to a consolidated assembly is exactly as
    writable as a fresh one. Measured retrieval of the FIRST of 10 assemblies
    stored in one area: 0.52 without, 0.83 with.
    """
    sub = mat[np.ix_(rows, cols)] if rows is not None else mat[:, cols]
    new = sub * (1.0 + (beta / (1.0 + meta * np.abs(sub)) if meta > 0 else beta))
    if rows is not None:
        mat[np.ix_(rows, cols)] = new
    else:
        mat[:, cols] = new


@dataclass
class FormationResult:
    """Outcome of one E%-WTA formation run."""

    winners: np.ndarray                 # final firing set F_T
    iterations: int                     # T
    formed: bool                        # all conditions satisfied
    density: float                      # D_A
    area_density: float                 # D_M = p_s
    stationary: bool                    # Eq. 8
    synchronized: bool                  # Eq. 9
    dense_enough: bool                  # Eq. 10
    big_enough: bool                    # |A| >= min_size
    size_history: List[int] = field(default_factory=list)
    new_winner_history: List[int] = field(default_factory=list)
    failure: Optional[str] = None

    @property
    def size(self) -> int:
        return int(len(self.winners))


def assembly_density(adjacency: np.ndarray, members: Sequence[int]) -> float:
    """Directed synaptic density of ``members`` (Eq. 11).

    ``D = |S| / (|N|(|N|-1))`` counting only synapses among the members.
    Self-loops are excluded, matching the ``|N|(|N|-1)`` denominator.
    """
    idx = np.asarray(sorted(set(int(m) for m in members)), dtype=int)
    n = len(idx)
    if n < 2:
        return 0.0
    sub = adjacency[np.ix_(idx, idx)]
    edges = float(np.count_nonzero(sub) - np.count_nonzero(np.diag(sub)))
    return edges / (n * (n - 1))


def form_assembly(
    *,
    n: int = 1000,
    stimulus_size: int = 200,
    p_s: float = 0.5,
    p_i: float = 0.2,
    w_inh: float = -0.2,
    beta: float = 0.01,
    epsilon: Optional[float] = None,
    selection: str = "epsilon",
    sigma_c: float = DEFAULT_SIGMA_C,
    metaplasticity: float = 0.0,
    max_iters: int = 200,
    min_size: int = DEFAULT_MIN_SIZE,
    seed: int = 0,
    rng: Optional[np.random.Generator] = None,
    adjacency: Optional[np.ndarray] = None,
    recurrent: Optional[np.ndarray] = None,
    stim_weights: Optional[np.ndarray] = None,
) -> FormationResult:
    """Run the paper's formation loop for a single assembly.

    A stimulus of ``stimulus_size`` neurons fires every step (they "continue
    to fire throughout the entire process"); the memory area receives that
    drive plus recurrence from its own previous firing set, and selection is
    E%-WTA. Iteration stops when Eq. 8 and Eq. 9 both hold; Eq. 10 and the
    size floor are then evaluated to decide whether an assembly formed.

    Passing ``adjacency``/``recurrent``/``stim_weights`` reuses an existing
    network, which is how multiple assemblies are stored in one area.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    eps = epsilon if epsilon is not None else DEFAULT_D_MS / DEFAULT_TAU_M_MS

    # Eq. 7: present with prob p_s; inhibitory among those with prob p_i.
    if recurrent is None:
        present = rng.random((n, n)) < p_s
        recurrent = present.astype(np.float64)
        inh = present & (rng.random((n, n)) < p_i)
        recurrent[inh] = w_inh
        np.fill_diagonal(recurrent, 0.0)
    if adjacency is None:
        adjacency = (recurrent != 0.0)
    if stim_weights is None:
        present_s = rng.random((stimulus_size, n)) < p_s
        stim_weights = present_s.astype(np.float64)
        inh_s = present_s & (rng.random((stimulus_size, n)) < p_i)
        stim_weights[inh_s] = w_inh

    ever_fired = np.zeros(n, dtype=bool)
    prev: np.ndarray = np.array([], dtype=int)
    sizes: List[int] = []
    new_counts: List[int] = []
    stationary = synchronized = False
    iterations = 0

    for t in range(1, max_iters + 1):
        iterations = t
        # h_j(t) = sum_i f_i(t-1) w_ij, over the stimulus (always on) and the
        # memory area's own previous firing set.
        h = stim_weights.sum(axis=0)
        if len(prev):
            h = h + recurrent[prev].sum(axis=0)

        cur = _select(h, mode=selection, eps=eps, sigma_c=sigma_c)

        # Hebbian update (Eq. 3) on the synapses that just co-fired.
        if len(cur):
            _potentiate(stim_weights, None, cur, beta, metaplasticity)
            if len(prev):
                _potentiate(recurrent, prev, cur, beta, metaplasticity)

        newly = cur[~ever_fired[cur]] if len(cur) else cur
        ever_fired[cur] = True
        sizes.append(int(len(cur)))
        new_counts.append(int(len(newly)))

        stationary = len(prev) == len(cur)            # Eq. 8
        synchronized = len(newly) == 0                # Eq. 9
        prev = cur
        if stationary and synchronized and len(cur) > 0:
            break

    density = assembly_density(adjacency, prev)
    dense_enough = density > p_s                      # Eq. 10
    big_enough = len(prev) >= min_size
    formed = bool(stationary and synchronized and dense_enough and big_enough)

    failure = None
    if not formed:
        if not (stationary and synchronized):
            failure = "did_not_converge"
        elif not big_enough:
            failure = "too_small"
        elif not dense_enough:
            failure = "density_below_area"

    return FormationResult(
        winners=prev, iterations=iterations, formed=formed,
        density=density, area_density=p_s,
        stationary=stationary, synchronized=synchronized,
        dense_enough=dense_enough, big_enough=big_enough,
        size_history=sizes, new_winner_history=new_counts, failure=failure,
    )


def recover_assembly(
    result: FormationResult,
    *,
    stim_weights: np.ndarray,
    recurrent: np.ndarray,
    epsilon: Optional[float] = None,
    selection: str = "epsilon",
    sigma_c: float = DEFAULT_SIGMA_C,
    rounds: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Recovered portion ``|A|_rec / |A|`` when the same stimulus is replayed.

    Plasticity is off: this reads the stored assembly rather than re-training
    it, which is what the paper's retrieval measurement does.
    """
    if result.size == 0:
        return 0.0
    rng = rng if rng is not None else np.random.default_rng(0)
    eps = epsilon if epsilon is not None else DEFAULT_D_MS / DEFAULT_TAU_M_MS

    prev: np.ndarray = np.array([], dtype=int)
    for _ in range(rounds):
        h = stim_weights.sum(axis=0)
        if len(prev):
            h = h + recurrent[prev].sum(axis=0)
        prev = _select(h, mode=selection, eps=eps, sigma_c=sigma_c)

    target = set(int(x) for x in result.winners)
    got = set(int(x) for x in prev)
    return len(target & got) / len(target)
