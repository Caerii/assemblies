# sparse_simulation.py

"""
Sparse simulation engine for neural assembly dynamics.

This module contains the complex sparse simulation algorithms extracted
from the root brain.py projection logic, including dynamic connectome expansion,
first-time winner tracking, and input distribution management.

Biological Context:
- Implements sparse simulation of neural assemblies
- Manages dynamic expansion of connectomes when new neurons fire
- Handles input distribution from multiple brain areas and stimuli
- Maintains synaptic connectivity patterns for learning

Assembly Calculus Context:
- Enables efficient simulation of large-scale neural networks
- Supports dynamic assembly formation and connectivity
- Models synaptic plasticity in sparse neural populations

Mathematical Foundation:
- Connectome expansion: Dynamic matrix resizing with padding
- Input distribution: Multinomial sampling across input sources
- Winner indexing: First-time vs repeat winner tracking
- Synaptic assignment: Bernoulli sampling for connectivity
"""

import math
from functools import lru_cache

import numpy as np
from typing import List, Dict, Optional, Tuple, Any

try:
    from ..core.backend import get_xp, to_cpu, to_xp, xp_by_name, xp_name
except ImportError:
    from core.backend import get_xp, to_cpu, to_xp, xp_by_name, xp_name


@lru_cache(maxsize=256)
def _binom_ppf_cached(quantile_num: int, quantile_den: int,
                      total_k: int, p: float) -> float:
    """Cached binomial inverse CDF, computed WITHOUT importing ``scipy.stats``.

    Uses integer numerator/denominator for the quantile so the cache key
    is exact (no floating-point hash issues).

    WHY NOT ``scipy.stats.binom.ppf``.  This is one scalar per parameter
    combination, and it used to cost the entire ``scipy.stats`` package: +1.09s
    and +429 modules ON TOP of the ``scipy.special`` this file already needs
    for ``ndtr``/``ndtri``.  Because it runs inside the FIRST projection, every
    process paid it -- measured at 981ms of a 1.7s lexicon build whose actual
    projection work was 132ms.  A sweep that spawns a process per trial paid it
    per trial.

    The body below is ``scipy.stats.binom._ppf`` itself, which is defined in
    terms of ``scipy.special`` only.  Not an approximation: 140/140 parameter
    combinations (``total_k`` in {50..5000} x ``p`` in {0.01..0.3} x five
    quantiles) return bit-identical values to ``binom.ppf``.  The
    ``bdtr(vals-1) >= q`` step is the off-by-one correction for ``bdtrik``
    landing exactly on an integer, and dropping it is what would make this an
    approximation rather than a reimplementation.
    """
    from scipy.special import bdtr, bdtrik

    q = quantile_num / quantile_den
    vals = np.ceil(bdtrik(q, total_k, p))
    vals_below = np.maximum(vals - 1, 0)
    return float(np.where(bdtr(vals_below, total_k, p) >= q, vals_below, vals))

class SparseSimulationEngine:
    """
    Sparse simulation engine for neural assembly dynamics.

    This class implements the complex algorithms for sparse neural simulation
    extracted from the root brain.py, including dynamic connectome expansion,
    winner tracking, and input distribution management.
    """

    def __init__(self, rng: np.random.Generator, xp=None):
        """
        Initialize the sparse simulation engine.

        Args:
            rng (np.random.Generator): Random number generator for reproducibility
            xp: Array module to build results with. Defaults to whatever
                ``get_xp()`` says AT CONSTRUCTION -- captured once, not re-read
                per call. The owning engine passes its own ``_xp`` so that a
                later ``set_backend`` elsewhere in the process cannot change
                what this simulator returns to an engine already running. That
                leak is what made a CuPy engine constructed anywhere retroactively
                hand CuPy arrays to an existing numpy engine.
        """
        self.rng = rng
        # A NAME, not the module -- see `backend.xp_name`.
        self._xp_name = xp_name(xp)

    @property
    def _xp(self):
        return xp_by_name(self._xp_name)

    def calculate_input_distribution(self, input_sizes: List[int],
                                  first_winner_inputs: List[float]) -> List[np.ndarray]:
        """
        Calculate how inputs should be distributed across multiple sources.

        This implements the complex input distribution logic from root brain.py
        lines 757-773, which determines how synaptic connections from each
        input source should be assigned to new winners.

        Args:
            input_sizes (List[int]): Size of each input source
            first_winner_inputs (List[float]): Input strength for each new winner

        Returns:
            List[np.ndarray]: Input distribution for each new winner

        Biological Context:
            Each new winner receives synaptic input from multiple brain areas
            and stimuli. This method determines the connection strength from
            each input source based on the winner's total input strength.

        Mathematical Context:
            Uses multinomial-like distribution across input sources.
            For each new winner with input strength s, sample s connections
            from the total input space, then count how many come from each source.
        """
        if not input_sizes or not first_winner_inputs:
            return []

        num_inputs = len(input_sizes)
        total_k = sum(input_sizes)
        num_first_winners = len(first_winner_inputs)

        inputs_by_first_winner_index = [None] * num_first_winners

        for i in range(num_first_winners):
            input_strength = int(first_winner_inputs[i])

            # Sample input indices from the total input space
            input_indices = self.rng.choice(
                range(total_k), input_strength, replace=False
            )

            # Count how many inputs come from each source
            num_connections_by_input_index = np.zeros(num_inputs)
            total_so_far = 0

            for j in range(num_inputs):
                # Count inputs that fall within this source's range
                num_connections_by_input_index[j] = sum(
                    total_so_far + input_sizes[j] > idx >= total_so_far
                    for idx in input_indices
                )
                total_so_far += input_sizes[j]

            inputs_by_first_winner_index[i] = num_connections_by_input_index

        return inputs_by_first_winner_index

    def expand_connectome_dynamic(self, current_connectome: np.ndarray,
                                num_new_winners: int,
                                axis: int = 1) -> np.ndarray:
        """
        Dynamically expand connectome to accommodate new winners.

        This implements the dynamic connectome expansion logic from root brain.py
        lines 782-784, 806-808, etc., which handles the expansion of synaptic
        weight matrices when new neurons become active.

        Args:
            current_connectome (np.ndarray): Current synaptic weight matrix
            num_new_winners (int): Number of new winners to add
            axis (int): Axis to expand (0=rows, 1=columns)

        Returns:
            np.ndarray: Expanded connectome matrix

        Biological Context:
            When new neurons fire for the first time, the synaptic weight
            matrices need to be expanded to include connections to/from
            these new neurons. This maintains the full connectivity structure.

        Mathematical Context:
            Matrix expansion with zero-padding to maintain connectivity
            structure while accommodating new neural elements.
        """
        if num_new_winners <= 0:
            return current_connectome

        if axis == 0:  # Expand rows
            pad_width = ((0, num_new_winners), (0, 0))
        elif axis == 1:  # Expand columns
            pad_width = ((0, 0), (0, num_new_winners))
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")

        expanded_connectome = np.pad(current_connectome, pad_width)
        return expanded_connectome

    def assign_synaptic_connections(self, connectome: np.ndarray,
                                  input_sources: List[np.ndarray],
                                  new_winner_indices: List[int],
                                  connection_probability: float = 0.05) -> np.ndarray:
        """
        Assign synaptic connections for new winners based on input distribution.

        This implements the synaptic connection assignment logic from root brain.py
        lines 788-790, 827-833, etc., which determines the initial synaptic
        weights for new winners based on their input distribution.

        Args:
            connectome (np.ndarray): Synaptic weight matrix to update
            input_sources (List[np.ndarray]): Input distributions for each winner
            new_winner_indices (List[int]): Indices of new winners
            connection_probability (float): Probability of random connections

        Returns:
            np.ndarray: Updated connectome matrix

        Biological Context:
            New winners establish synaptic connections based on their input
            patterns. Strong inputs become strong synapses, while weak or
            absent inputs may still form random connections.

        Mathematical Context:
            Direct assignment of connection strengths based on input distribution,
            with Bernoulli sampling for background connectivity.
        """
        updated_connectome = connectome.copy()
        num_inputs_processed = 0

        # Process each winner's input distribution
        for winner_idx, input_distribution in zip(new_winner_indices, input_sources):
            # Assign connections for this winner from all input sources
            for source_idx, strength in enumerate(input_distribution):
                # Direct connection strength assignment
                updated_connectome[winner_idx, source_idx] = strength

            num_inputs_processed += 1

        return updated_connectome

    def initialize_new_winner_synapses(self, connectome: np.ndarray,
                                     target_w: int,
                                     num_new_winners: int,
                                     input_sizes: List[int],
                                     p: float = 0.05) -> np.ndarray:
        """
        Initialize synaptic weights for new winners from external stimuli.

        This implements the logic from root brain.py lines 809-811, which
        initializes synaptic connections from stimuli that weren't active
        in the current projection round.

        Args:
            connectome (np.ndarray): Stimulus-to-area connectome
            target_w: Current number of ever-fired neurons
            num_new_winners (int): Number of new winners
            input_sizes (List[int]): Sizes of input stimuli
            connection_probability (float): Connection probability

        Returns:
            np.ndarray: Updated connectome with new winner synapses

        Biological Context:
            Even stimuli that weren't active in the current round can form
            connections to new winners through random synaptic formation.

        Mathematical Context:
            Bernoulli sampling to establish baseline connectivity from
            inactive stimuli to new neural elements.
        """
        if num_new_winners <= 0:
            return connectome

        updated_connectome = connectome.copy()

        # Initialize connections for new winners from external stimuli
        for i in range(num_new_winners):
            winner_idx = target_w + i
            for j, stim_size in enumerate(input_sizes):
                # Random connections from each stimulus
                connections = self.rng.binomial(
                    stim_size, p, size=1
                )[0]
                updated_connectome[winner_idx, j] = connections

        return updated_connectome

    def apply_plasticity_scaling(self, connectome: np.ndarray,
                               winner_indices: List[int],
                               plasticity_beta: float) -> np.ndarray:
        """
        Apply Hebbian plasticity scaling to synaptic weights.

        This implements the plasticity scaling logic from root brain.py
        lines 794-795, 837-839, which strengthens synapses to recently
        active neurons.

        Args:
            connectome (np.ndarray): Synaptic weight matrix
            winner_indices (List[int]): Indices of active winners
            plasticity_beta (float): Plasticity scaling factor

        Returns:
            np.ndarray: Connectome with applied plasticity scaling

        Biological Context:
            Hebbian plasticity strengthens synapses to neurons that
            participated in recent successful firing patterns.

        Mathematical Context:
            Multiplicative scaling of synaptic weights based on
            neural activity participation.
        """
        updated_connectome = connectome.copy()

        for i in winner_indices:
            for j in range(connectome.shape[0]):
                # Apply plasticity scaling
                updated_connectome[j, i] *= (1.0 + plasticity_beta)

        return updated_connectome

    def process_first_time_winners(self, all_potential_winners: List[float],
                                 target_area_w: int,
                                 target_area_k: int) -> Tuple[List[int], List[float], int]:
        """
        Process first-time winners and update their indices.

        This implements the first-time winner processing logic from root brain.py
        lines 736-747, which handles the indexing and tracking of neurons
        that fire for the first time.

        Args:
            all_potential_winners (List[float]): Input strengths for all potential winners
            target_area_w (int): Current number of ever-fired neurons
            target_area_k (int): Number of winners to select

        Returns:
            Tuple[List[int], List[float], int]: (new_winner_indices, first_winner_inputs, num_first_winners)

        Biological Context:
            First-time winners are assigned new indices in the neural population
            and their input strengths are tracked for connectome expansion.

        Mathematical Context:
            Index remapping to maintain consistent neural population structure
            while accommodating dynamic expansion.
        """
        # Select top k winners using heap algorithm
        # Select top k winners using argpartition (O(n) average)
        arr = np.asarray(all_potential_winners, dtype=np.float64)
        k = min(target_area_k, len(arr))
        if k >= len(arr):
            part_idx = np.arange(len(arr))
        else:
            part_idx = np.argpartition(-arr, k)[:k]
        sorted_order = np.argsort(-arr[part_idx])
        new_winner_indices = list(part_idx[sorted_order])

        first_winner_inputs = []
        num_first_winners_processed = 0

        # Process each winner
        for i in range(len(new_winner_indices)):
            if new_winner_indices[i] >= target_area_w:
                # First-time winner
                first_winner_inputs.append(all_potential_winners[new_winner_indices[i]])
                # Remap index to new position
                new_winner_indices[i] = target_area_w + num_first_winners_processed
                num_first_winners_processed += 1

        return new_winner_indices, first_winner_inputs, num_first_winners_processed

    def _draw_rng(self, key: Optional[Tuple]) -> np.random.Generator:
        """Generator for a candidate draw: content-addressed when keyed.

        `stable_seed` is crc32 of `repr(parts)`, so it is identical in every
        process -- `hash()` is not (PEP 456), and using it here would make the
        fix hold within a run and silently fail across runs, which is exactly
        the failure mode recorded for the lazy connectome seeds.
        """
        if key is None:
            return self.rng
        try:
            from ..core.numpy_engine._seeding import stable_seed
        except ImportError:                                   # pragma: no cover
            from core.numpy_engine._seeding import stable_seed
        return np.random.default_rng(stable_seed(*key))

    def sample_new_winner_inputs(
        self,
        input_sizes: List[int],
        n: int,
        w: int,
        k: int,
        p: float,
        key: Optional[Tuple] = None,
    ) -> np.ndarray:
        """
        Sample potential input strengths for k new winner candidates using
        the truncated-normal approximation from the original brain.py.

        Each new candidate neuron receives a random number of inputs from the
        total input pool (stimuli + source areas). The distribution is
        Binomial(total_k, p) truncated to the top-(k/effective_n) quantile,
        approximated via a truncated normal.

        Uses cached binom.ppf and direct inverse-CDF sampling via
        scipy.special.ndtri instead of scipy.stats.truncnorm for speed.

        CONTENT-ADDRESSED DRAW (`key`).  Without a key this consumes
        ``self.rng``, so the SAME input into the SAME area draws DIFFERENT
        candidates every call -- and because those candidates come from the
        top-(k/n) quantile by construction, they routinely displace the real
        incumbents. That breaks the model's defining invariant: drive is
        |{j in x : synapse j->i}|, a fixed property of the random graph, so at
        beta=0 a repeated projection must elect identical winners. Measured
        before this fix, n=20000 k=50 p=0.1, overlap against the first round:

            explicit  beta=0.0    1.000 1.000 1.000 1.000 1.000
            sparse    beta=0.0    1.000 0.620 0.340 0.320 0.260

        Passing `key` seeds the draw from the CONTENT of the projection --
        target area, size of the never-fired pool, and which sources fired
        with which assemblies -- via `stable_seed`, the same idiom the lazy
        connectome already uses. Identical input then reproduces identical
        candidates, while genuinely different input still gets fresh ones.

        This is the same class of defect, and the same fix, as the
        content-addressed synapse init: randomness that stands in for a fixed
        structural fact must be keyed on that fact, not drawn from a stream.

        Args:
            input_sizes: Size of each input source (stimulus sizes + source area k values).
            n: Total neuron count of the target area.
            w: Number of neurons that have ever fired in the target area.
            k: Assembly size (number of winners to select).
            p: Connection probability.
            key: Optional content key identifying this projection. When None,
                falls back to the stateful ``self.rng`` (pre-fix behaviour,
                retained so recorded goldens can be reproduced for comparison).

        Returns:
            1D array of length k with sampled input strengths for new candidates.
        """
        from scipy.special import ndtr, ndtri

        total_k = sum(input_sizes)
        effective_n = n - w

        # Graceful saturation. `effective_n = n - w` is the count of neurons that
        # have never fired -- the only source of brand-new winners. When it drops
        # to k or below, the area cannot recruit a full k of fresh winners, so we
        # recruit as many as remain (k_eff) and let the caller complete the
        # winner set from already-materialized incumbents (it top-k selects over
        # prev_winner_inputs + these). A biological area at capacity simply stops
        # recruiting; it must not crash mid-training. k_eff == k whenever
        # effective_n > k, so every non-saturated run stays bit-identical.
        k_eff = min(k, max(0, effective_n - 1))
        if k_eff <= 0:
            return self._xp.asarray(np.empty(0))

        # Cached ppf — integer num/den for exact hash key
        alpha = _binom_ppf_cached(effective_n - k_eff, effective_n, total_k, p)

        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        if std == 0:
            return self._xp.asarray(np.full(k_eff, mu))

        a = (alpha - mu) / std

        if key is not None:
            return self._xp.asarray(
                self._order_statistic_candidates(mu, std, n, w, k_eff,
                                                 total_k, key))

        # Fast truncated normal via inverse CDF: sample U ~ Uniform(Phi(a), 1)
        # then return mu + std * Phi_inv(U).  Avoids scipy.stats overhead.
        phi_a = float(ndtr(a))
        rng = self._draw_rng(key)
        u = rng.uniform(phi_a, 1.0, size=k_eff)
        np.clip(u, phi_a, 1.0 - 1e-12, out=u)  # guard against ndtri(1)=inf
        samples = (mu + ndtri(u) * std).round(0)
        np.clip(samples, 0, total_k, out=samples)
        return self._xp.asarray(samples)

    def _uniform_order_stats(self, key: Tuple, n: int, upto: int) -> np.ndarray:
        """The `upto` largest of n uniforms, descending, keyed and stable.

        Generated by exponential spacings: with E_j iid Exp(1),
        U_(i-th largest) = exp(-sum_{j<=i} E_j / (n - j + 1)).  Two properties
        matter and neither is optional:

        MONOTONE -- the sequence descends by construction, so recruiting the
        top m leaves the rest strictly below them. That is what makes a
        repeated projection idempotent.

        RANK-STABLE ACROSS CALLS -- the key does NOT include `w`. The E_j come
        from a fixed seed in a fixed order, so rank r has the same value no
        matter how many neurons have been recruited when we ask; `w` only
        indexes into the sequence.

        KEYED -- the first attempt at this fix used the EXPECTED order
        statistic, which is monotone and stable but identical for every input.
        With no input-specific variation, which neuron gets recruited is
        decided by sequential ID assignment rather than by which neuron would
        respond to THIS stimulus, and two measured properties broke: two
        independent stimuli collapsed onto shared cells (overlap 0.152 +/-
        0.200 against chance 0.005) and the coin's basin asymmetry stopped
        self-averaging (heads 0.95). Keying on the projection's content
        restores both while keeping the two properties above.
        """
        cache = getattr(self, "_ostat_cache", None)
        if cache is None:
            cache = self._ostat_cache = {}
        ck = (key, n)
        cached = cache.get(ck)
        if cached is not None and len(cached) >= upto:
            return cached[:upto]

        try:
            from ..core.numpy_engine._seeding import stable_seed
        except ImportError:                                   # pragma: no cover
            from core.numpy_engine._seeding import stable_seed
        rng = np.random.default_rng(stable_seed(*key))
        grow = max(upto, 2 * len(cached) if cached is not None else 0)
        e = rng.exponential(1.0, size=grow)
        denom = np.maximum(n - np.arange(grow, dtype=np.float64), 1.0)
        u = np.exp(-np.cumsum(e / denom))
        if len(cache) > 512:                       # bound: engines are pickled
            cache.clear()
        cache[ck] = u
        return u[:upto]

    def _order_statistic_candidates(self, mu: float, std: float, n: int,
                                    w: int, k_eff: int, total_k: int,
                                    key: Tuple) -> np.ndarray:
        """Candidate drives as ORDER STATISTICS of one fixed pool of n draws.

        THE BUG THIS REPLACES.  Drawing k fresh variates from the top-(k/n)
        tail on every call is drawing two INDEPENDENT top-k samples, when the
        model has one pool of n neurons whose drives are fixed by the random
        graph. Because k << n the tail threshold barely moves as w grows, so
        the second sample is statistically indistinguishable from the first --
        and roughly half of its candidates outbid the very incumbents the
        first sample produced. That is why a beta=0 projection was not
        idempotent (overlap 1.000 -> 0.620 -> 0.340 -> 0.320 -> 0.260) and why
        assemblies churned instead of converging.

        What the model actually says: the top-k of n draws, then the NEXT k of
        the SAME n draws, are ranks 1..k and k+1..2k -- strictly decreasing.
        So rank r gets the expected order statistic at quantile
        (n - r - 0.5)/n, and `w` -- the count already recruited -- is an OFFSET
        into that fixed sequence rather than part of a re-drawn threshold.

        Monotone by construction, which is what makes recruitment idempotent:
        the best remaining candidate is always strictly below the worst neuron
        already taken, so a repeated projection has nothing new to offer and
        the incumbents hold.

        This is deterministic, and deliberately so. The random graph is drawn
        ONCE in the model; re-randomising it per projection was the defect.
        Using expected order statistics approximates a single fixed
        realisation, at the cost of not modelling the spread between
        realisations -- an area's candidate profile is now a function of
        (n, w, mu, std) alone. Competition is against materialised incumbents,
        which do differ per area, so this does not make areas interchangeable.
        """
        from scipy.special import ndtri

        if k_eff <= 0:
            return np.empty(0)
        quantiles = self._uniform_order_stats(key, n, w + k_eff)[w:w + k_eff]
        quantiles = np.clip(quantiles, 1e-12, 1.0 - 1e-12)
        samples = (mu + ndtri(quantiles) * std).round(0)
        np.clip(samples, 0, total_k, out=samples)
        return samples

    def sample_new_winner_inputs_legacy(
        self,
        input_sizes: List[int],
        n: int,
        w: int,
        k: int,
        p: float,
        key: Optional[Tuple] = None,
    ) -> np.ndarray:
        """
        Legacy version of sample_new_winner_inputs using scipy.stats.truncnorm.

        Produces the exact same RNG sequence as the original brain.py code,
        ensuring bit-identical reproducibility for a given seed. Slower than
        the optimized version (~35x) due to scipy.stats object overhead.

        Use this when deterministic=True to preserve cross-version seed
        reproducibility.

        Args:
            input_sizes: Size of each input source.
            n: Total neuron count of the target area.
            w: Number of neurons that have ever fired in the target area.
            k: Assembly size (number of winners to select).
            p: Connection probability.

        Returns:
            1D array of length k with sampled input strengths for new candidates.
        """
        from scipy.stats import binom, truncnorm

        total_k = sum(input_sizes)
        effective_n = n - w

        # Graceful saturation -- see sample_new_winner_inputs for the rationale.
        # k_eff == k whenever effective_n > k, so the deterministic RNG stream is
        # unchanged for every non-saturated run (only the previously-crashing
        # case draws a different number of variates).
        k_eff = min(k, max(0, effective_n - 1))
        if k_eff <= 0:
            return self._xp.asarray(np.empty(0))

        alpha = float(binom.ppf(
            float(effective_n - k_eff) / effective_n, total_k, p
        ))

        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        if std == 0:
            return self._xp.asarray(np.full(k_eff, mu))

        a = (alpha - mu) / std
        samples = truncnorm.rvs(
            a, np.inf, loc=mu, scale=std, size=k_eff,
            random_state=self._draw_rng(key),
        ).round(0)
        np.clip(samples, 0, total_k, out=samples)
        return self._xp.asarray(samples)

    def compute_input_splits(
        self,
        input_sizes: List[int],
        first_winner_inputs: List[int],
    ) -> List[List[int]]:
        """
        Distribute each new winner's total input across source areas/stimuli
        proportional to their sizes.

        Vectorized: computes all winners at once via outer product + argmax
        remainder assignment instead of per-winner Python loops.

        Args:
            input_sizes: Size of each input source.
            first_winner_inputs: Total input for each first-time winner.

        Returns:
            List of per-winner split vectors (one int per input source).
        """
        total_k = sum(input_sizes)
        if total_k == 0:
            return [[] for _ in first_winner_inputs]
        if not first_winner_inputs:
            return []

        proportions = np.array(input_sizes, dtype=np.float64) / float(total_k)
        totals = np.array(first_winner_inputs, dtype=np.float64)

        # (n_winners, n_sources) outer product
        raw = totals[:, None] * proportions[None, :]
        base = np.floor(raw).astype(int)
        remainders = totals.astype(int) - base.sum(axis=1)

        # Distribute remainders to sources with largest fractional parts
        mask = remainders > 0
        if np.any(mask):
            frac = raw - base
            # For each winner with remainder, give +1 to the top sources
            n_sources = len(input_sizes)
            if n_sources <= 2:
                # Fast path: remainder is 0 or 1, give to argmax
                best = np.argmax(frac, axis=1)
                base[mask, best[mask]] += 1
            else:
                # General case: remainder can be > 1 with many sources
                order = np.argsort(-frac, axis=1)
                for i in np.where(mask)[0]:
                    rem = int(remainders[i])
                    for j in range(rem):
                        base[i, order[i, j % n_sources]] += 1

        return base.tolist()

    def get_simulation_method_info(self) -> Dict[str, Any]:
        """
        Get information about the sparse simulation methods.

        Returns:
            Dict[str, Any]: Information about the simulation methods
        """
        return {
            'module': 'sparse_simulation',
            'algorithms': [
                'dynamic_connectome_expansion',
                'input_distribution_calculation',
                'first_time_winner_processing',
                'synaptic_connection_assignment',
                'plasticity_scaling',
                'connectome_initialization'
            ],
            'biological_context': 'sparse_neural_assembly_dynamics',
            'mathematical_foundation': 'dynamic_matrix_operations_and_multinomial_sampling',
            'complexity': 'O(k * log k) for winner selection, O(total_k) for input distribution'
        }
