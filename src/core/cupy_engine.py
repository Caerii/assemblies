"""
CuPy-based sparse engine for assembly calculus.

CupySparseEngine: GPU-accelerated statistical sparse simulation.

This engine mirrors NumpySparseEngine's exact algorithm (same statistical
sampling, same Hebbian learning, same amortized buffer growth) but stores
all weight matrices and activation buffers as CuPy arrays on GPU.  It
produces the same assembly dynamics as numpy_sparse — unlike the hash-based
cuda_implicit engine which uses fundamentally different connectivity.

Key optimizations over numpy_sparse on CPU:
  - Weights and activations live on GPU (no CPU↔GPU copies for core compute)
  - Bernoulli sampling for connectome expansion uses CuPy RNG (GPU-native)
  - Top-k winner selection via cupy.argpartition (GPU-native)
  - Hebbian plasticity via GPU fancy indexing
  - project_rounds() keeps all state on GPU between rounds
  - Stim→area binomial approximated on GPU for large allocations

Requires: cupy (``pip install cupy-cuda12x`` or appropriate variant).
Falls back gracefully: module loads but engine is not registered without CuPy.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

from .engine import ComputeEngine, ProjectionResult, register_engine
from .connectome import Connectome

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

if _HAS_CUPY:
    # Only import parent and compute primitives if CuPy is available,
    # since we need CuPy backend active for proper initialization.
    from .numpy_engine import (
        NumpySparseEngine,
        _SparseAreaState,
        _StimulusState,
    )
    try:
        from ..compute.sparse_simulation import SparseSimulationEngine
        from ..compute.winner_selection import WinnerSelector
    except ImportError:
        from compute.sparse_simulation import SparseSimulationEngine
        from compute.winner_selection import WinnerSelector


    class CupySparseEngine(NumpySparseEngine):
        """GPU engine using CuPy with statistical sparse simulation.

        Subclasses NumpySparseEngine and overrides hot paths to use
        CuPy-native operations.  The algorithm is identical: same truncated-
        normal sampling for new candidates, same amortized doubling for
        connectome growth, same Hebbian plasticity rule.

        Truncated-normal sampling (scipy) runs on CPU — the k-length result
        is transferred to GPU.  All other operations (input accumulation,
        winner selection, plasticity, Bernoulli initialization) run on GPU.
        """

        def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                     deterministic: bool = False):
            import warnings
            warnings.warn(
                "cupy_sparse is deprecated; use torch_sparse for GPU "
                "acceleration (faster dispatch, CSR memory efficiency)",
                DeprecationWarning, stacklevel=2)
            # Activate CuPy backend before parent __init__ creates arrays
            from .backend import set_backend
            set_backend("cupy")

            # Parent creates all state using get_xp() which is now cupy
            super().__init__(p=p, seed=seed, w_max=w_max,
                             deterministic=deterministic)

            # GPU-native RNG for Bernoulli sampling in connectome expansion
            self._cp_rng = cp.random.default_rng(seed)

        # -----------------------------------------------------------------
        # Override: GPU-native Bernoulli in connectome expansion
        # -----------------------------------------------------------------

        def _expand_connectomes(self, target, from_stimuli, from_areas,
                                input_sizes, winners, first_winner_inputs,
                                new_w):
            """Expand connectivity for first-time winners.

            Same algorithm as NumpySparseEngine but Bernoulli matrices are
            sampled directly on GPU via CuPy RNG, avoiding CPU→GPU transfer
            of random data.
            """
            tgt = self._areas[target]
            inputs_names = list(from_stimuli) + list(from_areas)

            splits_per_new = self._sparse_sim.compute_input_splits(
                input_sizes, first_winner_inputs,
            )

            prior_w = tgt.w
            new_indices = [int(w) for w in winners if int(w) >= prior_w]

            stim_names = [n for n in inputs_names if n in self._stimuli]
            area_names = [n for n in inputs_names if n in self._areas]

            # --- Expand stim→area 1-D vectors ---
            for stim_name in self._stimuli.keys():
                conn = self._stim_conns[stim_name][target]
                if conn.sparse:
                    old = len(conn.weights)
                    if new_w > old:
                        add_len = new_w - old
                        if stim_name not in stim_names:
                            stim_size = self._stimuli[stim_name].size
                            # Binomial on CPU (CuPy lacks binomial), transfer
                            add = cp.asarray(
                                self._rng.binomial(
                                    stim_size, self.p, size=add_len
                                ).astype(np.float32)
                            )
                        else:
                            add = cp.zeros(add_len, dtype=cp.float32)
                        conn.weights = cp.concatenate([conn.weights, add])

            # Write allocations for firing stimuli
            for idx, win in enumerate(new_indices):
                if win >= new_w:
                    continue
                split = (splits_per_new[idx]
                         if idx < len(splits_per_new) else None)
                if split is None:
                    continue
                for j, name in enumerate(inputs_names):
                    alloc = int(split[j])
                    if name in self._stimuli:
                        conn = self._stim_conns[name][target]
                        if conn.sparse and win < len(conn.weights):
                            conn.weights[win] = alloc

            # --- Expand area→area 2-D matrices ---
            for src_name in area_names:
                conn = self._area_conns[src_name][target]
                if not conn.sparse:
                    continue
                src = self._areas[src_name]
                if conn.weights.ndim != 2:
                    conn.weights = cp.empty((0, 0), dtype=cp.float32)

                phys_rows, phys_cols = conn.weights.shape
                src_w_arr = cp.asarray(src.winners)
                max_src_idx = (
                    (int(cp.max(src_w_arr)) + 1)
                    if src_w_arr.size > 0 else 0
                )
                needed_rows = max(
                    max_src_idx,
                    new_w if src_name == target else src.w,
                )
                needed_cols = new_w

                if self._deterministic:
                    # Legacy exact-fit: CPU RNG for bit-identical sequences
                    if needed_rows > phys_rows:
                        nr = needed_rows - phys_rows
                        new_rows = cp.asarray(
                            (self._rng.random((nr, phys_cols)) < self.p
                             ).astype(np.float32)
                        )
                        conn.weights = (
                            cp.vstack([conn.weights, new_rows])
                            if phys_cols > 0
                            else cp.zeros((needed_rows, 0), dtype=cp.float32)
                        )
                        phys_rows = needed_rows
                    if needed_cols > phys_cols:
                        nc = needed_cols - phys_cols
                        new_cols = cp.asarray(
                            (self._rng.random((phys_rows, nc)) < self.p
                             ).astype(np.float32)
                        )
                        conn.weights = (
                            cp.hstack([conn.weights, new_cols])
                            if phys_rows > 0
                            else cp.zeros((0, needed_cols), dtype=cp.float32)
                        )
                        phys_cols = needed_cols
                else:
                    # Amortised buffer growth with GPU-native Bernoulli
                    log_rows = getattr(conn, '_log_rows', phys_rows)
                    log_cols = getattr(conn, '_log_cols', phys_cols)
                    if log_rows > phys_rows or log_cols > phys_cols:
                        log_rows = min(log_rows, phys_rows)
                        log_cols = min(log_cols, phys_cols)

                    new_pr, new_pc = phys_rows, phys_cols
                    need_realloc = False
                    if needed_rows > phys_rows:
                        new_pr = max(needed_rows, phys_rows * 2, 2 * src.k)
                        need_realloc = True
                    if needed_cols > phys_cols:
                        new_pc = max(needed_cols, phys_cols * 2, 2 * tgt.k)
                        need_realloc = True

                    if need_realloc:
                        buf = cp.zeros((new_pr, new_pc), dtype=cp.float32)
                        if phys_rows > 0 and phys_cols > 0:
                            buf[:phys_rows, :phys_cols] = conn.weights
                        conn.weights = buf
                        phys_rows, phys_cols = new_pr, new_pc

                    # GPU-native Bernoulli initialization
                    nr = needed_rows - log_rows
                    nc = needed_cols - log_cols
                    # Region A: new rows × old cols
                    if nr > 0 and log_cols > 0:
                        conn.weights[log_rows:needed_rows, :log_cols] = (
                            self._cp_rng.random(
                                (nr, log_cols), dtype=cp.float32
                            ) < self.p
                        ).astype(cp.float32)
                    # Region B: old rows × new cols
                    if nc > 0 and log_rows > 0:
                        conn.weights[:log_rows, log_cols:needed_cols] = (
                            self._cp_rng.random(
                                (log_rows, nc), dtype=cp.float32
                            ) < self.p
                        ).astype(cp.float32)
                    # Region C: new rows × new cols
                    if nr > 0 and nc > 0:
                        conn.weights[
                            log_rows:needed_rows, log_cols:needed_cols
                        ] = (
                            self._cp_rng.random(
                                (nr, nc), dtype=cp.float32
                            ) < self.p
                        ).astype(cp.float32)

                    conn._log_rows = needed_rows
                    conn._log_cols = needed_cols

                # Write specific allocations for first-time winners
                from_index = inputs_names.index(src_name)
                local_rng = np.random.default_rng(
                    self._rng.integers(0, 2**32))
                src_winners_cpu = np.asarray(
                    src.winners.get()
                    if hasattr(src.winners, 'get')
                    else src.winners
                )
                for idx, win in enumerate(new_indices):
                    alloc = (
                        int(splits_per_new[idx][from_index])
                        if idx < len(splits_per_new) else 0
                    )
                    if alloc <= 0 or src.w == 0:
                        continue
                    sample_size = min(alloc, len(src.winners))
                    if sample_size <= 0:
                        continue
                    chosen = local_rng.choice(
                        src_winners_cpu, size=sample_size, replace=False)
                    col_idx = win - prior_w
                    if 0 <= col_idx < phys_cols:
                        conn.weights[chosen, col_idx] = 1.0

        # -----------------------------------------------------------------
        # Override: GPU-native connection reset
        # -----------------------------------------------------------------

        def reset_area_connections(self, area: str) -> None:
            """Reset area→area connections involving *area*."""
            for src_name in list(self._area_conns.keys()):
                if area not in self._area_conns[src_name]:
                    continue
                conn = self._area_conns[src_name][area]
                if conn.sparse:
                    conn.weights = cp.empty((0, 0), dtype=cp.float32)
                    if hasattr(conn, '_log_rows'):
                        del conn._log_rows
                    if hasattr(conn, '_log_cols'):
                        del conn._log_cols
                else:
                    rows, cols = conn.weights.shape
                    conn.weights = (
                        self._cp_rng.random((rows, cols), dtype=cp.float32)
                        < self.p
                    ).astype(cp.float32)

        # -----------------------------------------------------------------
        # Override: Tight project_rounds with no per-round CPU copies
        # -----------------------------------------------------------------

        def project_rounds(
            self,
            target: str,
            from_stimuli: List[str],
            from_areas: List[str],
            rounds: int,
            plasticity_enabled: bool = True,
        ) -> ProjectionResult:
            """Multi-round projection keeping all state on GPU.

            Only the final round's result is converted to CPU for the
            ProjectionResult return.  Intermediate rounds skip the
            CPU copy, reducing GPU sync overhead.
            """
            if rounds <= 0:
                tgt = self._areas[target]
                return ProjectionResult(
                    winners=np.array(
                        tgt.winners.get()
                        if hasattr(tgt.winners, 'get')
                        else tgt.winners,
                        dtype=np.uint32,
                    ),
                    num_first_winners=0,
                    num_ever_fired=tgt.w,
                )
            # Run all rounds via parent's project_into (state stays on GPU
            # between rounds since tgt.winners is a CuPy array)
            result = None
            for _ in range(rounds):
                result = self.project_into(
                    target, from_stimuli, from_areas, plasticity_enabled)
            return result

        # -----------------------------------------------------------------
        # Override: Batch projection
        # -----------------------------------------------------------------

        def project_into_batch(
            self,
            configs: List[tuple],
            plasticity_enabled: bool = True,
        ) -> Dict[str, ProjectionResult]:
            """Project into multiple targets with shared GPU context.

            Currently sequential but all arrays stay on GPU between
            targets, avoiding redundant CPU↔GPU transfers for shared
            source area winners.
            """
            results = {}
            for target, stims, areas in configs:
                results[target] = self.project_into(
                    target, stims, areas, plasticity_enabled)
            return results

        # -----------------------------------------------------------------
        # Identity
        # -----------------------------------------------------------------

        @property
        def name(self) -> str:
            return "cupy_sparse"


    # Register the engine
    register_engine("cupy_sparse", CupySparseEngine)
