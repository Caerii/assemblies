"""
CUDA compute engine using implicit hash-based connectivity.

Wraps the CUDA kernels from ``src/core/kernels/implicit.py`` into
the :class:`ComputeEngine` interface for multi-area assembly simulations.

Key properties:
- No weight matrices for baseline connectivity (hash-based, O(1) memory)
- Learned weight deltas stored per-(source, target) pair in COO sparse format
- Memory: O(learned_connections) instead of O(n²)
- For n=100,000 with 100 words: ~25 MB instead of 40 GB

Requires: cupy, torch (optional, for faster top-k)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .engine import ComputeEngine, ProjectionResult, register_engine

# Guard imports — this module is only loaded if cupy is available
import cupy as cp

try:
    import torch
    _USE_TORCH_TOPK = torch.cuda.is_available()
except ImportError:
    _USE_TORCH_TOPK = False

# Import CUDA kernels
from .kernels.implicit import (
    implicit_projection_kernel,
    apply_learned_kernel,
    hebbian_update_kernel,
)

try:
    from .kernels.batched import batched_projection_kernel
    _HAS_BATCHED = True
except ImportError:
    _HAS_BATCHED = False


# ---------------------------------------------------------------------------
# Per-pair learned weight storage (COO sparse)
# ---------------------------------------------------------------------------

@dataclass
class _COOWeights:
    """Sparse COO storage for learned weight deltas on one (src, tgt) pair."""
    learned_src: object     # cp.ndarray uint32
    learned_dst: object     # cp.ndarray uint32
    learned_delta: object   # cp.ndarray float32
    num_learned: object     # cp.ndarray uint32 (scalar on GPU)
    max_learned: int

    @staticmethod
    def create(max_learned: int) -> "_COOWeights":
        return _COOWeights(
            learned_src=cp.zeros(max_learned, dtype=cp.uint32),
            learned_dst=cp.zeros(max_learned, dtype=cp.uint32),
            learned_delta=cp.zeros(max_learned, dtype=cp.float32),
            num_learned=cp.zeros(1, dtype=cp.uint32),
            max_learned=max_learned,
        )


@dataclass
class _CudaAreaState:
    """Per-area state for implicit CUDA simulation."""
    name: str
    n: int
    k: int
    beta: float
    winners: object = None          # cp.ndarray uint32
    prev_winners: object = None     # cp.ndarray uint32 (for Hebbian)
    activations: object = None      # cp.ndarray float32 (pre-allocated)
    fixed_assembly: bool = False
    beta_by_source: dict = field(default_factory=dict)
    # Pre-allocated PyTorch tensor for fast top-k
    _torch_buf: object = None
    _use_fp16: bool = True

    def __post_init__(self):
        self.activations = cp.zeros(self.n, dtype=cp.float32)
        if _USE_TORCH_TOPK:
            dtype = torch.float16 if self._use_fp16 else torch.float32
            self._torch_buf = torch.zeros(self.n, device="cuda", dtype=dtype)


@dataclass
class _StimulusState:
    name: str
    size: int


# ---------------------------------------------------------------------------
# CudaImplicitEngine
# ---------------------------------------------------------------------------

class CudaImplicitEngine(ComputeEngine):
    """GPU engine using implicit hash-based connectivity + CUDA kernels.

    Each (source, target) pair gets a deterministic seed derived from the
    global seed and the pair names.  Baseline connectivity is computed on the
    fly via a hash function — no weight matrix is stored.  Only Hebbian
    weight *deltas* are stored in sparse COO format.

    Parameters:
        p:        Connection probability.
        seed:     Global random seed.
        w_max:    Weight saturation limit.
        use_fp16: Use FP16 activations for faster PyTorch top-k.
        max_learned_per_pair: COO buffer size per (source, target) pair.
    """

    BLOCK_SIZE = 512  # CUDA block size (optimised for RTX 4090)

    def __init__(self, p: float, seed: int = 0, w_max: float = 10.0,
                 use_fp16: bool = True, max_learned_per_pair: int = 500_000):
        self.p = p
        self.global_seed = seed
        self.w_max = w_max
        self._use_fp16 = use_fp16
        self._max_learned = max_learned_per_pair

        self._areas: Dict[str, _CudaAreaState] = {}
        self._stimuli: Dict[str, _StimulusState] = {}
        # Per-pair connectivity seeds and learned COO weights
        self._pair_seeds: Dict[tuple, int] = {}
        self._pair_learned: Dict[tuple, _COOWeights] = {}
        # Pre-allocated batch buffers keyed by (n, batch_size, k_in)
        self._batch_bufs: Dict[tuple, tuple] = {}

    # -- Helpers ------------------------------------------------------------

    def _pair_seed(self, source: str, target: str) -> int:
        """Deterministic seed for a (source, target) pair."""
        key = (source, target)
        if key not in self._pair_seeds:
            h = hash((self.global_seed, source, target)) & 0xFFFFFFFF
            self._pair_seeds[key] = h
        return self._pair_seeds[key]

    def _ensure_pair(self, source: str, target: str):
        """Ensure COO storage exists for the pair."""
        key = (source, target)
        if key not in self._pair_learned:
            self._pair_learned[key] = _COOWeights.create(self._max_learned)
        self._pair_seed(source, target)  # also cache seed

    def _grid(self, n: int) -> int:
        return (n + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

    def _get_batch_bufs(self, n: int, batch_size: int, k_in: int):
        """Return (active_batch, result_batch, seeds) pre-allocated GPU buffers."""
        key = (n, batch_size, k_in)
        if key not in self._batch_bufs:
            self._batch_bufs[key] = (
                cp.zeros((batch_size, k_in), dtype=cp.uint32),
                cp.zeros((batch_size, n), dtype=cp.float32),
                cp.zeros(batch_size, dtype=cp.uint32),
            )
        return self._batch_bufs[key]

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float) -> None:
        area = _CudaAreaState(name=name, n=n, k=k, beta=beta,
                              _use_fp16=self._use_fp16)
        self._areas[name] = area

        # Create pair storage for every existing entity
        for stim_name in self._stimuli:
            self._ensure_pair(stim_name, name)
        for other_name in self._areas:
            self._ensure_pair(other_name, name)
            if other_name != name:
                self._ensure_pair(name, other_name)

    def add_stimulus(self, name: str, size: int) -> None:
        self._stimuli[name] = _StimulusState(name=name, size=size)
        for area_name in self._areas:
            self._ensure_pair(name, area_name)

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        self._ensure_pair(source, target)

    # -- Projection ---------------------------------------------------------

    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
    ) -> ProjectionResult:
        tgt = self._areas[target]

        # Filter source areas with no winners
        from_areas = [a for a in from_areas
                      if self._areas[a].winners is not None and len(self._areas[a].winners) > 0]

        if tgt.fixed_assembly and tgt.winners is not None:
            return ProjectionResult(
                winners=cp.asnumpy(tgt.winners).astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.n,
            )

        if len(from_stimuli) == 0 and len(from_areas) == 0:
            w = cp.asnumpy(tgt.winners).astype(np.uint32) if tgt.winners is not None else np.array([], dtype=np.uint32)
            return ProjectionResult(winners=w, num_first_winners=0, num_ever_fired=tgt.n)

        # --- Accumulate activations from all sources ---
        tgt.activations[:] = 0

        for stim_name in from_stimuli:
            stim = self._stimuli[stim_name]
            stim_active = cp.arange(stim.size, dtype=cp.uint32)
            seed = self._pair_seed(stim_name, target)
            self._run_projection(stim_active, tgt.activations, tgt.n, seed)
            self._apply_learned(stim_name, target, stim_active, tgt.activations)

        for src_name in from_areas:
            src = self._areas[src_name]
            seed = self._pair_seed(src_name, target)
            self._run_projection(src.winners, tgt.activations, tgt.n, seed)
            self._apply_learned(src_name, target, src.winners, tgt.activations)

        # --- Top-k winner selection ---
        winners = self._select_topk(tgt, tgt.k)

        # --- Hebbian update ---
        if plasticity_enabled and tgt.prev_winners is not None:
            beta = tgt.beta
            for src_name in from_stimuli:
                src_beta = tgt.beta_by_source.get(src_name, beta)
                stim = self._stimuli[src_name]
                stim_active = cp.arange(stim.size, dtype=cp.uint32)
                self._hebbian_update(src_name, target, stim_active, winners,
                                     tgt.k, src_beta)

            for src_name in from_areas:
                src_beta = tgt.beta_by_source.get(src_name, beta)
                src = self._areas[src_name]
                self._hebbian_update(src_name, target, src.winners, winners,
                                     tgt.k, src_beta)

        # --- Commit state ---
        tgt.prev_winners = tgt.winners
        tgt.winners = winners

        return ProjectionResult(
            winners=cp.asnumpy(winners).astype(np.uint32),
            num_first_winners=0,   # implicit: all neurons always addressable
            num_ever_fired=tgt.n,
        )

    # -- CUDA kernel wrappers -----------------------------------------------

    def _run_projection(self, active: "cp.ndarray", out: "cp.ndarray",
                        n: int, seed: int):
        """Run implicit_projection_kernel: hash-based projection."""
        k_in = len(active)
        shared_mem = k_in * 4  # one uint32 per active neuron
        implicit_projection_kernel(
            (self._grid(n),), (self.BLOCK_SIZE,),
            (active, out, cp.uint32(k_in), cp.uint32(n),
             cp.uint32(seed), cp.float32(self.p)),
            shared_mem=shared_mem,
        )

    def _apply_learned(self, source: str, target: str,
                       active: "cp.ndarray", out: "cp.ndarray"):
        """Apply learned COO weight deltas."""
        key = (source, target)
        coo = self._pair_learned.get(key)
        if coo is None:
            return
        num = int(coo.num_learned[0])
        if num == 0:
            return
        seed = self._pair_seed(source, target)
        learn_grid = (num + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        apply_learned_kernel(
            (learn_grid,), (self.BLOCK_SIZE,),
            (coo.learned_src, coo.learned_dst, coo.learned_delta,
             active, out,
             cp.uint32(num), cp.uint32(len(active)),
             cp.uint32(seed), cp.float32(self.p)),
        )

    def _hebbian_update(self, source: str, target: str,
                        prev_active: "cp.ndarray", new_active: "cp.ndarray",
                        k: int, beta: float):
        """Run Hebbian update kernel for one (source, target) pair."""
        key = (source, target)
        coo = self._pair_learned.get(key)
        if coo is None:
            return
        seed = self._pair_seed(source, target)
        update_grid = (k * k + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        hebbian_update_kernel(
            (update_grid,), (self.BLOCK_SIZE,),
            (coo.learned_src, coo.learned_dst, coo.learned_delta,
             coo.num_learned, prev_active, new_active,
             cp.uint32(k), cp.float32(beta), cp.float32(self.w_max),
             cp.uint32(coo.max_learned), cp.uint32(seed), cp.float32(self.p)),
        )

    def _select_topk(self, area: _CudaAreaState, k: int) -> "cp.ndarray":
        """Select top-k winners using PyTorch (fast) or CuPy fallback."""
        if _USE_TORCH_TOPK and area._torch_buf is not None:
            area._torch_buf.copy_(torch.as_tensor(area.activations, device="cuda"))
            _, top_idx = torch.topk(area._torch_buf, k, sorted=False)
            return cp.asarray(top_idx).astype(cp.uint32)
        return cp.argpartition(area.activations, -k)[-k:].astype(cp.uint32)

    # -- Batched projection --------------------------------------------------

    def project_into_batch(
        self,
        configs: List[tuple],
        plasticity_enabled: bool = True,
    ) -> Dict[str, ProjectionResult]:
        """Project into multiple target areas using batched GPU kernels.

        When multiple targets share the same ``(n, k)`` dimensions, the
        implicit projection kernel is launched once with a 2-D grid
        (neurons × batch) instead of sequentially, giving 4-6× speedup
        on large networks.

        Args:
            configs: List of ``(target_name, from_stimuli, from_areas)`` tuples.
            plasticity_enabled: Whether to apply Hebbian plasticity.

        Returns:
            Dict mapping target name to :class:`ProjectionResult`.
        """
        if not _HAS_BATCHED:
            # Fall back to sequential if batched kernel not available
            return {t: self.project_into(t, s, a, plasticity_enabled)
                    for t, s, a in configs}

        from collections import defaultdict

        # Pre-filter source areas with no winners
        filtered = []
        for target, stims, areas in configs:
            areas = [a for a in areas
                     if self._areas[a].winners is not None and len(self._areas[a].winners) > 0]
            filtered.append((target, list(stims), areas))

        # Handle trivial cases (fixed assemblies, no inputs) immediately
        results = {}
        batch_candidates = []
        for target, stims, areas in filtered:
            tgt = self._areas[target]
            if tgt.fixed_assembly and tgt.winners is not None:
                results[target] = ProjectionResult(
                    winners=cp.asnumpy(tgt.winners).astype(np.uint32),
                    num_first_winners=0, num_ever_fired=tgt.n)
            elif len(stims) == 0 and len(areas) == 0:
                w = cp.asnumpy(tgt.winners).astype(np.uint32) if tgt.winners is not None else np.array([], dtype=np.uint32)
                results[target] = ProjectionResult(winners=w, num_first_winners=0, num_ever_fired=tgt.n)
            else:
                batch_candidates.append((target, stims, areas))

        if not batch_candidates:
            return results

        # Group by (n, k) for batching
        groups = defaultdict(list)
        for target, stims, areas in batch_candidates:
            tgt = self._areas[target]
            groups[(tgt.n, tgt.k)].append((target, stims, areas))

        for (n, k), group in groups.items():
            batch_size = len(group)
            targets_in_batch = [g[0] for g in group]

            # Zero activations for all targets in batch
            for target, _, _ in group:
                self._areas[target].activations[:] = 0

            # Collect all unique sources across this batch
            all_stim_sources = set()
            all_area_sources = set()
            for _, stims, areas in group:
                all_stim_sources.update(stims)
                all_area_sources.update(areas)

            # For each unique source, do a batched projection into applicable targets
            for stim_name in all_stim_sources:
                stim = self._stimuli[stim_name]
                stim_active = cp.arange(stim.size, dtype=cp.uint32)
                k_in = len(stim_active)

                # Find which targets in this batch receive this stimulus
                applicable = [(i, g[0]) for i, g in enumerate(group) if stim_name in g[1]]
                if len(applicable) >= 2:
                    # Batched path with pre-allocated buffers
                    b = len(applicable)
                    active_batch, result_batch, seeds_buf = self._get_batch_bufs(n, b, k_in)
                    for j in range(b):
                        active_batch[j, :k_in] = stim_active
                    for j, (_, t) in enumerate(applicable):
                        seeds_buf[j] = self._pair_seed(stim_name, t)
                    result_batch[:b].fill(0)

                    grid_x = self._grid(n)
                    batched_projection_kernel(
                        (grid_x, b), (self.BLOCK_SIZE,),
                        (active_batch, result_batch,
                         cp.uint32(k_in), cp.uint32(n), cp.uint32(b),
                         seeds_buf, cp.float32(self.p)),
                        shared_mem=k_in * 4,
                    )
                    for j, (_, target) in enumerate(applicable):
                        self._areas[target].activations += result_batch[j]
                        self._apply_learned(stim_name, target, stim_active, self._areas[target].activations)
                else:
                    # Sequential fallback for single target
                    for _, target in applicable:
                        seed = self._pair_seed(stim_name, target)
                        self._run_projection(stim_active, self._areas[target].activations, n, seed)
                        self._apply_learned(stim_name, target, stim_active, self._areas[target].activations)

            for src_name in all_area_sources:
                src = self._areas[src_name]
                applicable = [(i, g[0]) for i, g in enumerate(group) if src_name in g[2]]
                k_in = len(src.winners)
                if len(applicable) >= 2:
                    b = len(applicable)
                    active_batch, result_batch, seeds_buf = self._get_batch_bufs(n, b, k_in)
                    for j in range(b):
                        active_batch[j, :k_in] = src.winners
                    for j, (_, t) in enumerate(applicable):
                        seeds_buf[j] = self._pair_seed(src_name, t)
                    result_batch[:b].fill(0)

                    grid_x = self._grid(n)
                    batched_projection_kernel(
                        (grid_x, b), (self.BLOCK_SIZE,),
                        (active_batch, result_batch,
                         cp.uint32(k_in), cp.uint32(n), cp.uint32(b),
                         seeds_buf, cp.float32(self.p)),
                        shared_mem=k_in * 4,
                    )
                    for j, (_, target) in enumerate(applicable):
                        self._areas[target].activations += result_batch[j]
                        self._apply_learned(src_name, target, src.winners, self._areas[target].activations)
                else:
                    for _, target in applicable:
                        seed = self._pair_seed(src_name, target)
                        self._run_projection(src.winners, self._areas[target].activations, n, seed)
                        self._apply_learned(src_name, target, src.winners, self._areas[target].activations)

            # Batched top-k: one torch.topk call for all areas in this group
            if _USE_TORCH_TOPK and batch_size > 1:
                # Stack activations into a (batch, n) tensor for batched top-k
                _bk_key = (n, batch_size)
                if _bk_key not in self._batch_bufs:
                    dtype = torch.float16 if self._use_fp16 else torch.float32
                    self._batch_bufs[_bk_key] = torch.zeros(
                        (batch_size, n), device="cuda", dtype=dtype)
                topk_buf = self._batch_bufs[_bk_key]
                for j, (target, _, _) in enumerate(group):
                    topk_buf[j].copy_(torch.as_tensor(
                        self._areas[target].activations, device="cuda"))
                _, all_top_idx = torch.topk(topk_buf[:batch_size], k, dim=1, sorted=False)
                all_top_cp = cp.asarray(all_top_idx)

                for j, (target, stims, areas) in enumerate(group):
                    tgt = self._areas[target]
                    winners = all_top_cp[j].astype(cp.uint32)

                    if plasticity_enabled and tgt.prev_winners is not None:
                        beta = tgt.beta
                        for s in stims:
                            sb = tgt.beta_by_source.get(s, beta)
                            sa = cp.arange(self._stimuli[s].size, dtype=cp.uint32)
                            self._hebbian_update(s, target, sa, winners, k, sb)
                        for a in areas:
                            ab = tgt.beta_by_source.get(a, beta)
                            self._hebbian_update(a, target, self._areas[a].winners, winners, k, ab)

                    tgt.prev_winners = tgt.winners
                    tgt.winners = winners
                    results[target] = ProjectionResult(
                        winners=cp.asnumpy(winners).astype(np.uint32),
                        num_first_winners=0, num_ever_fired=tgt.n)
            else:
                # Sequential top-k fallback
                for target, stims, areas in group:
                    tgt = self._areas[target]
                    winners = self._select_topk(tgt, k)

                    if plasticity_enabled and tgt.prev_winners is not None:
                        beta = tgt.beta
                        for s in stims:
                            sb = tgt.beta_by_source.get(s, beta)
                            sa = cp.arange(self._stimuli[s].size, dtype=cp.uint32)
                            self._hebbian_update(s, target, sa, winners, k, sb)
                        for a in areas:
                            ab = tgt.beta_by_source.get(a, beta)
                            self._hebbian_update(a, target, self._areas[a].winners, winners, k, ab)

                    tgt.prev_winners = tgt.winners
                    tgt.winners = winners
                    results[target] = ProjectionResult(
                        winners=cp.asnumpy(winners).astype(np.uint32),
                        num_first_winners=0, num_ever_fired=tgt.n)

        return results

    # -- State accessors ----------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        if st.winners is None:
            return np.array([], dtype=np.uint32)
        return cp.asnumpy(st.winners).astype(np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        self._areas[area].winners = cp.asarray(winners, dtype=cp.uint32)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].n  # all neurons are always addressable

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = beta

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or len(st.winners) == 0:
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    @property
    def name(self) -> str:
        return "cuda_implicit"

    def memory_usage(self) -> dict:
        """Get total memory usage across all pairs."""
        total_learned = 0
        total_bytes = 0
        for coo in self._pair_learned.values():
            n = int(coo.num_learned[0])
            total_learned += n
            total_bytes += n * 12  # 2 uint32 + 1 float32
        for area in self._areas.values():
            total_bytes += area.n * 4  # activation buffer
        return {
            "total_learned_connections": total_learned,
            "total_bytes": total_bytes,
            "num_pairs": len(self._pair_learned),
        }


# Register engine (only succeeds if cupy imported successfully above)
register_engine("cuda_implicit", CudaImplicitEngine)
