"""Per-area and per-stimulus state containers for TorchSparseEngine."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Threshold above which we use lazy ID generation instead of
# pre-computing a full permutation of n neuron IDs.
LAZY_ID_THRESHOLD = 1_000_000


@dataclass
class TorchAreaState:
    """Per-area state for TorchSparseEngine."""
    name: str
    n: int
    k: int
    beta: float
    w: int = 0
    winners: Optional[torch.Tensor] = None  # int32 on CUDA
    compact_to_neuron_id: list = field(default_factory=list)
    neuron_id_pool: Optional[np.ndarray] = None  # pre-computed for small n
    neuron_id_pool_ptr: int = 0
    # Lazy ID generation for large n (avoids O(n) permutation)
    _lazy_ids: bool = False
    _used_ids: Optional[set] = None
    _id_rng: Optional[np.random.Generator] = None
    fixed_assembly: bool = False
    beta_by_source: dict = field(default_factory=dict)
    # LRI
    refractory_period: int = 0
    inhibition_strength: float = 0.0
    _refractory_history: Optional[deque] = None
    # Refracted mode
    refracted: bool = False
    refracted_strength: float = 0.0
    _cumulative_bias: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.winners is None:
            self.winners = torch.empty(0, dtype=torch.int32, device='cuda')
        if self._refractory_history is None:
            self._refractory_history = deque(
                maxlen=max(self.refractory_period, 1))
        if self._cumulative_bias is None:
            self._cumulative_bias = torch.zeros(0, dtype=torch.float32,
                                                device='cuda')

    def next_neuron_id(self) -> int:
        """Get next unique neuron ID, either from pool or lazy sampling."""
        if self._lazy_ids:
            # Sample random unique ID — for n >> w, almost never retries
            while True:
                candidate = int(self._id_rng.integers(0, self.n))
                if candidate not in self._used_ids:
                    self._used_ids.add(candidate)
                    return candidate
        else:
            pid = self.neuron_id_pool_ptr
            if pid >= len(self.neuron_id_pool):
                raise RuntimeError(
                    f"Neuron id pool exhausted for area {self.name}")
            self.neuron_id_pool_ptr += 1
            return int(self.neuron_id_pool[pid])


@dataclass
class StimulusState:
    """Stimulus descriptor."""
    name: str
    size: int


class TorchConn:
    """Lightweight connectivity wrapper for stim->area (1-D weights)."""
    __slots__ = ('weights', 'sparse')

    def __init__(self, weights, sparse=True):
        self.weights = weights
        self.sparse = sparse
