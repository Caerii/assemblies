"""Internal state containers for NumPy engines."""

from dataclasses import dataclass, field

from ..backend import get_xp


@dataclass
class SparseAreaState:
    """Internal per-area state for sparse simulation."""
    name: str
    n: int
    k: int
    beta: float
    w: int = 0
    winners: object = None          # xp array, compact indices
    compact_to_neuron_id: list = field(default_factory=list)
    neuron_id_pool: object = None   # np.ndarray of shuffled neuron IDs
    neuron_id_pool_ptr: int = 0
    fixed_assembly: bool = False
    beta_by_source: dict = field(default_factory=dict)  # source_name -> beta
    # LRI (Long-Range Inhibition) — refractory suppression for sequences
    refractory_period: int = 0              # 0 = LRI disabled
    inhibition_strength: float = 0.0        # penalty magnitude
    _refractory_history: object = None      # deque of set[int] (compact indices)
    # Refracted mode — cumulative bias inhibition for FSM arc areas
    refracted: bool = False
    refracted_strength: float = 0.0
    _cumulative_bias: object = None         # xp float32 array, length w

    def __post_init__(self):
        from collections import deque
        if self.winners is None:
            xp = get_xp()
            self.winners = xp.array([], dtype=xp.uint32)
        if self._refractory_history is None:
            self._refractory_history = deque(
                maxlen=max(self.refractory_period, 1))
        if self._cumulative_bias is None:
            xp = get_xp()
            self._cumulative_bias = xp.zeros(0, dtype=xp.float32)


@dataclass
class ExplicitAreaState:
    """Internal per-area state for explicit simulation."""
    name: str
    n: int
    k: int
    beta: float
    w: int = 0
    winners: object = None
    ever_fired: object = None       # xp bool array of length n
    num_ever_fired: int = 0
    fixed_assembly: bool = False
    beta_by_source: dict = field(default_factory=dict)

    def __post_init__(self):
        xp = get_xp()
        if self.winners is None:
            self.winners = xp.array([], dtype=xp.uint32)
        if self.ever_fired is None:
            self.ever_fired = xp.zeros(self.n, dtype=bool)


@dataclass
class StimulusState:
    """Internal stimulus descriptor."""
    name: str
    size: int
