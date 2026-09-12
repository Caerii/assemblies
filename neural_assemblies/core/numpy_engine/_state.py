"""Internal state containers for NumPy engines."""

from dataclasses import dataclass, field
from typing import Any

from ..backend import get_xp
from ..activity import ActivityState


@dataclass
class SparseAreaState(ActivityState):
    """Internal per-area state for sparse simulation."""
    _activity_fields = ("winners", "w", "fixed_assembly", "explicit_source",
                        "_refractory_history", "_cumulative_bias")
    name: str
    n: int
    k: int
    beta: float
    w: int = 0
    winners: Any = None             # xp array, compact indices
    compact_to_neuron_id: list = field(default_factory=list)
    neuron_id_pool: Any = None      # np.ndarray of shuffled neuron IDs
    neuron_id_pool_ptr: int = 0
    fixed_assembly: bool = False
    beta_by_source: dict = field(default_factory=dict)  # source_name -> beta
    # LRI (Long-Range Inhibition) — refractory suppression for sequences
    refractory_period: int = 0              # 0 = LRI disabled
    inhibition_strength: float = 0.0        # penalty magnitude
    _refractory_history: Any = None         # deque of set[int] (compact indices)
    # Refracted mode — cumulative bias inhibition for FSM arc areas
    refracted: bool = False
    refracted_strength: float = 0.0
    _cumulative_bias: Any = None            # xp float32 array, length w
    #: MASKED READOUT ([[REFRACTION-ANTI-MERGING]]): a read (no plasticity)
    #: ranks the raw drive, the bias neither subtracted nor charged. Writes
    #: are never masked -- the bias is what keeps items apart while they
    #: are written -- so the flag is honoured only when plasticity is off.
    masked_readout: bool = False
    winner_policy: object = None
    input_noise_std: float = 0.0
    explicit_source: bool = False  # winners are real neuron IDs (explicit area)

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
class ExplicitAreaState(ActivityState):
    """Internal per-area state for explicit simulation."""
    _activity_fields = ("winners", "w", "ever_fired", "num_ever_fired", "fixed_assembly")
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
    slot_count: int = 0
    #: Competition rule; None is plain k-WTA. Present here for the same reason
    #: it was added to `numpy_exact` (#94): an emergent-size rule like E%-WTA is
    #: a claim about the DRIVE DISTRIBUTION, so it has to be runnable on an
    #: engine that does not invent drive. This is the dense ground-truth engine,
    #: and a policy being unavailable here is what makes a policy result
    #: uncheckable rather than merely unmeasured.
    winner_policy: object = None

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
