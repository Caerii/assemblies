"""NumpyExplicitEngine: CPU engine using dense explicit simulation.

All n neurons are tracked.  Connectivity is stored as full
(source_n, target_n) float32 matrices.  Suitable for small networks
where full fidelity is required.
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from ..backend import get_xp, to_cpu
from ..engine import ComputeEngine, ProjectionResult
from ..connectome import Connectome

try:
    from ...compute.winner_selection import WinnerSelector
except ImportError:
    from compute.winner_selection import WinnerSelector

from ._state import ExplicitAreaState, StimulusState


class NumpyExplicitEngine(ComputeEngine):
    """CPU engine using dense explicit simulation.

    All n neurons are tracked.  Connectivity is stored as full
    (source_n, target_n) float32 matrices.  Suitable for small networks
    where full fidelity is required.
    """

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False):
        self.p = p
        self.w_max = w_max
        self._rng = np.random.default_rng(seed)
        self._plasticity_enabled_global = True

        self._areas: Dict[str, ExplicitAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}
        self._stim_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)
        self._area_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)
        self._winner_sel = WinnerSelector(self._rng)

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0) -> None:
        area = ExplicitAreaState(name=name, n=n, k=k, beta=beta)
        self._areas[name] = area

        for stim_name, stim in self._stimuli.items():
            conn = Connectome(stim.size, n, self.p, sparse=False)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        for other_name, other in self._areas.items():
            if other_name == name:
                self._area_conns[name][name] = Connectome(n, n, self.p, sparse=False)
            else:
                self._area_conns[other_name][name] = Connectome(other.n, n, self.p, sparse=False)
                self._area_conns[name][other_name] = Connectome(n, other.n, self.p, sparse=False)
                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        self._stimuli[name] = StimulusState(name=name, size=size)
        for area_name, area in self._areas.items():
            conn = Connectome(size, area.n, self.p, sparse=False)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        pass

    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        record_activation: bool = False,
    ) -> ProjectionResult:
        xp = get_xp()
        tgt = self._areas[target]

        # Filter sourceless areas
        from_areas = [a for a in from_areas
                      if self._areas[a].winners.size > 0]

        if tgt.fixed_assembly:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.num_ever_fired,
            )

        # Accumulate inputs into a full n-vector
        prev_winner_inputs = xp.zeros(tgt.n, dtype=xp.float32)

        for stim in from_stimuli:
            conn = self._stim_conns[stim][target]
            if conn.weights.ndim == 2:
                prev_winner_inputs += conn.weights.sum(axis=0).astype(xp.float32)
            else:
                prev_winner_inputs += conn.weights.astype(xp.float32, copy=False)

        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            src = self._areas[src_name]
            if src.winners.size > 0 and int(xp.max(src.winners)) >= conn.weights.shape[0]:
                raise IndexError(
                    f"Source area {src_name!r} has winner index "
                    f"{int(xp.max(src.winners))} exceeding connectome "
                    f"rows ({conn.weights.shape[0]})")
            if src.winners.size > 0:
                prev_winner_inputs += conn.weights[src.winners].sum(axis=0)

        # Select top-k winners
        winners, _, _, _ = self._winner_sel.select_combined_winners(
            prev_winner_inputs, tgt.w, tgt.k,
        )

        # Apply plasticity
        if plasticity_enabled and self._plasticity_enabled_global:
            for stim_name in from_stimuli:
                conn = self._stim_conns[stim_name][target]
                beta = tgt.beta_by_source.get(stim_name, tgt.beta)
                if beta != 0:
                    valid = xp.array([int(w) for w in winners if int(w) < conn.weights.shape[1]])
                    if len(valid) > 0:
                        conn.weights[:, valid] *= (1 + beta)
                    if self.w_max is not None:
                        xp.clip(conn.weights, 0, self.w_max, out=conn.weights)

            for src_name in from_areas:
                conn = self._area_conns[src_name][target]
                beta = tgt.beta_by_source.get(src_name, tgt.beta)
                if beta != 0:
                    src = self._areas[src_name]
                    valid_rows = xp.array([int(fw) for fw in src.winners
                                           if int(fw) < conn.weights.shape[0]])
                    valid_cols = xp.array([int(w) for w in winners
                                           if int(w) < conn.weights.shape[1]])
                    if len(valid_rows) > 0 and len(valid_cols) > 0:
                        ix = xp.ix_(valid_rows, valid_cols)
                        conn.weights[ix] *= (1 + beta)
                        if self.w_max is not None:
                            sub = conn.weights[ix]
                            xp.clip(sub, 0, self.w_max, out=sub)
                            conn.weights[ix] = sub

        # Update state
        winners = xp.asarray(winners, dtype=xp.uint32)
        tgt.winners = winners
        tgt.ever_fired[winners] = True
        tgt.num_ever_fired = int(xp.sum(tgt.ever_fired))
        tgt.w = len(winners)

        total_act = float(to_cpu(prev_winner_inputs[winners]).sum())

        return ProjectionResult(
            winners=np.array(to_cpu(xp.asarray(winners, dtype=xp.uint32)), dtype=np.uint32),
            num_first_winners=0,
            num_ever_fired=tgt.num_ever_fired,
            total_activation=total_act,
        )

    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        return np.array(to_cpu(st.winners), dtype=np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        xp = get_xp()
        st = self._areas[area]
        st.winners = xp.asarray(winners, dtype=xp.uint32)
        st.w = len(st.winners)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].num_ever_fired

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = beta

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or (hasattr(st.winners, '__len__') and len(st.winners) == 0):
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    def reset_area_connections(self, area: str) -> None:
        """Reset area->area connections involving *area* to initial state."""
        xp = get_xp()
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            conn = self._area_conns[src_name][area]
            rows, cols = conn.weights.shape
            conn.weights = xp.asarray(
                (np.random.default_rng().random((rows, cols)) < self.p
                 ).astype(np.float32),
            )

    @property
    def name(self) -> str:
        return "numpy_explicit"
