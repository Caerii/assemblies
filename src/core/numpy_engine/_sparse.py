"""NumpySparseEngine: CPU engine using statistical sparse simulation.

This is the extraction of brain.py's ``_project_into_legacy`` sparse path.
Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
generates candidate activations for new neurons.
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from ..backend import get_xp, to_cpu, to_xp
from ..engine import ComputeEngine, ProjectionResult
from ..connectome import Connectome

try:
    from ...compute.sparse_simulation import SparseSimulationEngine
    from ...compute.winner_selection import WinnerSelector
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine
    from compute.winner_selection import WinnerSelector

from ._state import SparseAreaState, StimulusState


class NumpySparseEngine(ComputeEngine):
    """CPU engine using statistical sparse simulation.

    Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
    numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
    generates candidate activations for new neurons.

    Parameters mirror ``Brain.__init__``.
    """

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False):
        self.p = p
        self.w_max = w_max
        self._deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        self._plasticity_enabled_global = True

        # Internal state
        self._areas: Dict[str, SparseAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}

        # Connectivity: stim_name -> area_name -> Connectome (1-D weights)
        self._stim_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)
        # Connectivity: src_area -> tgt_area -> Connectome (2-D weights)
        self._area_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)

        # Reusable math primitives
        self._sparse_sim = SparseSimulationEngine(self._rng)
        self._winner_sel = WinnerSelector(self._rng)

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0) -> None:
        xp = get_xp()
        area = SparseAreaState(name=name, n=n, k=k, beta=beta,
                               refractory_period=refractory_period,
                               inhibition_strength=inhibition_strength)
        area.neuron_id_pool = self._rng.permutation(np.arange(n, dtype=np.uint32))
        area.neuron_id_pool_ptr = 0
        self._areas[name] = area

        # Initialize stim->area connectomes for every already-registered stimulus
        for stim_name, stim in self._stimuli.items():
            conn = Connectome(stim.size, n, self.p, sparse=True)
            conn.weights = xp.empty(0, dtype=xp.float32)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        # Initialize area->area connectomes (both directions) for every existing area
        for other_name, other in self._areas.items():
            if other_name == name:
                self_conn = Connectome(n, n, self.p, sparse=True)
                self_conn.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[name][name] = self_conn
            else:
                conn_fwd = Connectome(other.n, n, self.p, sparse=True)
                conn_fwd.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[other_name][name] = conn_fwd

                conn_rev = Connectome(n, other.n, self.p, sparse=True)
                conn_rev.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[name][other_name] = conn_rev

                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        xp = get_xp()
        self._stimuli[name] = StimulusState(name=name, size=size)

        # Initialize stim->area for every already-registered area
        for area_name, area in self._areas.items():
            conn = Connectome(size, area.n, self.p, sparse=True)
            conn.weights = xp.empty(0, dtype=xp.float32)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        pass

    # -- Projection ---------------------------------------------------------

    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
    ) -> ProjectionResult:
        xp = get_xp()
        tgt = self._areas[target]
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter out source areas with no assembly
        from_areas = [a for a in from_areas
                      if self._areas[a].winners.size > 0 and self._areas[a].w > 0]

        # Fixed assembly — short-circuit
        if tgt.fixed_assembly:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # No inputs -> keep assembly unchanged
        if len(from_stimuli) == 0 and len(from_areas) == 0:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # --- Accumulate inputs from previous winners ---
        prev_winner_inputs = xp.zeros(tgt.w, dtype=xp.float32)

        # Stimulus inputs (1-D slice up to w)
        limit = tgt.w
        for stim in from_stimuli:
            stim_w = self._stim_conns[stim][target].weights
            end = min(limit, len(stim_w))
            if end > 0:
                prev_winner_inputs[:end] += stim_w[:end]

        # Area inputs (2-D, vectorised fancy-index)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            if conn.weights.shape[1] == 0:
                continue
            src = self._areas[src_name]
            src_w = xp.asarray(src.winners)
            internal = src_w[src_w < conn.weights.shape[0]]
            if len(internal) > 0 and limit > 0:
                col_end = min(limit, conn.weights.shape[1])
                prev_winner_inputs[:col_end] += conn.weights[internal, :col_end].sum(axis=0)

        # Zero signal -> preserve current assembly
        if len(prev_winner_inputs) > 0 and float(xp.sum(prev_winner_inputs)) == 0.0:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # --- Sample new winner candidates via truncated normal ---
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [self._areas[a].k for a in from_areas]
        )

        old_rng = self._sparse_sim.rng
        self._sparse_sim.rng = rng
        if self._deterministic:
            potential_new = self._sparse_sim.sample_new_winner_inputs_legacy(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p,
            )
        else:
            potential_new = self._sparse_sim.sample_new_winner_inputs(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p,
            )
        self._sparse_sim.rng = old_rng

        potential_new = to_xp(potential_new)
        if len(prev_winner_inputs) > 0:
            all_inputs = xp.concatenate([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

        # --- LRI: penalise recently-fired neurons ---
        if (tgt.refractory_period > 0
                and tgt.inhibition_strength > 0
                and len(tgt._refractory_history) > 0):
            n_inputs = len(all_inputs)
            for steps_ago_idx, winner_set in enumerate(
                    reversed(list(tgt._refractory_history))):
                steps_ago = steps_ago_idx + 1
                decay = 1.0 - (steps_ago - 1) / tgt.refractory_period
                penalty = tgt.inhibition_strength * decay
                for cidx in winner_set:
                    if cidx < n_inputs:
                        all_inputs[cidx] -= penalty

        # --- Refracted mode: cumulative bias penalty ---
        if tgt.refracted and tgt._cumulative_bias is not None:
            bias = tgt._cumulative_bias
            end = min(len(bias), len(all_inputs))
            if end > 0:
                all_inputs[:end] -= bias[:end]

        # --- Select top-k winners ---
        new_winner_indices = self._winner_sel.heapq_select_top_k(
            all_inputs, tgt.k
        ).tolist()

        # --- Process first-time winners ---
        num_first = 0
        first_winner_inputs = []
        for i in range(tgt.k):
            if new_winner_indices[i] >= tgt.w:
                first_winner_inputs.append(int(all_inputs[new_winner_indices[i]]))
                if tgt.neuron_id_pool is not None:
                    pid = tgt.neuron_id_pool_ptr
                    if pid >= len(tgt.neuron_id_pool):
                        raise RuntimeError(f"Neuron id pool exhausted for area {tgt.name}")
                    actual_id = int(tgt.neuron_id_pool[pid])
                    tgt.neuron_id_pool_ptr += 1
                else:
                    actual_id = tgt.w + num_first
                tgt.compact_to_neuron_id.append(actual_id)
                new_winner_indices[i] = tgt.w + num_first
                num_first += 1

        new_w = tgt.w + num_first

        # --- Apply plasticity ---
        if plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(target, from_stimuli, from_areas, new_winner_indices)

        # --- Expand connectomes for new winners ---
        if num_first > 0:
            self._expand_connectomes(
                target, from_stimuli, from_areas,
                input_sizes, new_winner_indices,
                first_winner_inputs, new_w,
            )

        # --- Commit state ---
        tgt.winners = xp.asarray(new_winner_indices, dtype=xp.uint32)
        tgt.w = new_w

        # --- Update LRI refractory history ---
        if tgt.refractory_period > 0:
            tgt._refractory_history.append(
                set(int(i) for i in new_winner_indices))

        # --- Update refracted cumulative bias ---
        if tgt.refracted and tgt.refracted_strength > 0:
            if len(tgt._cumulative_bias) < new_w:
                old = tgt._cumulative_bias
                tgt._cumulative_bias = xp.zeros(new_w, dtype=xp.float32)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            for cidx in new_winner_indices:
                if cidx < len(tgt._cumulative_bias):
                    tgt._cumulative_bias[cidx] += tgt.refracted_strength

        total_act = float(xp.sum(all_inputs[new_winner_indices]))

        return ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w,
            total_activation=total_act,
        )

    # -- Plasticity ---------------------------------------------------------

    def _apply_plasticity(self, target, from_stimuli, from_areas, winners):
        """Hebbian learning: w *= (1 + beta), clamped at w_max."""
        xp = get_xp()
        tgt = self._areas[target]
        winners_arr = xp.asarray(winners, dtype=xp.int64)

        # Stimulus -> area (1-D weights)
        for stim_name in from_stimuli:
            conn = self._stim_conns[stim_name][target]
            beta = tgt.beta_by_source.get(stim_name, tgt.beta)
            if beta == 0:
                continue
            valid = winners_arr[winners_arr < len(conn.weights)]
            if len(valid) > 0:
                conn.weights[valid] *= (1 + beta)
            if self.w_max is not None:
                xp.clip(conn.weights, 0, self.w_max, out=conn.weights)

        # Area -> area (2-D weights)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            if beta == 0:
                continue
            src = self._areas[src_name]
            src_w = xp.asarray(src.winners)
            if conn.weights.ndim == 2:
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                valid_cols = winners_arr[winners_arr < conn.weights.shape[1]]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    ix = xp.ix_(valid_rows, valid_cols)
                    conn.weights[ix] *= (1 + beta)
                    if self.w_max is not None:
                        sub = conn.weights[ix]
                        xp.clip(sub, 0, self.w_max, out=sub)
                        conn.weights[ix] = sub
            else:
                valid = winners_arr[winners_arr < len(conn.weights)]
                if len(valid) > 0:
                    conn.weights[valid] *= (1 + beta)
                if self.w_max is not None:
                    xp.clip(conn.weights, 0, self.w_max, out=conn.weights)

    # -- Connectome expansion for new winners --------------------------------

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs, new_w):
        """Expand connectivity for first-time winners.

        Uses amortised buffer growth for 2-D area->area matrices: physical
        capacity doubles when exceeded, avoiding repeated vstack/hstack
        reallocation on every step.
        """
        xp = get_xp()
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs,
        )

        prior_w = tgt.w
        new_indices = [int(w) for w in winners if int(w) >= prior_w]

        # --- Expand stim->area 1-D vectors ---
        stim_names = [name for name in inputs_names if name in self._stimuli]
        area_names = [name for name in inputs_names if name in self._areas]

        for stim_name in self._stimuli.keys():
            conn = self._stim_conns[stim_name][target]
            if conn.sparse:
                old = len(conn.weights)
                if new_w > old:
                    add_len = new_w - old
                    if stim_name not in stim_names:
                        stim_size = self._stimuli[stim_name].size
                        add = to_xp(self._rng.binomial(stim_size, self.p, size=add_len).astype(np.float32))
                    else:
                        add = xp.zeros(add_len, dtype=xp.float32)
                    conn.weights = xp.concatenate([conn.weights, add])

        # Write allocations for firing stimuli
        for idx, win in enumerate(new_indices):
            if win >= new_w:
                continue
            split = splits_per_new[idx] if idx < len(splits_per_new) else None
            if split is None:
                continue
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self._stimuli:
                    conn = self._stim_conns[name][target]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc

        # --- Expand area->area 2-D matrices ---
        for src_name in area_names:
            conn = self._area_conns[src_name][target]
            if not conn.sparse:
                continue
            src = self._areas[src_name]
            if conn.weights.ndim != 2:
                conn.weights = xp.empty((0, 0), dtype=xp.float32)

            phys_rows, phys_cols = conn.weights.shape
            src_w_arr = xp.asarray(src.winners)
            max_src_idx = (int(xp.max(src_w_arr)) + 1) if src_w_arr.size > 0 else 0
            needed_rows = max(max_src_idx, new_w if src_name == target else src.w)
            needed_cols = new_w

            if self._deterministic:
                if needed_rows > phys_rows:
                    nr = needed_rows - phys_rows
                    new_rows = to_xp(
                        (self._rng.random((nr, phys_cols)) < self.p).astype(np.float32)
                    )
                    conn.weights = xp.vstack([conn.weights, new_rows]) if phys_cols > 0 else xp.zeros((needed_rows, 0), dtype=xp.float32)
                    phys_rows = needed_rows
                if needed_cols > phys_cols:
                    nc = needed_cols - phys_cols
                    new_cols = to_xp(
                        (self._rng.random((phys_rows, nc)) < self.p).astype(np.float32)
                    )
                    conn.weights = xp.hstack([conn.weights, new_cols]) if phys_rows > 0 else xp.zeros((0, needed_cols), dtype=xp.float32)
                    phys_cols = needed_cols
            else:
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
                    buf = xp.zeros((new_pr, new_pc), dtype=xp.float32)
                    if phys_rows > 0 and phys_cols > 0:
                        buf[:phys_rows, :phys_cols] = conn.weights
                    conn.weights = buf
                    phys_rows, phys_cols = new_pr, new_pc

                nr = needed_rows - log_rows
                nc = needed_cols - log_cols
                if nr > 0 and log_cols > 0:
                    conn.weights[log_rows:needed_rows, :log_cols] = to_xp(
                        (self._rng.random((nr, log_cols)) < self.p).astype(np.float32)
                    )
                if nc > 0 and log_rows > 0:
                    conn.weights[:log_rows, log_cols:needed_cols] = to_xp(
                        (self._rng.random((log_rows, nc)) < self.p).astype(np.float32)
                    )
                if nr > 0 and nc > 0:
                    conn.weights[log_rows:needed_rows, log_cols:needed_cols] = to_xp(
                        (self._rng.random((nr, nc)) < self.p).astype(np.float32)
                    )

                conn._log_rows = needed_rows
                conn._log_cols = needed_cols

            # -- Write specific allocations for first-time winners --
            from_index = inputs_names.index(src_name)
            local_rng = np.random.default_rng(self._rng.integers(0, 2**32))
            src_winners_cpu = np.asarray(
                to_cpu(src.winners) if hasattr(src.winners, 'get') else src.winners
            )
            for idx, win in enumerate(new_indices):
                alloc = int(splits_per_new[idx][from_index]) if idx < len(splits_per_new) else 0
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(src_winners_cpu, size=sample_size, replace=False)
                col_idx = win - prior_w
                if 0 <= col_idx < phys_cols:
                    conn.weights[chosen, col_idx] = 1.0

    # -- State accessors ----------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        return np.array(to_cpu(st.winners), dtype=np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        xp = get_xp()
        st = self._areas[area]
        st.winners = xp.asarray(winners, dtype=xp.uint32)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].w

    def get_neuron_id_mapping(self, area: str) -> list:
        """Return the compact_to_neuron_id list for stable winner IDs."""
        return self._areas[area].compact_to_neuron_id

    # -- Plasticity control -------------------------------------------------

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = beta

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    # -- Assembly fixation --------------------------------------------------

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or (hasattr(st.winners, '__len__') and len(st.winners) == 0):
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    # -- Connection reset ---------------------------------------------------

    def reset_area_connections(self, area: str) -> None:
        """Reset area->area connections involving *area* to initial state."""
        xp = get_xp()
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            conn = self._area_conns[src_name][area]
            if conn.sparse:
                conn.weights = xp.empty((0, 0), dtype=xp.float32)
                if hasattr(conn, '_log_rows'):
                    del conn._log_rows
                if hasattr(conn, '_log_cols'):
                    del conn._log_cols
            else:
                rows, cols = conn.weights.shape
                conn.weights = xp.asarray(
                    (self._rng.random((rows, cols)) < self.p
                     ).astype(np.float32),
                )

    # -- LRI control --------------------------------------------------------

    def clear_refractory(self, area: str) -> None:
        """Clear refractory history for an area."""
        self._areas[area]._refractory_history.clear()

    def set_lri(self, area: str, refractory_period: int,
                inhibition_strength: float) -> None:
        """Update LRI parameters for an area at runtime."""
        from collections import deque
        st = self._areas[area]
        st.refractory_period = refractory_period
        st.inhibition_strength = inhibition_strength
        st._refractory_history = deque(
            maxlen=max(refractory_period, 1))

    # -- Refracted mode control ---------------------------------------------

    def set_refracted(self, area: str, enabled: bool,
                      strength: float = 0.0) -> None:
        """Enable or disable refracted mode for an area."""
        st = self._areas[area]
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and len(st._cumulative_bias) == 0:
            xp = get_xp()
            st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    def clear_refracted_bias(self, area: str) -> None:
        """Reset accumulated refracted bias to zero."""
        xp = get_xp()
        st = self._areas[area]
        st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: str = None) -> None:
        """Column-normalize weights into *target* so each neuron sums to 1.0."""
        xp = get_xp()
        eps = 1e-8

        def _norm_conn(conn):
            w = conn.weights
            if w.ndim == 2 and w.size > 0:
                col_sums = w.sum(axis=0, keepdims=True)
                col_sums = xp.maximum(col_sums, eps)
                conn.weights = w / col_sums
            elif w.ndim == 1 and w.size > 0:
                total = float(xp.sum(w))
                if total > eps:
                    conn.weights = w / total

        if source is not None:
            if source in self._stim_conns and target in self._stim_conns[source]:
                _norm_conn(self._stim_conns[source][target])
            if source in self._area_conns and target in self._area_conns[source]:
                _norm_conn(self._area_conns[source][target])
            return

        for stim_name in self._stim_conns:
            if target in self._stim_conns[stim_name]:
                _norm_conn(self._stim_conns[stim_name][target])
        for src_name in self._area_conns:
            if target in self._area_conns[src_name]:
                _norm_conn(self._area_conns[src_name][target])

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "numpy_sparse"
