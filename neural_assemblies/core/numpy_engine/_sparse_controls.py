"""Per-area state and controls for the numpy engine: winners and neuron ids, the extent of
what is materialized, projection fidelity (compiled topology), input noise and competition,
learning rates, fixed assemblies, connection reset, LRI and refraction.

A mixin of NumpySparseEngine (_sparse.py), which owns the state these methods read."""


import numpy as np

from ..index_spaces import CompactIdx, NeuronIds, validated_indices


from ..backend import to_cpu


from .._homeostasis import (validate_refraction_strength,
                            check_area_homeostasis, validate_lri_parameters)


from ..registration import (validate_input_noise, validate_plasticity_rate)


from ..projection_fidelity import ProjectionFidelity


try:
    from ...compute.winner_policies import validate_competition_policy
except ImportError:
    from compute.winner_policies import validate_competition_policy

from ._growth import GrowthMixin, _self_fiber_deferred_init  # noqa: F401


from ._sparse_switches import (  # noqa: F401  re-exported: the torch engine and _exact read them here
    _PRUNE_MAX_FRACTION, _explicit_src_norm_enabled, _fixed_target_plasticity_enabled,
    _strict_drive_enabled, _warn_fixed_target_enabled,
)


from ._drive_cache import (  # noqa: F401
    DriveCacheMixin, _csr_storage_available, _CSR_MIN_CELLS,
    _CSR_MAX_DENSITY,
)


class NumpyAreaControls:
    """Accessors and switches of an area (mixed into NumpySparseEngine)."""

    # -- Connectome expansion for new winners --------------------------------


    def get_winners(self, area: str) -> CompactIdx:
        st = self._areas[area]
        return CompactIdx(np.array(to_cpu(st.winners), dtype=np.uint32))

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-winner-inputs"""
        if isinstance(winners, NeuronIds) and self.get_neuron_id_mapping(area):
            raise TypeError("sparse engine winner inputs require compact indices")
        xp = self._xp
        st = self._areas[area]
        st.winners = CompactIdx(validated_indices(winners, upper=st.n,
                                                  label=f"{area} winners",
                                                  xp=xp, unique=True))

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].w

    def get_neuron_id_mapping(self, area: str) -> list:
        """Return the compact_to_neuron_id list for stable winner IDs."""
        return self._areas[area].compact_to_neuron_id

    # -- Materialization (see ComputeEngine.fiber_extent for the rationale) --

    def materialized_count(self, area: str):
        st = self._areas.get(area)
        return None if st is None else int(st.w)

    def fiber_extent(self, source: str, target: str):
        """Logical column watermark of ``source -> target``.

        Returns ``None`` when the fiber is dense (every column exists, so the
        watermark is vacuous) or absent, and the integer watermark when it is
        lazily materialized. The PHYSICAL shape is deliberately not returned:
        growth doubles capacity, so it over-runs the logical content and
        columns past the watermark are allocated-but-uninitialised zeros.
        """
        conn = self._area_conns.get(source, {}).get(target)
        if conn is None or not getattr(conn, "sparse", False):
            return None
        w = getattr(conn, "weights", None)
        if w is None or getattr(w, "ndim", 0) != 2:
            return None
        return int(min(getattr(conn, "_log_cols", w.shape[1]), w.shape[1]))

    # -- Projection fidelity ------------------------------------------------

    def set_projection_fidelity(self, fidelity: str) -> None:
        self._projection_fidelity = ProjectionFidelity.normalize(fidelity)

    def get_projection_fidelity(self) -> str:
        return self._projection_fidelity.value

    def preallocate_stim_targets(self, target: str, min_columns: int) -> None:
        """Extend all stim→*target* vectors to at least *min_columns* (zeros)."""
        if min_columns <= 0 or target not in self._areas:
            return
        xp = self._xp
        n = self._areas[target].n
        for _stim_name, tgt_map in self._stim_conns.items():
            conn = tgt_map.get(target)
            if conn is None or not conn.sparse:
                continue
            old = len(conn.weights)
            if min_columns > old:
                if self._stim_fastpath:
                    self._grow_stim_vector(conn, n, old, min_columns, None)
                else:
                    add = xp.zeros(min_columns - old, dtype=xp.float32)
                    conn.weights = (
                        xp.concatenate([conn.weights, add])
                        if old > 0 else add
                    )

    def _use_compiled_projection(self, tgt) -> bool:
        """True when compiled/fuzzy top-k on pregrown columns should run."""
        if not getattr(tgt, "_freeze_connectome_growth", False):
            return False
        if getattr(tgt, "_force_exact_projection", False):
            return False
        if tgt.w < tgt.k:
            return False
        if getattr(tgt, "_plasticity_only_mode", False):
            return True
        return self._projection_fidelity == ProjectionFidelity.COMPILED

    # -- Plasticity control -------------------------------------------------

    def set_input_noise(self, area: str, std: float) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-input-noise"""
        self._areas[area].input_noise_std = validate_input_noise(std)

    def set_competition_policy(self, area: str, policy) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-runtime-policy"""
        validate_competition_policy(self._areas[area].n, policy)
        self._areas[area].winner_policy = policy

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = validate_plasticity_rate(beta)

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
        self.invalidate_csr_drive()
        xp = self._xp
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
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        st = self._areas[area]
        st.refractory_period = refractory_period
        st.inhibition_strength = inhibition_strength
        st._refractory_history = deque(
            maxlen=max(refractory_period, 1))

    # -- Refracted mode control ---------------------------------------------

    def set_refracted(self, area: str, enabled: bool,
                      strength: float = 0.0) -> None:
        """Enable or disable refracted mode for an area."""
        if type(enabled) is not bool:
            raise TypeError("refracted enabled flag must be a bool")
        strength = validate_refraction_strength(strength)
        st = self._areas[area]
        check_area_homeostasis(area, refracted=enabled, synaptic_scaling=self.synaptic_scaling)
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and len(st._cumulative_bias) == 0:
            xp = self._xp
            st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    def clear_refracted_bias(self, area: str) -> None:
        """Reset accumulated refracted bias to zero."""
        xp = self._xp
        st = self._areas[area]
        st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)
