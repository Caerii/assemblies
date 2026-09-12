"""NumpyExplicitEngine: CPU engine using dense explicit simulation.

All n neurons are tracked.  Connectivity is stored as full
(source_n, target_n) float32 matrices.  Suitable for small networks
where full fidelity is required.
"""

import numpy as np
from typing import Any, Dict, List, cast
from collections import defaultdict

from ..backend import to_cpu
from ..engine import (
    ComputeEngine,
    ProjectionResult,
    validate_deterministic_allocation,
)
from ..registration import (validate_input_noise, validate_stimulus_registration,
                            validate_area_registration, validate_slot_configuration,
                            validate_plasticity_rate)
from ..connectome import Connectome
from ..index_spaces import validated_indices
from ..index_spaces import CompactIdx
from ..semantics import (
    ArithmeticMode, CandidateDomain, ConnectomeMode, ModelSemantics,
    NormalizationMode, PlasticityRule, StimulusDriveLaw, TieBreakRule,
)

try:
    from ...compute.winner_selection import WinnerSelector, select_slot_winners
except ImportError:
    from compute.winner_selection import WinnerSelector, select_slot_winners

from ._exact import _reject_unsupported
from ._state import ExplicitAreaState, StimulusState


class NumpyExplicitEngine(ComputeEngine):
    """CPU engine using dense explicit simulation.

    All n neurons are tracked.  Connectivity is stored as full
    (source_n, target_n) float32 matrices.  Suitable for small networks
    where full fidelity is required.
    """

    supports_slots = True
    supports_fiber_learning_masks = True

    def describe_model_semantics(self) -> ModelSemantics:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics"""
        return ModelSemantics(
            connectome=ConnectomeMode.FIXED_DENSE_CONTENT_ADDRESSED,
            candidate_domain=CandidateDomain.ALL_NEURONS,
            stimulus_drive=StimulusDriveLaw.FIXED_BERNOULLI_AFFERENT_COUNT,
            default_tie_break=TieBreakRule.LOWEST_NEURON_ID,
            arithmetic=ArithmeticMode.FLOAT32,
            normalization=NormalizationMode.NONE,
            plasticity=(
                PlasticityRule.MULTIPLICATIVE_UNBOUNDED
                if self.w_max is None
                else PlasticityRule.MULTIPLICATIVE_CLIPPED
            ),
            weight_ceiling=self.w_max,
        )

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False):
        validate_deterministic_allocation(type(self), deterministic)
        self.p = p
        self.w_max = w_max
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._plasticity_enabled_global = True

        self._areas: Dict[str, ExplicitAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}
        self._stim_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)
        self._area_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)
        self._winner_sel = WinnerSelector(self._rng)

    def _fiber_seed(self, source: str, target: str) -> int:
        """Content-addressed identity for one fiber -- door 5 of
        [[content-addressed-synapse-init]].

        This engine draws its dense connectomes from a single `_rng` STREAM, so
        a fiber's wiring depended on how many draws preceded it: two Brains
        with the same seed and the same areas, created in opposite order,
        agreed on X->X wiring at 0.905, exactly chance for p=0.05. Keying on
        (seed, source, target) makes wiring a function of WHICH fiber it is.

        The explicit engine is where explicit areas' connectomes are actually
        born -- `Brain` adopts the same objects -- so patching Brain alone left
        this open, verified by the object identities matching.
        """
        from ._seeding import fnv1a_pair_seed
        return fnv1a_pair_seed(self.seed, source, target)

    #: Per-area mechanisms `Brain.add_area` forwards that this engine does not
    #: implement, with the value meaning "not requested". Same rationale as
    #: `NumpyExactEngine._UNSUPPORTED_AREA`.
    #:
    #: `refractory_period` and `inhibition_strength` were in the SIGNATURE and
    #: went nowhere -- `ExplicitAreaState` has no field for either, so LRI was
    #: silently off on this engine while the caller's configuration said it was
    #: on. That is [[silent-no-op-dead-fibers]] exactly: configured, wired,
    #: never runs, and the symptom is "LRI seems to have little effect".
    _UNSUPPORTED_AREA = {
        "refractory_period": 0,
        "inhibition_strength": 0.0,
        "input_noise_std": 0.0,
    }

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 winner_policy=None,
                 input_noise_std: float = 0.0,
                 *, slot_count: int = 0) -> None:
        """Dense area registration.

        `winner_policy` and `input_noise_std` ARE IN THIS SIGNATURE because
        `Brain.add_area` forwards them to the primary engine unconditionally.
        Without them `Brain(engine="numpy_explicit")` raised TypeError on the
        first `add_area` -- so the dense engine, which is the ground truth every
        sampler result is checked against, could not be constructed through the
        public API at all. It was reachable only as an `explicit=True` area
        inside a sparse brain, by a different call with a different kwarg set.

        `winner_policy` is implemented here rather than refused: this engine has
        an RNG and a `WinnerSelector` already, and a competition rule that
        cannot run on the exact-drive engine is a rule whose results cannot be
        checked (#94). `input_noise_std` is refused loudly instead of ignored.
        """
        input_noise_std = validate_input_noise(input_noise_std)
        n, k = validate_area_registration(name, n, k, existing=self._areas, reserved=self._stimuli)
        from ...compute.winner_policies import validate_competition_policy
        validate_competition_policy(n, winner_policy)
        beta = validate_plasticity_rate(beta)
        slot_count = validate_slot_configuration(n, slot_count, winner_policy)
        _reject_unsupported(
            f"NumpyExplicitEngine.add_area({name!r})", self._UNSUPPORTED_AREA,
            dict(refractory_period=refractory_period,
                 inhibition_strength=inhibition_strength,
                 input_noise_std=input_noise_std))
        area = ExplicitAreaState(name=name, n=n, k=k, beta=beta,
                                 slot_count=slot_count,
                                 winner_policy=winner_policy,
                                 backend_name="numpy")
        self._areas[name] = area

        for stim_name, stim in self._stimuli.items():
            conn = Connectome(stim.size, n, self.p, sparse=False,
                              rng=self._rng,
                              pair_seed=self._fiber_seed(stim_name, name))
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        for other_name, other in self._areas.items():
            if other_name == name:
                self._area_conns[name][name] = Connectome(
                    n, n, self.p, sparse=False, rng=self._rng,
                    pair_seed=self._fiber_seed(name, name))
            else:
                self._area_conns[other_name][name] = Connectome(
                    other.n, n, self.p, sparse=False, rng=self._rng,
                    pair_seed=self._fiber_seed(other_name, name))
                self._area_conns[name][other_name] = Connectome(
                    n, other.n, self.p, sparse=False, rng=self._rng,
                    pair_seed=self._fiber_seed(name, other_name))
                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        size = validate_stimulus_registration(name, size, existing=self._stimuli,
                                             reserved=self._areas)
        self._stimuli[name] = StimulusState(name=name, size=size)
        for area_name, area in self._areas.items():
            conn = Connectome(size, area.n, self.p, sparse=False,
                              rng=self._rng,
                              pair_seed=self._fiber_seed(name, area_name))
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        """Per-fiber connection probability -- NOT supported by this engine.

        Same rationale as `numpy_sparse.add_connectivity`: this was `pass`
        everywhere while the interface advertised it, so callers silently got
        the global `p`. Here the connectomes are drawn dense at construction
        from a shared RNG stream, so a per-fiber density would have to be
        threaded into `add_area`/`add_stimulus` ordering -- exactly the
        draw-order coupling this engine already has open as door 5 (#81).
        Requesting the global `p` stays a no-op; anything else raises.
        """
        if float(p) != float(self.p):
            raise NotImplementedError(
                f"numpy_explicit does not support per-fiber connectivity: "
                f"add_connectivity({source!r}, {target!r}, p={p}) differs from "
                f"the engine's p={self.p}. Use numpy_exact, which implements "
                f"it, rather than assuming this call took effect.")

    def validate_projection_inputs(self, target, from_stimuli, from_areas, external_drive=None):
        """Nonmutating numerical preflight shared by execution and IR lowering.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-inputs
        Returns validated source caps and the float32 drive, without publishing
        either into engine state. Profile and fiber restrictions remain in IR.
        """
        xp = np
        if target not in self._areas:
            raise ValueError(f"Unknown target area {target!r}")
        tgt = self._areas[target]
        for names, registry in ((from_areas, self._areas), (from_stimuli, self._stimuli)):
            if len(set(names)) != len(names) or any(name not in registry for name in names):
                raise ValueError("Projection sources must be distinct registered names")
        self._validated_winners(target, tgt.winners)
        source_winners = {name: self._validated_winners(name, self._areas[name].winners)
                          for name in from_areas}
        if external_drive is not None:
            external_drive = xp.asarray(external_drive)
            if external_drive.shape != (tgt.n,) or external_drive.dtype.kind not in "fiu":
                raise ValueError("External drive must be a real vector of target population length")
            with np.errstate(over="ignore"):
                external_drive = external_drive.astype(xp.float32, copy=False)
            if not bool(xp.isfinite(external_drive).all()):
                raise ValueError("External drive must be finite and representable as float32")

        return source_winners, external_drive

    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        record_activation: bool = False,
        external_drive: np.ndarray | None = None,
    ) -> ProjectionResult:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-inputs

        Validate numerical inputs even when the target is clamped. IR and legacy
        callers share this boundary; profile restrictions remain with the IR.
        """
        xp = np
        source_winners, external_drive = self.validate_projection_inputs(
            target, from_stimuli, from_areas, external_drive)
        tgt = self._areas[target]

        # Empty sources contribute neither drive nor learning.
        from_areas = [name for name in from_areas if source_winners[name].size > 0]

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
            winners_in = source_winners[src_name]
            if winners_in.size > 0 and int(xp.max(winners_in)) >= conn.weights.shape[0]:
                raise IndexError(
                    f"Source area {src_name!r} has winner index "
                    f"{int(xp.max(winners_in))} exceeding connectome "
                    f"rows ({conn.weights.shape[0]})")
            if winners_in.size > 0:
                prev_winner_inputs += conn.weights[winners_in].sum(axis=0)

        if external_drive is not None:
            prev_winner_inputs += xp.asarray(
                external_drive, dtype=xp.float32,
            )

        # Explicit areas index global neuron ids 0..n-1; do not remap to sparse
        # contiguous slots (that path is for refractory sparse areas only).
        if tgt.slot_count and tgt.slot_count > 1:
            winners = select_slot_winners(
                prev_winner_inputs, tgt.k, tgt.slot_count,
            )
        else:
            winners = self._select_winners(prev_winner_inputs, tgt)

        winners = self._validated_winners(target, winners)

        # Apply plasticity
        if plasticity_enabled and self._plasticity_enabled_global:
            for stim_name in from_stimuli:
                conn = self._stim_conns[stim_name][target]
                beta = tgt.beta_by_source.get(stim_name, tgt.beta)
                if beta != 0 and self.fiber_learning_allowed(stim_name, target):
                    conn.weights[:, winners] *= (1 + beta)
                    if self.w_max is not None:
                        xp.clip(conn.weights, 0, self.w_max, out=conn.weights)

            for src_name in from_areas:
                conn = self._area_conns[src_name][target]
                beta = tgt.beta_by_source.get(src_name, tgt.beta)
                if beta != 0 and self.fiber_learning_allowed(src_name, target):
                    ix = xp.ix_(source_winners[src_name], winners)
                    cast(Any, conn.weights)[ix] *= (1 + beta)
                    if self.w_max is not None:
                        sub = cast(Any, conn.weights)[ix]
                        xp.clip(sub, 0, self.w_max, out=sub)
                        cast(Any, conn.weights)[ix] = sub

        # Update state
        winners = xp.asarray(winners, dtype=xp.uint32)
        tgt.winners = winners
        cast(Any, tgt).ever_fired[winners] = True
        tgt.num_ever_fired = int(xp.sum(cast(Any, tgt).ever_fired))
        tgt.w = len(winners)

        total_act = float(to_cpu(prev_winner_inputs[winners]).sum())
        pre_kwta = (
            np.array(to_cpu(prev_winner_inputs), dtype=np.float32, copy=True)
            if record_activation else None
        )

        return ProjectionResult(
            winners=np.array(to_cpu(xp.asarray(winners, dtype=xp.uint32)), dtype=np.uint32),
            num_first_winners=0,
            num_ever_fired=tgt.num_ever_fired,
            total_activation=total_act,
            pre_kwta_inputs=pre_kwta,
            pre_kwta_prev_only=(pre_kwta.copy() if pre_kwta is not None else None),
            pre_kwta_total=(float(to_cpu(prev_winner_inputs).sum())
                            if record_activation else 0.0),
            pre_kwta_count=tgt.n if record_activation else 0,
        )

    def _select_winners(self, drive, tgt):
        """k-WTA unless the area carries a `winner_policy`.

        Policies go through the SHARED `compute.winner_selection`
        implementation, exactly as `NumpyExactEngine._select_winners` does, so
        the engines cannot come to disagree about what a policy MEANS -- the
        failure mode of [[pricing-law-implemented-twice]], where the same law
        was implemented twice and one copy missed two fixes.

        Unlike the exact engine this one HAS an RNG, so a policy that samples
        works here rather than needing a stand-in generator.
        """
        policy = getattr(tgt, "winner_policy", None)
        if policy is None:
            _, _, _, winners = self._winner_sel.select_combined_winners(
                drive, tgt.n, tgt.k)
            return winners
        from ...compute.winner_policies import TopKPolicy
        if isinstance(policy, TopKPolicy) and policy.k == tgt.k:
            _, _, _, winners = self._winner_sel.select_combined_winners(
                drive, tgt.n, tgt.k)
            return winners
        return self._winner_sel.select_with_policy(drive, policy)

    def get_winners(self, area: str) -> CompactIdx:
        st = self._areas[area]
        return CompactIdx(np.array(to_cpu(st.winners), dtype=np.uint32))

    def _validated_winners(self, area: str, winners):
        xp = np
        st = self._areas[area]
        if not 0 < st.k <= st.n:
            raise ValueError("Area requires 0 < k <= n")
        return validated_indices(winners, upper=st.n, label=f"{area} neuron IDs",
                                 xp=xp, unique=True)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-inputs"""
        ids = self._validated_winners(area, winners)
        st = self._areas[area]
        st.winners = CompactIdx(ids)
        st.w = len(ids)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].num_ever_fired

    def set_competition_policy(self, area: str, policy) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-runtime-policy"""
        state = self._areas[area]
        from ...compute.winner_policies import validate_competition_policy
        validate_competition_policy(state.n, policy)
        validate_slot_configuration(state.n, state.slot_count, policy)
        state.winner_policy = policy

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = validate_plasticity_rate(beta)

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or (hasattr(st.winners, '__len__') and len(cast(Any, st.winners)) == 0):
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    def reset_area_connections(self, area: str) -> None:
        """Reset area->area connections involving *area* to initial state."""
        xp = np
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            conn = self._area_conns[src_name][area]
            rows, cols = conn.weights.shape
            conn.weights = xp.asarray(
                (self._rng.random((rows, cols)) < self.p
                 ).astype(np.float32),
            )

    @property
    def name(self) -> str:
        return "numpy_explicit"
