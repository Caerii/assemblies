"""
Named Assembly Calculus operations -- the instruction set of the NEMO model.

Everything in this file is a *schedule of projections*.  The underlying brain
model has exactly one primitive: in each discrete time step, every area sums
the synaptic input arriving from currently-firing presynaptic neurons, the top
``k`` neurons by input win (winners-take-all, a stand-in for local inhibition),
and every synapse that carried input from a firing presynaptic neuron to a
winner is multiplied by ``(1 + beta)`` (Hebbian plasticity).  An "operation"
in the assembly calculus is nothing more than a choice of *which fibers are
open on which time step*.  That is why every function below is a short
sequence of ``brain.project(...)`` calls and nothing else -- the science lives
in the schedule, not in any per-operation math.

The five primitives of the calculus, and where they are implemented:

    project           :func:`project`           stimulus -> area
    reciprocal_project:func:`reciprocal_project`area -> area (a "copy")
    associate         :func:`associate`         two assemblies pulled together
    merge             :func:`merge`             two assemblies -> one conjunct
    pattern_complete  :func:`pattern_complete`  partial cue -> full assembly

plus the sequence extension of [SEQ25] (:func:`sequence_memorize` /
:func:`ordered_recall`), which needs long-range inhibition (a refractory
period) to break the attractor that the other operations rely on.

Two properties hold for all of them and are worth stating once:

* **Convergence is empirical, not enforced.**  The PNAS 2020 analysis shows
  assemblies stabilise in O(log n) rounds *with high probability* under a
  random G(n, p) connectome.  Nothing here checks that it happened; ``rounds``
  is a budget, not a convergence criterion.  Callers that need a guarantee use
  :func:`learn_assembly` / :func:`learn_assembly_from_pattern`, which loop
  until consecutive snapshots overlap above a threshold.
* **Mutation is part of the operation contract.**  Projection, reciprocal
  projection, association and merge apply the brain's live plasticity policy.
  Pattern completion requires an explicit ``plastic``, ``frozen`` or
  ``read-only`` observation policy. The returned :class:`Assembly` is an
  immutable snapshot precisely because live winners may move or be restored.

All functions:

- Take a Brain as first argument (purely functional, no subclassing)
- Return an Assembly snapshot of the result
- Accept ``rounds`` for stabilization control

Reference:
    [PNAS20] Papadimitriou, Vempala, Mitropolsky, Collins, Maass.
    "Brain Computation by Assemblies of Neurons." PNAS 117(25), 2020.

Bracketed tags such as [PNAS20] / [COIN24] / [SEQ25] / [ACREF] are citation
keys resolving against ``research/literature/index.json`` (field ``cite_tag``),
which carries the full reference and a local PDF path where one is checked in.
``tests/test_literature_index.py`` fails if a tag used here has no entry, so a
citation cannot quietly become a dead string.
"""

from contextlib import contextmanager
from numbers import Integral, Real

import numpy as np

from .assembly import Assembly, overlap
from .contracts import (
    ASSOCIATION_CONTRACT, COMPLETION_CONTRACT, MERGE_CONTRACT,
    ORDERED_RECALL_CONTRACT,
    SEPARATION_CONTRACT,
    PROJECTION_CONTRACT, RECIPROCAL_PROJECTION_CONTRACT, AssociationPlan,
    CompletionPlan, MergePlan, ProjectionPlan, ReciprocalProjectionPlan,
    OrderedRecallPlan, SequenceMemorizePlan, SeparationPlan,
    SEQUENCE_MEMORIZE_CONTRACT,
    implements,
)
from ..core.index_spaces import NeuronIds, to_neuron_ids, validated_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(brain, area_name) -> Assembly:
    """Take an immutable snapshot of the current assembly in an area.

    THE TWO INDEX SPACES.  The sparse engine never materialises the full
    ``n``-neuron population of an area (``n`` is typically 10^4-10^6 while an
    assembly is ``k`` ~ 10^2).  It instead *simulates lazily*: a neuron gets a
    row in the connectome only once it has actually won something.  So an area
    has two coordinate systems:

        compact index  -- position in the engine's materialised arrays; dense,
                          contiguous, and GROWS as new neurons are recruited.
        neuron ID      -- the stable identity of that neuron in the notional
                          population of ``n``; sparse and permanent.

    ``area.winners`` holds *compact* indices.  ``Assembly.winners`` holds
    *neuron IDs*.  This function is the one-way door between them, and the
    reason it must be used for every snapshot: a compact index is only
    meaningful relative to how many neurons the area had materialised at that
    instant, so a compact-indexed snapshot silently decays into garbage as
    soon as the area recruits anyone else.  Neuron IDs stay comparable across
    timepoints and across cloned brains.

    Explicit areas materialise all ``n`` neurons up front, so compact index
    and neuron ID coincide and no remapping is applied.

    Invalid compact indices raise; passing them through would invent a neuron ID.
    """
    area = brain.areas[area_name]
    winners = area.winners
    if area.explicit:
        return Assembly(area_name, winners.copy())
    engine = brain._engine_for(area)
    mapping = engine.get_neuron_id_mapping(area_name) if hasattr(
        engine, "get_neuron_id_mapping",
    ) else None
    if mapping is not None and len(mapping) > 0:
        mapped = to_neuron_ids(winners, mapping)
        # THE one-way door between the index spaces: compact -> neuron IDs.
        return Assembly(area_name, NeuronIds(mapped))
    # No mapping table means an EXPLICIT area, where the index already IS the
    # neuron id -- so this is a relabel, not a conversion.
    return Assembly(area_name, NeuronIds(winners.copy()))


def _compact_index(engine, area_name: str):
    """Real-neuron-id -> compact-index dict for *area_name*, or None.

    Returns None when the engine has no mapping, or the mapping is empty (in
    which case real IDs are used verbatim). This inversion --
    ``{int(nid): i for i, nid in enumerate(mapping)}`` -- was hand-rolled at
    several call sites and has twice been a bug source: comparing real IDs
    against a compact-indexed weight matrix (a false test failure), and
    injecting a snapshot whose IDs no longer map after consolidation. Centralise
    it so the real/compact boundary lives in exactly one place.
    """
    if not hasattr(engine, "get_neuron_id_mapping"):
        return None
    mapping = engine.get_neuron_id_mapping(area_name)
    if not mapping:
        return None

    # CACHED, because this is O(area size) and ran per PROJECTION: every
    # `_cue_state` inverts the whole table to activate one k-neuron assembly.
    # Measured on the Z60 organ (n_state=4200, arc n=20000) it was 0.81s of an
    # 11.7s GPU build and 3.9s of a 93.9s numpy build -- pure Python dict
    # construction, on both engines, for a table that rarely changes.
    #
    # WHY THE KEY IS SOUND, which is the whole difficulty. The mapping is NOT
    # append-only: `materialize_area` and the explicit-dense bootstrap REPLACE
    # the list wholesale (`compact_to_neuron_id = list(neuron_ids)`), and a
    # stale inverse here is exactly the compact-vs-neuron-id confusion that has
    # cost this project three results. So the entry keeps a STRONG REFERENCE to
    # the list it was built from and validates with `is`. Holding the reference
    # is what makes identity sound: the object cannot be freed while cached, so
    # a new list cannot be allocated at the same address and compare equal.
    # Length is checked too, which catches the append/extend growth paths
    # (recruitment, `materialize_area`'s fill). No site mutates an element in
    # place without changing the length -- the grep is `compact_to_neuron_id`
    # in core/, and every writer appends, extends, or rebinds.
    cache = getattr(engine, "_compact_index_cache", None)
    if cache is None:
        cache = {}
        try:
            engine._compact_index_cache = cache
        except AttributeError:          # __slots__ engine: stay uncached
            return {int(nid): i for i, nid in enumerate(mapping)}
    hit = cache.get(area_name)
    if hit is not None:
        cached_mapping, cached_len, inverse = hit
        if cached_mapping is mapping:
            if cached_len == len(mapping):
                return inverse
            if cached_len < len(mapping):
                # Same list, longer: growth is append-only, so the existing
                # entries are still correct and only the tail is new.
                for i in range(cached_len, len(mapping)):
                    inverse[int(mapping[i])] = i
                cache[area_name] = (mapping, len(mapping), inverse)
                return inverse
    inverse = {int(nid): i for i, nid in enumerate(mapping)}
    cache[area_name] = (mapping, len(mapping), inverse)
    return inverse


def activate_assembly(brain, assembly: Assembly) -> None:
    """Inject a lexicon assembly snapshot into an area's active winners.

    The inverse of :func:`_snap`: maps stable neuron IDs back to the compact
    engine indices the engine actually computes with.  Used during
    lexicon-grounded context accumulation to skip the phon->core projection
    when core lexicons are already stabilised -- replaying the stored winners
    directly is both faster and exactly reproducible.

    FAILURE MODE.  The ``ValueError`` below ("Assembly neuron N not in area
    mapping") fires when the snapshot names a neuron the target brain has
    never materialised.  It means the assembly and the brain have diverged:
    the snapshot came from a *different* brain (or from before a reset), so
    that neuron ID has no compact slot here.  It is a genuine consistency
    check, not a lookup that should be made tolerant -- silently dropping the
    missing neurons would inject a truncated, subtly wrong assembly.
    """
    area_name = assembly.area
    if area_name not in brain.areas:
        raise ValueError(f"Unknown area {area_name!r}")

    neuron_ids = np.asarray(assembly.winners, dtype=np.uint32)
    area = brain.areas[area_name]
    neuron_ids = validated_indices(neuron_ids, upper=area.n, label='assembly neuron IDs')
    if area.explicit:
        area.winners = neuron_ids.copy()
        brain._engine_for(area).set_winners(area_name, neuron_ids)
        return
    engine = brain._engine_for(area)
    neuron_to_compact = _compact_index(engine, area_name)
    if neuron_to_compact is not None:
        try:
            compact = [neuron_to_compact[int(n)] for n in neuron_ids]
        except KeyError as exc:
            raise ValueError(
                f"Assembly neuron {exc.args[0]!r} not in area {area_name!r} "
                f"mapping (len={len(neuron_to_compact)})",
            ) from exc
        winners_arr = np.array(compact, dtype=np.uint32)
    else:
        winners_arr = neuron_ids.copy()

    area.winners = winners_arr
    engine.set_winners(area_name, winners_arr)


def assembly_is_current(brain, assembly) -> bool:
    """Whether ``assembly`` can still be injected into its area.

    True iff every neuron the snapshot names has a compact slot in the area's
    CURRENT mapping -- i.e. ``activate_assembly`` would succeed. Returns False
    instead of raising, so callers can decide whether to re-project rather than
    crash.

    The mapping can move out from under a snapshot: consolidation
    (``consolidation.prepare_area_for_replay``) deliberately resets
    ``compact_to_neuron_id`` and re-issues neuron IDs, orphaning every
    pre-consolidation snapshot for that area. This is the cheap guard that lets
    a training pass detect that and fall back.
    """
    area_name = assembly.area
    if area_name not in brain.areas:
        return False
    area = brain.areas[area_name]
    if area.explicit:
        return True
    engine = brain._engine_for(area)
    neuron_to_compact = _compact_index(engine, area_name)
    if neuron_to_compact is None:
        # No mapping yet: activate_assembly injects real IDs verbatim, so
        # there is nothing to be stale against.
        return True
    return all(int(n) in neuron_to_compact for n in assembly.winners)


def learn_assembly_from_pattern(
    brain,
    src_area: str,
    pattern: np.ndarray,
    dst_area: str,
    *,
    max_epochs: int = 20,
    project_rounds: int = 8,
    tau: float = 0.90,
    stability_window: int = 2,
    external_drive: dict | None = None,
    recurrent: bool = True,
) -> tuple[Assembly, int, float]:
    """Converge a dst assembly from a fixed src pattern (E8).

    Wraps winner injection + ``project`` until consecutive snapshots
    overlap ≥ *tau*.  Used for explicit LOW→HIGH MNIST encoding.

    This is :func:`project` with a CONVERGENCE CRITERION instead of a fixed
    budget, and it exists because ``rounds`` is only a heuristic: the O(log n)
    stabilisation result is probabilistic, so a caller that needs a formed
    assembly rather than "ten rounds' worth of settling" has to check.  Each
    epoch resets the destination and re-drives it from the same source
    pattern; when consecutive epochs agree to within *tau* the assembly is
    reproducible from that input, which is the operational meaning of
    "formed" here.

    Note this is a WEAKER criterion than the E%-WTA formation conditions in
    ``epwta.py``, which additionally require that no neuron fires for the
    first time and that the assembly's internal synaptic density exceeds the
    host area's.  Consecutive-snapshot agreement alone can be satisfied by a
    set that is still churning between rounds within an epoch.

    Returns ``(assembly, epochs_used, persistence)``; ``epochs_used ==
    max_epochs`` means it did NOT converge, and the returned persistence is
    the last observed agreement rather than a success value.  Check it.
    """
    if src_area not in brain.areas:
        raise KeyError(f"learn_assembly_from_pattern source area is unknown: {src_area!r}")
    if dst_area not in brain.areas:
        raise KeyError(f"learn_assembly_from_pattern target area is unknown: {dst_area!r}")
    if isinstance(pattern, (str, bytes)):
        raise TypeError("pattern must be a one-dimensional numeric array")
    pattern = np.asarray(pattern)
    if pattern.ndim != 1 or pattern.shape[0] != brain.areas[src_area].n:
        raise ValueError(
            f"pattern must have shape ({brain.areas[src_area].n},), got {pattern.shape}"
        )
    if not np.issubdtype(pattern.dtype, np.number) or not np.all(np.isfinite(pattern)):
        raise ValueError("pattern must contain finite numeric values")
    if not np.any(pattern > 0):
        raise ValueError("pattern must activate at least one source neuron")
    for label, value in (("max_epochs", max_epochs), ("project_rounds", project_rounds),
                         ("stability_window", stability_window)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if stability_window < 2:
        raise ValueError("stability_window must be at least two")
    if (isinstance(tau, bool) or not isinstance(tau, Real)
            or not np.isfinite(float(tau)) or not 0.0 <= float(tau) <= 1.0):
        raise ValueError("tau must be a finite real number in [0, 1]")
    if type(recurrent) is not bool:
        raise ValueError("recurrent must be an explicit boolean")

    drive = external_drive or {}
    history: list[Assembly] = []
    src_winners = np.flatnonzero(pattern > 0).astype(np.uint32)
    for epoch in range(1, max_epochs + 1):
        brain.areas[dst_area].unfix_assembly()
        brain.areas[dst_area].winners = np.array([], dtype=np.uint32)
        brain._engine_for(brain.areas[dst_area]).set_winners(
            dst_area, np.array([], dtype=np.uint32),
        )
        brain.areas[src_area].unfix_assembly()
        brain.areas[src_area].winners = src_winners
        brain._engine_for(brain.areas[src_area]).set_winners(src_area, src_winners)
        projections = {src_area: [dst_area]}
        if recurrent:
            projections[dst_area] = [dst_area]
        for _ in range(project_rounds):
            brain.project(
                external_inputs={src_area: src_winners},
                projections=projections,
                external_drive=drive,
            )
        snap = _snap(brain, dst_area)
        history.append(snap)
        if len(history) >= stability_window:
            pairs = [
                overlap(history[i], history[i + 1])
                for i in range(len(history) - stability_window, len(history) - 1)
            ]
            if all(p >= tau for p in pairs):
                return snap, epoch, min(pairs)
    final_pers = (
        overlap(history[-2], history[-1]) if len(history) > 1 else 0.0
    )
    return history[-1], max_epochs, final_pers


def _fix(brain, *area_names):
    """Fix assemblies in the given areas (prevent winner changes).

    Fixing is how a SOURCE is held steady while a target settles.  Without it
    the source drifts under its own recurrence during the rounds it is
    supposed to be driving, so the synapses potentiated in early rounds belong
    to a pattern that is no longer firing by the last one -- the binding is
    written against a moving target and does not survive.

    CRITICAL ASYMMETRY, and the reason every operation in this file fixes
    sources but never targets: the sparse engine SHORT-CIRCUITS a projection
    into a fixed area.  It returns before inputs are summed and before
    plasticity is applied, so a projection into a fixed target writes nothing
    at all -- silently, with no error and no winner change to notice.  Fixing
    a target therefore does not "hold the answer in place while it learns"; it
    disables the learning entirely.  ``assembly_calculus.binding.bind``
    documents a measured instance of this (weight sum unchanged, 315 -> 315).

    This low-level helper does not scope its mutation. Reciprocal projection,
    association and merge use ``_fixed_sources`` to restore the caller's
    facade and engine clamp flags, including after exceptions.
    """
    for name in area_names:
        brain.areas[name].fix_assembly()


def _unfix(brain, *area_names):
    """Unfix assemblies in the given areas."""
    for name in area_names:
        brain.areas[name].unfix_assembly()


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

@contextmanager
def _fixed_sources(brain, *names):
    """Borrow source clamps; restore facade and engine state even on error."""
    saved = [(name, brain.areas[name].fixed_assembly,
              brain._engine_for(brain.areas[name]).is_fixed(name)) for name in names]
    try:
        _fix(brain, *names)
        yield
    finally:
        for name, fixed, engine_fixed in saved:
            area = brain.areas[name]
            area.fixed_assembly = fixed
            engine = brain._engine_for(area)
            (engine.fix_assembly if engine_fixed else engine.unfix_assembly)(name)


@implements(PROJECTION_CONTRACT)
def project(brain, stimulus, target, rounds=10, recurrent=False) -> Assembly:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-projection

    Execute a stimulus schedule and return the final neuron-ID Assembly.

    ``stimulus`` and ``target`` must already exist; ``rounds`` is a positive
    integer. The first round is stimulus-only. ``recurrent=True`` explicitly
    adds target recurrence on later rounds. ``recurrent=False`` supplies no
    self-edge, regardless of Brain's legacy fast-path recurrence settings.
    The compatibility default is False; it trains no recurrent structure.

    Plasticity follows the brain/backend settings. This operation evaluates
    neither stability nor partial-cue recovery. Use a registered training
    schedule and a matched negative control before claiming assembly formation.
    """
    ProjectionPlan(stimulus, target, rounds, recurrent).execute(brain)
    return _snap(brain, target)


#: Recurrent rounds in :func:`bind`. Round 1 is always feed-forward, so the
#: total is ``1 + BIND_TAIL_ROUNDS``. Kept at 1 to match
#: ``parser_mixins._shared._ROLE_BINDING_ROUNDS = 2``, which faced the same
#: shared-area problem and was swept: on the real path the value barely matters
#: and is not monotonic, while deep reinforcement demonstrably merges items in a
#: shared area (94 constituents into n=1000: pairwise overlap 0.752 at 10
#: rounds, 0.078 at 3, 0.051 at 1).
BIND_TAIL_ROUNDS = 1


def bind(brain, source_area, target_area, source_assembly=None, *,
         source_stimulus=None, project_rounds=10,
         tail_rounds=BIND_TAIL_ROUNDS, fix_source=True) -> Assembly:
    """Bind the content of *source_area* into a SHARED *target_area*.

    THE ONE IMPLEMENTATION OF THIS PROTOCOL. It existed as three hand-rolled
    copies -- ``assembly_calculus.parser.train_roles``,
    ``emergent.parser_mixins.roles.train_roles`` and
    ``emergent.parser_mixins.generation`` -- which drifted into three DIFFERENT
    states, only one of them correct. That drift is what produced the bug: the
    emergent copy had been fixed, the other two had not, and nothing connected
    them. Duplicated protocol logic rots silently, one copy at a time, so this
    function is the fix for the class rather than for the instance.

    Three properties, each of which one of the copies got wrong:

    1. **NO** ``reset_area_connections`` on the target. Zeroing the connectome
       leaves every candidate neuron at equal input, so the deterministic index
       tie-break in winner selection returns the SAME k winners for every
       source -- all stored assemblies become bit-identical, retrieval reads
       exactly chance with a unit margin, and no value of beta can separate
       them. Measured on the demo parser: the reset fired 138 times over 40
       sentences and all 138 zeroed a pathway that was carrying weight,
       leaving one live area->area pathway in the entire brain.

    2. **Replay** a stored *source_assembly* rather than re-projecting a
       stimulus into the source. ``project(phon, core)`` carries plasticity, so
       the source assembly drifts between storing a binding and reading it
       back, and retrieval then misses the target it was trained on.
       ``activate_assembly`` is exactly reproducible.

    3. **Round 1 feed-forward**, recurrence only in a short tail. Round 1
       input-driven makes the bound assembly a function of the source;
       self-recurrence in a shared area is this project's documented collapse
       channel, because the first item's self-connections potentiate until they
       beat every later item's input.

    SYMMETRY IS THE POINT. Training and readout must call THIS function, the
    readout inside ``brain.read_only()``. Two call sites that merely agree today
    are the situation that produced the bug; one function cannot disagree with
    itself. A readout running different dynamics from the drive that built the
    assembly reads at chance no matter how healthy the representation is.

    Args:
        source_area: area holding the filler.
        target_area: shared area the binding is stored in.
        source_assembly: snapshot to replay into *source_area*. If None, the
            caller has already established the source's activity.
        tail_rounds: recurrent rounds after the feed-forward one.
        fix_source: hold the source steady during the projection. Leave True
            unless the source is a stimulus-driven area you want to keep live.

    Returns the bound assembly as an immutable snapshot (NEURON IDs).
    """
    for label, area_name in (("source_area", source_area),
                             ("target_area", target_area)):
        if area_name not in brain.areas:
            raise KeyError(f"bind {label} is unknown: {area_name!r}")
    if (isinstance(project_rounds, bool)
            or not isinstance(project_rounds, Integral)
            or project_rounds < 1):
        raise ValueError("bind project_rounds must be a positive integer")
    if (isinstance(tail_rounds, bool)
            or not isinstance(tail_rounds, Integral)
            or tail_rounds < 0):
        raise ValueError("bind tail_rounds must be a nonnegative integer")
    if type(fix_source) is not bool:
        raise ValueError("bind fix_source must be an explicit boolean")
    project_rounds = int(project_rounds)
    tail_rounds = int(tail_rounds)
    # STALE SNAPSHOTS ARE EXPECTED, not exceptional, so the guard belongs here
    # rather than in each caller. `consolidation.prepare_area_for_replay`
    # deliberately resets an area's compact_to_neuron_id and re-issues neuron
    # IDs, which orphans every snapshot taken before it -- its own docstring
    # says "do not carry pre-consolidation lexicons across". The novel-chat
    # curriculum really does consolidate between `train_lexicon` (which fills
    # the lexicons) and the role pass (which replays them), so an unguarded
    # `activate_assembly` raises there.
    #
    # `training/batch.py` was the only copy of this protocol that had the guard,
    # which made it the STRONGEST implementation -- so unifying had to absorb it
    # upward. Routing that caller through an unguarded `bind` would have turned
    # a cleanup into a crash on any consolidated parser.
    replayed = False
    if source_assembly is not None and assembly_is_current(brain,
                                                           source_assembly):
        activate_assembly(brain, source_assembly)
        replayed = True
    if not replayed and source_stimulus is not None:
        # Same fallback every caller had hand-rolled: no usable snapshot, so
        # drive the source from its stimulus and accept the drift this
        # function's docstring warns about.
        project(brain, source_stimulus, source_area, rounds=project_rounds)
    if not replayed and source_stimulus is None:
        # A missing snapshot and stimulus is only valid when the caller has
        # explicitly established live source activity.  Proceeding with an
        # empty/stale source makes bind look successful while training a
        # target from no evidence at all.
        live_winners = getattr(brain.areas[source_area], "winners", None)
        if live_winners is None or len(live_winners) == 0:
            raise ValueError(
                "bind requires source_assembly, source_stimulus, or a "
                f"nonempty live assembly in source area {source_area!r}"
            )
    if fix_source:
        brain.areas[source_area].fix_assembly()
    try:
        brain.project({}, {source_area: [target_area]})
        for _ in range(max(0, tail_rounds)):
            brain.project({}, {source_area: [target_area],
                               target_area: [target_area]})
        return _snap(brain, target_area)
    finally:
        if fix_source:
            brain.areas[source_area].unfix_assembly()


def read_binding(brain, source_area, target_area, source_assembly=None, *,
                 tail_rounds=BIND_TAIL_ROUNDS) -> Assembly:
    """Re-drive a binding for READOUT, with plasticity and recruitment off.

    Identical dynamics to :func:`bind` by construction -- it calls it -- wrapped
    in ``brain.read_only()`` so the measurement cannot create the structure it
    is trying to detect. ``frozen()`` is NOT equivalent: it stops weights
    changing but not ``w``, and two probe orders that recruit differently are
    structurally different brains.

    Compare the result against stored snapshots with
    ``diagnostics.assembly_overlap``. Both sides are then neuron IDs; comparing
    a snapshot against ``area.winners`` (compact engine indices) reads exactly
    chance and has silently voided three results in this project.
    """
    with brain.read_only():
        return bind(brain, source_area, target_area, source_assembly,
                    tail_rounds=tail_rounds)


@implements(RECIPROCAL_PROJECTION_CONTRACT)
def reciprocal_project(brain, source, target, rounds=10, *,
                       fix_source=True) -> Assembly:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-reciprocal-projection

    Project ``source`` into ``target`` with a return edge on later rounds.

    The first round is source -> target. Subsequent rounds also project target
    to itself and source. ``source`` must have an established assembly.
    With ``fix_source=True`` its winners are held steady for the schedule;
    its original clamp flags are restored, including on exception.

    Return a final neuron-ID Assembly in target. Return-edge learning depends
    on the backend's fixed-target plasticity policy; the edge's presence does
    not certify source recovery. Measure that separately with free winners.
    """
    plan = ReciprocalProjectionPlan(source, target, rounds, fix_source)
    plan.preflight(brain)
    with _fixed_sources(brain, *((source,) if plan.fix_source else ())):
        plan.execute_steps(brain)
        return _snap(brain, target)


def consolidate_pair(
    brain,
    area_a: str,
    assembly_a: Assembly,
    area_b: str,
    assembly_b: Assembly,
    *,
    rounds: int = 5,
    a_to_b: bool = True,
    b_to_a: bool = True,
) -> tuple[Assembly, Assembly]:
    """Sleep-replay pairing: stabilize bidirectional assemblies (E4).

    Alternates ``reciprocal_project`` in both directions so forward
    discriminative paths are not the only trained association.
    """
    if a_to_b:
        activate_assembly(brain, assembly_a)
        reciprocal_project(brain, area_a, area_b, rounds=rounds)
    if b_to_a:
        activate_assembly(brain, assembly_b)
        reciprocal_project(brain, area_b, area_a, rounds=rounds)
    return _snap(brain, area_a), _snap(brain, area_b)


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------

@implements(ASSOCIATION_CONTRACT)
def associate(brain, source_a, source_b, target,
              stim_a=None, stim_b=None, rounds=10, *,
              cofire_rounds=None) -> Assembly:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-association

    Train two pathways sequentially, then coactivate them into ``target``.

    Each source receives ``rounds`` pathway-training steps. Joint activation
    receives ``cofire_rounds`` steps, defaulting to rounds; zero is the
    no-coactivation control. Excessive joint training can collapse the two
    pathways into one representation, so the schedule is part of the claim.

    ``stim_a`` and ``stim_b`` drive their respective sources when both are
    provided. If both are absent, source clamps are scoped and restored. A
    partial stimulus pair is rejected before mutation because it names neither
    of those two protocols.

    Return the final joint neuron-ID Assembly. No before/after singly-cued
    comparison is performed; this function alone does not measure association.
    """
    plan = AssociationPlan(
        source_a, source_b, target, stim_a, stim_b, rounds, cofire_rounds,
    )
    plan.preflight(brain)
    fixed = (source_a, source_b) if plan.fix_sources else ()
    with _fixed_sources(brain, *fixed):
        plan.execute_steps(brain)

    return _snap(brain, target)


@implements(MERGE_CONTRACT)
def merge(brain, source_a, source_b, target,
          stim_a=None, stim_b=None, rounds=10, *,
          parent_self=True, target_self=True, back_project=True,
          unstimulated_source_mode=None) -> Assembly:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-merge

    Coactivate ``source_a`` and ``source_b`` into ``target`` from the first round.

    ``parent_self`` enables source recurrence from the first round.
    ``target_self`` and ``back_project`` enable target recurrence and return
    edges on subsequent rounds. All default True. Record all three switches:
    conjunction formation, forward retrieval and parent recovery are distinct
    claims with different controls, particularly when many items share an area.

    ``stim_a`` and ``stim_b`` drive their respective sources. When both are
    absent, both current sources are clamped temporarily. With exactly one
    stimulus, ``unstimulated_source_mode`` must explicitly be ``require-fixed``,
    ``fix-current`` or ``evolving``. Return-edge learning also depends on the
    backend's fixed-target plasticity policy. Borrowed clamps are restored.

    Return the final target neuron-ID Assembly, without a retrieval test.
    """
    plan = MergePlan(
        source_a, source_b, target, stim_a, stim_b, rounds,
        parent_self, target_self, back_project, unstimulated_source_mode,
    )
    plan.preflight(brain)
    with _fixed_sources(brain, *plan.fixed_sources):
        plan.execute_steps(brain)

    return _snap(brain, target)


@implements(COMPLETION_CONTRACT)
def pattern_complete(
    brain, area, fraction=0.5, rounds=5, seed=None, *, observation_mode=None,
):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-completion

    Return ``(recovered_assembly, overlap_with_entry_assembly)`` after a partial cue.

    Snapshot the current winners of ``area`` as the reference; retain
    floor(len(reference) * fraction) winners using ``random.Random(seed)``,
    then perform ``rounds`` recurrent projections. The cue is not clamped.
    Its initial overlap is therefore not a floor on the final score.

    ``seed`` and ``observation_mode`` are mandatory protocol inputs. ``plastic``
    retains historical live learning, ``frozen`` disables learning while retaining
    activity and recruitment, and ``read-only`` restores all supported observation
    state. Neither this schedule nor its score establishes learned recovery without
    a stated regime and matched negative control.
    """
    plan = CompletionPlan(area, fraction, rounds, seed, observation_mode)
    prepared = plan.prepare(brain)
    with plan.observation_scope(brain):
        prepared.inject_cue(brain)
        for step in plan.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())
        recovered = _snap(brain, area)
        recovery = overlap(recovered, prepared.reference)
    return recovered, recovery


@implements(SEPARATION_CONTRACT)
def separate(brain, stim_a, stim_b, target, rounds=10):
    """Project two different stimuli into the same area and measure overlap.

    Each stimulus is projected with its own recurrent stabilization.
    Between projections, the area→area recurrent connections are reset
    to prevent the first stimulus's attractor from dominating the second.
    The stim→area connections (independent per stimulus) are preserved.

    Args:
        brain: Brain instance with both stimuli and target already added.
        stim_a: Name of the first stimulus.
        stim_b: Name of the second stimulus.
        target: Name of the target area.
        rounds: Number of projection rounds per stimulus (default 10).

    Returns:
        (assembly_a, assembly_b, overlap) tuple.

    Theory:
        Two independent stimuli should produce assemblies with overlap
        near chance level (k/n), verifying the area has sufficient
        capacity for distinct representations.

    Note on the recurrent reset:
        Wiping area->area weights between the two stimuli is a MEASUREMENT
        DEVICE, not a claim about biology.  Without it, stimulus A's
        recurrent attractor is still in the connectome when B arrives, and B
        gets pulled partway into it -- so the measured overlap would conflate
        "these stimuli are similar" with "A ran first".  Resetting isolates
        the former.  It also means ``separate`` is destructive: the brain that
        comes back has no recurrent trace of stimulus A, so do not call it on
        a brain you intend to keep using.
    """
    # Validate the complete schedule before the first projection.
    plan = SeparationPlan(stim_a, stim_b, target, rounds)
    plan.preflight(brain)
    stim_a, stim_b, target, rounds = (
        plan.stim_a, plan.stim_b, plan.target, plan.rounds,
    )

    # Project stimulus A
    assembly_a = project(brain, stim_a, target, rounds=rounds)

    # Reset area→area connections for the target to remove recurrent
    # attractor from stimulus A.  This gives stimulus B a fresh recurrent
    # landscape while preserving stim→area connections.
    _reset_recurrent(brain, target)

    # Project stimulus B
    assembly_b = project(brain, stim_b, target, rounds=rounds)

    return assembly_a, assembly_b, overlap(assembly_a, assembly_b)


def learn_assembly(
    brain,
    stimulus,
    target,
    max_epochs: int = 12,
    project_rounds: int = 6,
    stability_window: int = 2,
    convergence: float = 0.90,
):
    """Online assembly learning for one stimulus (Dabagia et al. COLT 2022).

    Repeatedly projects ``stimulus → target`` until consecutive assemblies
    stabilize above ``convergence`` overlap.

    The stimulus-driven counterpart of :func:`learn_assembly_from_pattern`
    (which drives from an explicit source pattern), and the same caveat
    applies: ``epochs_used == max_epochs`` means it ran out of budget, not
    that it converged.  The returned persistence is the evidence; a caller
    that ignores it cannot distinguish a formed assembly from a failure.

    Note that epochs here do NOT reset the target.  Each epoch projects into
    an area still carrying the previous epoch's recurrent weights, so
    convergence is partly self-fulfilling -- the assembly is being deepened by
    the very measurement that checks whether it is stable.  It is still a
    meaningful test (a stimulus that cannot carve out a consistent assembly
    will not converge), but the epoch count is a lower bound on the difficulty,
    not an unbiased one.

    Returns:
        (assembly, epochs_used, final_persistence) tuple.
    """
    for label, value in (("max_epochs", max_epochs),
                         ("project_rounds", project_rounds),
                         ("stability_window", stability_window)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if stability_window < 2:
        raise ValueError("stability_window must be at least two")
    if (isinstance(convergence, bool) or not isinstance(convergence, Real)
            or not np.isfinite(float(convergence))
            or not 0.0 <= float(convergence) <= 1.0):
        raise ValueError("convergence must be a finite real number in [0, 1]")

    history = []
    for epoch in range(1, max_epochs + 1):
        asm = project(brain, stimulus, target, rounds=project_rounds)
        history.append(asm)
        if len(history) >= stability_window:
            pairs = [
                overlap(history[i], history[i + 1])
                for i in range(len(history) - stability_window, len(history) - 1)
            ]
            if all(o >= convergence for o in pairs):
                return asm, epoch, min(pairs)
    final_pers = (
        overlap(history[-2], history[-1]) if len(history) > 1 else 0.0
    )
    return history[-1], max_epochs, final_pers


def _reset_recurrent(brain, area_name):
    """Reset area→area connections involving an area to their initial state.

    Delegates to the engine's ``reset_area_connections`` method, which
    preserves stimulus→area connections while reverting area→area weights.

    THE RULE FOR WHEN THIS IS CORRECT, since the same call is load-bearing in
    one place and destructive in another and that cost this project weeks:

    **CORRECT** on an area whose next input is a per-item **STIMULUS**. Each
    item has its own stimulus, so the k-WTA tie-break is never reached, and
    without the reset the first item's recurrent attractor wins against every
    later item's drive. Measured on a lexicon, n=1000 k=50 beta=0.1:
    M=64 gives 64 distinct assemblies with the reset and 7 without
    (spread 0.0499 vs 0.9153). This is why ``train_lexicon`` and the core-area
    probes in ``classify``/``morphosyntax``/``incremental`` all reset.

    **DESTRUCTIVE** on a shared area whose input is another **AREA** — role
    areas, VP, any binding target. Zeroing that connectome leaves every
    candidate neuron at equal input, so the deterministic index tie-break
    returns the SAME k winners for every source: all stored assemblies become
    bit-identical, retrieval reads exactly chance with a unit margin, and no
    value of beta can separate them. Measured on the demo parser, the reset
    fired 138 times over 40 sentences and all 138 zeroed a pathway that was
    carrying weight, leaving one live area→area pathway in the whole brain.

    So: **stimulus-driven target, reset; area-driven target, never.** For the
    second case use :func:`bind`, which is the one implementation of that
    protocol. ``ASSEMBLIES_STRICT_DRIVE=1`` makes a violation say so at runtime
    instead of returning plausible winners.
    """
    brain.reset_area_connections(area_name)


# ---------------------------------------------------------------------------
# Sequence operations [SEQ25]
#
# NOTE the citation tags: this file refers to two different Dabagia et al.
# papers, and both used to read "Dabagia et al. 2024".
#   [SEQ25]  sequences -- Neural Computation 2025, ALT 2024, arXiv:2306.03812
#   [COIN24] the assembly DEFINITION quoted above -- arXiv:2406.07715 (2024)
# Tags resolve against research/literature/index.json; see validate_index.py.
# ---------------------------------------------------------------------------

from .sequence import Sequence


@implements(SEQUENCE_MEMORIZE_CONTRACT)
def sequence_memorize(brain, stimuli, target, rounds_per_step=10,
                      repetitions=1, phase_b_ratio=None,
                      beta_boost=None) -> Sequence:
    """Memorize an ordered sequence of stimuli in a target area.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-sequence-memory

    For each repetition, each stimulus is projected into the target area
    with recurrent stabilization.  Hebbian plasticity naturally links
    consecutive assemblies: when stimulus s_{i+1} arrives, the recurrent
    weights from the x_i assembly are still warm, creating an x_i -> x_{i+1}
    bridge that enables later ordered recall.

    Args:
        brain: Brain instance with stimuli and target area already added.
        stimuli: Ordered list of stimulus names (the sequence to memorize).
        target: Name of the target area.
        rounds_per_step: Projection rounds per stimulus (default 10).
        repetitions: Number of times to replay the full sequence for
            strengthening (default 1).
        phase_b_ratio: Fraction of rounds_per_step for Phase B (recurrence).
            If None, uses legacy default (2 rounds regardless of total).
            A value of 0.5 with rounds_per_step=10 gives 5:5 split.

            FOR ORDERED RECALL, PASS THIS **TOGETHER WITH** ``repetitions>=3``.
            The legacy default never builds the inter-assembly bridge at all
            (see the root-cause comment in the body), and this parameter is what
            opens the recurrent fiber across the transition -- but it reads as a
            dead parameter at the default ``repetitions=1``, which is how it was
            previously dismissed. 0.5, 0.8 and 1.0 all take recall from 1 of 3
            to 2 of 3 once repetitions is 3 or more.
        beta_boost: Temporary plasticity boost for recurrent connections
            during Phase B.  If None, uses the area's current beta.
            A value of 0.5 strengthens inter-assembly bridges.  Best measured
            result is ``phase_b_ratio=1.0, repetitions=3, beta_boost=0.5``
            (2.33 of 3); note it is NOT monotone in repetitions -- the same
            setting falls back to 1.00 by repetitions=8.

    Returns:
        Sequence of Assembly snapshots (one per stimulus, from last repetition).

    Reference:
        Dabagia, Papadimitriou, Vempala.
        "Computation with Sequences of Assemblies in a Model of the Brain."
        Neural Computation (2025).  arXiv:2306.03812.
    """
    if isinstance(stimuli, (str, bytes)):
        raise TypeError("stimuli must be an ordered collection of stimulus names")
    try:
        stimuli = list(stimuli)
    except TypeError as exc:
        raise TypeError(
            "stimuli must be an ordered collection of stimulus names"
        ) from exc
    plan = SequenceMemorizePlan(
        tuple(stimuli), target, rounds_per_step, repetitions,
        phase_b_ratio, beta_boost,
    )
    plan.preflight(brain)
    stimuli = list(plan.stimuli)
    target = plan.target
    rounds_per_step = plan.rounds_per_step
    repetitions = plan.repetitions
    phase_b_ratio = plan.phase_b_ratio
    beta_boost = plan.beta_boost

    assemblies = []

    for _rep in range(repetitions):
        # Only the LAST repetition's snapshots are returned; earlier passes
        # exist purely to deepen the Hebbian bridges, and their assemblies are
        # discarded because they are the same assemblies observed earlier in
        # training (the connectome, not the winner set, is what accumulates).
        assemblies = []
        for stim_name in stimuli:
            # Compute Phase A / Phase B split
            if phase_b_ratio is not None:
                recur_rounds = max(1, int(rounds_per_step * phase_b_ratio))
                stim_rounds = rounds_per_step - recur_rounds
            else:
                # Legacy default: 2 recurrence rounds regardless of total, so
                # the A:B ratio drifts with rounds_per_step (8:2 at 10, 3:2 at
                # 5).  Retained as the default because published numbers in
                # this repo were measured with it; pass phase_b_ratio for a
                # proportional split.
                stim_rounds = max(1, rounds_per_step - 2)
                recur_rounds = rounds_per_step - stim_rounds

            # Phase A: stimulus-only rounds to establish the new assembly.
            # This anchors the winners to the stimulus input so that the
            # recurrent attractor from the previous assembly doesn't
            # dominate.
            for _ in range(stim_rounds):
                brain.project({stim_name: [target]}, {})

            # Phase B: stimulus + recurrence rounds, intended to build the
            # inter-assembly Hebbian bridge (x_{i-1} -> x_i). Raising beta here
            # deepens those bridges without over-strengthening the
            # within-assembly recurrence Phase A already built.
            #
            # MEASURED, AND THE BRIDGE COMES OUT FAR TOO WEAK -- see #56. In the
            # A->A connectome after memorising L=3 at n=5000, k=80, T=8 (mean
            # weight of nonzero synapses, ambient = 1.0):
            #
            #     reps     within x_i->x_i     bridge x_i->x_i+1     ratio
            #        1              1.0636                1.0051     1.06x
            #        3              1.5242                1.0229     1.49x
            #       10              5.8003                1.1895     4.88x
            #       25             20.0000                1.6763    11.93x
            #
            # Within-assembly weights run to the w_max ceiling of 20 while the
            # bridge barely leaves ambient, so after LRI suppresses x_i the
            # x_i+1 neurons draw about (k*p) * 1.68 ~ 6.7 while the extreme
            # value over n never-fired candidates is ~10.8 -- fresh neurons win,
            # recall lands on noise, and ordered_recall stops after ONE step.
            #
            # ROOT CAUSE, isolated 2026-07-30. The bridge is not "too weak"; it
            # is STRUCTURALLY NEVER WRITTEN. Measured: the target->target
            # connectome is EMPTY after stimulus-only rounds, so Phase A never
            # touches it. The x_{i-1} -> x_i transition happens DURING Phase A --
            # the one moment when prev=x_{i-1} and new=x_i coincide is exactly
            # the moment the recurrent fiber is closed. Phase B opens it only
            # after the winners are already x_i, so every round it runs
            # potentiates the ATTRACTOR (x_i -> x_i). That is why within-assembly
            # weights reach w_max while the bridge stays at ambient.
            #
            # THE FIX NEEDS NO NEW CODE: `phase_b_ratio` already shrinks Phase A
            # (at 1.0, `stim_rounds` is 0 and every round carries the recurrent
            # fiber), so the transition is potentiated. It was previously
            # recorded as having no effect, but that was measured at the DEFAULT
            # `repetitions=1`, where nothing helps. Steps recalled of L=3, at
            # n=5000, k=80, p=0.05, beta=0.1, T=10, mean over 3 seeds:
            #
            #     phase_b_ratio   reps=1   reps=3   reps=5   reps=8
            #     legacy (None)     1.00     1.00     1.00     1.00
            #     0.5 / 0.8 / 1.0   1.00     2.00     2.00     2.00
            #     1.0 + boost 0.5   1.00     2.33     2.00     1.00
            #
            # So it is the INTERACTION that matters: pass `phase_b_ratio` AND
            # `repetitions >= 3`. Either alone reads 1.00 and looks like a dead
            # parameter.
            #
            # STILL OPEN: this reaches 2 of 3, not 3 of 3, and it is NOT
            # monotone -- raising repetitions further degrades it again (2.00 at
            # 5, 1.33 at 10, 1.00 at 20) while the bridge keeps growing. The
            # obvious explanation is ruled out: the assemblies do NOT merge.
            # Pairwise overlap stays at 0.000-0.004, below the 0.016 chance
            # level, at every repetition count up to 40. The remaining suspect is
            # the within/bridge RATIO (1.11 at reps=5 vs 1.76 at reps=40): the
            # attractor grows faster than the bridge, and recall has to escape
            # the attractor to advance.
            if beta_boost is not None:
                # NOTE: saves the AREA-WIDE default beta but restores it into
                # the target->target pathway specifically.  If a caller had
                # set a distinct target->target beta before calling, that
                # value is not what gets restored.  Left as-is: current
                # callers never do, and changing it would alter results.
                original_beta = brain.areas[target].beta
                brain.update_plasticity(target, target, beta_boost)
            try:
                for _ in range(recur_rounds):
                    brain.project({stim_name: [target]}, {target: [target]})
            finally:
                if beta_boost is not None:
                    # The boost is a scoped protocol setting. Restore it even
                    # when a backend raises, otherwise a failed experiment
                    # contaminates every later run on this Brain.
                    brain.update_plasticity(target, target, original_beta)

            assemblies.append(_snap(brain, target))

    return Sequence(area=target, assemblies=assemblies)


@implements(ORDERED_RECALL_CONTRACT)
def ordered_recall(brain, area, cue, max_steps=20,
                   known_assemblies=None, convergence_threshold=0.9,
                   rounds_per_step=1, *, novelty_threshold=0.3) -> Sequence:
    """Recall a memorized sequence from a cue using LRI.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-sequence-memory

    Activates the cue in the area, then repeatedly self-projects.
    Long-Range Inhibition (LRI) suppresses the current assembly so the
    next assembly in the memorized chain fires.  Recall stops when a
    cycle is detected, a novel (unrecognised) assembly appears, or
    *max_steps* is reached.

    Why LRI is not optional.  ``sequence_memorize`` builds two things at once:
    strong WITHIN-assembly recurrence (x_i holds itself up) and weaker
    BETWEEN-assembly bridges (x_i -> x_{i+1}).  The within-assembly weights
    always win a plain top-k competition, so an area left to self-project
    simply sits on x_i forever.  The refractory period removes the neurons
    that just fired from candidacy, which deletes the within-assembly term
    from the competition and lets the second-strongest signal -- the bridge to
    x_{i+1} -- take the area.  Recall is thus *driven by inhibition*, and a
    refractory period shorter than the sequence lets the chain loop back onto
    itself, which is what the cycle check below catches.

    Requires:
        The target area must have ``refractory_period > 0`` (LRI enabled).
        Without LRI, self-recurrence converges back to the current
        assembly (attractor dynamics) and the sequence cannot advance.

    Args:
        brain: Brain instance.
        area: Name of the area containing the memorized sequence.
        cue: Stimulus name (str) to activate as the starting cue.
        max_steps: Maximum recall steps (default 20).
        known_assemblies: Optional list of Assembly snapshots from
            ``sequence_memorize``.  If provided, recall stops when a
            novel assembly (low overlap with all known) is encountered.
        convergence_threshold: If the new assembly overlaps > this with
            any previously recalled assembly, it is considered a cycle
            and recall stops.
        rounds_per_step: Self-projection rounds per recall step (default 1).
        novelty_threshold: Minimum overlap with a known assembly required to
            continue when ``known_assemblies`` is provided (default 0.3).

    Returns:
        Sequence of Assembly snapshots in recall order.

    Raises:
        ValueError: If the area has ``refractory_period == 0``.
    """
    plan = OrderedRecallPlan(
        area, cue, max_steps, convergence_threshold,
        rounds_per_step, novelty_threshold,
    )
    plan.preflight(brain)
    area = plan.area
    cue = plan.cue
    max_steps = plan.max_steps
    convergence_threshold = plan.convergence_threshold
    rounds_per_step = plan.rounds_per_step
    novelty_threshold = plan.novelty_threshold
    if known_assemblies is not None:
        known_assemblies = tuple(known_assemblies)
        malformed = [item for item in known_assemblies if not isinstance(item, Assembly)]
        if malformed:
            raise TypeError("known_assemblies must contain Assembly snapshots")
        wrong_area = [item.area for item in known_assemblies if item.area != area]
        if wrong_area:
            raise ValueError(
                f"known_assemblies must belong to recall area {area!r}; "
                f"found {wrong_area!r}"
            )

    # Clear refractory history from any previous operations
    brain.clear_refractory(area)

    # Activate cue
    brain.project({cue: [area]}, {})

    recalled = [_snap(brain, area)]

    for _step in range(max_steps):
        # Self-project with LRI active
        for _ in range(rounds_per_step):
            brain.project({}, {area: [area]})

        current = _snap(brain, area)

        # Check for cycle
        is_cycle = any(
            overlap(current, prev) > convergence_threshold
            for prev in recalled
        )
        if is_cycle:
            break

        # Check for novel (unrecognised) assembly: the chain has run off the
        # end of what was memorised and the area is now settling on noise.
        # novelty_threshold is a deliberately separate "clearly not one of
        # ours" floor: far above chance overlap k/n but below the ~0.8+
        # genuine recall steps produce. Keeping it explicit makes this
        # termination rule reproducible and lets a registered protocol tune
        # it without editing the operator.
        if known_assemblies is not None and len(known_assemblies) > 0:
            max_known_overlap = max(
                overlap(current, k) for k in known_assemblies
            )
            if max_known_overlap < novelty_threshold:
                break

        recalled.append(current)

    return Sequence(area=area, assemblies=recalled)
