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
* **Operations mutate the brain.**  Plasticity is applied on every call, so
  the same operation run twice on the same brain does not give the same
  result -- the second run starts from a connectome the first run reshaped.
  The returned :class:`Assembly` is an immutable snapshot precisely because
  the live ``area.winners`` will be overwritten by the next operation.

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

import random
from contextlib import contextmanager

import numpy as np

from .assembly import Assembly, overlap
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


def project(brain, stimulus, target, rounds=10, recurrent=False) -> Assembly:
    """Run a stimulus projection schedule and return its final winner snapshot.

    Completion of the schedule does not certify stability or attractor formation.

    ``recurrent`` OPTS IN TO THE PROTOCOL AS DOCUMENTED BELOW, and defaults
    False because the default path does NOT implement it. ``Brain.project_rounds``
    filters the projection map with ``a != target`` unless
    ``Brain(recurrent_projection=True)``, which itself defaults False -- so by
    default this function runs stimulus-only on every round, with no
    ``target -> target`` recurrence at all. What that builds is not an assembly
    in the defining sense: [COIN24] §2 defines an assembly as "sets of k neurons
    in a single brain area ... when the internal synaptic weights of the set have
    been suf[f]iciently strengthened", and stimulus-only projection strengthens
    no internal weight at all.

    The default is kept WRONG on purpose. The repository -- goldens, the parser,
    and ~40 tests -- is calibrated on the stimulus-only path; flipping it is a
    migration with its own re-baseline, not a bug fix. Measured when it was
    flipped: ~40 test failures and the "not slow" suite hanging at 54%, because
    below the stability threshold recurrence recruits without bound, ``w`` grows
    every step, and projection cost scales with ``w`` -- callers do not fail,
    they crawl.

    PASS ``recurrent=True`` FOR NEW WORK, and mind the training window, which
    has two walls pointing opposite ways
    (``research/experiments/recurrent_assembly_decay.py``):

    * a lone assembly re-selecting itself needs ``(1+beta)^rounds`` ABOVE the
      population maximum -- roughly 6 at n=1e4, and the threshold RISES with n
      (overlap 0.630 / 0.150 / 0.007 at n=2000 / 1e4 / 5e4 for a fixed 2.6);
    * many assemblies sharing one target need the incumbent BELOW the level
      where it dominates the k-cap -- past about 3, the first assembly stored
      wins every later merge.

    Use ``research/experiments/_substrate.py`` rather than calling this
    directly: it wires the opt-in, the correct readout, and the window together.

    This branch is bit-identical to ``project_rounds`` run with
    ``Brain(recurrent_projection=True)``; it is the same fast-path recurrent
    mode, reached without changing a global default.

    Protocol::

        1. stimulus → target                          (initial activation)
        2. (stimulus → target) + (target → target)    × (rounds - 1)

    Args:
        brain: Brain instance with stimulus and target already added.
        stimulus: Name of the stimulus.
        target: Name of the target area.
        rounds: Number of projection rounds (default 10).

    Returns:
        Assembly snapshot of the stabilized assembly in target.

    Theory (Papadimitriou 2020, §2):
        After O(log n) rounds, the assembly stabilizes: the set of
        winners converges to a fixed set with overlap > 0.95 between
        consecutive rounds.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    brain.project({stimulus: [target]}, {})
    if recurrent:
        # Driven directly rather than through project_rounds, whose `a != target`
        # filter would strip the recurrence this argument exists to request.
        for _ in range(rounds - 1):
            brain.project({stimulus: [target]}, {target: [target]})
    elif rounds > 1:
        brain.project_rounds(
            target=target,
            areas_by_stim={stimulus: [target]},
            dst_areas_by_src_area={target: [target]},
            rounds=rounds - 1,
        )
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


def reciprocal_project(brain, source, target, rounds=10, *,
                       fix_source=True) -> Assembly:
    """Project source into target, and train the RETURN path while doing it.

    Protocol (the reference's [ACREF], ``simulations.fixed_assembly_recip_proj``)::

        source is held fixed
        1. source → target                                        (feed-forward)
        2. (source → target) + (target → [target, source])         × (rounds - 1)

    THE ``target → source`` EDGE IS THE WHOLE POINT, and it was missing. Without
    it this function was a plain one-way ``project`` with target recurrence, so
    nothing ever wrote the target→source synapses -- while its own docstring
    promised "the source assembly can be recovered by projecting back". It could
    not be: there was nothing there to recover it with.

    Holding the source fixed is what makes the return path meaningful rather
    than merely present. The back-projection is written against a STATIONARY
    source assembly, so those synapses encode the pattern that is to be
    restored; if the source drifts under its own recurrence meanwhile, each
    round potentiates toward a different target and the sum restores nothing.
    (This depends on projections into a fixed area still applying plasticity,
    which is the reference's behaviour -- see the sparse engine's
    ``_fixed_target_plasticity_enabled``. It did not hold here until it was
    fixed alongside this function, which is why adding the edge alone was not
    enough.)

    MEASURED against [ACREF]. Restoration overlap after projecting back, at
    ``tests/test_assembly_calculus.py``'s parameters (n=1e4, k=100, p=0.05,
    beta=0.1, rounds=10) -- reference implementation 0.75, this function 0.00
    before the fix. At the reference's own defaults (n=1e5, k=317, p=0.01,
    beta=0.05) the reference restores 0.246 on the first back-projection rising
    to 0.344, matching the expectation written into that function's header
    comment. Restoration is PARTIAL by nature: perfect restoration is the
    signature of a dead fiber, not of a working one, and three of this repo's
    conformance tests were green for exactly that reason.

    Args:
        brain: Brain instance.
        source: Name of the source area (must have an established assembly).
        target: Name of the target area.
        rounds: Number of projection rounds (default 10).
        fix_source: hold the source steady for the duration, so the return path
            is written against one pattern. Leave True unless the source is
            stimulus-driven and you are keeping it live yourself.

    Returns:
        Assembly snapshot of the new assembly in target.
    """
    with _fixed_sources(brain, *((source,) if fix_source else ())):
        brain.project({}, {source: [target]})
        for _ in range(max(0, rounds - 1)):
            # Not `project_rounds`: that helper stabilises a SINGLE named
            # target, and this step has two destinations -- the target and,
            # via the return edge, the source.
            brain.project({}, {source: [target], target: [target, source]})
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

def associate(brain, source_a, source_b, target,
              stim_a=None, stim_b=None, rounds=10, *,
              cofire_rounds=None) -> Assembly:
    """Associate two source assemblies through a shared target area.

    Protocol::

        Phase 1: source_a → target         (rounds steps, with recurrence)
        Phase 2: source_b → target         (rounds steps, with recurrence)
        Phase 3: both sources → target     (rounds steps, simultaneous)

    Phase 3 is what does the associating.  Phases 1 and 2 only carve out the
    two pathways; it is the co-firing in Phase 3 -- both sources driving the
    same winners at the same time -- that potentiates a→target and b→target
    synapses onto a SHARED winner set, pulling the two target assemblies
    toward each other.  Note that this is a partial pull, not a merge: the
    result is elevated overlap between the two, not identity.

    If ``stim_a`` / ``stim_b`` are provided, stimuli remain active during
    their respective phases to maintain source assemblies. Otherwise, source
    assemblies are fixed before projection to prevent drift.

    Args:
        brain: Brain instance.
        source_a: Name of the first source area.
        source_b: Name of the second source area.
        target: Name of the target area.
        stim_a: Optional stimulus name that drives source_a.
        stim_b: Optional stimulus name that drives source_b.
        rounds: Number of rounds per phase (default 10).
        cofire_rounds: Number of CO-ACTIVATIONS in phase 3, defaulting to
            ``rounds``. This is [PNAS20]'s own independent variable -- it says
            post-association overlap "increases with the extent of cooccurrence
            (the number of consecutive simultaneous activations of the two
            parents)" -- and it could not be varied before, because ``rounds``
            moved the pathway training and the co-firing together. ``0`` gives
            the natural control: both pathways trained, nothing associated.

    Returns:
        Assembly snapshot of the associated assembly in target.

    Theory [PNAS20] §3:
        After association, activating source_a alone and projecting to target
        produces an assembly that significantly overlaps with the one source_b
        alone produces -- "an overlap between associated assemblies in the MTL of
        about 8 to 10% of the size of an assembly".

        ASSOCIATION IS PARTIAL BY DEFINITION, and there is a budget past which
        this operation stops being association at all. [PNAS20] treats associate
        and merge as different operations: association leaves a partial overlap,
        merge yields one assembly with both parents. Measured at n=1e4, k=50,
        p=0.05, beta=0.1, as overlap between the two singly-cued readouts
        (chance 0.005, 12 seeds; see tests/test_ac_conformance.py):

            cofire_rounds      0   0.0167     5   0.0200
                               1   0.0167    10   0.1217   <- the paper's band
                               2   0.0167    20   0.9317   <- a merge
                               3   0.0183

        and against merge at the same parameters: rounds=10 gives associate
        0.1425 vs merge 1.0000 (separated), rounds=20 gives 0.9850 vs 1.0000
        (indistinguishable). So the default of 10 sits in the regime the paper
        describes, and raising it does not "associate harder" -- it merges.

        Consequence worth knowing: research/literature/parity records its goldens
        at rounds=20, so associate_cue_overlap is pinned at 0.9875 against a
        merge_cue_overlap of 1.0. That value is not association.
    """
    use_fix = (stim_a is None and stim_b is None)
    with _fixed_sources(brain, *((source_a, source_b) if use_fix else ())):
        _associate_body(
            brain, source_a, source_b, target, stim_a, stim_b, rounds, use_fix,
            cofire_rounds,
        )

    return _snap(brain, target)


def _associate_body(brain, source_a, source_b, target,
                    stim_a, stim_b, rounds, use_fix, cofire_rounds=None):
    """Projection phases for :func:`associate`; see it for the contract.

    NO ``project_rounds`` FAST PATH HERE, and that is the fix for association
    rather than an optimisation given up. Its fast path filters
    ``dst_areas_by_src_area`` with ``a != target`` unless
    ``Brain(recurrent_projection=True)``, so the ``target: [target]`` entry
    every phase below passes it was SILENTLY DROPPED -- the argument was
    accepted and discarded. An assembly is defined by its strengthened internal
    weights ([COIN24] §2), so removing target recurrence removes the thing being
    built, and the co-fired winners of phase 3 never consolidate.

    MEASURED at n=1e4, k=100, p=0.05, beta=0.1, rounds=10, over 5 seeds, as
    post-association overlap between the assembly cued by source_a alone and the
    one cued by source_b alone (the claim as PNAS20 states it, against a chance
    of 0.0100):

        fixed sources, via project_rounds     0.0100  ->  1.0x chance
        fixed sources, explicit projections   0.1840  -> 18.4x chance
        stimulus-driven sources (unchanged)   0.2160  -> 21.6x chance

    So ``associate`` did nothing at all whenever it was called WITHOUT stimuli,
    which is the default and what the conformance test uses. The stimulus path
    worked only because it had already been written as explicit projections and
    therefore kept its recurrence -- the two branches were not two spellings of
    one protocol, one of them was broken.

    The previous note here claimed the branches differed only by "the
    source_a -> source_a fiber" and that "winners are unaffected". Both were
    wrong: the winners differed completely, and the recurrence was the reason.

    The remaining asymmetry IS principled and is kept: a fixed source needs no
    self-recurrence because its winners cannot move, while a stimulus-driven
    source needs it to hold its assembly across rounds.
    """
    def _phase(stim_dict, src_dsts, rounds_):
        """One phase: feed-forward round, then rounds_-1 with target recurrence."""
        for i in range(max(1, rounds_)):
            dsts = dict(src_dsts)
            if i > 0:
                dsts[target] = [target]
            brain.project(stim_dict, dsts)

    # Phase 1: Establish source_a → target pathway
    stim_dict_a = {stim_a: [source_a]} if stim_a else {}
    a_dsts = ({source_a: [target]} if use_fix
              else {source_a: [source_a, target]})
    _phase(stim_dict_a, a_dsts, rounds)

    # Phase 2: Establish source_b → target pathway
    stim_dict_b = {stim_b: [source_b]} if stim_b else {}
    b_dsts = ({source_b: [target]} if use_fix
              else {source_b: [source_b, target]})
    _phase(stim_dict_b, b_dsts, rounds)

    # Phase 3: Both sources drive target simultaneously (this is the step
    # that creates the shared winners; see the docstring).
    stim_dict_both = {}
    if stim_a:
        stim_dict_both[stim_a] = [source_a]
    if stim_b:
        stim_dict_both[stim_b] = [source_b]
    both_dsts = ({source_a: [target], source_b: [target]} if use_fix
                 else {source_a: [source_a, target],
                       source_b: [source_b, target]})
    # Target recurrence from the FIRST round here: phases 1 and 2 have already
    # established the target assembly, so there is no feed-forward-only round
    # to carve out -- this phase is consolidating a blend of the two.
    cofire = rounds if cofire_rounds is None else max(0, cofire_rounds)
    for _ in range(cofire):
        brain.project(stim_dict_both, {**both_dsts, target: [target]})


def merge(brain, source_a, source_b, target,
          stim_a=None, stim_b=None, rounds=10, *,
          parent_self=True, target_self=True, back_project=True) -> Assembly:
    """Merge assemblies from two source areas into a target area.

    Protocol::

        1. (source_a → target) + (source_b → target)       (simultaneous)
        2. Same + (target → target) + (target → sources)    × (rounds - 1)

    Key difference from associate: merge projects both sources
    SIMULTANEOUSLY from step 1, and feeds the target back to sources
    to create a single conjunctive assembly.

    If ``stim_a`` / ``stim_b`` are provided, stimuli remain active to
    maintain source assemblies. Otherwise, sources are fixed.

    Args:
        brain: Brain instance.
        source_a: Name of the first source area.
        source_b: Name of the second source area.
        target: Name of the target area.
        stim_a: Optional stimulus name that drives source_a.
        stim_b: Optional stimulus name that drives source_b.
        rounds: Number of projection rounds (default 10).
        parent_self: Keep ``source -> source`` (default True).
        target_self: Keep ``target -> target`` (default True).
        back_project: Keep ``target -> sources`` (default True).

    Returns:
        Assembly snapshot of the merged assembly in target.

    THE THREE RECURRENT CHANNELS, and when to gate them:
        The defaults reproduce the protocol above exactly and are correct for
        what this operation ports -- ONE merge, of one pair, as in
        ``.reference/dmitropolsky-assemblies simulations.merge_sim``.

        They are wrong for MANY merges through shared areas, which is this
        project's extension and not the reference's. Each channel accumulates
        potentiation across merges until it beats the next item's stimulus in
        k-WTA, and all three collapse their areas at different points.
        Measured n=1000 k=50 beta=0.1, 16 merges into one shared target, at
        the merge round T where each first bites (mean pairwise overlap of the
        16 items, against a random-pair floor of 0.0500; recall is rank-1
        identity from cueing one parent):

            channel                gates           collapses      by T
            parent_self  source -> source          the PARENTS       5
            target_self  target -> target          the TARGET       10
            back_project target -> sources         the parents      20

        parent_self is by far the strongest: gating it alone takes parent
        overlap from 0.9431 to 0.0518 at T=5. back_project -- the channel this
        docstring spends the most words on, and the one [PNAS20]'s "two-way
        connectivity" refers to -- is the weakest, and gating it changed parent
        overlap only from 0.9431 to 0.8266. The 3.47x parent potentiation
        measured above is a genuine SINGLE-merge property; it does not survive
        contact with sixteen of them.

        With all three gated -- repeated stimulus-driven feed-forward
        projection -- recall is 1.0000 and fidelity 0.9069 at T=10, against
        0.9271 / 0.3825 for the defaults at their own best setting (T=2).
        The merge CRITERION still holds without the back-projection: the
        composed assembly is returned from EITHER parent alone, acc 1.0000
        cueing source_b. So the two-way connectivity is not what the criterion
        needs, at least for retrieval from a partial cue.

        Rule of thumb: gate all three when the target holds MANY composed items;
        keep the defaults when the areas hold one thing at a time. Same law as
        ``core/brain.py:project_rounds`` documents for the lexicon -- the
        potentiation that makes ONE assembly persist is what makes MANY
        assemblies merge. See ``research/experiments/merge_recurrence_channels``
        and ``merge_capacity_ladder``.

    Theory (Papadimitriou 2020, §3):
        The merged assembly in target responds to EITHER source alone.
        This differs from association where two separate pathways are
        created sequentially.

    Why the feedback ``target -> [source_a, source_b]`` matters:
        Without it the sources would be inert inputs and the target would
        just be a downstream readout.  With it, all three areas settle
        jointly: the target reshapes its own sources, and the fixed point is
        a mutually-consistent triple.  This is the structural difference from
        :func:`associate`, which never projects back, and it is what makes
        merge the calculus's binding operation (the parser uses it to attach
        a modifier to a head).
    """
    # DO NOT use _fix here to hold the parents steady, even though that is what
    # every other operation in this file does. The engine SHORT-CIRCUITS a
    # projection into a fixed area, returning before plasticity is applied (see
    # _fix). Merge is the one operation that projects BACK into its sources --
    # `target: [..., source_a, source_b]` below -- so fixing them silently
    # discards exactly the two-way connectivity merge exists to create.
    # Measured: weight potentiation between the merged assembly and its parents
    # is 0.000 with fixed parents and 3.19x when they are driven.
    #
    # The reference (.reference/dmitropolsky-assemblies simulations.merge_sim)
    # keeps the parent STIMULI firing on every round instead of pinning:
    #     project({stimA:[A], stimB:[B]}, {A:[A,C], B:[B,C], C:[C,A,B]})
    # When callers supply stimuli we do the same. When they do not, we
    # reproduce the effect by re-activating the parent assemblies after each
    # round -- stable like _fix, but still writable.
    # PASS stim_a AND stim_b IF YOU NEED THE TWO-WAY CONNECTIVITY.
    #
    # Merge is the only operation here that projects BACK into its sources
    # (`target: [..., source_a, source_b]` below), and that back-projection is
    # what [PNAS20] means by "strong two-way synaptic connectivity between x
    # and z". The engine short-circuits a projection into a FIXED area before
    # plasticity is applied (see _fix), so the stimulus-less path cannot write
    # those synapses. Measured potentiation between the merged assembly and its
    # parents, n=10000 k=100 p=0.01 beta=0.05, 50 rounds:
    #
    #     stimulus-driven parents   B->A 3.47  A->B 2.52  C->A 3.52  A->C 2.75
    #     fixed parents             no potentiation on the back-projection
    #
    # The reference (.reference/dmitropolsky-assemblies simulations.merge_sim)
    # always drives the parents:
    #     project({stimA:[A], stimB:[B]}, {A:[A,C], B:[B,C], C:[C,A,B]})
    #
    # Holding the parents steady by re-activating their assemblies each round
    # instead of fixing them does NOT rescue this -- it is worse (ratios
    # 0.64-1.38, i.e. none). Plasticity follows the winners the projection
    # actually selects, so overwriting them afterwards potentiates the wrong
    # cells. Driving the parents is the only way to keep them stable AND
    # writable. The forward direction still works fixed, so the stimulus-less
    # path remains valid for a one-way merge.
    use_fix = (stim_a is None and stim_b is None)

    stim_dict = {}
    if stim_a:
        stim_dict[stim_a] = [source_a]
    if stim_b:
        stim_dict[stim_b] = [source_b]

    src_map = {
        source_a: ([source_a] if parent_self else []) + [target],
        source_b: ([source_b] if parent_self else []) + [target],
    }
    tgt_list = (([target] if target_self else [])
                + ([source_a, source_b] if back_project else []))

    with _fixed_sources(brain, *((source_a, source_b) if use_fix else ())):
        # Step 1: Simultaneous projection (no target recurrence yet)
        brain.project(stim_dict, dict(src_map))

        # Steps 2+: Add target recurrence and feedback to sources
        for _ in range(rounds - 1):
            brain.project(
                stim_dict,
                {**src_map, **({target: tgt_list} if tgt_list else {})},
            )

    return _snap(brain, target)


def pattern_complete(brain, area, fraction=0.5, rounds=5, seed=None):
    """Test pattern completion from partial activation.

    Protocol::

        1. Record current assembly as reference
        2. Randomly subsample ``fraction`` of winners
        3. Set subsampled winners, project area → area for ``rounds``
        4. Measure overlap with reference

    Args:
        brain: Brain instance.
        area: Name of the area with an established assembly.
        fraction: Fraction of assembly neurons to keep (default 0.5).
        rounds: Number of recurrent completion rounds (default 5).
        seed: Optional random seed for reproducible subsampling.

    Returns:
        (recovered_assembly, overlap_with_original) tuple.

    Theory:
        A well-trained assembly is an attractor: partial activation
        flows back to the full assembly through strengthened recurrent
        connections. This function does not establish that the assembly was
        well trained or enforce a recovery threshold; those require a stated
        regime and a matched control.

    Caveat on interpreting the score:
        The initial cue overlaps the reference by ``fraction``, but subsequent
        winners are free to change: this is not a floor on the final score.
        Compare against a matched learning-disabled or mechanism-lesioned
        control before interpreting recovery as evidence of learned recurrence.

    Note:
        Because plasticity is on, measuring pattern completion also
        strengthens the assembly being measured.  Repeated calls report
        rising recovery partly because of the earlier calls.
    """
    reference = _snap(brain, area)
    k = len(reference)

    # Subsample from COMPACT indices (area.winners), not mapped real IDs
    # (reference.winners).  The engine uses compact indexing internally.
    compact_winners = list(brain.areas[area].winners)
    rng = random.Random(seed)
    subsample_size = int(k * fraction)
    subsample = rng.sample(compact_winners, subsample_size)
    brain.areas[area].winners = np.array(subsample, dtype=np.uint32)
    # Winner sync to engine is handled by _project_impl

    # Recurrent completion
    for _ in range(rounds):
        brain.project({}, {area: [area]})

    recovered = _snap(brain, area)
    recovery = overlap(recovered, reference)
    return recovered, recovery


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
    brain._engine.reset_area_connections(area_name)


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


def sequence_memorize(brain, stimuli, target, rounds_per_step=10,
                      repetitions=1, phase_b_ratio=None,
                      beta_boost=None) -> Sequence:
    """Memorize an ordered sequence of stimuli in a target area.

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

            for _ in range(recur_rounds):
                brain.project({stim_name: [target]}, {target: [target]})

            if beta_boost is not None:
                brain.update_plasticity(target, target, original_beta)

            assemblies.append(_snap(brain, target))

    return Sequence(area=target, assemblies=assemblies)


def ordered_recall(brain, area, cue, max_steps=20,
                   known_assemblies=None, convergence_threshold=0.9,
                   rounds_per_step=1) -> Sequence:
    """Recall a memorized sequence from a cue using LRI.

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

    Returns:
        Sequence of Assembly snapshots in recall order.

    Raises:
        ValueError: If the area has ``refractory_period == 0``.
    """
    area_obj = brain.areas[area]
    if area_obj.refractory_period == 0:
        raise ValueError(
            f"ordered_recall requires refractory_period > 0 for area {area!r}. "
            f"Add the area with refractory_period=N to enable LRI."
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
        # 0.3 is a hardcoded "clearly not one of ours" floor -- far above
        # chance overlap k/n (~0.005 at k=100, n=20000) but far below the
        # ~0.8+ a genuine recall step produces, so the gap is wide and the
        # exact value is not delicate.  Unlike convergence_threshold it is
        # deliberately not exposed; no caller has needed to tune it.
        if known_assemblies is not None and len(known_assemblies) > 0:
            max_known_overlap = max(
                overlap(current, k) for k in known_assemblies
            )
            if max_known_overlap < 0.3:
                break

        recalled.append(current)

    return Sequence(area=area, assemblies=recalled)
