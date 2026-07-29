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

plus the sequence extension of Dabagia et al. (:func:`sequence_memorize` /
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
    Papadimitriou, Vempala, Mitropolsky, Collins, Maass.
    "Brain Computation by Assemblies of Neurons." PNAS 117(25), 2020.
"""

import random

import numpy as np

from .assembly import Assembly, overlap


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

    The ``int(idx) < len(mapping)`` guard passes through indices past the end
    of the mapping unchanged.  That should not happen -- a winner always has a
    compact slot -- so it is a defensive fallback, not a modelled case.
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
        mapped = np.array(
            [mapping[int(idx)] if int(idx) < len(mapping) else int(idx)
             for idx in winners],
            dtype=np.uint32,
        )
        return Assembly(area_name, mapped)
    return Assembly(area_name, winners.copy())


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
    return {int(nid): i for i, nid in enumerate(mapping)}


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

    Fixing is also not scoped: it persists until ``_unfix``.  Every caller
    here pairs them, but note that none use try/finally, so an exception
    mid-operation leaves the source areas fixed and every later projection out
    of them quietly wrong.
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

def project(brain, stimulus, target, rounds=10, recurrent=False) -> Assembly:
    """Project a stimulus into a target area, forming a stable assembly.

    ``recurrent`` OPTS IN TO THE PROTOCOL AS DOCUMENTED BELOW, and defaults
    False because the default path does NOT implement it. ``Brain.project_rounds``
    filters the projection map with ``a != target`` unless
    ``Brain(recurrent_projection=True)``, which itself defaults False -- so by
    default this function runs stimulus-only on every round, with no
    ``target -> target`` recurrence at all. What that builds is not an assembly
    in the defining sense (Dabagia et al. 2024: k neurons whose INTERNAL weights
    have been strengthened).

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


def reciprocal_project(brain, source, target, rounds=10) -> Assembly:
    """Project assembly from source area into target area.

    Protocol::

        1. source → target                          (initial projection)
        2. (source → target) + (target → target)    × (rounds - 1)

    Args:
        brain: Brain instance.
        source: Name of the source area (must have an established assembly).
        target: Name of the target area.
        rounds: Number of projection rounds (default 10).

    Returns:
        Assembly snapshot of the new assembly in target.

    Theory:
        The target assembly is a "copy" of the source assembly in the
        new area's neural population. After stabilization, the source
        assembly can be recovered by projecting back (target → source).
    """
    brain.project({}, {source: [target]})
    if rounds > 1:
        brain.project_rounds(
            target=target,
            areas_by_stim={},
            dst_areas_by_src_area={source: [target], target: [target]},
            rounds=rounds - 1,
        )
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
              stim_a=None, stim_b=None, rounds=10) -> Assembly:
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

    Returns:
        Assembly snapshot of the associated assembly in target.

    Theory (Papadimitriou 2020, §3):
        After association, activating source_a alone and projecting to
        target produces an assembly that significantly overlaps with the
        assembly produced by activating source_b alone. The overlap is
        well above chance level (k/n).
    """
    use_fix = (stim_a is None and stim_b is None)
    if use_fix:
        _fix(brain, source_a, source_b)

    try:
        _associate_body(
            brain, source_a, source_b, target, stim_a, stim_b, rounds, use_fix,
        )
    finally:
        # A raise mid-projection must not leave the sources fixed: the areas
        # would stay pinned for the rest of the session and every later
        # projection out of them would be silently wrong (a projection INTO a
        # fixed area is short-circuited before plasticity; see _fix).
        if use_fix:
            _unfix(brain, source_a, source_b)

    return _snap(brain, target)


def _associate_body(brain, source_a, source_b, target,
                    stim_a, stim_b, rounds, use_fix):
    """Projection phases for :func:`associate`; see it for the contract."""
    # Phase 1: Establish source_a → target pathway
    stim_dict_a = {stim_a: [source_a]} if stim_a else {}
    brain.project(stim_dict_a, {source_a: [source_a, target]})
    # Fast path: when sources are fixed, only target changes — project_rounds
    # handles it in a tight GPU loop.  When stims drive sources, each round
    # must also update the source area, so we fall back to brain.project().
    #
    # The two branches are NOT synapse-for-synapse identical: the fast path
    # omits the source_a -> source_a fiber that the slow path keeps open.
    # Winners are unaffected (source_a is fixed either way), but the slow path
    # additionally potentiates source_a's recurrent weights.  Documented
    # rather than reconciled, since aligning them would change measured
    # results for every caller that passes stimuli.
    if rounds > 1 and use_fix:
        brain.project_rounds(
            target=target,
            areas_by_stim={},
            dst_areas_by_src_area={source_a: [target], target: [target]},
            rounds=rounds - 1,
        )
    else:
        for _ in range(rounds - 1):
            brain.project(stim_dict_a, {source_a: [source_a, target], target: [target]})

    # Phase 2: Establish source_b → target pathway
    stim_dict_b = {stim_b: [source_b]} if stim_b else {}
    brain.project(stim_dict_b, {source_b: [source_b, target]})
    if rounds > 1 and use_fix:
        brain.project_rounds(
            target=target,
            areas_by_stim={},
            dst_areas_by_src_area={source_b: [target], target: [target]},
            rounds=rounds - 1,
        )
    else:
        for _ in range(rounds - 1):
            brain.project(stim_dict_b, {source_b: [source_b, target], target: [target]})

    # Phase 3: Both sources drive target simultaneously (this is the step
    # that creates the shared winners; see the docstring).
    stim_dict_both = {}
    if stim_a:
        stim_dict_both[stim_a] = [source_a]
    if stim_b:
        stim_dict_both[stim_b] = [source_b]
    if use_fix:
        brain.project_rounds(
            target=target,
            areas_by_stim={},
            dst_areas_by_src_area={source_a: [target], source_b: [target], target: [target]},
            rounds=rounds,
        )
    else:
        for _ in range(rounds):
            brain.project(
                stim_dict_both,
                {source_a: [source_a, target], source_b: [source_b, target], target: [target]},
            )


def merge(brain, source_a, source_b, target,
          stim_a=None, stim_b=None, rounds=10) -> Assembly:
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

    Returns:
        Assembly snapshot of the merged assembly in target.

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
    if use_fix:
        _fix(brain, source_a, source_b)

    stim_dict = {}
    if stim_a:
        stim_dict[stim_a] = [source_a]
    if stim_b:
        stim_dict[stim_b] = [source_b]

    try:
        # Step 1: Simultaneous projection (no target recurrence yet)
        brain.project(
            stim_dict,
            {source_a: [source_a, target], source_b: [source_b, target]},
        )

        # Steps 2+: Add target recurrence and feedback to sources
        for _ in range(rounds - 1):
            brain.project(
                stim_dict,
                {
                    source_a: [source_a, target],
                    source_b: [source_b, target],
                    target: [target, source_a, source_b],
                },
            )
    finally:
        # See associate: a raise must not leave the sources fixed for the rest
        # of the session.
        if use_fix:
            _unfix(brain, source_a, source_b)

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
        connections. At fraction=0.5, recovery should exceed 0.8 for
        well-trained assemblies.

    Caveat on interpreting the score:
        The kept half of the assembly is *itself* part of the reference, so
        the arithmetic floor of the returned overlap is ``fraction`` (0.5 by
        default) even if completion recruits nothing but noise.  Only the
        excess above ``fraction`` is evidence of attractor dynamics.

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
    """
    brain._engine.reset_area_connections(area_name)


# ---------------------------------------------------------------------------
# Sequence operations (Dabagia et al. 2024)
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
        beta_boost: Temporary plasticity boost for recurrent connections
            during Phase B.  If None, uses the area's current beta.
            A value of 0.5 strengthens inter-assembly bridges.

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

            # Phase B: stimulus + recurrence rounds to build the
            # inter-assembly Hebbian bridge (x_{i-1} -> x_i).  The bridge is
            # what makes recall possible: while x_i is being driven into
            # place, the x_{i-1} neurons are still firing, so target->target
            # synapses from x_{i-1} onto x_i get potentiated.  Later,
            # activating x_{i-1} alone drives x_i harder than anything else
            # in the area.  Raising beta here deepens exactly those bridges
            # without over-strengthening the within-assembly recurrence that
            # Phase A already built.
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
