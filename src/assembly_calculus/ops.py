"""
Named Assembly Calculus operations.

Each function encapsulates a projection pattern from:
Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (PNAS 2020)

All functions:
- Take a Brain as first argument (purely functional, no subclassing)
- Return an Assembly snapshot of the result
- Accept ``rounds`` for stabilization control
"""

import random

import numpy as np

from .assembly import Assembly, overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(brain, area_name) -> Assembly:
    """Take an immutable snapshot of the current assembly in an area.

    Maps compact engine indices to real neuron IDs so that snapshots
    from different timepoints or different brain instances are comparable.
    """
    winners = brain.areas[area_name].winners
    # Map compact indices → real neuron IDs (sparse engine uses compact indexing)
    mapping = brain._engine.get_neuron_id_mapping(area_name)
    if mapping:
        mapped = np.array(
            [mapping[int(idx)] if int(idx) < len(mapping) else int(idx)
             for idx in winners],
            dtype=np.uint32,
        )
        return Assembly(area_name, mapped)
    return Assembly(area_name, winners.copy())


def _fix(brain, *area_names):
    """Fix assemblies in the given areas (prevent winner changes)."""
    for name in area_names:
        brain.areas[name].fix_assembly()


def _unfix(brain, *area_names):
    """Unfix assemblies in the given areas."""
    for name in area_names:
        brain.areas[name].unfix_assembly()


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

def project(brain, stimulus, target, rounds=10) -> Assembly:
    """Project a stimulus into a target area, forming a stable assembly.

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
    if rounds > 1:
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


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------

def associate(brain, source_a, source_b, target,
              stim_a=None, stim_b=None, rounds=10) -> Assembly:
    """Associate two source assemblies through a shared target area.

    Protocol::

        Phase 1: source_a → target         (rounds steps, with recurrence)
        Phase 2: source_b → target         (rounds steps, with recurrence)
        Phase 3: both sources → target     (rounds steps, interleaved)

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

    # Phase 1: Establish source_a → target pathway
    stim_dict_a = {stim_a: [source_a]} if stim_a else {}
    brain.project(stim_dict_a, {source_a: [source_a, target]})
    # Fast path: when sources are fixed, only target changes — project_rounds
    # handles it in a tight GPU loop.  When stims drive sources, each round
    # must also update the source area, so we fall back to brain.project().
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

    # Phase 3: Interleave both sources → target
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

    if use_fix:
        _unfix(brain, source_a, source_b)

    return _snap(brain, target)


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
    """
    use_fix = (stim_a is None and stim_b is None)
    if use_fix:
        _fix(brain, source_a, source_b)

    stim_dict = {}
    if stim_a:
        stim_dict[stim_a] = [source_a]
    if stim_b:
        stim_dict[stim_b] = [source_b]

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
        assemblies = []
        for stim_name in stimuli:
            # Compute Phase A / Phase B split
            if phase_b_ratio is not None:
                recur_rounds = max(1, int(rounds_per_step * phase_b_ratio))
                stim_rounds = rounds_per_step - recur_rounds
            else:
                # Legacy default: 2 recurrence rounds
                stim_rounds = max(1, rounds_per_step - 2)
                recur_rounds = rounds_per_step - stim_rounds

            # Phase A: stimulus-only rounds to establish the new assembly.
            # This anchors the winners to the stimulus input so that the
            # recurrent attractor from the previous assembly doesn't
            # dominate.
            for _ in range(stim_rounds):
                brain.project({stim_name: [target]}, {})

            # Phase B: stimulus + recurrence rounds to build the
            # inter-assembly Hebbian bridge (x_{i-1} -> x_i).
            if beta_boost is not None:
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

        # Check for novel (unrecognised) assembly
        if known_assemblies is not None and len(known_assemblies) > 0:
            max_known_overlap = max(
                overlap(current, k) for k in known_assemblies
            )
            if max_known_overlap < 0.3:
                break

        recalled.append(current)

    return Sequence(area=area, assemblies=recalled)
