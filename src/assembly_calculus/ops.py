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
    mapping = (brain._engine.get_neuron_id_mapping(area_name)
               if hasattr(brain._engine, 'get_neuron_id_mapping') else None)
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
    for _ in range(rounds - 1):
        brain.project({stimulus: [target]}, {target: [target]})
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
    for _ in range(rounds - 1):
        brain.project({}, {source: [target], target: [target]})
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
    for _ in range(rounds - 1):
        brain.project(stim_dict_a, {source_a: [source_a, target], target: [target]})

    # Phase 2: Establish source_b → target pathway
    stim_dict_b = {stim_b: [source_b]} if stim_b else {}
    brain.project(stim_dict_b, {source_b: [source_b, target]})
    for _ in range(rounds - 1):
        brain.project(stim_dict_b, {source_b: [source_b, target], target: [target]})

    # Phase 3: Interleave both sources → target
    stim_dict_both = {}
    if stim_a:
        stim_dict_both[stim_a] = [source_a]
    if stim_b:
        stim_dict_both[stim_b] = [source_b]
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

    # Subsample
    rng = random.Random(seed)
    subsample_size = int(k * fraction)
    subsample = rng.sample(list(reference.winners), subsample_size)
    brain.areas[area].winners = np.array(subsample, dtype=np.uint32)

    # Sync subsampled winners to engine
    brain._engine.set_winners(area, np.array(subsample, dtype=np.uint32))

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

    In the Assembly Calculus, the separation theorem assumes each stimulus
    is projected into a fresh area (no prior attractor).  This helper
    simulates that by reverting area→area weight matrices to their
    pre-learning state.  Stimulus→area connections are preserved since
    they are independent per stimulus.

    Neurobiological motivation:  In cortex, strong attractors from prior
    stimuli are weakened by synaptic decay, short-term depression, and
    inhibitory feedback.  Resetting the recurrent weights is a discrete
    approximation of these continuous processes.
    """
    engine = brain._engine

    if hasattr(engine, '_area_conns'):
        is_sparse = hasattr(engine, '_sparse_sim')

        for src_name in list(engine._area_conns.keys()):
            if area_name not in engine._area_conns[src_name]:
                continue
            conn = engine._area_conns[src_name][area_name]

            if is_sparse and hasattr(conn, 'sparse') and conn.sparse:
                # Sparse engine: revert to initial empty shape
                conn.weights = np.empty((0, 0), dtype=np.float32)
                # Clear amortised growth bookkeeping so expansion
                # starts fresh after reset.
                if hasattr(conn, '_log_rows'):
                    del conn._log_rows
                if hasattr(conn, '_log_cols'):
                    del conn._log_cols
            elif not is_sparse:
                # Explicit engine: re-randomize
                rows, cols = conn.weights.shape
                conn.weights = np.asarray(
                    (np.random.default_rng().random((rows, cols)) < engine.p
                     ).astype(np.float32),
                )
