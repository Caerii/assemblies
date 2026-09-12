"""
Systems consolidation and context accumulation for Assembly Calculus.

**Consolidation** replays projection protocols *without*
``reset_area_connections``, strengthening persistent area→area Hebbian
pathways.  Episodic training (with reset between items) learns
category-level bindings; consolidation (replay without reset) corresponds
to systems consolidation in the complementary learning systems framework
(McClelland, McNaughton & O'Reilly 1995).

**Context accumulation** implements incremental merge of word assemblies
into a running context representation — the emergent analogue of sustained
semantic context during sequential phon input in NEMO language acquisition
(Mitropolsky & Papadimitriou 2025).

The single mechanistic difference between the two training regimes in this
repo is one line: whether ``reset_area_connections`` is called between items.

    episodic (reset between items)   -- each item is learned against a fresh
        recurrent landscape, so what survives is the stimulus->area mapping
        and category-level structure.  Nothing accumulates across items,
        which is exactly what you want when measuring one item in isolation.
    consolidation (no reset)         -- items are replayed into a connectome
        that already carries the traces of every earlier item, so area->area
        pathways compound.  Cross-item regularities get written into
        cortico-cortical weights.

That maps onto the complementary learning systems picture: fast,
interference-avoiding episodic encoding versus slow interleaved replay that
extracts what is common across episodes.  The mapping is an ANALOGY at the
level of protocol, not a claim that these areas are hippocampus and cortex.

These are *protocol-level* operations: they compose ``project`` and ``merge``
from Papadimitriou et al. (PNAS 2020).  Parity with full episodic training
must be validated per application (see ``neural_assemblies/tests/test_consolidation.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import List, Sequence, Set, Tuple, Union

from .assembly import Assembly
from .ops import _fix, _snap, _unfix, activate_assembly, merge, project
from .contracts import (
    CONTEXT_ACCUMULATION_CONTRACT,
    CONTEXT_STEP_CONTRACT,
    CONSOLIDATION_PROTOCOL_CONTRACT,
    ContextAccumulationPlan,
    ContextAccumulationStepPlan,
    ConsolidationProtocolPlan,
    implements,
)

# ---------------------------------------------------------------------------
# Consolidation step types (declarative replay protocols)
# ---------------------------------------------------------------------------

PathwayEdge = Tuple[str, str]


@dataclass(frozen=True)
class PathwayReplay:
    """Replay stimulus→source then source→target with recurrence.

    Models core→role consolidation: fix the source assembly, project into
    the target with self-recurrence, unfix — **no** connectome reset.
    """

    source_area: str
    target_area: str
    stimulus: str | None = None
    rounds: int = 10


@dataclass(frozen=True)
class MergeReplay:
    """Replay merge of two sources into a conjunctive target area.

    Models subject+verb→VP consolidation after episodic phrase training.
    """

    source_a: str
    source_b: str
    target: str
    stimulus_a: str | None = None
    stimulus_b: str | None = None
    rounds: int = 10


@dataclass(frozen=True)
class MultiProjectReplay:
    """Replay several fixed sources projecting into one target.

    Used when auxiliary areas (e.g. NUMBER) co-project with core areas
    into structural targets during consolidation.
    """

    sources: Tuple[str, ...]
    target: str
    stimulus_projections: Tuple[Tuple[str, str], ...] = ()
    """Optional (stimulus, area) pairs to activate before the merge step."""
    rounds: int = 10


ConsolidationStep = Union[PathwayReplay, MergeReplay, MultiProjectReplay]


def _validate_replay_step(brain, step: ConsolidationStep) -> None:
    """Reject a replay schedule before its first backend mutation."""
    if isinstance(step, PathwayReplay):
        names = (step.source_area, step.target_area)
        stimuli = (step.stimulus,)
        if step.source_area == step.target_area:
            raise ValueError("pathway replay requires distinct source and target areas")
    elif isinstance(step, MergeReplay):
        names = (step.source_a, step.source_b, step.target)
        stimuli = (step.stimulus_a, step.stimulus_b)
        if len(set(names)) != 3:
            raise ValueError("merge replay requires three distinct areas")
    elif isinstance(step, MultiProjectReplay):
        names = (*step.sources, step.target)
        stimuli = tuple(stim for stim, _ in step.stimulus_projections)
        if not step.sources:
            raise ValueError("multi-project replay requires at least one source")
        if len(set(step.sources)) != len(step.sources):
            raise ValueError("multi-project replay sources must be distinct")
        if step.target in step.sources:
            raise ValueError("multi-project replay target must differ from sources")
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in step.stimulus_projections):
            raise TypeError("multi-project replay stimuli must be (stimulus, area) pairs")
        if any(area not in names for _, area in step.stimulus_projections):
            raise ValueError("multi-project replay stimulus area must be a replay area")
    else:
        raise TypeError(f"unsupported consolidation step: {type(step).__name__}")

    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("consolidation replay area names must be nonempty strings")
    if isinstance(step.rounds, bool) or not isinstance(step.rounds, Integral) or step.rounds < 1:
        raise ValueError("consolidation replay rounds must be a positive integer")
    for stimulus in stimuli:
        if stimulus is not None and stimulus not in brain.stimuli:
            raise KeyError(f"consolidation replay stimulus is unknown: {stimulus!r}")
    for name in names:
        if name not in brain.areas:
            raise KeyError(f"consolidation replay area is unknown: {name!r}")


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------

def replay_pathway(brain, step: PathwayReplay) -> PathwayEdge:
    """Execute one pathway consolidation step; return the strengthened edge."""
    _validate_replay_step(brain, step)
    if step.stimulus is not None:
        project(brain, step.stimulus, step.source_area, rounds=step.rounds)

    _fix(brain, step.source_area)
    brain.project(
        {},
        {step.source_area: [step.target_area], step.target_area: [step.target_area]},
    )
    if step.rounds > 1:
        brain.project_rounds(
            target=step.target_area,
            areas_by_stim={},
            dst_areas_by_src_area={
                step.source_area: [step.target_area],
                step.target_area: [step.target_area],
            },
            rounds=step.rounds - 1,
        )
    _unfix(brain, step.source_area)
    return (step.source_area, step.target_area)


def replay_merge(brain, step: MergeReplay) -> Set[PathwayEdge]:
    """Execute one merge consolidation step."""
    _validate_replay_step(brain, step)
    edges: Set[PathwayEdge] = set()

    if step.stimulus_a is not None:
        project(brain, step.stimulus_a, step.source_a, rounds=step.rounds)
    if step.stimulus_b is not None:
        project(brain, step.stimulus_b, step.source_b, rounds=step.rounds)

    merge(brain, step.source_a, step.source_b, step.target, rounds=step.rounds)
    edges.add((step.source_a, step.target))
    edges.add((step.source_b, step.target))
    return edges


def replay_multi_project(brain, step: MultiProjectReplay) -> Set[PathwayEdge]:
    """Execute a multi-source projection into one target with recurrence."""
    _validate_replay_step(brain, step)
    edges: Set[PathwayEdge] = set()

    for stim, area in step.stimulus_projections:
        project(brain, stim, area, rounds=step.rounds)

    for src in step.sources:
        _fix(brain, src)

    dst_map = {src: [step.target] for src in step.sources}
    dst_map[step.target] = [step.target]

    brain.project({}, dst_map)
    if step.rounds > 1:
        brain.project_rounds(
            target=step.target,
            areas_by_stim={},
            dst_areas_by_src_area=dst_map,
            rounds=step.rounds - 1,
        )

    for src in step.sources:
        _unfix(brain, src)
        edges.add((src, step.target))

    return edges


def _replay_step(brain, step: ConsolidationStep) -> Set[PathwayEdge]:
    if isinstance(step, PathwayReplay):
        return {replay_pathway(brain, step)}
    if isinstance(step, MergeReplay):
        return replay_merge(brain, step)
    return replay_multi_project(brain, step)


def inhibit_all_areas(brain) -> None:
    """Clear activity in all areas while preserving learned connectomes."""
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])


def prepare_area_for_replay(brain, area_name: str) -> None:
    """Reset winners and column counters before a consolidation replay step.

    After episodic ``reset_area_connections``, connectome weights are cleared
    but ``w`` may still reflect prior firing.  Replaying projections with a
    stale ``w`` leaves new weights disconnected from the sparse matrix.

    What ``w`` is and why it must be rewound.  ``w`` counts how many neurons of
    the area have been materialised so far (see ``ops._snap`` for the compact
    index / neuron ID distinction).  It is the width of every incoming
    connectome block.  Clearing the weights without clearing ``w`` leaves the
    area claiming ``w`` materialised neurons whose weight rows no longer exist,
    so a replayed projection writes into columns nothing reads.  That is the
    "weights disconnected from the sparse matrix" failure: it produces silence,
    not an error.

    The counters are rewound in BOTH the Area object and the engine's private
    per-area state, because the two are separate mirrors of the same quantity
    and the engine is what actually indexes the matrices.  Touching only one
    leaves them disagreeing.

    INVARIANT THIS BREAKS.  Resetting ``compact_to_neuron_id`` and the neuron
    ID pool pointer means the area will re-issue the same neuron IDs to
    different neurons.  Any :class:`Assembly` snapshot taken before this call
    is therefore no longer valid for this area: injecting it via
    ``ops.activate_assembly`` will either raise "not in area mapping" or,
    worse, silently name unrelated neurons.  Re-snapshot after consolidation;
    do not carry pre-consolidation lexicons across.
    """
    brain.inhibit_areas([area_name])
    brain.unfix_assembly(area_name)
    brain.reset_area_population_cursor(area_name, preserve_mapping=False)


def drop_stale_assemblies(parser) -> int:
    """Purge parser lexicon entries orphaned by consolidation.

    :func:`prepare_area_for_replay` re-issues neuron IDs, so every Assembly
    snapshot stored before it (in ``core_lexicons``, ``role_lexicons``,
    ``vp_assemblies``, ...) may now name neurons the area no longer maps.
    Replaying such a snapshot via ``ops.activate_assembly`` raises "not in area
    mapping". Its docstring already prescribes the remedy -- "do not carry
    pre-consolidation lexicons across" -- but the emergent curriculum fills
    those lexicons in one stage and replays them several stages later, so the
    orphaned entries must be dropped after consolidation runs. A dropped entry
    is not lost: every replay site falls back to re-projecting phon -> core,
    which produces a fresh, valid assembly.

    Returns the number of entries dropped (for logging/tests).
    """
    from .ops import assembly_is_current

    brain = getattr(parser, "brain", None)
    if brain is None:
        return 0

    dropped = 0
    for attr in ("core_lexicons", "role_lexicons"):
        lexicons = getattr(parser, attr, None)
        if not isinstance(lexicons, dict):
            continue
        for area, entries in lexicons.items():
            if not isinstance(entries, dict):
                continue
            stale = [w for w, asm in entries.items()
                     if asm is not None and not assembly_is_current(brain, asm)]
            for w in stale:
                del entries[w]
                dropped += 1

    vp = getattr(parser, "vp_assemblies", None)
    if isinstance(vp, dict):
        stale = [k for k, asm in vp.items()
                 if asm is not None and not assembly_is_current(brain, asm)]
        for k in stale:
            del vp[k]
            dropped += 1

    return dropped


def _prepare_step_areas(brain, step: ConsolidationStep) -> None:
    if isinstance(step, PathwayReplay):
        for name in (step.source_area, step.target_area):
            prepare_area_for_replay(brain, name)
    elif isinstance(step, MergeReplay):
        for name in (step.source_a, step.source_b, step.target):
            prepare_area_for_replay(brain, name)
    else:
        for name in (*step.sources, step.target):
            prepare_area_for_replay(brain, name)
        for _, area in step.stimulus_projections:
            prepare_area_for_replay(brain, area)


@implements(CONSOLIDATION_PROTOCOL_CONTRACT)
def consolidate(
    brain,
    steps: Sequence[ConsolidationStep],
    *,
    passes: int = 1,
    clear_activity: bool = True,
    prepare_areas: bool = False,
) -> Set[PathwayEdge]:
    """Replay a consolidation protocol without resetting area connections.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-consolidation-protocol

    Args:
        brain: Brain instance with areas and stimuli configured.
        steps: Ordered consolidation steps (one exposure episode).
        passes: Number of full replays through ``steps`` (developmental
            experience / sleep cycles).
        clear_activity: Inhibit all areas after consolidation (default).
        prepare_areas: Rewind ``w`` and the neuron-ID mapping of every area a
            step touches, via :func:`prepare_area_for_replay`. Opt-in for the
            EPISODIC-RESET case only, where the connectome was cleared and a
            stale ``w`` would leave replayed weights disconnected.
            **Destructive on a live connectome**, and the two senses of
            "reset" are easy to conflate: this function does not reset
            CONNECTIONS, but preparing an area does reset its INDEX SPACE,
            which invalidates every Assembly snapshot of it.

            Default False because every production caller replays onto a
            LIVE connectome and the old True default had to be overridden by
            all of them -- a default all callers override is wrong. Measured
            on the curriculum at ``DIALOGUE``: preparing the SOURCE area of
            a role replay rewound ``NOUN_CORE`` from w=2493 to w=976 and
            left 48 of 74 stored nouns unmappable (34 of 44 verbs), so a
            later parse raised "Assembly neuron N not in area mapping". The
            source of a role replay IS the core lexicon; rewinding it
            destroys the assemblies the replay exists to strengthen.

    Returns:
        Set of ``(source_area, target_area)`` edges strengthened.

    Reference:
        IMPLICATIONS_AND_PREDICTIONS.md §2 — consolidation as cortical replay.
    """
    plan = ConsolidationProtocolPlan(tuple(steps), passes, clear_activity, prepare_areas)
    invalid = [type(step).__name__ for step in plan.steps
               if not isinstance(step, (PathwayReplay, MergeReplay, MultiProjectReplay))]
    if invalid:
        raise TypeError(
            "consolidation steps must be PathwayReplay, MergeReplay, or "
            f"MultiProjectReplay; got {invalid!r}"
        )
    for step in plan.steps:
        _validate_replay_step(brain, step)
    steps = plan.steps
    passes = plan.passes
    clear_activity = plan.clear_activity
    prepare_areas = plan.prepare_areas

    strengthened: Set[PathwayEdge] = set()
    for _ in range(passes):
        for step in steps:
            if prepare_areas:
                _prepare_step_areas(brain, step)
            strengthened |= _replay_step(brain, step)

    if clear_activity:
        inhibit_all_areas(brain)

    return strengthened


# ---------------------------------------------------------------------------
# Context accumulation (incremental prefix assembly)
# ---------------------------------------------------------------------------

@implements(CONTEXT_STEP_CONTRACT)
def accumulate_context_step(
    brain,
    *,
    phon: str | None = None,
    core_area: str,
    context_area: str,
    core_assembly: Assembly | None = None,
    rounds: int = 10,
) -> Assembly:
    """Advance context by one word: phon→core (or lexicon core), then core→context.

    This is the direct (ungated) context protocol used during bridge
    training and ``predict_next``.  Each step merges the current word's
    core assembly into the running context assembly with recurrence.

    When ``core_assembly`` is provided (lexicon-grounded mode), the
    phon→core projection is skipped and the stabilized lexicon assembly
    is injected directly — NEMO Property 1: reuse stabilized word
    assemblies after lexicon training.

    Args:
        phon: Stimulus name for the word's phonological code (optional
            when ``core_assembly`` is given).
        core_area: Core area for the word's syntactic category.
        context_area: Area holding the running context assembly.
        core_assembly: Optional lexicon snapshot to inject instead of
            projecting from ``phon``.
        rounds: Projection rounds per step.

    Returns:
        Snapshot of the context assembly after this step.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-context-accumulation-step
    """
    plan = ContextAccumulationStepPlan(
        core_area=core_area,
        context_area=context_area,
        phon=phon,
        core_assembly=core_assembly,
        rounds=rounds,
    )
    plan.preflight(brain)
    core_area = plan.core_area
    context_area = plan.context_area
    phon = plan.phon
    core_assembly = plan.core_assembly
    rounds = plan.rounds
    if core_assembly is not None:
        if core_assembly.area != core_area:
            raise ValueError(
                f"core_assembly area {core_assembly.area!r} != "
                f"core_area {core_area!r}",
            )
        activate_assembly(brain, core_assembly)
    elif phon is not None:
        project(brain, phon, core_area, rounds=rounds)
    else:
        raise ValueError(
            "accumulate_context_step requires phon or core_assembly",
        )

    brain.project(
        {},
        {core_area: [context_area], context_area: [context_area]},
    )
    if rounds > 1:
        brain.project_rounds(
            target=context_area,
            areas_by_stim={},
            dst_areas_by_src_area={
                core_area: [context_area],
                context_area: [context_area],
            },
            rounds=rounds - 1,
        )
    return _snap(brain, context_area)


@implements(CONTEXT_ACCUMULATION_CONTRACT)
def accumulate_context(
    brain,
    word_steps: Sequence[Tuple[str, str]],
    *,
    context_area: str,
    core_assemblies: Sequence[Assembly | None] | None = None,
    rounds: int = 10,
) -> Assembly:
    """Build a context assembly from an ordered list of words.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-context-accumulation

    Args:
        word_steps: ``(phon_stimulus, core_area)`` pairs in sentence order.
        context_area: Target context area name.
        core_assemblies: Optional parallel sequence of lexicon snapshots;
            ``None`` entries fall back to phon→core projection.
        rounds: Projection rounds per word.

    Returns:
        Final context assembly after processing all words.
    """
    normalized_steps = tuple(
        (phon_stimulus, core_area) for phon_stimulus, core_area in word_steps
    )
    plan = ContextAccumulationPlan(
        normalized_steps,
        context_area,
        None if core_assemblies is None else tuple(core_assemblies),
        rounds,
    )
    plan.preflight(brain)
    word_steps = plan.word_steps
    context_area = plan.context_area
    core_assemblies = plan.core_assemblies
    rounds = plan.rounds

    result: Assembly | None = None
    for i, (phon, core_area) in enumerate(word_steps):
        core_asm = core_assemblies[i] if core_assemblies is not None else None
        result = accumulate_context_step(
            brain,
            phon=phon if core_asm is None else None,
            core_area=core_area,
            context_area=context_area,
            core_assembly=core_asm,
            rounds=rounds,
        )
    assert result is not None
    return result


def build_context_word_steps(
    words: Sequence[str],
    stim_map: dict,
    core_resolver,
) -> List[Tuple[str, str]]:
    """Compile surface words into ``(phon, core_area)`` pairs.

    Args:
        words: Token strings in order.
        stim_map: Word → phon stimulus name.
        core_resolver: Callable ``word -> core_area`` (e.g. parser method).
    """
    steps: List[Tuple[str, str]] = []
    for word in words:
        phon = stim_map.get(word)
        if phon is None:
            continue
        steps.append((phon, core_resolver(word)))
    return steps
