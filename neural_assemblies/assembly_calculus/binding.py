"""Associative binding between assemblies across fibers.

Every subsystem in this package that needed "fire this state later and get that
assembly back" rediscovered the same four failure modes, and each one shows up
as a suspiciously round number -- an overlap of exactly 0.000 or exactly 1.000:

1. **Unmaterialized fiber -> 0.000.** In sparse mode a projection can only
   strengthen synapses that already exist. A fiber's columns are created the
   first time it carries traffic, and only for neurons that actually fire. If
   the source area is empty when the fiber is first projected, no columns are
   allocated, so a later Hebbian pairing has nothing to write to and recall
   never lands on the target.

2. **Unpinned target -> the binding goes somewhere else.** Co-firing a teacher
   stimulus alongside the source lets both drive the competition, so the
   winners are a blend. At recall only the source fires, producing a different
   assembly than the one stored, and readout misses.

3. **Unpinned source -> the binding decays to noise.** Plasticity strengthens
   (source winners -> target winners). If the source is free to drift between
   the pairing and the recall, the strengthened synapses belong to a pattern
   that never fires again.

4. **Recurrence during binding -> 1.000.** Letting the target area's own
   recurrence run while many different sources are bound into it pulls every
   binding into that area's dominant attractor, so all stored assemblies
   become the same set. Worse, with plasticity disabled every weight is 1, all
   candidates tie, and the deterministic index tie-break in winner selection
   re-selects an identical winner set every time.

`bind` and `recall` below enforce all four, so callers get the invariants for
free instead of rediscovering them.

Timing note: binding is a *traversal* of already-stabilized assemblies, not the
formation of a new one. Papadimitriou et al. (PNAS 2020) report "a stable
assembly is formed after about T = 10 steps" for formation; Mitropolsky &
Papadimitriou (2025) fire a word for tau = 2 steps to traverse a trained
pathway. TAU below is that traversal constant.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from .assembly import Assembly, overlap
from .ops import activate_assembly, _snap
from .contracts import (
    BINDING_RECALL_CONTRACT,
    BINDING_STRENGTH_CONTRACT,
    INPUT_DRIVE_CONTRACT,
    SOURCE_BINDING_CONTRACT,
    BindingRecallPlan,
    BindingStrengthPlan,
    InputDrivePlan,
    SourceBindingPlan,
    implements,
)

# Per-step traversal of a trained pathway (Mitropolsky & Papadimitriou 2025,
# Fig. 3a). Not the ~10 steps needed to form a new assembly.
TAU = 2

__all__ = [
    "TAU", "materialize_fiber", "bind", "recall", "bind_strength", "binding_strength",
    "input_drive",
]


def _activate_all(
    brain,
    source_assemblies: Optional[Mapping[str, Assembly]],
) -> None:
    if not source_assemblies:
        return
    for assembly in source_assemblies.values():
        activate_assembly(brain, assembly)


def materialize_fiber(
    brain,
    src_area: str,
    dst_area: str,
    *,
    src_assembly: Optional[Assembly] = None,
) -> bool:
    """Ensure the ``src_area -> dst_area`` connectome has columns.

    Returns True if the fiber carried traffic. A fiber projected while the
    source area is empty allocates nothing, which is failure mode 1 above, so
    this refuses to pretend it succeeded.
    """
    if src_area not in brain.areas:
        raise KeyError(f"materialize_fiber source area is unknown: {src_area!r}")
    if dst_area not in brain.areas:
        raise KeyError(f"materialize_fiber target area is unknown: {dst_area!r}")
    if src_assembly is not None:
        activate_assembly(brain, src_assembly)
    if len(brain.areas[src_area].winners) == 0:
        return False

    # frozen(), NOT probe() -- and this is the clearest case in the repo of the
    # distinction. This function exists to ALLOCATE COLUMNS, and columns are
    # allocated as a side effect of the target recruiting. `probe()` under
    # isolation suppresses recruitment, which would turn the whole function
    # into a silent no-op that still returns True. Plasticity-off with
    # recruitment-on is exactly what is wanted here.
    with brain.frozen():
        brain.project({}, {src_area: [dst_area]})

    # A projection only allocates columns as a side effect of the target
    # recruiting new neurons, so a fiber first used after its target has
    # already grown stays shaped (0, 0) and delivers zero input forever --
    # which multiplicative plasticity can never repair. Allocate explicitly.
    engine = getattr(brain, "_engine", None)
    ensure = getattr(engine, "ensure_area_conn", None)
    if ensure is not None:
        ensure(src_area, dst_area)
    return True


@implements(SOURCE_BINDING_CONTRACT)
def bind(
    brain,
    *,
    sources: Iterable[str],
    target_area: str,
    teachers: Iterable[str] = (),
    source_assemblies: Optional[Mapping[str, Assembly]] = None,
    rounds: int = TAU,
) -> bool:
    """Hebbian-associate a cue with whatever the teacher drives in the target.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-source-binding

    Cue and teacher fire into ``target_area`` TOGETHER, with the target left
    free to settle. The teacher determines which assembly wins; plasticity
    then strengthens cue -> those winners. This is the paper's own mechanism:

        "the firing of the verb's phonological assembly will propagate through
        firing into ROLE_action; at this point, neurons from SUBJ (from the
        previous word) also fire into ROLE_action, and this is how the fact
        that verb comes after subject is recorded in the synapses between SUBJ
        and ROLE_action."

    Do NOT pin the target to force a particular assembly. The sparse engine
    short-circuits a projection into a fixed area::

        # Fixed assembly -- short-circuit
        if tgt.fixed_assembly:
            return ProjectionResult(...)

    returning before inputs are summed and before plasticity is applied, so a
    pinned-target "binding" writes nothing at all. Measured directly: weight
    sum 315 -> 315 with the target fixed, versus a real update with it free.
    An earlier version of this function pinned the target and was therefore a
    silent no-op; it appeared to work only because a `recall` between training
    steps performed the unfixed projection that actually wrote the weights.

    Args:
        sources: Areas carrying the cue.
        target_area: Area the association is written into.
        teachers: Areas that drive the target to the assembly being taught.
        source_assemblies: Optional {area: assembly} to activate first.
        rounds: Co-firing steps; defaults to TAU.

    Returns:
        True if the pairing was applied.
    """
    sources = tuple(sources)
    teachers = tuple(teachers)
    plan = SourceBindingPlan(sources, target_area, teachers, rounds)
    plan.preflight(brain)
    rounds = plan.rounds
    if not sources:
        return False

    _activate_all(brain, source_assemblies)
    live = [a for a in sources + teachers if len(brain.areas[a].winners) > 0]
    if not live:
        # Failure mode 1: nothing fires, so nothing can be bound.
        return False

    # Size every fiber to the CURRENT target before writing. Materializing
    # once at setup is not enough: the target keeps recruiting neurons during
    # training (a role area grew 362 -> 421 in one run), and the columns added
    # by that growth are unallocated for this source. Multiplicative
    # plasticity cannot grow them from zero, so the pairing writes nothing.
    #
    # This was masked in an ugly way. Interleaving a `bind_strength` call
    # between training steps made the binding "work" (0.000 -> 0.833), because
    # the projection inside `recall` grew the area and lazily sized the fiber.
    # The measurement was repairing what it measured; without it the same
    # training left the transition at 0.000.
    engine = getattr(brain, "_engine", None)
    ensure = getattr(engine, "ensure_area_conn", None)
    if ensure is not None:
        for area in live:
            ensure(area, target_area)

    # Sources are pinned so their winners are identical at binding time and
    # at recall; the TARGET is deliberately left free (see above).
    for area in live:
        brain.areas[area].fix_assembly()
    try:
        for _ in range(max(1, rounds)):
            brain.project({}, {area: [target_area] for area in live})
    finally:
        for area in live:
            brain.areas[area].unfix_assembly()
    return True


@implements(BINDING_RECALL_CONTRACT)
def recall(
    brain,
    *,
    sources: Iterable[str],
    target_area: str,
    source_assemblies: Optional[Mapping[str, Assembly]] = None,
    clear_target: bool = True,
) -> Optional[Assembly]:
    """Fire the cue and return what ``target_area`` produces.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-binding-recall

    A read, so it runs under `Brain.probe()`. "Plasticity is off, so recall
    does not reshape what it reads" was HALF TRUE: `frozen()` stops weights
    changing but not the target GROWING, and recruitment reshapes the
    connectome just as surely. Sources are pinned for the same reason they are
    pinned in `bind`.

    ``clear_target`` silences the target first, and defaults True because
    otherwise the measurement is dominated by incumbency: whatever the target
    already holds tends to be re-selected, so a cue that was never paired
    scores exactly as high as one that was. Measured -- without clearing, a
    paired cue and an unpaired control cue returned *identical* overlap at
    every training level (difference +0.000 from 0 to 16 pairings); with
    clearing, the paired cue reached 0.740 while the control stayed at 0.000.
    Set it False only when the retained state is deliberately part of what is
    being read.
    """
    plan = BindingRecallPlan(tuple(sources), target_area, clear_target)
    plan.preflight(brain)
    sources = list(plan.sources)
    clear_target = plan.clear_target
    if not sources:
        return None

    _activate_all(brain, source_assemblies)
    live = [a for a in sources if len(brain.areas[a].winners) > 0]
    if not live:
        return None

    with brain.probe():
        if clear_target:
            brain.inhibit_areas([target_area])
        for area in live:
            brain.areas[area].fix_assembly()
        try:
            brain.project({}, {area: [target_area] for area in live})
            return _snap(brain, target_area)
        finally:
            for area in live:
                brain.areas[area].unfix_assembly()


@implements(INPUT_DRIVE_CONTRACT)
def input_drive(
    brain,
    *,
    sources: Iterable[str],
    target_areas: Iterable[str],
    source_assemblies: Optional[Mapping[str, Assembly]] = None,
    metric: str = "pre_kwta",
) -> Dict[str, float]:
    """Total synaptic drive the cue delivers to each candidate area.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-pre-kwta-observation

    `metric` selects what "drive" means:

    * ``"pre_kwta"`` (default) -- global energy over every candidate neuron
      *before* winner selection, divided by the number of candidates.
    * ``"winners"`` -- summed input over the selected winners only.

    The pre-k-WTA figure is normalized per candidate because areas do not
    have the same number of recruited neurons: two role areas measured here
    differed by 391 vs 449, and comparing raw sums across them reverses the
    ranking purely on size. Any cross-area comparison must divide out the
    candidate count.

    ``w`` WAS THE WRONG STAND-IN FOR THAT COUNT, and it decided the scale of
    every number this function returned (#104). This implementation reads the
    engine-reported observation pair. The intent above is right --
    make areas commensurable -- but ``w`` is the MATERIALISED count, an
    artifact of lazy instantiation with no counterpart in the calculus, where
    an area has a fixed ``n``. Measured across arms that vary how much of an
    area competes (research/experiments/erp_full_substrate.log): ``w`` grows
    7.8x, the P600 gap this feeds falls 12.5x, while the rank statistic barely
    moves. So the SCALE tracks training history, which is why a threshold set
    once ended up 11.9x above anything observable.

    RATIOS BETWEEN AREAS MEASURED IN ONE CALL ARE STILL FINE -- that is what
    this function is for, and it is why the defect stayed invisible. It is
    absolute magnitudes, and thresholds on them, that do not survive.
    See research/notes/language/erp_scale_is_an_implementation_detail.md.

    The distinction is not cosmetic. This repository's N400 work found global
    pre-k-WTA energy to be the robust quantity (Cohen's d = -25.2) while
    neuron-specific, post-selection measures *reversed* direction, because
    k-WTA makes related assemblies compete for shared neurons. A competition
    scored on the post-selection sum is therefore reading a quantity known to
    behave badly, which is why "winners" is not the default.

    Returns {area: drive}. This is the quantity area-level competition is
    actually decided on -- ``Brain._apply_mutual_inhibition`` keeps the area
    with the highest total activation -- so it is the right score for "which
    area fires next".

    Use this rather than `bind_strength` whenever the question is *which area*
    rather than *which assembly*. A single cue bound over time to many
    different target assemblies cannot reproduce any one of them, so
    per-assembly overlap saturates (every candidate returns ~1.0) and carries
    no signal, while the drive it delivers still differs.

    All targets are driven in ONE projection so their scores are commensurable
    and any registered mutual inhibition resolves between them exactly as it
    would during normal operation.
    """
    plan = InputDrivePlan(tuple(sources), tuple(target_areas), metric)
    plan.preflight(brain)
    sources = list(plan.sources)
    targets = list(plan.target_areas)
    metric = plan.metric

    _activate_all(brain, source_assemblies)
    live = [a for a in sources if len(brain.areas[a].winners) > 0]
    if not live:
        raise ValueError("input_drive requires at least one active source assembly")

    # probe() owns plasticity AND recruitment; record_activation is a separate
    # flag saved here. This function IS the P600 measurement (see
    # `erp.adapters.anchored_p600_live`), so it is a read in the strict sense.
    prev_rec = getattr(brain, "record_activation", False)
    with brain.probe():
        if metric == "pre_kwta":
            brain.record_activation = True
        for area in live:
            brain.areas[area].fix_assembly()
        try:
            brain.project({}, {area: list(targets) for area in live})
            if metric == "pre_kwta":
                scores = dict(getattr(brain, "last_pre_kwta_totals", {}) or {})
                if not scores:  # engine did not record; fall back
                    scores = dict(getattr(brain, "last_activation_scores", {}) or {})
                else:
                    normalized = {}
                    for area in scores:
                        if area not in brain.areas:
                            continue
                        observation = brain.pre_kwta_observation(area)
                        if observation is None:
                            raise RuntimeError(
                                f"pre-k-WTA total for {area!r} has no candidate count"
                            )
                        normalized[area] = observation.mean
                    scores = normalized
            else:
                scores = dict(getattr(brain, "last_activation_scores", {}) or {})
        finally:
            for area in live:
                brain.areas[area].unfix_assembly()
            brain.record_activation = prev_rec

    missing = [area for area in targets if area not in scores]
    if missing:
        raise RuntimeError(
            "input_drive engine observation omitted target area(s): "
            f"{missing!r}"
        )
    return {area: float(scores[area]) for area in targets}


@implements(BINDING_STRENGTH_CONTRACT)
def bind_strength(
    brain,
    *,
    sources: Iterable[str],
    target_area: str,
    target_assembly: Assembly,
    source_assemblies: Optional[Mapping[str, Assembly]] = None,
) -> float:
    """How well the cue currently reproduces ``target_assembly``, in [0, 1].

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-binding-strength

    This is the competition score to compare across candidate targets, and the
    diagnostic to assert on: a value of exactly 0.0 means the fiber was never
    materialized, and 1.0 across *different* targets means they have collapsed
    onto one assembly.
    """
    plan = BindingStrengthPlan(tuple(sources), target_area, target_assembly)
    plan.preflight(brain)
    sources = list(plan.sources)
    _activate_all(brain, source_assemblies)
    if not any(len(brain.areas[name].winners) > 0 for name in sources):
        # A numeric zero is reserved for a measured but unsuccessful recall;
        # an inactive cue is an invalid measurement domain.
        raise ValueError("bind_strength requires at least one active source assembly")
    got = recall(
        brain,
        sources=sources,
        target_area=target_area,
        source_assemblies=source_assemblies,
    )
    if got is None:
        raise RuntimeError("recall produced no target assembly after active cue")
    return float(overlap(got, target_assembly))


# Canonical descriptive spelling; retain ``bind_strength`` for compatibility.
binding_strength = bind_strength
