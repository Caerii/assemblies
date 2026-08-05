"""Live and fresh-stimulus ERP adapters on EmergentParser.

P600/N400 RAW QUANTITIES ARE PRE-k-WTA ENERGY, NOT POST-k-WTA CHURN.

An earlier version read the P600 (winner-set Jaccard instability) and N400
(winner overlap) from POST-k-WTA winner sets. That family of measures reverses
sign whenever the k-WTA competition is reweighted -- documented at
``binding.py`` (see ``input_drive``): "neuron-specific, post-selection measures
*reversed* direction, because k-WTA makes related assemblies compete for shared
neurons". ``norm_init`` (the Brain default) reweights the competition by design
via per-neuron in-degree normalization, so the churn ordering flips and the
grammatical/violation separation inverts (measured Cohen's d = -1.94).

The robust quantity is GLOBAL PRE-k-WTA ENERGY -- the summed synaptic drive
over candidate neurons BEFORE winner selection, normalized per candidate. This
package's N400 work found it robust (Cohen's d = -25.2) where post-selection
measures reversed. Both components below are now energy DEFICITS: a grammatical
(trained) pathway delivers HIGH pre-k-WTA energy, a category/syntactic
violation routes a wrongly-typed core through an UNTRAINED pathway and delivers
LESS, so ``1 - normalized_energy`` is LARGER for violations, which is the
correct P600/N400 sign.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from neural_assemblies.assembly_calculus.binding import input_drive
from neural_assemblies.assembly_calculus.metrics.instability import (
    mean_jaccard_instability,
)
from neural_assemblies.assembly_calculus.ops import _compact_index
from neural_assemblies.core.measurement import Measured, defined_values

from ...core.areas import (
    ADJP,
    CATEGORY_TO_CORE,
    CONTEXT,
    NP,
    NUMBER,
    PREDICTION,
    ROLE_AGENT,
    ROLE_PATIENT,
    VP,
)
from .gates import ErpReadiness, assess_erp_readiness, areas_with_active_assembly
from .protocol import ErpProtocol
from .probe_util import probe_context

if TYPE_CHECKING:
    from ...parser import EmergentParser

P600_SETTLING_ROUNDS = 10
P600_SETTLING_MINING = 3
PHRASE_STABILITY_ROUNDS_CALIBRATION = 3
PHRASE_STABILITY_ROUNDS_MINING = 1
N400_WEIGHT = 0.55
P600_WEIGHT = 0.45

_ERP_DEBUG = os.environ.get("ERP_DEBUG", "").strip() not in ("", "0", "false")

ProbeDepth = str  # "calibration" | "mining"


def settling_rounds_for_depth(probe_depth: ProbeDepth) -> int:
    if probe_depth == "mining":
        return P600_SETTLING_MINING
    return P600_SETTLING_ROUNDS


def phrase_stability_rounds_for_depth(probe_depth: ProbeDepth) -> int:
    if probe_depth == "mining":
        return PHRASE_STABILITY_ROUNDS_MINING
    return PHRASE_STABILITY_ROUNDS_CALIBRATION


def _self_recurrent_energy(brain, area: str) -> Measured:
    """Normalized self-recurrent PRE-k-WTA energy of the assembly in *area*.

    The assembly projects back into its own area; the summed drive over
    candidate neurons before winner selection, divided by area size, measures
    how strongly the assembly reinforces itself. A well-integrated assembly
    scores high, a weakly-bound one low.

    The area is deliberately NOT fixed: the sparse engine short-circuits a
    projection into a FIXED target and returns before inputs are summed (see
    ``binding.bind``), which would leave ``pre_kwta_total`` unrecorded. Reading
    energy therefore requires the free projection.

    "Plasticity is off (frozen), so this reads without reshaping the
    connectome" -- THAT WAS HALF TRUE AND THE MISSING HALF MATTERS. frozen()
    stops weights changing; it does not stop the area GROWING, and growth
    reshapes the connectome just as surely. See `probe_util.probe_context`.

    THE DIVISOR IS WRONG AND IT SETS THE WHOLE SCALE (#104). ``area.w`` is the
    MATERIALISED COUNT -- how many neurons lazy instantiation has got around to
    creating -- which is an implementation detail with no counterpart in the
    calculus, where an area has a fixed ``n``. Measured over arms that vary how
    much of the area competes (research/experiments/erp_full_substrate.log):
    ``w`` grows 7.8x and the p600 gap falls 12.5x, span 5.9x, while the rank
    statistic barely moves (AUC 1.000 -> 0.922). So every training that grows
    the parser shrinks this number, which is how ``P600_EXCESS_MARGIN`` ended
    up 11.9x above anything observable without anyone editing it -- and it is
    also the mechanism behind the saturation, since ``1 - drive/w`` is pushed
    toward 1.0 as ``w`` grows.

    The same divisor is in ``binding.input_drive``, which this function calls,
    so it is one choice on two paths.

    AND REPLACING IT DOES NOT FIX ANYTHING -- MEASURED, hypothesis refuted
    (research/experiments/erp_denominator_invariance.log). Six statistics off
    the same sum across three materialisation levels: ``w`` drifts 40.6x, and
    then EVERY scale-free alternative lands on 7.76-7.78 -- including two that
    are ratios of quantities from the SAME pool, where pool size cancels
    algebraically. A statistic whose normalisation cannot matter still drifts,
    so the drift is not in the normalisation.

    THE POOL IS THE PROBLEM. Concentration (top-k drive over pool mean) reads
    1.23 when the area is lazily materialised and 9.57 when it is full, because

        lazy pool = 42 candidates, k = 30  ->  71% OF THE POOL IS THE ASSEMBLY

    leaving nothing for the assembly to stand out from. Hence
    ``pre_kwta_pool_ratio`` below: a number read at pool/k ~ 1.4 is not a
    measurement, and this package's history is of exactly such numbers being
    reported as results. See
    research/notes/erp_scale_is_an_implementation_detail.md.

    RETURNS `Measured`, AND THAT IS THE POINT OF THIS FUNCTION'S HISTORY. Four
    different conditions used to return the bare float 0.0 -- area absent, no
    winners, projection failed, and genuinely-zero drive -- and a fifth, THE
    FIBER DOES NOT EXIST, was indistinguishable from the fourth. `VP -> VP` is
    shape (0,0) because nothing ever declares it, so this read exactly 0.000000
    and `1 - energy` was a CONSTANT 1.0 for the whole P600 violation arm. Every
    magnitude published off that arm was measured with a dead probe.

    Each of those is now an UNDEFINED measurement naming its own precondition,
    so the caller cannot average it into a result. See `core/measurement`.
    """
    if area not in brain.areas:
        return Measured.undefined(f"area {area!r} is not in this brain")
    winners = brain.areas[area].winners
    if winners is None or len(winners) == 0:
        return Measured.undefined(f"{area} has no winners; nothing is firing")

    # THE STRUCTURAL CHECK, ahead of the projection. A zero-synapse fiber still
    # produces a projection and still returns k winners -- the totalizing
    # substrate has no way to refuse -- so asking afterwards whether the drive
    # was zero cannot distinguish "no fiber" from "weak assembly".
    engine = getattr(brain, "_engine", None)
    extent = None
    if engine is not None and hasattr(engine, "fiber_extent"):
        try:
            extent = engine.fiber_extent(area, area)
        except (AttributeError, KeyError, RuntimeError):
            extent = None
    if extent == 0:
        return Measured.undefined(
            f"{area} has no self-fiber ({area}->{area} is unmaterialized), so "
            f"self-recurrent energy is not defined for it -- this is the #108 "
            f"defect that made the P600 violation arm a constant",
            area=area, extent=0,
        )

    prev_rec = getattr(brain, "record_activation", False)
    with probe_context(brain):
        brain.record_activation = True
        try:
            brain.project({}, {area: [area]})
            totals = getattr(brain, "last_pre_kwta_totals", {}) or {}
            counts = getattr(brain, "last_pre_kwta_counts", {}) or {}
            _record_pool_ratio(brain, area, counts.get(area))
            w = max(int(brain.areas[area].w), 1)
            return Measured.of(float(totals.get(area, 0.0)) / w)
        except (RuntimeError, IndexError, ValueError) as exc:
            return Measured.undefined(
                f"projection into {area} failed: {type(exc).__name__}",
                area=area,
            )
        finally:
            brain.record_activation = prev_rec


#: Below this, a concentration/energy read cannot separate a trained pathway
#: from an untrained one because most of the candidate pool IS the assembly.
#: Not tuned: it is the point where the assembly stops being a majority of what
#: it is being compared against. At pool/k = 2 half the pool is the assembly;
#: measured concentration is 1.23 at pool/k = 1.4 and 9.57 at pool/k = 100.
MIN_POOL_RATIO = 2.0


def _record_pool_ratio(brain, area: str, count) -> None:
    """Stash candidates/k for the last probe so callers can check it.

    A SEPARATE RECORD RATHER THAN A RAISE, deliberately. Raising here would
    break every existing caller on a substrate that has always been in this
    regime; returning a silently untrustworthy float is what the package
    already does. So the ratio is reported and
    `gates.assess_erp_readiness` can gate on it -- the same shape as
    `parse_errors.Stability.trustworthy`, which exists for the identical reason
    (pool <= k makes a k-cap unable to move).
    """
    if count is None:
        return
    k = max(int(getattr(brain.areas[area], "k", 0)), 1)
    ratios = getattr(brain, "last_erp_pool_ratio", None)
    if ratios is None:
        ratios = {}
        brain.last_erp_pool_ratio = ratios
    ratios[area] = float(count) / k


def _live_sources_into(brain, area: str) -> List[str]:
    """Areas that BOTH have winners and own a real fiber into *area*.

    Both halves matter. A source with no winners contributes nothing; a source
    whose fiber into *area* has never been materialised contributes nothing
    either, and the second case is invisible -- k-WTA still returns k winners
    ([[silent-no-op-dead-fibers]]), which is the whole reason the self-recurrent
    probe read 0.0 for a year without anyone noticing.
    """
    engine = brain._engine_for(brain.areas[area])
    conns = getattr(engine, "_area_conns", None)
    if conns is None:
        return []
    out = []
    for src in sorted(brain.areas):            # sorted: recruitment order (#80)
        if src == area:
            continue
        conn = conns.get(src, {}).get(area)
        w = getattr(conn, "weights", None)
        if w is None or getattr(w, "shape", (0, 0))[0] == 0:
            continue
        if len(brain.areas[src].winners) == 0:
            continue
        out.append(src)
    return out


def afferent_energy(brain, area: str) -> float:
    """Normalized PRE-k-WTA energy *area* RECEIVES FROM ITS SOURCES.

    THE ALTERNATIVE TO `_self_recurrent_energy`, and the reason it exists is
    measured (#108). VP -- the P600 violation arm -- has NO self-fiber at all:
    `VP -> VP` is shape (0,0) with zero synapses, because `_build_circuit`
    never declares it. So the self-recurrent probe returned exactly 0.000000
    and `1 - energy` was a constant 1.0 whatever the sentence was.

    VP is not unbuilt; it is richly built the other way round::

        VERB_CORE -> VP   (2543, 480)   50032 synapses
        SUBJ      -> VP    (296, 480)    4360
        OBJ       -> VP    (120, 656)    2736
        VP        -> VP        (0, 0)       0

    The grammatical arm probes ROLE_PATIENT, which DOES have a self-fiber
    (960x960, 24173 synapses) only because `_pregrow_role_pathways` explicitly
    opens `{core: [role], role: [role]}`. So the two arms were never measuring
    comparable quantities -- one had the probed fiber and the other did not.

    Afferent drive is defined for BOTH arms and needs no new structure. It is
    NOT wired in by default: whether it separates grammatical from violation is
    an empirical question, and the last structural change made here inverted
    seed 42 below chance. Compare with
    research/experiments/erp_afferent_vs_recurrent.py before adopting.
    """
    if area not in brain.areas:
        return 0.0
    sources = _live_sources_into(brain, area)
    if not sources:
        return 0.0
    prev_rec = getattr(brain, "record_activation", False)
    with probe_context(brain):
        brain.record_activation = True
        try:
            brain.project({}, {src: [area] for src in sources})
            totals = getattr(brain, "last_pre_kwta_totals", {}) or {}
            counts = getattr(brain, "last_pre_kwta_counts", {}) or {}
            _record_pool_ratio(brain, area, counts.get(area))
            # Divided by the candidates ACTUALLY SUMMED, not by `area.w` --
            # the shipped divisor is the materialised count, which is a
            # lazy-instantiation artifact (#104).
            n = max(int(counts.get(area, 0)), 1)
            return float(totals.get(area, 0.0)) / n
        except (RuntimeError, IndexError, ValueError):
            return 0.0
        finally:
            brain.record_activation = prev_rec


def phrase_stability(
    brain,
    area: str,
    *,
    rounds: int = 3,
    k: Optional[int] = None,
) -> Measured:
    """Phrase integration as normalized self-recurrent PRE-k-WTA energy.

    RETURNS `Measured`, which may be UNDEFINED -- most often because the area
    has no self-fiber. Do not average it with `sum(...)/len(...)`; use
    `measurement.defined_values` and report how many readings were dropped.

    Replaces the recurrent winner-overlap churn (``overlap / k``), which
    reverses sign under ``norm_init`` because it is a post-k-WTA quantity.
    Higher = better-integrated phrase; ``measure_live_integration`` reads
    ``1 - stability`` as the (deficit) phrase-instability term. ``rounds`` /
    ``k`` are retained for call-site compatibility but no longer used: energy
    is a single-projection quantity, not a settling one.

    ``ERP_AFFERENT_ENERGY=1`` switches to `afferent_energy`, which is the only
    quantity DEFINED for an area with no self-fiber -- see #108 and that
    function's docstring. Off by default: it is an A/B seam, not an adoption.
    """
    if ErpProtocol.from_environment().afferent_energy:
        return Measured.of(afferent_energy(brain, area))
    return _self_recurrent_energy(brain, area)


def _phrase_areas_for_category(category: str, *, verb_seen: bool) -> List[str]:
    if category == "VERB":
        return [VP]
    if category in ("NOUN", "PRON"):
        return [ROLE_AGENT if not verb_seen else ROLE_PATIENT, VP, NP]
    if category == "ADJ":
        return [ADJP, NP, VP]
    if category in ("ADV", "PREP"):
        return [VP]
    return [VP]


def structural_role_area(category: str, *, verb_seen: bool) -> str:
    """The area implied by the word's OBSERVED category.

    THIS IS THE CONFOUND, not merely a convention. A category violation IS a
    word whose observed category differs from the expected one, so dispatching
    the probe area on `category` makes the violation arm read a DIFFERENT AREA
    from its grammatical control by construction -- in every frame set, at every
    seed, under every definition of energy. Measured on both shipped frame sets:
    grammatical -> ROLE_PATIENT x3, category_violation -> VP x3.

    Area identity alone reproduces the headline AUC with the condition held
    constant (research/notes/p600_is_confounded_with_area_identity.md), which is
    why `_self_recurrent_energy` reads a constant 1.0 on VP and `afferent_energy`
    reads AUC exactly 0.000 with ZERO seed variance. Both are the same fact.

    Kept as the default and as the fallback for positions where the grammar
    licenses more than one continuation. See `expected_role_area`.
    """
    if category == "VERB":
        return VP
    if category in ("NOUN", "PRON"):
        return ROLE_PATIENT if verb_seen else ROLE_AGENT
    return VP


def expected_role_area(
    *, verb_seen_before: bool, object_open: bool,
) -> Optional[str]:
    """The slot the PARSE predicts here, independent of what word arrived.

    Returns None where the grammar licenses more than one continuation and no
    single area is predicted; the caller falls back to `structural_role_area`.

    ONLY the post-verb object position is claimed. After a verb that licenses an
    object, the next content word is expected in ROLE_PATIENT whether it turns
    out to be `cat` or `finds` -- so both ERP arms read the same area and the
    contrast becomes "did the word deliver drive into the slot the grammar
    predicted?", which is what a P600 is. Before the verb, both a verb (VP) and
    further subject material (ROLE_AGENT) are licensed, so there is no unique
    expectation and claiming one would be inventing structure.

    `verb_seen_before` MUST be the state BEFORE the word was consumed. Taking it
    after would make the verb itself expect an object slot and probe
    ROLE_PATIENT, which is the same class of error this function exists to fix.

    `object_open` comes from `_verb_takes_an_object`, the predicate built for the
    paper's empty-project detector (#24) -- this reuses that mechanism rather
    than adding a parallel one. Unknown verbs keep the slot open there, so an
    untrained parser behaves exactly as before.
    """
    if verb_seen_before and object_open:
        return ROLE_PATIENT
    return None


def _expected_slot_enabled() -> bool:
    """A/B seam, DEFAULT OFF. `ERP_EXPECTED_SLOT=1` dispatches on the predicted
    slot; the shipped default keeps the observed-category dispatch.

    ADOPTION WAS ATTEMPTED AND ROLLED BACK, AND THE BLOCKER IS NOT THIS CODE.
    Flipping the default failed 4 ERP tests, but THE FAILURES ARE
    ORDER-DEPENDENT and do not reproduce in isolation:

        pytest test_erp_metric_range.py                  3 passed, 1 xfailed
        pytest test_erp_calibration.py                   3 passed, 1 xpassed
        pytest test_erp_calibration.py test_erp_metric_range.py
                                                         6 passed, 1 xfailed
        pytest -k erp   (full selection)                 4 FAILED

    all with ERP_EXPECTED_SLOT=1. A cold single-arm process
    (research/experiments/erp_cold_vs_warm_arm.py) reads seed 11 exp raw p600
    AUC 1.000 on gram [0.988, 0.9923, 0.9881] vs catv [0.9927, 0.993, 0.9927],
    while the in-suite failure read gram [0.9932, 0.9928, 0.9945] vs catv
    [0.9927, 0.9925, 0.9924] -- a different parser state entirely.

    RESOLVED: IT IS THE BACKBONE CACHE, AND THE DISPATCH REALLY DOES INVERT ON A
    FRESHLY-TRAINED PARSER.

        warm cache, default path            62 passed     75-128s
        warm cache, ERP_EXPECTED_SLOT=1     62 passed
        COLD cache, default path            62 passed     430s
        COLD cache, ERP_EXPECTED_SLOT=1      4 FAILED     339s

    The DEFAULT path is consistent warm and cold, so the cache is not broadly
    poisoning results -- but a cached parser and a freshly-trained one differ in
    some structure that THIS dispatch reads and the shipped one does not. And
    the 10-seed A/B below ran entirely on `get_parser_cache().fork()`, i.e. on
    cached parsers, which is exactly why it looked good. AN A/B BUILT ON CACHED
    PARSERS IS EVIDENCE ABOUT CACHED PARSERS ONLY.

    Likely mechanism, unconfirmed: same family as the VP self-fiber finding --
    pre-grown pathways wire the neurons materialised at bootstrap and later
    training recruits DIFFERENT ones, so what ROLE_PATIENT can reach is
    training-path dependent. Fixing that is prior to re-testing this dispatch.

    THREE EXPLANATIONS RULED OUT on the way, recorded so they are not re-derived:
      * cross-test state leakage: all four precursor files together pass.
      * raw vs excess: A FALSE PREMISE I held briefly. `calibration.py` builds
        `separation["p600_auc"]` from `catv_p600_raw, gram_p600_raw` -- the RAW
        p600, the same quantity the tests rank. The two harnesses measure the
        SAME thing and still disagreed, which is what leaves the cache standing.
      * warm-up order in the A/B harness (obs first, exp second, one process):
        refuted -- the COLD exp arm reads 1.000, not 0.000.

    RUNTIME IS THE TELL: 75-128s is a disk hit, 340-430s is a retrain. If a
    result moves and the runtime jumped, suspect the substrate before the code.

        p600_auc   obs 0.9056 +/- 0.0268   exp 0.7167 +/- 0.0805
                   delta -0.1889 +/- 0.0627   (CI excludes zero)

        per-seed exp: 1.000, 0.667 x8, 0.833 -- none below chance, not constant

    The DROP is the point. The shipped 0.9056 is confounded: area identity alone
    reproduces that AUC with the condition held constant. Area-matched, the
    effect survives at 0.7167 with CI 0.636-0.797, clearly above the 0.5 null.
    So the P600 is REAL and was inflated by ~0.19 AUC.

    Pre-registered bar, all three met: arms area-match; every seed above chance
    (42 is the seed that inverted the last structural change); the violation arm
    is no longer constant (zero seed variance was the `afferent_energy` defect).

    CAVEAT ON GRANULARITY: 3 grammatical x 3 violation = 9 pairs, so AUC moves
    in steps of 1/9 and 8 of 10 seeds read exactly 6/9. The estimate is coarse
    by construction; widen the frame set before reading finer differences.
    """
    return ErpProtocol.from_environment().expected_slot


def anchored_p600_live(
    parser: "EmergentParser",
    core_area: str,
    role_area: str,
    *,
    subject_core: Optional[str] = None,
    n_settling: int = P600_SETTLING_ROUNDS,
) -> float:
    """Live-anchored P600 as a PRE-k-WTA ENERGY DEFICIT into the role area.

    The fixed core assemblies (plus the subject core and NUMBER when live) fire
    into the role area through their core->role pathway; ``input_drive`` reads
    the summed synaptic drive over the role area's candidate neurons BEFORE
    winner selection, normalized per candidate. A grammatical pairing traverses
    a trained pathway and delivers HIGH energy; a category violation routes a
    wrongly-typed core through an untrained pathway and delivers LESS. P600 is
    the DEFICIT ``1 - energy`` -- larger for violations.

    This replaces the settle-and-measure-winner-churn protocol, which reversed
    sign under ``norm_init`` (see module docstring / ``binding.input_drive``).
    ``input_drive`` owns the frozen()/fix/unfix and record_activation handling.
    ``n_settling`` is retained for call-site compatibility; pre-k-WTA energy is
    the immediate single-projection drive, so no settling loop is run.
    """
    brain = parser.brain
    if core_area not in brain.areas or role_area not in brain.areas:
        return 0.0

    # THE SOURCE SET MUST NOT DEPEND ON THE CONTRAST BEING MEASURED.
    #
    # This previously appended `subject_core` under the guard
    # `subject_core != core_area`. That condition IS the grammatical/violation
    # distinction: on a grammatical pairing the probed core IS the subject's
    # core, so nothing was appended; on a category violation the wrongly-typed
    # core differs from the subject's, so one EXTRA trained source fired. The
    # violation arm was therefore summing drive from more trained pathways than
    # the grammatical arm, and that surplus outweighed and REVERSED the effect
    # the metric exists to measure -- Cohen's d -2.4 where +1.9 was expected.
    #
    # It is the same family as the sign inversion fixed in #23 (the two arms
    # measured different AREAS) but a distinct instance: here the arms measure
    # the same area with a different NUMBER OF SUMMED SOURCES. Area-matching
    # alone does not catch it, which is why it survived that fix.
    #
    # `subject_core` is kept in the signature for call-site compatibility and is
    # deliberately unused: any re-introduction must fire it in BOTH arms.
    sources = [core_area]
    if (
        NUMBER in brain.areas
        and brain.areas[NUMBER].winners is not None
        and len(brain.areas[NUMBER].winners) > 0
    ):
        sources.append(NUMBER)

    try:
        drives = input_drive(brain, sources=sources, target_areas=[role_area])
    except (RuntimeError, IndexError, ValueError):
        return 0.0
    energy = float(drives.get(role_area, 0.0))
    return max(0.0, 1.0 - energy)


def measure_live_integration(
    parser: "EmergentParser",
    word: str,
    category: str,
    *,
    verb_seen: bool,
    subject_core: Optional[str] = None,
    readiness: Optional[ErpReadiness] = None,
    probe_depth: ProbeDepth = "calibration",
    verb_seen_before: Optional[bool] = None,
    object_open: bool = True,
) -> Tuple[float, str, float]:
    """P600 on live parse: phrase instability + live-anchored role settling.

    ``ERP_EXPECTED_SLOT=1`` dispatches the probed area on the slot the PARSE
    PREDICTS rather than the observed word's category, which is the only way to
    area-match the grammatical/violation contrast -- see `expected_role_area`.
    OFF by default: adoption was attempted and rolled back, see
    `_expected_slot_enabled`. The shipped default is CONFOUNDED and its
    magnitudes must not be quoted as effect sizes either.

    ``verb_seen_before`` is the pre-consumption state and is required for that
    path; callers that do not supply it keep the observed-category dispatch,
    so the flag silently does nothing rather than reading the wrong state.
    """
    readiness = readiness or assess_erp_readiness(parser)
    core = CATEGORY_TO_CORE.get(category)
    if core is None:
        return 0.0, VP, 1.0

    # The PHRASE areas must be matched too, not only the role area. Measured:
    # phrase_stability is PERFECTLY determined by which area is read (0.0049 on
    # every ROLE_PATIENT probe, 0.0000 on every VP probe, in ALL conditions), so
    # leaving `_phrase_areas_for_category` on the observed category would carry
    # the confound straight into the stability term and area-match only half the
    # metric. When the slot is predicted, the category it predicts is nominal.
    role_area = None
    phrase_category = category
    if _expected_slot_enabled() and verb_seen_before is not None:
        role_area = expected_role_area(
            verb_seen_before=verb_seen_before, object_open=object_open,
        )
        if role_area is not None:
            phrase_category = "NOUN"
    if role_area is None:
        role_area = structural_role_area(category, verb_seen=verb_seen)

    if not readiness.p600_ready:
        return 0.0, role_area, 1.0

    phrase_areas = areas_with_active_assembly(
        parser.brain,
        _phrase_areas_for_category(phrase_category, verb_seen=verb_seen),
    )
    stab_rounds = phrase_stability_rounds_for_depth(probe_depth)
    if probe_depth == "mining" and phrase_areas:
        phrase_areas = [role_area] if role_area in phrase_areas else phrase_areas[:1]
    readings = [
        phrase_stability(parser.brain, area, rounds=stab_rounds)
        for area in phrase_areas
    ]
    # THE HISTORICAL FALLBACK IS KEPT ON PURPOSE, AND IS NOW VISIBLE.
    #
    # An undefined reading (usually a dead self-fiber) has always entered this
    # mean as a hard 0.0, pulling stability down and p600 up -- a gap wearing
    # the costume of a finding. Dropping them instead is the CORRECT
    # aggregation, and `defined_values` below does exactly that.
    #
    # But it is not behaviour-preserving: measured, switching to the dropped
    # form moves every P600 magnitude and fails five ERP tests, including the
    # strict xfail that pins the metric's range. Changing every published number
    # is a measured change with its own A/B, not a side effect of a typing
    # refactor -- and today already produced two adoptions that had to be rolled
    # back for exactly that kind of unmeasured coupling.
    #
    # So `.or_else(0.0)` reproduces the old arithmetic EXACTLY while making the
    # choice explicit at the call site, and `stabilities_dropped` records what
    # the honest version would have discarded. Flip to `defined_values` behind a
    # measurement; the plumbing is already here.
    stabilities = [r.or_else(0.0) for r in readings]
    stabilities_dropped = sum(1 for r in readings if not r.defined)
    _ = defined_values  # the corrected aggregation, pending its own A/B
    mean_stability = (
        sum(stabilities) / len(stabilities) if stabilities else 1.0
    )
    phrase_instability = 1.0 - mean_stability

    anchored = anchored_p600_live(
        parser, core, role_area, subject_core=subject_core,
        n_settling=settling_rounds_for_depth(probe_depth),
    )

    p600 = N400_WEIGHT * phrase_instability + P600_WEIGHT * anchored
    if _ERP_DEBUG:
        dropped = ""
        if stabilities_dropped:
            # Name the reasons: a run where most areas are undefined is a probe
            # failure wearing the costume of a low-stability result.
            why = "; ".join(sorted({r.why for r in readings if not r.defined}))
            dropped = (f" DROPPED={stabilities_dropped}/{len(readings)} "
                       f"({why})")
        print(
            f"[P600] w={word!r} cat={category} role={role_area} "
            f"self_energy={mean_stability:.4f} instab={phrase_instability:.4f} "
            f"anchored_deficit={anchored:.4f} p600={p600:.4f}{dropped}",
        )
    return round(p600, 4), role_area, round(mean_stability, 4)


def _predicted_energy(brain, entry) -> float:
    """Mean PRE-k-WTA drive the settled context delivers to *entry*'s neurons.

    ``entry`` is a stored PREDICTION assembly (STABLE neuron IDs). A final
    recording projection into PREDICTION exposes the per-neuron pre-k-WTA input
    vector (compact-indexed); ``_compact_index`` maps the entry's stable IDs to
    compact positions so the drive landing on exactly the word's assembly can be
    read. High when the context predicts the word, low when it does not -- the
    word-specific counterpart of global pre-k-WTA energy (this package's robust
    N400 quantity), needed because two frames sharing a context deliver
    identical GLOBAL PREDICTION energy and only differ on the target word.
    """
    area = PREDICTION
    if area not in brain.areas:
        return 0.0
    engine = brain._engine_for(brain.areas[area])
    eng_areas = getattr(engine, "_areas", {})
    from_areas = [a for a in (CONTEXT, area) if a in eng_areas]
    prev_rec = getattr(brain, "record_activation", False)
    try:
        brain.record_activation = True
        result = engine.project_into(
            area,
            from_stimuli=[],
            from_areas=from_areas,
            plasticity_enabled=False,
            record_activation=True,
        )
    except (RuntimeError, IndexError, ValueError):
        return 0.0
    finally:
        brain.record_activation = prev_rec

    vec = getattr(result, "pre_kwta_inputs", None)
    if vec is None or len(vec) == 0:
        return 0.0
    vec = np.asarray(vec)
    entry_ids = np.asarray(entry.winners, dtype=np.int64)
    n2c = _compact_index(engine, area)
    if n2c is None:
        idx = [int(i) for i in entry_ids if 0 <= int(i) < len(vec)]
    else:
        idx = [
            n2c[int(i)]
            for i in entry_ids
            if int(i) in n2c and n2c[int(i)] < len(vec)
        ]
    if not idx:
        return 0.0
    return float(np.mean(vec[idx]))


def measure_lexical_surprise(
    parser: "EmergentParser",
    prefix: Tuple[str, ...],
    word: str,
    *,
    readiness: Optional[ErpReadiness] = None,
) -> float:
    """N400 adapter: pre-k-WTA ENERGY DEFICIT at the word's PREDICTION assembly.

    Replaces ``1 - overlap(context->PREDICTION, entry)``, a post-k-WTA overlap
    that reverses sign under ``norm_init``. The context is settled into
    PREDICTION exactly as before; the readout is then the mean pre-k-WTA drive
    landing on the word's stored assembly. A semantically/syntactically expected
    word has its neurons strongly driven (high energy => low N400); an anomaly
    does not (low energy => high N400). N400 = ``1 - energy`` is therefore
    larger for violations, the correct sign.
    """
    if not prefix:
        return 0.0
    readiness = readiness or assess_erp_readiness(parser)
    if not readiness.n400_ready:
        return 0.0
    if not hasattr(parser, "prediction_lexicon"):
        return 0.0

    parser._ensure_prediction_lexicon([word])
    entry = parser.prediction_lexicon.get(word)
    if entry is None:
        return 1.0

    brain = parser.brain
    prev_fid = brain.projection_fidelity
    brain.projection_fidelity = "exact"
    with probe_context(brain):
        try:
            parser._bootstrap_prediction_connectivity()
            parser.build_context_incremental(list(prefix), reset=True, direct=True)
            parser._clear_prediction_activity()
            infer = parser.inference_rounds
            brain.project({}, {CONTEXT: [PREDICTION]})
            if infer > 1:
                brain.project_rounds(
                    target=PREDICTION,
                    areas_by_stim={},
                    dst_areas_by_src_area={
                        CONTEXT: [PREDICTION],
                        PREDICTION: [PREDICTION],
                    },
                    rounds=infer - 1,
                )
            energy = _predicted_energy(brain, entry)
            n400 = 1.0 - energy
            if _ERP_DEBUG:
                print(
                    f"[N400] w={word!r} ctx={' '.join(prefix)!r} "
                    f"pred_energy={energy:.4f} n400={n400:.4f}",
                )
            return n400
        finally:
            brain.projection_fidelity = prev_fid


def measure_fresh_stimulus_integration(
    parser: "EmergentParser",
    word: str,
    category: str,
    *,
    verb_seen: bool = False,
    n_rounds: int = P600_SETTLING_ROUNDS,
) -> Tuple[float, str]:
    """Fresh-stimulus P600 probe (matches package anchored protocol)."""
    core = CATEGORY_TO_CORE.get(category)
    if core is None:
        return 0.0, VP

    role_area = structural_role_area(category, verb_seen=verb_seen)
    phon = parser.stim_map.get(word)
    if phon is None or core not in parser.brain.areas:
        return 0.0, role_area

    brain = parser.brain
    with probe_context(brain):
        try:
            brain.inhibit_areas([core, role_area])
            for _ in range(2):
                brain.project({phon: [core]}, {core: [core]})

            brain.inhibit_areas([role_area])
            brain.project({phon: [core, role_area]}, {core: [role_area]})
            winners = [set(int(w) for w in brain.areas[role_area].winners.tolist())]

            for _ in range(max(1, n_rounds - 1)):
                brain.project(
                    {},
                    {core: [role_area], role_area: [role_area, core]},
                )
                winners.append(
                    set(int(w) for w in brain.areas[role_area].winners.tolist()),
                )

            return mean_jaccard_instability(winners), role_area
        except (RuntimeError, IndexError, ValueError):
            return 1.0, role_area


measure_n400_surprise = measure_lexical_surprise
measure_integration_instability = measure_fresh_stimulus_integration
