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


def _self_recurrent_energy(brain, area: str) -> float:
    """Normalized self-recurrent PRE-k-WTA energy of the assembly in *area*.

    The assembly projects back into its own area; the summed drive over
    candidate neurons before winner selection, divided by area size, measures
    how strongly the assembly reinforces itself. A well-integrated assembly
    scores high, a weakly-bound one low.

    The area is deliberately NOT fixed: the sparse engine short-circuits a
    projection into a FIXED target and returns before inputs are summed (see
    ``binding.bind``), which would leave ``pre_kwta_total`` unrecorded. Reading
    energy therefore requires the free projection. Plasticity is off (frozen),
    so this reads without reshaping the connectome.
    """
    if area not in brain.areas:
        return 0.0
    winners = brain.areas[area].winners
    if winners is None or len(winners) == 0:
        return 0.0
    prev_rec = getattr(brain, "record_activation", False)
    with brain.frozen():
        brain.record_activation = True
        try:
            brain.project({}, {area: [area]})
            totals = getattr(brain, "last_pre_kwta_totals", {}) or {}
            w = max(int(brain.areas[area].w), 1)
            return float(totals.get(area, 0.0)) / w
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
) -> float:
    """Phrase integration as normalized self-recurrent PRE-k-WTA energy.

    Replaces the recurrent winner-overlap churn (``overlap / k``), which
    reverses sign under ``norm_init`` because it is a post-k-WTA quantity.
    Higher = better-integrated phrase; ``measure_live_integration`` reads
    ``1 - stability`` as the (deficit) phrase-instability term. ``rounds`` /
    ``k`` are retained for call-site compatibility but no longer used: energy
    is a single-projection quantity, not a settling one.
    """
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
    if category == "VERB":
        return VP
    if category in ("NOUN", "PRON"):
        return ROLE_PATIENT if verb_seen else ROLE_AGENT
    return VP


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
) -> Tuple[float, str, float]:
    """P600 on live parse: phrase instability + live-anchored role settling."""
    readiness = readiness or assess_erp_readiness(parser)
    core = CATEGORY_TO_CORE.get(category)
    if core is None:
        return 0.0, VP, 1.0

    role_area = structural_role_area(category, verb_seen=verb_seen)

    if not readiness.p600_ready:
        return 0.0, role_area, 1.0

    phrase_areas = areas_with_active_assembly(
        parser.brain,
        _phrase_areas_for_category(category, verb_seen=verb_seen),
    )
    stab_rounds = phrase_stability_rounds_for_depth(probe_depth)
    if probe_depth == "mining" and phrase_areas:
        phrase_areas = [role_area] if role_area in phrase_areas else phrase_areas[:1]
    stabilities = [
        phrase_stability(parser.brain, area, rounds=stab_rounds)
        for area in phrase_areas
    ]
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
        print(
            f"[P600] w={word!r} cat={category} role={role_area} "
            f"self_energy={mean_stability:.4f} instab={phrase_instability:.4f} "
            f"anchored_deficit={anchored:.4f} p600={p600:.4f}",
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
    with brain.frozen():
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
    with brain.frozen():
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
