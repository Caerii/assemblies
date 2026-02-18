"""
ERP Measurement Functions

Shared measurement primitives for N400 and P600 experiments.
These are the core building blocks that all sentence-processing
experiments use to measure energy, instability, and competition metrics.

measure_critical_word():
    Standard N400 + P600 measurement. Processes context, measures
    N400 energy at critical word, measures P600 instability in
    structural areas.

measure_agreement_word():
    Extended measurement with NUMBER co-projection. Also tracks
    subject core area, VP competition margin, and VP assembly
    for paired distance computation.
"""

import numpy as np
from typing import Dict, List, Any, Set

from research.experiments.metrics.instability import (
    compute_jaccard_instability,
    measure_p600_settling,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import (
    NOUN_CORE, VERB_CORE, ROLE_AGENT, ROLE_PATIENT, VP, NUMBER,
)
from src.assembly_calculus.ops import project


def measure_critical_word(
    parser: EmergentParser,
    context_words: List[str],
    critical_word: str,
    p600_areas: List[str],
    rounds: int,
    p600_settling_rounds: int = 5,
) -> Dict[str, Any]:
    """Process context words, then measure energy and settling at the critical word.

    N400 measurement: On the first recurrent projection step of the critical
    word into its core area, capture global pre-k-WTA energy. Self-recurrence
    from the subject noun's residual assembly (still active in NOUN_CORE)
    provides context-dependent facilitation.

    P600 measurement: After the word settles in its core area, fix the core
    assembly and run settling dynamics in structural areas. Measure assembly
    instability (primary P600 metric) and cumulative energy (secondary).

    Returns dict with:
      n400_energy: float  -- global energy in core area (N400 metric)
      core_instability: float  -- Jaccard instability in core during settling
      p600_instability: {area: float}  -- instability per structural area
      p600_cumulative_energy: {area: float}  -- cumulative energy per area
      p600_mean_instability: float  -- mean instability across active areas
      core_area: str  -- which core area the word projected into
    """
    engine = parser.brain._engine
    brain = parser.brain

    # Clear all areas to start fresh for each sentence
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    # Process context words normally (builds sentence context)
    for word in context_words:
        core = parser._word_core_area(word)
        phon = parser.stim_map.get(word)
        if phon is not None:
            project(brain, phon, core, rounds=rounds)

    # Now measure the critical word
    crit_core = parser._word_core_area(critical_word)
    crit_phon = parser.stim_map.get(critical_word)
    if crit_phon is None:
        return {
            "n400_energy": 0.0,
            "p600_instability": {a: 0.0 for a in p600_areas},
            "p600_cumulative_energy": {a: 0.0 for a in p600_areas},
            "p600_mean_instability": 0.0,
            "core_area": crit_core,
        }

    # --- N400 + core instability: manual per-round projection ---
    # Track winners at every round to compute core-area Jaccard instability
    core_round_winners = []

    # Round 1: stimulus only (no recurrence), just like normal project()
    brain.project({crit_phon: [crit_core]}, {})
    core_round_winners.append(
        set(int(w) for w in brain.areas[crit_core].winners))

    # Round 2: stimulus + self-recurrence WITH record_activation
    # This is where context (residual assembly from subject noun) provides
    # facilitation via Hebbian-trained weights within the core area
    n400_result = engine.project_into(
        crit_core,
        from_stimuli=[crit_phon],
        from_areas=[crit_core],
        plasticity_enabled=True,
        record_activation=True,
    )
    n400_energy = n400_result.pre_kwta_total

    # Sync winners back to brain area
    engine.set_winners(crit_core, n400_result.winners)
    brain.areas[crit_core].winners = n400_result.winners
    brain.areas[crit_core].w = n400_result.num_ever_fired
    core_round_winners.append(
        set(int(w) for w in n400_result.winners))

    # Remaining rounds: stimulus + self-recurrence, track winners each round
    for _r in range(2, rounds):
        rr = engine.project_into(
            crit_core,
            from_stimuli=[crit_phon],
            from_areas=[crit_core],
            plasticity_enabled=True,
            record_activation=False,
        )
        engine.set_winners(crit_core, rr.winners)
        brain.areas[crit_core].winners = rr.winners
        brain.areas[crit_core].w = rr.num_ever_fired
        core_round_winners.append(
            set(int(w) for w in rr.winners))

    # Core-area instability: how much the assembly changes during settling
    # Trained words (via self-recurrence) should converge faster
    core_instability = compute_jaccard_instability(core_round_winners)

    # --- P600: settling dynamics in structural areas ---
    # Fix core assembly (stable lexical representation by ~600ms)
    engine.fix_assembly(crit_core)

    # Measure settling instability across structural areas
    p600_results = measure_p600_settling(
        engine, brain, crit_core, p600_areas, p600_settling_rounds,
    )

    # Unfix core assembly
    engine.unfix_assembly(crit_core)

    # Extract metrics
    p600_instability = {a: p600_results[a]["instability"] for a in p600_areas}
    p600_cum_energy = {
        a: p600_results[a]["cumulative_energy"] for a in p600_areas
    }

    # Mean instability across areas with non-zero values
    nonzero_inst = [v for v in p600_instability.values() if v > 0]
    mean_instability = float(np.mean(nonzero_inst)) if nonzero_inst else 0.0

    return {
        "n400_energy": n400_energy,
        "core_instability": core_instability,
        "p600_instability": p600_instability,
        "p600_cumulative_energy": p600_cum_energy,
        "p600_mean_instability": mean_instability,
        "core_area": crit_core,
    }


def measure_agreement_word(
    parser: EmergentParser,
    context_words: List[str],
    critical_word: str,
    p600_areas: List[str],
    rounds: int,
    p600_settling_rounds: int = 5,
) -> Dict[str, Any]:
    """Process context with NUMBER co-projection, then measure at critical word.

    Like measure_critical_word(), but also projects number stimuli into the
    NUMBER area during context processing.

    Key difference: at verb position, we fix the subject's NOUN_CORE assembly
    alongside VERB_CORE and NUMBER during settling. This matches the
    consolidation pattern:
      - VP consolidation: NOUN_CORE + VERB_CORE + NUMBER -> VP
      - ROLE consolidation: NOUN_CORE + NUMBER -> ROLE

    Returns dict with:
      n400_energy: float
      core_instability: float
      p600_instability: {area: float}
      p600_cumulative_energy: {area: float}
      p600_mean_instability: float
      vp_margin: float  -- gap between k-th winner and (k+1)-th loser
      vp_winners: set   -- VP assembly for paired distance computation
      core_area: str
      subject_core: str or None
    """
    engine = parser.brain._engine
    brain = parser.brain

    number_stims = {"SG": "number_SG", "PL": "number_PL"}

    # Clear all areas
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    # Process context words with NUMBER co-projection.
    # Track last noun's core area as subject (for fixing during settling).
    subject_core = None
    for word in context_words:
        core = parser._word_core_area(word)
        phon = parser.stim_map.get(word)
        if phon is not None:
            project(brain, phon, core, rounds=rounds)

        # Co-project number for content words
        grounding = parser.word_grounding.get(word)
        if grounding and grounding.dominant_modality in ("visual", "motor"):
            num = parser.detect_number(word)
            num_stim = number_stims[num]
            brain.project({num_stim: [NUMBER]}, {})
            if rounds > 1:
                brain.project_rounds(
                    target=NUMBER,
                    areas_by_stim={num_stim: [NUMBER]},
                    dst_areas_by_src_area={NUMBER: [NUMBER]},
                    rounds=rounds - 1,
                )
            # Track noun core as subject
            if grounding.dominant_modality == "visual":
                subject_core = core

    # Now measure the critical word
    crit_core = parser._word_core_area(critical_word)
    crit_phon = parser.stim_map.get(critical_word)
    if crit_phon is None:
        return {
            "n400_energy": 0.0,
            "core_instability": 0.0,
            "p600_instability": {a: 0.0 for a in p600_areas},
            "p600_cumulative_energy": {a: 0.0 for a in p600_areas},
            "p600_mean_instability": 0.0,
            "vp_margin": 0.0,
            "vp_winners": set(),
            "core_area": crit_core,
            "subject_core": subject_core,
        }

    # NOTE: We do NOT project the critical word's number into NUMBER.
    # The NUMBER area retains the SUBJECT's number from context.
    # For grammatical sentences (SG subj + SG verb), the SG NUMBER
    # signal is compatible with SG-consolidated structural patterns.
    # For agreement violations (PL subj + SG verb), the PL NUMBER
    # signal CONFLICTS with the SG-verb consolidated patterns in VP.

    # --- N400 + core instability: manual per-round projection ---
    core_round_winners = []

    # Round 1: stimulus only
    brain.project({crit_phon: [crit_core]}, {})
    core_round_winners.append(
        set(int(w) for w in brain.areas[crit_core].winners))

    # Round 2: stimulus + self-recurrence WITH record_activation
    n400_result = engine.project_into(
        crit_core,
        from_stimuli=[crit_phon],
        from_areas=[crit_core],
        plasticity_enabled=True,
        record_activation=True,
    )
    n400_energy = n400_result.pre_kwta_total

    engine.set_winners(crit_core, n400_result.winners)
    brain.areas[crit_core].winners = n400_result.winners
    brain.areas[crit_core].w = n400_result.num_ever_fired
    core_round_winners.append(
        set(int(w) for w in n400_result.winners))

    # Remaining rounds
    for _r in range(2, rounds):
        rr = engine.project_into(
            crit_core,
            from_stimuli=[crit_phon],
            from_areas=[crit_core],
            plasticity_enabled=True,
            record_activation=False,
        )
        engine.set_winners(crit_core, rr.winners)
        brain.areas[crit_core].winners = rr.winners
        brain.areas[crit_core].w = rr.num_ever_fired
        core_round_winners.append(
            set(int(w) for w in rr.winners))

    core_instability = compute_jaccard_instability(core_round_winners)

    # --- P600: settling dynamics in structural areas ---
    # Fix critical word's core and NUMBER
    engine.fix_assembly(crit_core)
    engine.fix_assembly(NUMBER)

    # Also fix subject's core if available and different from critical word.
    # At verb position: subject_core=NOUN_CORE, crit_core=VERB_CORE -> both fixed.
    # At object position: subject_core=NOUN_CORE but crit overwrites it -> skip.
    has_subject = subject_core is not None and subject_core != crit_core
    if has_subject:
        engine.fix_assembly(subject_core)

    # Measure each area group with source configuration matching consolidation:
    # VP consolidation was: NOUN_CORE + VERB_CORE + NUMBER -> VP
    # ROLE consolidation was: NOUN_CORE + NUMBER -> ROLE
    vp_areas = [a for a in p600_areas if a == VP]
    role_areas = [a for a in p600_areas if a in (ROLE_AGENT, ROLE_PATIENT)]

    p600_results = {}

    if vp_areas:
        extra = [NUMBER]
        if has_subject:
            extra.append(subject_core)
        p600_results.update(measure_p600_settling(
            engine, brain, crit_core, vp_areas, p600_settling_rounds,
            additional_source_areas=extra,
        ))

    if role_areas:
        if has_subject:
            # Use subject_core as primary (matches NOUN_CORE + NUMBER -> ROLE)
            p600_results.update(measure_p600_settling(
                engine, brain, subject_core, role_areas, p600_settling_rounds,
                additional_source_areas=[NUMBER],
            ))
        else:
            # Fallback for object position: crit_core + NUMBER
            p600_results.update(measure_p600_settling(
                engine, brain, crit_core, role_areas, p600_settling_rounds,
                additional_source_areas=[NUMBER],
            ))

    # --- VP competition metrics (captures agreement-specific P600) ---
    vp_margin = 0.0
    vp_winners = set()
    if VP in p600_areas:
        brain.inhibit_areas([VP])  # Fresh projection (no self-recurrence)
        vp_sources = [crit_core, NUMBER]
        if has_subject:
            vp_sources.append(subject_core)
        comp_result = engine.project_into(
            VP,
            from_stimuli=[],
            from_areas=vp_sources,
            plasticity_enabled=False,
            record_activation=True,
        )
        vp_winners = set(int(w) for w in comp_result.winners)
        if (comp_result.pre_kwta_inputs is not None
                and len(comp_result.pre_kwta_inputs) > len(comp_result.winners)):
            sorted_act = np.sort(comp_result.pre_kwta_inputs)[::-1]
            k = len(comp_result.winners)
            vp_margin = float(sorted_act[k - 1] - sorted_act[k])

    engine.unfix_assembly(crit_core)
    engine.unfix_assembly(NUMBER)
    if has_subject:
        engine.unfix_assembly(subject_core)

    # Extract metrics
    p600_instability = {a: p600_results[a]["instability"] for a in p600_areas}
    p600_cum_energy = {
        a: p600_results[a]["cumulative_energy"] for a in p600_areas
    }

    nonzero_inst = [v for v in p600_instability.values() if v > 0]
    mean_instability = float(np.mean(nonzero_inst)) if nonzero_inst else 0.0

    return {
        "n400_energy": n400_energy,
        "core_instability": core_instability,
        "p600_instability": p600_instability,
        "p600_cumulative_energy": p600_cum_energy,
        "p600_mean_instability": mean_instability,
        "vp_margin": vp_margin,
        "vp_winners": vp_winners,
        "core_area": crit_core,
        "subject_core": subject_core,
    }
