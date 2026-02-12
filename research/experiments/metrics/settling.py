"""
Settling Dynamics — Cumulative Pre-k-WTA Energy Across Rounds

Models the Cheyette & Plaut (2017) transient over-activation account
of the N400: cumulative energy across multiple projection rounds
captures the total "work" of settling into a stable assembly.

Related primes reduce cumulative energy (fewer rounds to settle,
lower energy per round) compared to unrelated primes.

References:
  - Cheyette & Plaut 2017: N400 = transient over-activation during settling
  - research/claims/N400_GLOBAL_ENERGY.md (Path 3)
"""

from typing import Dict, Any

import numpy as np

from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.ops import project


def measure_settling_dynamics(
    parser: EmergentParser,
    prime: str,
    target: str,
    core_lexicon: Dict,
    max_rounds: int = 10,
) -> Dict[str, Any]:
    """Track total activation energy across rounds during target processing.

    1. Project prime -> core (establishes prime context)
    2. Project target round-by-round, recording pre_kwta_total each round
    3. Compute cumulative energy, peak, overshoot
    """
    target_info = core_lexicon[target]
    target_core = target_info["core_area"]
    prime_core = parser._word_core_area(prime)
    engine = parser.brain._engine

    # Clear activation only — keep trained Hebbian weights
    parser.brain.inhibit_areas([target_core])
    if prime_core != target_core:
        parser.brain.inhibit_areas([prime_core])
    project(parser.brain, parser.stim_map[prime], prime_core,
            rounds=max_rounds)

    # Project target round-by-round with activation recording
    phon = parser.stim_map[target]
    round_data = []

    for r in range(max_rounds):
        if r == 0:
            from_areas = [target_core] if prime_core == target_core else []
            result = engine.project_into(
                target_core,
                from_stimuli=[phon],
                from_areas=from_areas,
                plasticity_enabled=True,
                record_activation=True,
            )
        else:
            result = engine.project_into(
                target_core,
                from_stimuli=[phon],
                from_areas=[target_core],
                plasticity_enabled=True,
                record_activation=True,
            )

        # Apply result so next round sees these winners
        engine.set_winners(target_core, result.winners)
        parser.brain.areas[target_core].winners = result.winners
        parser.brain.areas[target_core].w = result.num_ever_fired

        round_data.append({
            "round": r,
            "pre_kwta_total": result.pre_kwta_total,
            "winner_activation": result.total_activation,
            "mean_winner_act": (
                result.total_activation / max(len(result.winners), 1)),
        })

    # Compute settling metrics
    pre_kwta_totals = [d["pre_kwta_total"] for d in round_data]
    winner_acts = [d["winner_activation"] for d in round_data]

    cumulative_energy = float(np.sum(pre_kwta_totals))
    peak = float(np.max(pre_kwta_totals)) if pre_kwta_totals else 0.0
    final = pre_kwta_totals[-1] if pre_kwta_totals else 0.0
    overshoot = peak - final

    cumulative_winner = float(np.sum(winner_acts))
    peak_winner = float(np.max(winner_acts)) if winner_acts else 0.0
    final_winner = winner_acts[-1] if winner_acts else 0.0

    return {
        "cumulative_energy": cumulative_energy,
        "peak_activation": peak,
        "final_activation": final,
        "transient_overshoot": overshoot,
        "cumulative_winner_energy": cumulative_winner,
        "peak_winner_activation": peak_winner,
        "final_winner_activation": final_winner,
        "round_data": round_data,
    }
