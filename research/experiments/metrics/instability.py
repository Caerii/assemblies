"""
Assembly Instability — P600 Analogue

The P600 in Assembly Calculus maps to assembly instability during
structural integration: sum(1 - Jaccard(winners[r], winners[r-1]))
across settling rounds in structural areas.

Trained pathways (consolidated Hebbian connections) produce stable
assemblies that converge quickly (low instability = small P600).
Random pathways (bootstrapped baseline weights) produce oscillating
assemblies (high instability = large P600).

References:
  - Vosse & Kempen 2000: P600 = settling time
  - Brouwer & Crocker 2017: P600 = integration update cost
  - Hagoort 2005: P600 = unification difficulty
  - research/plans/P600_REANALYSIS.md
"""

from typing import Dict, List, Any, Set


def compute_jaccard_instability(round_winners: List[Set[int]]) -> float:
    """Compute sum of (1 - Jaccard) across consecutive rounds.

    Args:
        round_winners: List of winner sets, one per round.

    Returns:
        Total instability: sum of (1 - Jaccard) for consecutive pairs.
        Higher values mean more instability (assembly changing more).
    """
    instability = 0.0
    for i in range(1, len(round_winners)):
        prev_set = round_winners[i - 1]
        curr_set = round_winners[i]
        union = prev_set | curr_set
        if len(union) > 0:
            jaccard = len(prev_set & curr_set) / len(union)
        else:
            jaccard = 1.0  # both empty = stable
        instability += (1.0 - jaccard)
    return instability


def measure_p600_settling(
    engine,
    brain,
    core_area: str,
    structural_areas: List[str],
    n_rounds: int = 5,
    additional_source_areas: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Measure assembly instability during structural integration.

    For each structural area, runs n_rounds of integration
    (core -> area + self-recurrence), recording winners at each round.

    PRIMARY P600 metric: Assembly instability
      sum(1 - Jaccard(winners[r], winners[r-1])) over rounds
      Trained pathway (consolidated): converges quickly -> low instability
      Random pathway (bootstrapped only): oscillates -> high instability
      Requires consolidation pass to create trained/untrained asymmetry.

    Secondary metric: Cumulative energy
      sum(pre_kwta_total) across rounds
      Note: After consolidation, energy is REVERSED (trained > random),
      so this metric only works without consolidation.

    Args:
        core_area: Primary source area (must be FIXED before calling).
        structural_areas: Target areas to measure settling in.
        n_rounds: Number of settling rounds.
        additional_source_areas: Optional extra fixed source areas to project
            from (e.g., NUMBER for number-aware settling). These must also
            be fixed by the caller. Default None for backward compatibility.

    Plasticity is OFF (measurement probe, not learning).
    """
    source_areas = [core_area]
    if additional_source_areas:
        source_areas.extend(additional_source_areas)

    results = {}

    for struct_area in structural_areas:
        if struct_area not in brain.areas:
            results[struct_area] = {
                "instability": 0.0,
                "cumulative_energy": 0.0,
                "round_energies": [],
            }
            continue

        # Clear structural area activation (keep weights)
        brain.inhibit_areas([struct_area])

        round_winners = []
        round_energies = []

        for r in range(n_rounds):
            try:
                result = engine.project_into(
                    struct_area,
                    from_stimuli=[],
                    from_areas=source_areas + [struct_area],
                    plasticity_enabled=False,
                    record_activation=True,
                )

                # Sync winners for next round's self-recurrence
                engine.set_winners(struct_area, result.winners)
                brain.areas[struct_area].winners = result.winners
                brain.areas[struct_area].w = result.num_ever_fired

                round_winners.append(set(int(w) for w in result.winners))
                energy = (result.pre_kwta_total
                          if result.pre_kwta_total is not None else 0.0)
                round_energies.append(energy)
            except (IndexError, KeyError, ValueError):
                round_winners.append(set())
                round_energies.append(0.0)

        # Compute instability using shared helper
        instability = compute_jaccard_instability(round_winners)
        cumulative_energy = sum(round_energies)

        results[struct_area] = {
            "instability": instability,
            "cumulative_energy": cumulative_energy,
            "round_energies": round_energies,
        }

    return results
