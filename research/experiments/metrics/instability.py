"""
Assembly Instability Metrics — P600 Analogue

Provides metrics for measuring structural integration difficulty,
the Assembly Calculus analogue of the P600 ERP component.

Core insight: P600 reflects how well a structural pathway can sustain
an assembly pattern without continued external (stimulus) input.

Two main metrics:

  compute_jaccard_instability: Raw sum of (1 - Jaccard) across rounds.
      Works when both conditions produce nonzero signal, but fails for
      untrained pathways that have zero weights (paradoxically stable).

  compute_anchored_instability: Principled P600 metric that solves the
      zero-signal problem. Phase A primes the role area with stimulus
      co-projection; Phase B settles with area-to-area connections only.
      Trained pathways sustain the pattern (low instability = low P600);
      untrained pathways cannot (high instability = high P600).

References:
  - Vosse & Kempen 2000: P600 = settling time
  - Brouwer & Crocker 2017: P600 = integration update cost
  - Hagoort 2005: P600 = unification difficulty
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


def compute_anchored_instability(
    brain,
    word: str,
    core_area: str,
    role_area: str,
    n_settling_rounds: int = 10,
    activate_rounds: int = 3,
) -> Dict[str, Any]:
    """Compute anchored instability — principled P600 metric.

    Addresses the problem where untrained pathways have zero signal and thus
    paradoxically show zero Jaccard instability. By first anchoring with a
    stimulus-driven co-projection (Phase A), we create an initial pattern that
    trained pathways can sustain but untrained pathways cannot.

    Phase A (1 round): Co-project stimulus + core -> role to create initial
        pattern via both stimulus and area-to-area connections.
    Phase B (n_settling_rounds - 1 rounds): Settle with area-to-area only
        (core <-> role bidirectional). No stimulus.

    Instability is measured only over Phase B rounds, capturing how well the
    structural pathway can sustain the initial pattern without external input.

    Trained pathways: sustain pattern -> low instability -> low P600
    Untrained pathways: can't sustain -> high instability -> high P600

    Args:
        brain: Brain instance (plasticity must be OFF before calling).
        word: Word to bind (stimulus name will be PHON_{word}).
        core_area: Source area with active word assembly (e.g. "NOUN_CORE").
        role_area: Target role area (e.g. "ROLE_PATIENT").
        n_settling_rounds: Total rounds (Phase A + Phase B).
        activate_rounds: Rounds to activate word in core area before anchoring.

    Returns:
        Dict with:
          - instability: float — anchored instability score
          - round_winners: list of sets — per-round winners (all phases)
    """
    # Activate word in core area
    brain.inhibit_areas([core_area])
    for _ in range(activate_rounds):
        brain.project({f"PHON_{word}": [core_area]}, {core_area: [core_area]})

    # Phase A: one round with stimulus co-projection to create anchored pattern
    brain.inhibit_areas([role_area])
    brain.project(
        {f"PHON_{word}": [core_area, role_area]},
        {core_area: [role_area]},
    )
    all_round_winners = [set(brain.areas[role_area].winners.tolist())]

    # Phase B: settle without stimulus (area-to-area only)
    for _ in range(n_settling_rounds - 1):
        brain.project(
            {},
            {core_area: [role_area],
             role_area: [role_area, core_area]},
        )
        all_round_winners.append(
            set(brain.areas[role_area].winners.tolist()))

    # Instability over Phase B rounds only (index 1 onward includes the
    # transition from Phase A -> first Phase B round, which is informative)
    instability = compute_jaccard_instability(all_round_winners)

    return {
        "instability": instability,
        "round_winners": all_round_winners,
    }


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
