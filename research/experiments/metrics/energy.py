"""
Global Pre-k-WTA Energy — N400 Analogue

The N400 in Assembly Calculus maps to global pre-k-WTA energy:
sum(all_inputs) across all neurons before winner-take-all selection.

Related primes reduce total energy (smaller N400) because Hebbian-trained
pathways create redundant activation with the upcoming target. Unrelated
primes produce additive, non-redundant activation (larger N400).

References:
  - Kutas & Federmeier 2011: N400 = ease of semantic memory access
  - Nour Eddine et al. 2024: N400 = lexico-semantic prediction error
  - research/claims/N400_GLOBAL_ENERGY.md
"""

from typing import Dict

import numpy as np

from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.ops import project


def measure_pre_kwta_activation(
    parser: EmergentParser,
    prime: str,
    target: str,
    core_lexicon: Dict,
    rounds: int,
) -> Dict[str, float]:
    """Measure total pre-k-WTA input to target's known assembly neurons.

    1. Clear core activation (keep trained Hebbian weights!)
    2. Project prime -> core (using trained weights + plasticity)
    3. Project target with record_activation=True
    4. Index pre_kwta_inputs at target's known neuron positions
    5. Return mean and sum of those inputs

    CRITICAL: We do NOT reset_area_connections, which would destroy trained
    Hebbian weights.  The trained weights encode semantic structure: shared
    ANIMAL features between dog/cat strengthen connections between their
    assembly neurons.  Resetting them leaves only stimulus input, which
    is identical regardless of prime.
    """
    target_info = core_lexicon[target]
    target_core = target_info["core_area"]
    target_indices = target_info["compact_winners"]
    prime_core = parser._word_core_area(prime)

    engine = parser.brain._engine

    # Clear activation only — keep trained Hebbian weights!
    parser.brain.inhibit_areas([target_core])
    if prime_core != target_core:
        parser.brain.inhibit_areas([prime_core])

    # Project prime using trained weights (this activates prime's assembly,
    # and Hebbian learning further strengthens prime-specific pathways)
    project(parser.brain, parser.stim_map[prime], prime_core, rounds=rounds)

    # Single target projection step with activation recording.
    # Recurrence from prime's active assembly through trained Hebbian weights
    # provides extra input to neurons that share features with the prime.
    phon = parser.stim_map[target]
    from_areas = [target_core] if prime_core == target_core else []
    result = engine.project_into(
        target_core,
        from_stimuli=[phon],
        from_areas=from_areas,
        plasticity_enabled=False,
        record_activation=True,
    )

    pre_kwta = result.pre_kwta_inputs
    if pre_kwta is None:
        return {"mean_input": 0.0, "max_input": 0.0,
                "global_energy": 0.0, "target_rank_frac": 0.0,
                "n_valid": 0}

    # Index into known assembly neuron positions
    valid_idx = target_indices[target_indices < len(pre_kwta)]
    if len(valid_idx) == 0:
        return {"mean_input": 0.0, "max_input": 0.0,
                "global_energy": 0.0, "target_rank_frac": 0.0,
                "n_valid": 0}

    target_inputs = pre_kwta[valid_idx]

    # Metric 1: Mean input at target assembly neuron positions
    mean_input = float(np.mean(target_inputs))

    # Metric 2: Max input at target positions (best-activated target neuron)
    max_input = float(np.max(target_inputs))

    # Metric 3: Global energy (sum of all pre-k-WTA inputs)
    global_energy = float(np.sum(pre_kwta))

    # Metric 4: What fraction of top-k are target assembly neurons?
    # Sort all neuron indices by their pre-k-WTA input (descending)
    k = len(target_indices)
    top_k_indices = set(np.argpartition(pre_kwta, -k)[-k:].tolist())
    target_set = set(valid_idx.tolist())
    overlap_count = len(top_k_indices & target_set)
    target_rank_frac = overlap_count / max(k, 1)

    return {
        "mean_input": mean_input,
        "max_input": max_input,
        "global_energy": global_energy,
        "target_rank_frac": target_rank_frac,
        "n_valid": int(len(valid_idx)),
    }
