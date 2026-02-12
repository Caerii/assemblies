"""
Prediction Error — Cosine Similarity Between Recurrence and Target

N400 as prediction error: how well does the prime's recurrence prediction
(via core->core Hebbian weights) align with the target's assembly signature?

This metric captures the "prediction error" interpretation of the N400
(Nour Eddine et al. 2024), though empirically it shows weak/null effects
compared to global energy (Path 1c).

References:
  - Nour Eddine et al. 2024: N400 = lexico-semantic prediction error
  - research/claims/N400_GLOBAL_ENERGY.md (Path 2)
"""

from typing import Dict, Optional

import numpy as np

from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.ops import project


def _compute_weight_projection(
    engine, source_area: str, target_area: str,
    source_winners: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute predicted input to target from source's connection weights.

    Returns a float32 vector of length target_w, or None if connections
    don't exist or are empty.
    """
    if not hasattr(engine, '_area_conns'):
        return None

    conn_dict = engine._area_conns.get(source_area)
    if conn_dict is None:
        return None
    conn = conn_dict.get(target_area)
    if conn is None:
        return None

    # Numpy engine: Connectome with .weights ndarray
    if hasattr(conn, 'weights') and hasattr(conn.weights, 'shape'):
        w = conn.weights
        if w.ndim == 2 and w.shape[0] > 0 and w.shape[1] > 0:
            valid = source_winners[source_winners < w.shape[0]]
            if len(valid) > 0:
                result = w[valid, :].sum(axis=0)
                result = np.asarray(result, dtype=np.float32).flatten()
                return result
    # Torch engine: CSRConn with accumulate_rows
    elif hasattr(conn, 'accumulate_rows'):
        try:
            import torch
            src_t = torch.tensor(
                source_winners, dtype=torch.long,
                device=getattr(conn, '_device', 'cpu'))
            ncols = getattr(conn, '_ncols', 0)
            if ncols > 0 and src_t.numel() > 0:
                result = conn.accumulate_rows(src_t, ncols)
                return result.cpu().numpy().astype(np.float32)
        except Exception:
            pass
    return None


def measure_prediction_error(
    parser: EmergentParser,
    prime: str,
    target: str,
    core_lexicon: Dict,
    rounds: int,
) -> Dict[str, float]:
    """N400 as prediction error: ||recurrence_prediction - target_signature||.

    Uses core->core recurrence weights as the prediction mechanism:
    1. Clear core, project prime (activates prime's assembly)
    2. Compute prediction: sum core->core weight rows for prime's active
       neurons -> this is the recurrence input pattern the prime "predicts"
    3. Compare prediction with target's known assembly signature
    4. prediction_error = 1 - cosine(prediction, target_indicator)

    The core->core weights ARE trained (Hebbian learning during training
    strengthens connections between co-occurring assemblies).
    """
    target_info = core_lexicon[target]
    target_core = target_info["core_area"]
    target_indices = target_info["compact_winners"]
    prime_core = parser._word_core_area(prime)
    engine = parser.brain._engine

    # Clear activation (keep trained weights)
    parser.brain.inhibit_areas([target_core])
    if prime_core != target_core:
        parser.brain.inhibit_areas([prime_core])

    # Project prime to activate its assembly
    project(parser.brain, parser.stim_map[prime], prime_core, rounds=rounds)

    # Read prime's active winners
    prime_winners = np.array(
        parser.brain.areas[prime_core].winners, dtype=np.int64)

    # Compute prediction: what does the prime's assembly predict via
    # core->core recurrence weights?
    prediction = _compute_weight_projection(
        engine, target_core, target_core, prime_winners)

    if prediction is None or len(prediction) == 0:
        return {
            "prediction_error": float('nan'),
            "cosine_sim": float('nan'),
            "pred_magnitude": 0.0,
        }

    # Build target signature vector: 1.0 at target's assembly positions, 0 elsewhere
    dim = len(prediction)
    target_sig = np.zeros(dim, dtype=np.float64)
    valid_idx = target_indices[target_indices < dim]
    if len(valid_idx) == 0:
        return {
            "prediction_error": float('nan'),
            "cosine_sim": float('nan'),
            "pred_magnitude": 0.0,
        }
    target_sig[valid_idx] = 1.0

    pred = prediction.astype(np.float64)
    pred_mag = float(np.linalg.norm(pred))

    # Cosine similarity: how well does the recurrence prediction align
    # with the target's assembly pattern?
    if pred_mag > 0:
        cosine = float(np.dot(pred, target_sig) / (
            pred_mag * np.linalg.norm(target_sig)))
    else:
        cosine = 0.0

    # Also: mean predicted input at target positions vs non-target positions
    pred_at_target = float(np.mean(pred[valid_idx])) if len(valid_idx) > 0 else 0.0
    non_target_mask = np.ones(dim, dtype=bool)
    non_target_mask[valid_idx] = False
    pred_at_other = float(np.mean(pred[non_target_mask])) if non_target_mask.any() else 0.0

    return {
        "prediction_error": 1.0 - cosine,  # lower = better prediction = smaller N400
        "cosine_sim": cosine,
        "pred_magnitude": pred_mag,
        "pred_at_target": pred_at_target,
        "pred_at_other": pred_at_other,
    }
