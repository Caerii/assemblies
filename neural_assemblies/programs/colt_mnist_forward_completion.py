"""
Forward encoding from partial LOW cues (AC-native generative completion).

``pattern_complete`` on explicit MNIST recovers ~93–99% after E2 metric fixes.
The viable occlusion-native path remains **partial LOW → project → HIGH**.
"""

from __future__ import annotations

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs.colt_mnist_absence import apply_absence_mask, apply_bundle_absence
from neural_assemblies.programs.colt_mnist_brain_util import clear_area_winners, set_kcap_winners
from neural_assemblies.programs.colt_mnist_hierarchical_brain import HIGH, LOW, NUM_DIGITS


def readout_prototypes(bundle) -> np.ndarray:
    """Prototypes aligned with generative forward path when available."""
    gen = bundle.parameters.get("generative_prototypes")
    return gen if gen is not None else bundle.prototypes


def has_generative_head(bundle) -> bool:
    return bundle.parameters.get("generative_prototypes") is not None


def prototype_predict(hv: np.ndarray, prototypes: np.ndarray, k: int) -> int:
    scores = [
        float(np.dot(hv, prototypes[d])) / max(int(np.count_nonzero(prototypes[d])), 1)
        for d in range(NUM_DIGITS)
    ]
    return int(np.argmax(scores))


def predict_high_vector(
    bundle,
    hv: np.ndarray,
    *,
    head: str = "discriminative",
    from_forward: bool = False,
) -> int:
    """Classify a HIGH vector using connectome or prototype readout."""
    if from_forward or head == "generative":
        protos = readout_prototypes(bundle) if head == "generative" else bundle.prototypes
        return prototype_predict(hv, protos, bundle.k)
    from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict
    return connectome_predict(hv, bundle.brain, k=bundle.k)


def encode_and_predict(
    bundle,
    low_pattern: np.ndarray,
    *,
    head: str = "auto",
    high_rounds: int = 3,
) -> tuple[int, np.ndarray, dict]:
    """
    Generative/predictive inference: partial or full LOW → HIGH → digit.

    ``head='auto'`` uses generative prototypes when the bundle has them
    (attractor dual-head), else discriminative connectome readout.
    """
    hv = forward_high_from_low(
        bundle.brain, low_pattern, bundle.high_bias, high_rounds=high_rounds,
    )
    if head == "auto":
        head = "generative" if has_generative_head(bundle) else "discriminative"
    pred = predict_high_vector(bundle, hv, head=head, from_forward=True)
    meta = {"head": head, "high_active": int(np.count_nonzero(hv))}
    return pred, hv, meta


def forward_high_from_low(
    brain,
    low_pattern: np.ndarray,
    high_bias: np.ndarray,
    *,
    high_rounds: int = 3,
) -> np.ndarray:
    """Encode a (possibly partial) LOW pattern into HIGH."""
    from neural_assemblies.programs.colt_mnist_hierarchical_brain import MID

    if MID in brain.areas:
        from neural_assemblies.programs.colt_mnist_spatial_ventral import forward_high_spatial
        return forward_high_spatial(
            brain, low_pattern, high_bias, high_rounds=high_rounds,
        )
    clear_area_winners(brain, HIGH)
    winners = set_kcap_winners(brain, LOW, low_pattern)
    brain.project(
        external_inputs={LOW: winners},
        projections={LOW: [HIGH], HIGH: [HIGH]},
        external_drive={HIGH: high_bias},
    )
    for _ in range(max(high_rounds - 1, 0)):
        brain.project({}, {HIGH: [HIGH]})
    vec = np.zeros(brain.areas[HIGH].n, dtype=np.float32)
    snap = _snap(brain, HIGH)
    if snap.winners.size:
        vec[np.asarray(snap.winners, dtype=int)] = 1.0
    return vec


def forward_low_from_high(
    brain,
    high_vec: np.ndarray,
    *,
    rounds: int = 5,
) -> np.ndarray:
    """Top-down HIGH→LOW regeneration."""
    clear_area_winners(brain, HIGH)
    clear_area_winners(brain, LOW)
    set_kcap_winners(brain, HIGH, high_vec)
    for _ in range(rounds):
        brain.project({}, {HIGH: [LOW]})
    vec = np.zeros(brain.areas[LOW].n, dtype=np.float32)
    snap = _snap(brain, LOW)
    if snap.winners.size:
        vec[np.asarray(snap.winners, dtype=int)] = 1.0
    return vec


def prototype_overlap(hv: np.ndarray, prototype: np.ndarray, k: int) -> float:
    active = max(int(np.count_nonzero(prototype)), 1)
    return float(np.dot(hv, prototype)) / active


def low_assembly_overlap(hv: np.ndarray, reference: np.ndarray, k: int) -> float:
    a = Assembly(LOW, np.flatnonzero(hv > 0).astype(np.uint32))
    b = Assembly(LOW, np.flatnonzero(reference > 0).astype(np.uint32))
    return float(overlap(a, b))


def measure_forward_completion(
    bundle,
    *,
    digit: int,
    protocol: str,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Return (prototype overlap, LOW-regeneration overlap) for one digit/protocol.

    Uses forward encoding from structurally masked LOW — the generative path
    that works on the explicit engine.
    """
    brain = bundle.brain
    k = bundle.k
    protos = readout_prototypes(bundle)
    proto_overs: list[float] = []
    low_overs: list[float] = []
    rng = np.random.default_rng(seed)

    for j in range(bundle.n_examples):
        full_pat = bundle.examples[digit, j]
        partial = apply_bundle_absence(bundle, full_pat, protocol, rng=rng)
        hv = forward_high_from_low(brain, partial, bundle.high_bias)
        proto_overs.append(prototype_overlap(hv, protos[digit], k))
        if float(np.sum(brain.connectomes[HIGH][LOW].weights)) > 0:
            lv = forward_low_from_high(brain, hv)
            low_overs.append(low_assembly_overlap(lv, full_pat, k))

    mean_proto = float(np.mean(proto_overs)) if proto_overs else 0.0
    mean_low = float(np.mean(low_overs)) if low_overs else 0.0
    return mean_proto, mean_low


def measure_digit_forward_completion_battery(
    bundle,
    digit: int,
    protocols: tuple[str, ...],
    *,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Per-protocol forward completion metrics for one digit."""
    out: dict[str, dict[str, float]] = {}
    for i, proto in enumerate(protocols):
        po, lo = measure_forward_completion(
            bundle, digit=digit, protocol=proto, seed=seed + digit * 10 + i,
        )
        out[proto] = {"prototype_overlap": po, "low_regeneration": lo}
    return out
