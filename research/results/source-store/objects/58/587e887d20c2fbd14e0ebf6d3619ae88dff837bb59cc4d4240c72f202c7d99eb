"""
Spatial ventral stream — local RF LOW→MID→HIGH (CNN-hierarchy analogue).

Feeds structured spatial input into assemblies via local receptive fields
instead of dense random LOW→MID.  Pairs with structured absence curriculum
and forward-completion inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes
from neural_assemblies.programs.colt_mnist_brain_util import (
    clear_area_winners,
    reinforce_class_slot,
    renorm_connectome_columns,
    set_kcap_winners,
)
from neural_assemblies.programs.colt_mnist_data import find_mnist_dir, load_mnist_arrays
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    LOW,
    MID,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
    _init_two_layer_weights,
)
from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples
from neural_assemblies.programs.colt_mnist_spatial_connectome import (
    init_spatial_low_mid_weights,
    spatial_connectome_stats,
)
from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict
from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
    _sync_two_layer,
)
from neural_assemblies.assembly_calculus.ops import _snap


@dataclass
class SpatialVentralResult(ColtMnistHierarchicalBrainResult):
    tier: str = "R"
    method: str = "spatial_local_rf_ventral"
    spatial_stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _sync_spatial_two_layer(brain, a_lm: np.ndarray, w_mh: np.ndarray) -> None:
    _sync_two_layer(brain, a_lm, w_mh)


def forward_high_spatial(
    brain,
    low_pattern: np.ndarray,
    high_bias: np.ndarray,
    *,
    mid_rounds: int = 1,
    high_rounds: int = 1,
) -> np.ndarray:
    """LOW → MID → HIGH forward (spatial ventral path)."""
    clear_area_winners(brain, MID)
    clear_area_winners(brain, HIGH)
    set_kcap_winners(brain, LOW, low_pattern)
    for _ in range(mid_rounds):
        brain.project({}, {LOW: [MID]})
    for _ in range(high_rounds):
        brain.project({}, {MID: [HIGH]}, external_drive={HIGH: high_bias})
    vec = np.zeros(brain.areas[HIGH].n, dtype=np.float32)
    snap = _snap(brain, HIGH)
    if snap.winners.size:
        vec[np.asarray(snap.winners, dtype=int)] = 1.0
    return vec


def train_spatial_ventral_brain(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    p: float = 0.1,
    beta: float = 1.0,
    n_low: int = 784,
    n_mid: int = 2000,
    n_high: int = 2000,
    n_class: int = 2000,
    class_bias: float = -1.0,
    class_passes: int = 5,
    class_beta: float = 3.0,
    rf_radius: int = 2,
    radii: tuple[int, ...] | None = None,
    train_iters: int | None = None,
    digit3_center_passes: int = 3,
    examples: np.ndarray | None = None,
    patch_curriculum: tuple[str, ...] | None = None,
    patch_graph=None,
    patch_curriculum_passes: int = 0,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, dict]:
    """
    Train LOW(local RF)→MID→HIGH with repeated exposure + optional patch curriculum.

    When ``radii`` is set (e.g. ``(1, 3)``), uses multi-scale pyramid LOW→MID init.
    Pass ``examples`` directly to train on Fashion-MNIST or other datasets.
    """
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.colt_mnist_absence import apply_absence_mask
    from neural_assemblies.programs.colt_mnist_spatial_connectome import (
        init_multiscale_low_mid_weights,
    )

    if examples is None:
        train_imgs, train_labels, _, _ = load_mnist_arrays(n_examples)
        examples = preprocess_mnist_examples(
            train_imgs, train_labels, n_examples=n_examples, cap_size=k,
        )
    if train_iters is None:
        train_iters = n_examples

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    brain.add_area(LOW, n_low, k, beta, explicit=True)
    brain.add_area(MID, n_mid, k, beta, explicit=True)
    brain.add_area(HIGH, n_high, k, beta, explicit=True)
    brain.add_area(CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS)

    rng = np.random.default_rng(seed)
    if radii is not None:
        a_lm = init_multiscale_low_mid_weights(
            rng, n_low=n_low, n_mid=n_mid, radii=radii,
        )
    else:
        a_lm = init_spatial_low_mid_weights(
            rng, n_low=n_low, n_mid=n_mid, rf_radius=rf_radius,
        )
    _, w_mh = _init_two_layer_weights(rng, n_low, n_mid, n_high, p)
    _sync_spatial_two_layer(brain, a_lm, w_mh)
    stats = spatial_connectome_stats(a_lm)

    high_bias = np.zeros(n_high, dtype=np.float32)
    for digit in range(NUM_DIGITS):
        clear_area_winners(brain, MID)
        clear_area_winners(brain, HIGH)
        for j in range(train_iters):
            set_kcap_winners(brain, LOW, examples[digit, j % n_examples])
            brain.project({}, {LOW: [MID]})
            brain.project({}, {MID: [HIGH]}, external_drive={HIGH: high_bias})
        snap = _snap(brain, HIGH)
        if len(snap.winners) > 0:
            high_bias[snap.winners] += class_bias
        renorm_connectome_columns(brain, LOW, MID)
        renorm_connectome_columns(brain, MID, HIGH)

    saved = brain.disable_plasticity
    brain.disable_plasticity = False
    if patch_curriculum and patch_graph is not None and patch_curriculum_passes > 0:
        occ = np.random.default_rng(seed + 808)
        for _ in range(patch_curriculum_passes):
            for digit in range(NUM_DIGITS):
                for j in range(n_examples):
                    pat = examples[digit, j]
                    if occ.random() < 0.40:
                        proto = patch_curriculum[int(occ.integers(0, len(patch_curriculum)))]
                        pat = patch_graph.apply_absence(pat, proto, rng=occ)
                    forward_high_spatial(brain, pat, high_bias)
            renorm_connectome_columns(brain, LOW, MID)
            renorm_connectome_columns(brain, MID, HIGH)
    elif digit3_center_passes > 0:
        for _ in range(digit3_center_passes):
            for j in range(n_examples):
                pat = apply_absence_mask(examples[3, j], "center_band")
                forward_high_spatial(brain, pat, high_bias)
            renorm_connectome_columns(brain, LOW, MID)
            renorm_connectome_columns(brain, MID, HIGH)
    brain.disable_plasticity = saved

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    for _ in range(class_passes):
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                forward_high_spatial(brain, examples[digit, j], high_bias)
                reinforce_class_slot(brain, HIGH, CLASS, digit, k, beta=class_beta)
            renorm_connectome_columns(brain, HIGH, CLASS)

    high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
    brain.disable_plasticity = True
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = forward_high_spatial(
                brain, examples[digit, j], high_bias,
            )
    brain.disable_plasticity = saved

    prototypes = np.zeros((NUM_DIGITS, n_high))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0

    return brain, high_outputs, prototypes, high_bias, k, examples, stats


def run_spatial_ventral_mnist(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    **kwargs,
) -> SpatialVentralResult:
    """Evaluate spatial RF ventral stream."""
    brain, high_outputs, prototypes, high_bias, k, examples, stats = train_spatial_ventral_brain(
        seed=seed, n_examples=n_examples, k=k, **kwargs,
    )
    wire_class_from_prototypes(brain, prototypes, HIGH, CLASS, k)

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(high_outputs[digit, j], brain, k=k) == digit
            for j in range(n_examples)
        )
        correct[digit] = hits / n_examples

    from neural_assemblies.programs.colt_mnist_absence import apply_absence_mask

    d3_hits = 0
    for j in range(n_examples):
        pat = apply_absence_mask(examples[3, j], "center_band")
        hv = forward_high_spatial(brain, pat, high_bias)
        d3_hits += connectome_predict(hv, brain, k=k) == 3
    d3_center = d3_hits / n_examples

    return SpatialVentralResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        parameters={"seed": seed, "n_examples": n_examples, "k": k, **kwargs},
        backend="spatial_ventral_stream",
        spatial_stats=stats,
        extra={
            "digit3_center_band_accuracy": d3_center,
            "confused_digit_accuracy": float(np.mean([correct[d] for d in (2, 3, 5, 8)])),
        },
    )
