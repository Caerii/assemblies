"""
COLT 2022 hierarchical MNIST — simple 4-area Brain illustration.

LOW → MID → HIGH → CLASS feedforward chain matching the two-layer numpy
protocol (~64% mean accuracy at n=50).  Default readout is prototype
dot-product on HIGH.

This module is intentionally minimal: it demonstrates that assembly-calculus
``project``, Hebbian plasticity, per-class negative bias, and column renorm
suffice for above-chance multi-class learning, but **not** for notebook-level
accuracy.  See ``colt_mnist_ventral_theory`` (hypothesis H1) for why recurrence
or repeated exposure is required, and ``colt_mnist_visual_advanced`` for models
that implement those fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs.colt_mnist_brain_util import (
    clear_area_winners,
    read_slot_scores,
    reinforce_class_slot,
    renorm_connectome_columns,
    set_kcap_winners,
)
from neural_assemblies.programs.colt_mnist_data import (
    find_mnist_dir as _find_mnist_dir,
    load_mnist_arrays as _load_mnist_arrays,
)
from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples

LOW = "LOW"
MID = "MID"
HIGH = "HIGH"
CLASS = "CLASS"
NUM_DIGITS = 10


@dataclass
class ColtMnistHierarchicalBrainResult:
    per_class_accuracy: np.ndarray
    mean_accuracy: float
    data_source: str
    parameters: dict
    backend: str


def _init_two_layer_weights(
    rng: np.random.Generator,
    n_low: int,
    n_mid: int,
    n_high: int,
    sparsity: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask_lm = rng.random((n_low, n_mid)) < sparsity
    a_lm = mask_lm.astype(np.float64)
    a_lm /= np.maximum(a_lm.sum(axis=0, keepdims=True), 1e-12)
    mask_mh = rng.random((n_mid, n_high)) < sparsity
    w_mh = mask_mh.astype(np.float64)
    w_mh /= np.maximum(w_mh.sum(axis=0, keepdims=True), 1e-12)
    return a_lm, w_mh


def _sync_two_layer(brain, a_lm: np.ndarray, w_mh: np.ndarray) -> None:
    a32 = a_lm.astype(np.float32)
    w32 = w_mh.astype(np.float32)
    brain.connectomes[LOW][MID].weights = a32
    brain.connectomes[MID][HIGH].weights = w32
    zero_hc = np.zeros((w_mh.shape[0], brain.areas[CLASS].n), dtype=np.float32)
    brain.connectomes[HIGH][CLASS].weights = zero_hc
    if brain._explicit_engine is not None:
        eng = brain._explicit_engine
        eng._area_conns[LOW][MID].weights = a32
        eng._area_conns[MID][HIGH].weights = w32
        eng._area_conns[HIGH][CLASS].weights = zero_hc


def _forward_high(
    brain,
    pattern: np.ndarray,
    high_bias: np.ndarray,
) -> np.ndarray:
    clear_area_winners(brain, MID)
    clear_area_winners(brain, HIGH)
    set_kcap_winners(brain, LOW, pattern)
    brain.project({}, {LOW: [MID]})
    brain.project({}, {MID: [HIGH]}, external_drive={HIGH: high_bias})
    vec = np.zeros(brain.areas[HIGH].n, dtype=np.float32)
    vec[_snap(brain, HIGH).winners] = 1.0
    return vec


def _set_high_from_vector(brain, high_vec: np.ndarray) -> None:
    winners = np.flatnonzero(high_vec > 0).astype(np.uint32)
    brain.areas[HIGH].winners = winners
    brain.areas[HIGH].w = len(winners)
    if brain._explicit_engine is not None:
        brain._explicit_engine.set_winners(HIGH, winners)


def _predict_digit(
    brain,
    high_vec: np.ndarray,
    prototypes: np.ndarray,
    k: int,
    *,
    readout: str,
) -> int:
    proto_scores = prototypes @ high_vec
    if readout == "prototype":
        return int(np.argmax(proto_scores))

    _set_high_from_vector(brain, high_vec)
    clear_area_winners(brain, CLASS)
    brain.project({}, {HIGH: [CLASS]})
    slot_scores = read_slot_scores(_snap(brain, CLASS).winners, NUM_DIGITS, k)
    if readout == "class":
        return int(np.argmax(slot_scores))

    scale = float(np.max(proto_scores)) if np.max(proto_scores) > 0 else 1.0
    combined = proto_scores + np.asarray(slot_scores) * scale
    return int(np.argmax(combined))


def run_colt_mnist_hierarchical_brain(
    *,
    seed: int = 42,
    n_low: int = 784,
    n_mid: int = 2000,
    n_high: int = 2000,
    n_class: int = 2000,
    k: int = 200,
    beta: float = 1.0,
    n_rounds: int = 5,
    n_examples: int = 50,
    p: float = 0.1,
    class_bias: float = -1.0,
    readout: str = "prototype",
) -> ColtMnistHierarchicalBrainResult:
    """Four-area feedforward Brain; prototype readout by default."""
    from neural_assemblies.core.brain import Brain

    if readout not in ("hybrid", "prototype", "class"):
        raise ValueError(f"readout must be hybrid, prototype, or class; got {readout!r}")

    mnist_dir = _find_mnist_dir()
    data_source = "mnist_csv" if mnist_dir else "synthetic_fallback"
    train_imgs, train_labels, _, _ = _load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    brain.add_area(LOW, n_low, k, beta, explicit=True)
    brain.add_area(MID, n_mid, k, beta, explicit=True)
    brain.add_area(HIGH, n_high, k, beta, explicit=True)
    brain.add_area(
        CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS,
    )

    rng = np.random.default_rng(seed)
    a_lm, w_mh = _init_two_layer_weights(rng, n_low, n_mid, n_high, p)
    _sync_two_layer(brain, a_lm, w_mh)

    high_bias = np.zeros(n_high, dtype=np.float32)
    for digit in range(NUM_DIGITS):
        clear_area_winners(brain, MID)
        clear_area_winners(brain, HIGH)
        for j in range(n_rounds):
            set_kcap_winners(brain, LOW, examples[digit, j])
            brain.project({}, {LOW: [MID]})
            brain.project(
                {},
                {MID: [HIGH]},
                external_drive={HIGH: high_bias},
            )
        snap = _snap(brain, HIGH)
        if len(snap.winners) > 0:
            high_bias[snap.winners] += class_bias
        renorm_connectome_columns(brain, LOW, MID)
        renorm_connectome_columns(brain, MID, HIGH)

    saved_plasticity = brain.disable_plasticity
    brain.disable_plasticity = True
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            _forward_high(brain, examples[digit, j], high_bias)
            reinforce_class_slot(brain, HIGH, CLASS, digit, k, beta=beta)
        renorm_connectome_columns(brain, HIGH, CLASS)

    high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = _forward_high(
                brain, examples[digit, j], high_bias,
            )

    prototypes = np.zeros((NUM_DIGITS, n_high))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0

    try:
        correct = np.zeros(NUM_DIGITS)
        for digit in range(NUM_DIGITS):
            hits = 0
            for j in range(n_examples):
                pred = _predict_digit(
                    brain,
                    high_outputs[digit, j],
                    prototypes,
                    k,
                    readout=readout,
                )
                if pred == digit:
                    hits += 1
            correct[digit] = hits / n_examples
    finally:
        brain.disable_plasticity = saved_plasticity

    params = {
        "seed": seed,
        "n_low": n_low,
        "n_mid": n_mid,
        "n_high": n_high,
        "n_class": n_class,
        "k": k,
        "beta": beta,
        "n_rounds": n_rounds,
        "n_examples": n_examples,
        "p": p,
        "class_bias": class_bias,
        "readout": readout,
    }
    return ColtMnistHierarchicalBrainResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=data_source,
        parameters=params,
        backend="brain_hierarchical",
    )
