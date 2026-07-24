"""
COLT 2022 MNIST — faithful port of learning-with-assemblies/MNIST.ipynb.

The notebook trains a sparse Hebbian recurrent classifier with:
  1. Per-class Hebbian updates over ``n_rounds`` example presentations
  2. Negative bias ``b=-1`` on hidden neurons active after each class block
  3. Column renormalization of ``W`` and ``A`` after each class block
  4. Prototype readout from summed hidden activations (round 1)

Without steps 2–3 the model collapses to chance (~10% one-hot digit).

Theory and roadmap: ``colt_mnist_ventral_theory``.  Advanced models:
``colt_mnist_visual_advanced``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from neural_assemblies.programs.colt_mnist_data import (
    find_mnist_dir as _find_mnist_dir,
    k_cap as _k_cap,
    load_mnist_arrays as _load_mnist_arrays,
)


@dataclass
class ColtMnistProtocolResult:
    per_class_accuracy: np.ndarray
    mean_accuracy: float
    data_source: str
    parameters: dict


def preprocess_mnist_examples(
    train_imgs: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_examples: int,
    cap_size: int,
) -> np.ndarray:
    """Blur + k-cap per class → ``(10, n_examples, 784)``."""
    # Imported here rather than at module scope: this module is reachable from
    # ``import neural_assemblies`` (via programs/__init__ -> blocks_bridge), and
    # scipy costs ~3s to import while convolve is only needed by this one
    # function. Same call, same numerics.
    from scipy.signal import convolve

    examples = np.zeros((10, n_examples, 784))
    for digit in range(10):
        class_imgs = train_imgs[train_labels == digit][:n_examples].reshape(-1, 28, 28)
        blurred = convolve(
            class_imgs, np.ones((1, 3, 3)), mode="same",
        ).reshape(-1, 784)
        examples[digit] = _k_cap(blurred, cap_size)
    return examples


def init_protocol_weights(
    rng: np.random.Generator,
    n_in: int,
    n_neurons: int,
    sparsity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (
        (rng.random((n_neurons, n_neurons)) < sparsity)
        & np.logical_not(np.eye(n_neurons, dtype=bool))
    )
    w = mask.astype(np.float64)
    w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    mask_a = rng.random((n_in, n_neurons)) < sparsity
    a = mask_a.astype(np.float64)
    a /= np.maximum(a.sum(axis=0, keepdims=True), 1e-12)
    return w, a


def _renorm_columns(mat: np.ndarray) -> None:
    mat /= np.maximum(mat.sum(axis=0, keepdims=True), 1e-12)


def run_colt_mnist_protocol(
    *,
    seed: int = 42,
    n_in: int = 784,
    n_neurons: int = 2000,
    cap_size: int = 200,
    sparsity: float = 0.1,
    beta: float = 1.0,
    n_rounds: int = 5,
    n_examples: int = 500,
    class_bias: float = -1.0,
) -> ColtMnistProtocolResult:
    """Train + classify per COLT MNIST.ipynb (cells 4–7, 16)."""
    rng = np.random.default_rng(seed)
    mnist_dir = _find_mnist_dir()
    data_source = "mnist_csv" if mnist_dir else "synthetic_fallback"

    train_imgs, train_labels, _, _ = _load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=cap_size,
    )

    w, a = init_protocol_weights(rng, n_in, n_neurons, sparsity)
    bias = np.zeros(n_neurons, dtype=np.float64)

    for class_idx in range(10):
        act_h = np.zeros(n_neurons, dtype=np.float64)
        for round_idx in range(n_rounds):
            inp = examples[class_idx, round_idx]
            act_h_new = _k_cap(act_h @ w + inp @ a + bias, cap_size)
            a[(inp > 0)[:, np.newaxis] & (act_h_new > 0)[np.newaxis, :]] *= 1 + beta
            w[(act_h > 0)[:, np.newaxis] & (act_h_new > 0)[np.newaxis, :]] *= 1 + beta
            act_h = act_h_new
        bias[act_h > 0] += class_bias
        _renorm_columns(a)
        _renorm_columns(w)

    outputs = np.zeros((10, n_rounds + 1, n_examples, n_neurons))
    for class_idx in range(10):
        for round_idx in range(n_rounds):
            outputs[class_idx, round_idx + 1] = _k_cap(
                outputs[class_idx, round_idx] @ w + examples[class_idx] @ a + bias,
                cap_size,
            )

    prototypes = np.zeros((10, n_neurons))
    for class_idx in range(10):
        support = outputs[class_idx, 1].sum(axis=0)
        prototypes[class_idx, support.argsort()[-cap_size:]] = 1.0

    predictions = (outputs[:, 1] @ prototypes.T).argmax(axis=-1)
    per_class = (
        predictions == np.arange(10)[:, np.newaxis]
    ).sum(axis=-1) / n_examples

    params = {
        "seed": seed,
        "n_in": n_in,
        "n_neurons": n_neurons,
        "cap_size": cap_size,
        "sparsity": sparsity,
        "beta": beta,
        "n_rounds": n_rounds,
        "n_examples": n_examples,
        "class_bias": class_bias,
    }
    return ColtMnistProtocolResult(
        per_class_accuracy=per_class,
        mean_accuracy=float(per_class.mean()),
        data_source=data_source,
        parameters=params,
    )


def run_colt_mnist_two_layer_protocol(
    *,
    seed: int = 42,
    n_in: int = 784,
    n_mid: int = 2000,
    n_high: int = 2000,
    cap_size: int = 200,
    sparsity: float = 0.1,
    beta: float = 1.0,
    n_rounds: int = 5,
    n_examples: int = 50,
    class_bias: float = -1.0,
) -> ColtMnistProtocolResult:
    """LOW → MID → HIGH feedforward Hebbian chain with bias+renorm on HIGH."""
    rng = np.random.default_rng(seed)
    mnist_dir = _find_mnist_dir()
    data_source = "mnist_csv" if mnist_dir else "synthetic_fallback"

    train_imgs, train_labels, _, _ = _load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=cap_size,
    )

    mask_lm = rng.random((n_in, n_mid)) < sparsity
    a_lm = mask_lm.astype(np.float64)
    a_lm /= np.maximum(a_lm.sum(axis=0, keepdims=True), 1e-12)
    mask_mh = rng.random((n_mid, n_high)) < sparsity
    w_mh = mask_mh.astype(np.float64)
    w_mh /= np.maximum(w_mh.sum(axis=0, keepdims=True), 1e-12)
    bias_high = np.zeros(n_high, dtype=np.float64)

    for class_idx in range(10):
        for round_idx in range(n_rounds):
            inp = examples[class_idx, round_idx]
            mid = _k_cap(inp @ a_lm, cap_size)
            high = _k_cap(mid @ w_mh + bias_high, cap_size)
            a_lm[(inp > 0)[:, np.newaxis] & (mid > 0)[np.newaxis, :]] *= 1 + beta
            w_mh[(mid > 0)[:, np.newaxis] & (high > 0)[np.newaxis, :]] *= 1 + beta
        bias_high[high > 0] += class_bias
        _renorm_columns(a_lm)
        _renorm_columns(w_mh)

    outputs = np.zeros((10, n_examples, n_high))
    for class_idx in range(10):
        mid_batch = _k_cap(examples[class_idx] @ a_lm, cap_size)
        outputs[class_idx] = _k_cap(mid_batch @ w_mh + bias_high, cap_size)

    prototypes = np.zeros((10, n_high))
    for class_idx in range(10):
        support = outputs[class_idx].sum(axis=0)
        prototypes[class_idx, support.argsort()[-cap_size:]] = 1.0

    predictions = (outputs @ prototypes.T).argmax(axis=-1)
    per_class = (
        predictions == np.arange(10)[:, np.newaxis]
    ).sum(axis=-1) / n_examples

    params = {
        "seed": seed,
        "n_in": n_in,
        "n_mid": n_mid,
        "n_high": n_high,
        "cap_size": cap_size,
        "sparsity": sparsity,
        "beta": beta,
        "n_rounds": n_rounds,
        "n_examples": n_examples,
        "class_bias": class_bias,
    }
    return ColtMnistProtocolResult(
        per_class_accuracy=per_class,
        mean_accuracy=float(per_class.mean()),
        data_source=data_source,
        parameters=params,
    )
