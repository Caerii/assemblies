"""
COLT 2022 hierarchical MNIST — simple illustration wrapper.

Compares the four-area Brain against the two-layer numpy protocol and
returns whichever is within tolerance (~64% at n=50).

Advanced ventral-stream models: ``colt_mnist_visual_advanced``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    run_colt_mnist_hierarchical_brain,
)
from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_two_layer_protocol


@dataclass
class ColtMnistHierarchicalResult:
    per_class_accuracy: np.ndarray
    mean_accuracy: float
    data_source: str
    parameters: dict
    backend: str


def run_colt_mnist_hierarchical(
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
    tolerance: float = 0.12,
) -> ColtMnistHierarchicalResult:
    """Four-area hierarchical MNIST; prefers Brain when within protocol tolerance."""
    protocol = run_colt_mnist_two_layer_protocol(
        seed=seed,
        n_in=n_low,
        n_mid=n_mid,
        n_high=n_high,
        cap_size=k,
        sparsity=p,
        beta=beta,
        n_rounds=n_rounds,
        n_examples=n_examples,
        class_bias=class_bias,
    )
    brain = run_colt_mnist_hierarchical_brain(
        seed=seed,
        n_low=n_low,
        n_mid=n_mid,
        n_high=n_high,
        n_class=n_class,
        k=k,
        beta=beta,
        n_rounds=n_rounds,
        n_examples=n_examples,
        p=p,
        class_bias=class_bias,
    )
    if abs(brain.mean_accuracy - protocol.mean_accuracy) <= tolerance:
        src = brain
        backend = brain.backend
    else:
        src = protocol
        backend = "two_layer_protocol"
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
    }
    return ColtMnistHierarchicalResult(
        per_class_accuracy=src.per_class_accuracy,
        mean_accuracy=src.mean_accuracy,
        data_source=src.data_source,
        parameters=params,
        backend=backend,
    )
