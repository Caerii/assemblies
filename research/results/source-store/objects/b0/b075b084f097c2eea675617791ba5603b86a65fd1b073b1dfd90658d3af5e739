"""
COLT 2022 MNIST — Brain API (native explicit when within protocol tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_assemblies.programs.colt_mnist_brain_explicit import run_colt_mnist_brain_explicit
from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_protocol


@dataclass
class ColtMnistBrainResult:
    per_class_accuracy: np.ndarray
    mean_accuracy: float
    data_source: str
    parameters: dict
    backend: str = "protocol"


def run_colt_mnist_brain(
    *,
    seed: int = 42,
    n_in: int = 784,
    n_neurons: int = 2000,
    cap_size: int = 200,
    beta: float = 1.0,
    n_rounds: int = 5,
    n_examples: int = 50,
    p: float = 0.1,
    class_bias: float = -1.0,
    tolerance: float = 0.08,
) -> ColtMnistBrainResult:
    """Train + classify; prefers native explicit Brain if within protocol tolerance."""
    protocol = run_colt_mnist_protocol(
        seed=seed,
        n_in=n_in,
        n_neurons=n_neurons,
        cap_size=cap_size,
        sparsity=p,
        beta=beta,
        n_rounds=n_rounds,
        n_examples=n_examples,
        class_bias=class_bias,
    )
    explicit = run_colt_mnist_brain_explicit(
        seed=seed,
        n_in=n_in,
        n_neurons=n_neurons,
        cap_size=cap_size,
        beta=beta,
        n_rounds=n_rounds,
        n_examples=n_examples,
        p=p,
        class_bias=class_bias,
    )
    if abs(explicit.mean_accuracy - protocol.mean_accuracy) <= tolerance:
        src = explicit
        backend = explicit.backend
    else:
        src = protocol
        backend = "protocol_fallback"
    params = dict(src.parameters)
    params["p"] = p
    return ColtMnistBrainResult(
        per_class_accuracy=src.per_class_accuracy,
        mean_accuracy=src.mean_accuracy,
        data_source=src.data_source,
        parameters=params,
        backend=backend,
    )
