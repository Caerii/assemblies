"""
COLT 2022 MNIST — native explicit-Brain port of MNIST.ipynb.

Single recurrent HIDDEN area; combined INPUT+self projection per round
matches ``k_cap(act_h @ W + inp @ A + bias, k)`` in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs.colt_mnist_brain_util import (
    clear_area_winners,
    renorm_connectome_columns,
    set_kcap_winners,
    sync_protocol_weights,
)
from neural_assemblies.programs.colt_mnist_data import (
    find_mnist_dir as _find_mnist_dir,
    load_mnist_arrays as _load_mnist_arrays,
)
from neural_assemblies.programs.colt_mnist_protocol import (
    init_protocol_weights,
    preprocess_mnist_examples,
)


@dataclass
class ColtMnistBrainExplicitResult:
    per_class_accuracy: np.ndarray
    mean_accuracy: float
    data_source: str
    parameters: dict
    backend: str


def run_colt_mnist_brain_explicit(
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
) -> ColtMnistBrainExplicitResult:
    from neural_assemblies.core.brain import Brain

    mnist_dir = _find_mnist_dir()
    data_source = "mnist_csv" if mnist_dir else "synthetic_fallback"
    train_imgs, train_labels, _, _ = _load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=cap_size,
    )

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    input_area, hidden_area = "INPUT", "HIDDEN"
    brain.add_area(input_area, n_in, cap_size, beta, explicit=True)
    brain.add_area(hidden_area, n_neurons, cap_size, beta, explicit=True)

    rng = np.random.default_rng(seed)
    w, a = init_protocol_weights(rng, n_in, n_neurons, p)
    sync_protocol_weights(brain, w, a, input_area, hidden_area)

    hidden_bias = np.zeros(n_neurons, dtype=np.float32)

    for digit in range(10):
        clear_area_winners(brain, hidden_area)
        for j in range(n_rounds):
            winners = set_kcap_winners(brain, input_area, examples[digit, j])
            brain.project(
                external_inputs={input_area: winners},
                projections={input_area: [hidden_area], hidden_area: [hidden_area]},
                external_drive={hidden_area: hidden_bias},
            )
        snap = _snap(brain, hidden_area)
        if len(snap.winners) > 0:
            hidden_bias[snap.winners] += class_bias
        renorm_connectome_columns(brain, input_area, hidden_area)
        renorm_connectome_columns(brain, hidden_area, hidden_area)

    saved_plasticity = brain.disable_plasticity
    brain.disable_plasticity = True
    try:
        hidden_outputs = np.zeros((10, n_examples, n_neurons))
        for digit in range(10):
            for j in range(n_examples):
                clear_area_winners(brain, hidden_area)
                winners = set_kcap_winners(brain, input_area, examples[digit, j])
                brain.project(
                    external_inputs={input_area: winners},
                    projections={input_area: [hidden_area], hidden_area: [hidden_area]},
                    external_drive={hidden_area: hidden_bias},
                )
                snap = _snap(brain, hidden_area)
                hidden_outputs[digit, j, snap.winners] = 1.0

        proto_mat = np.zeros((10, n_neurons))
        for digit in range(10):
            support = hidden_outputs[digit].sum(axis=0)
            proto_mat[digit, support.argsort()[-cap_size:]] = 1.0

        correct = np.zeros(10)
        for digit in range(10):
            scores = hidden_outputs[digit] @ proto_mat.T
            correct[digit] = (scores.argmax(axis=-1) == digit).mean()
    finally:
        brain.disable_plasticity = saved_plasticity

    params = {
        "seed": seed,
        "n_in": n_in,
        "n_neurons": n_neurons,
        "cap_size": cap_size,
        "beta": beta,
        "n_rounds": n_rounds,
        "n_examples": n_examples,
        "p": p,
        "class_bias": class_bias,
    }
    return ColtMnistBrainExplicitResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=data_source,
        parameters=params,
        backend="brain_explicit",
    )
