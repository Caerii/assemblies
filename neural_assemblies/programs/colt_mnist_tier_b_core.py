"""Core ventral training helper shared by tier B/C and cross-domain modules."""

from __future__ import annotations

import numpy as np

from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    LOW,
    MID,
    NUM_DIGITS,
)


def _ventral_brain_and_outputs(**kwargs):
    """Train ventral model; return brain, high_outputs, prototypes, examples meta."""
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.colt_mnist_data import load_mnist_arrays
    from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples
    from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
        _extend_hierarchical_training,
        _init_two_layer_weights,
        _sync_two_layer,
    )
    from neural_assemblies.programs.colt_mnist_brain_util import renorm_connectome_columns

    seed = kwargs.get("seed", 42)
    n_examples = kwargs.get("n_examples", 50)
    k = kwargs.get("k", 200)
    p = kwargs.get("p", 0.1)
    beta = kwargs.get("beta", 1.0)
    class_bias = kwargs.get("class_bias", -1.0)
    class_passes = kwargs.get("class_passes", 5)
    class_beta = kwargs.get("class_beta", 3.0)
    enable_high_recurrence = kwargs.get("enable_high_recurrence", False)

    train_imgs, train_labels, _, _ = load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )
    train_occlusion_fraction = float(kwargs.get("train_occlusion_fraction", 0.0))
    if train_occlusion_fraction > 0:
        occ_rng = np.random.default_rng(seed + 999)
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                mask = occ_rng.random(784) > train_occlusion_fraction
                examples[digit, j] = examples[digit, j] * mask
    n_low, n_mid, n_high, n_class = 784, 2000, 2000, 2000

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    brain.add_area(LOW, n_low, k, beta, explicit=True)
    brain.add_area(MID, n_mid, k, beta, explicit=True)
    brain.add_area(HIGH, n_high, k, beta, explicit=True)
    brain.add_area(CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS)
    rng = np.random.default_rng(seed)
    _sync_two_layer(brain, *_init_two_layer_weights(rng, n_low, n_mid, n_high, p))

    if enable_high_recurrence:
        mask_hh = rng.random((n_high, n_high)) < p
        w_hh = mask_hh.astype(np.float64)
        w_hh /= np.maximum(w_hh.sum(axis=0, keepdims=True), 1e-12)
        w32 = w_hh.astype(np.float32)
        brain.connectomes[HIGH][HIGH].weights = w32
        if brain._explicit_engine is not None:
            brain._explicit_engine._area_conns[HIGH][HIGH].weights = w32

    if enable_high_recurrence:
        from neural_assemblies.programs.colt_mnist_brain_util import (
            clear_area_winners,
            reinforce_class_slot,
            set_kcap_winners,
        )
        from neural_assemblies.programs.colt_mnist_hierarchical_brain import _forward_high

        high_bias = np.zeros(n_high, dtype=np.float32)
        for digit in range(NUM_DIGITS):
            clear_area_winners(brain, MID)
            clear_area_winners(brain, HIGH)
            for j in range(n_examples):
                set_kcap_winners(brain, LOW, examples[digit, j])
                brain.project({}, {LOW: [MID]})
                brain.project({}, {MID: [HIGH]}, external_drive={HIGH: high_bias})
                for _ in range(3):
                    brain.project({}, {HIGH: [HIGH]})
            snap = _forward_high(brain, examples[digit, 0], high_bias)
            snap_winners = np.flatnonzero(snap > 0)
            if snap_winners.size:
                high_bias[snap_winners] += class_bias
            renorm_connectome_columns(brain, LOW, MID)
            renorm_connectome_columns(brain, MID, HIGH)
            renorm_connectome_columns(brain, HIGH, HIGH)

        saved = brain.disable_plasticity
        brain.disable_plasticity = True
        for _ in range(class_passes):
            for digit in range(NUM_DIGITS):
                for j in range(n_examples):
                    _forward_high(brain, examples[digit, j], high_bias)
                    for _ in range(2):
                        brain.project({}, {HIGH: [HIGH]})
                    reinforce_class_slot(brain, HIGH, CLASS, digit, k, beta=class_beta)
                renorm_connectome_columns(brain, HIGH, CLASS)

        high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                hv = _forward_high(brain, examples[digit, j], high_bias)
                for _ in range(2):
                    brain.project({}, {HIGH: [HIGH]})
                vec = np.zeros(n_high, dtype=np.float32)
                vec[np.asarray(brain.areas[HIGH].winners, dtype=int)] = 1.0
                high_outputs[digit, j] = vec
        brain.disable_plasticity = saved
    else:
        high_bias, high_outputs = _extend_hierarchical_training(
            brain, examples,
            n_examples=n_examples,
            train_iters=n_examples,
            class_bias=class_bias,
            class_passes=class_passes,
            class_beta=class_beta,
        )

    prototypes = np.zeros((NUM_DIGITS, n_high))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0
    return brain, high_outputs, prototypes, k, n_examples, examples, high_bias
