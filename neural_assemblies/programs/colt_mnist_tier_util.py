"""
Shared ventral training bundle for tier A/B/C and cross-domain experiments.

Avoids redundant full re-trains when the evidence suite runs many experiments
on the same seed / n_examples configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from neural_assemblies.programs.colt_mnist_data import find_mnist_dir
from neural_assemblies.programs.colt_mnist_hierarchical_brain import NUM_DIGITS


# Stroke-confusable digit pairs (H3 diagnostic / H5 LRI targets).
CONFUSED_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 8), (3, 8), (3, 5), (4, 9), (5, 6), (7, 9),
)

CONFUSED_DIGITS: frozenset[int] = frozenset(d for pair in CONFUSED_PAIRS for d in pair)


@dataclass
class VentralBundle:
    """One ventral-stream training run — shared across tier experiments."""

    brain: Any
    high_outputs: np.ndarray
    prototypes: np.ndarray
    high_bias: np.ndarray
    k: int
    n_examples: int
    examples: np.ndarray
    data_source: str
    parameters: dict


_BUNDLE_CACHE: dict[tuple, VentralBundle] = {}
_RECURRENT_CACHE: dict[tuple, VentralBundle] = {}
_ATTRACTOR_CACHE: dict[tuple, VentralBundle] = {}
_SPATIAL_CACHE: dict[tuple, VentralBundle] = {}
_MULTISCALE_CACHE: dict[tuple, VentralBundle] = {}
_FASHION_CACHE: dict[tuple, VentralBundle] = {}


def _build_prototypes_from_outputs(high_outputs: np.ndarray, k: int) -> np.ndarray:
    prototypes = np.zeros((NUM_DIGITS, high_outputs.shape[2]))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0
    return prototypes


def refresh_bundle_representations(bundle: VentralBundle) -> None:
    """Re-forward all exemplars and rebuild prototypes after encoding updates."""
    from neural_assemblies.programs.colt_mnist_attractor import refresh_high_outputs_from_forward

    refresh_high_outputs_from_forward(
        bundle.brain,
        bundle.high_bias,
        bundle.examples,
        bundle.high_outputs,
        bundle.n_examples,
    )
    bundle.prototypes = _build_prototypes_from_outputs(bundle.high_outputs, bundle.k)


def load_ventral_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    enable_high_recurrence: bool = False,
    use_cache: bool = True,
    **kwargs,
) -> VentralBundle:
    """Train (or retrieve cached) ventral bundle."""
    key = (seed, n_examples, k, enable_high_recurrence, frozenset(kwargs.items()))
    if use_cache and key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    from neural_assemblies.programs.colt_mnist_tier_b_core import _ventral_brain_and_outputs

    brain, high_outputs, prototypes, k, n_examples, examples, high_bias = (
        _ventral_brain_and_outputs(
            seed=seed,
            n_examples=n_examples,
            k=k,
            enable_high_recurrence=enable_high_recurrence,
            **kwargs,
        )
    )
    bundle = VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        parameters={"seed": seed, "n_examples": n_examples, "k": k, **kwargs},
    )
    if use_cache:
        _BUNDLE_CACHE[key] = bundle
    return bundle


def load_recurrent_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    use_cache: bool = True,
    **kwargs,
) -> VentralBundle:
    """Train visual_advanced recurrent cortex (~80% at n=50) as a VentralBundle.

    Prefer this over ``load_ventral_bundle(enable_high_recurrence=True)``, which
    uses tier_b_core and reaches only ~67% connectome readout (H8).
    """
    key = (seed, n_examples, k, frozenset(kwargs.items()))
    if use_cache and key in _RECURRENT_CACHE:
        return _RECURRENT_CACHE[key]

    from neural_assemblies.programs.colt_mnist_data import find_mnist_dir, load_mnist_arrays
    from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples
    from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
        _attach_class_to_recurrent,
    )

    n_rounds = kwargs.pop("n_rounds", 5)
    p = kwargs.pop("p", 0.1)
    beta = kwargs.pop("beta", 1.0)
    class_bias = kwargs.pop("class_bias", -1.0)
    class_passes = kwargs.pop("class_passes", 5)
    class_beta = kwargs.pop("class_beta", 3.0)
    absence_exposure_prob = kwargs.pop("absence_exposure_prob", 0.0)
    enable_generative_head = kwargs.pop("enable_generative_head", True)
    digit3_center_passes = kwargs.pop("digit3_center_passes", 4)
    absence_curriculum_prob = kwargs.pop("absence_curriculum_prob", 0.35)
    absence_exposure_passes = kwargs.pop("absence_exposure_passes", 1)
    n_low = kwargs.pop("n_low", 784)
    n_high = kwargs.pop("n_high", 2000)
    n_class = kwargs.pop("n_class", 2000)

    train_imgs, train_labels, _, _ = load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )
    brain, high_outputs, high_bias = _attach_class_to_recurrent(
        seed=seed,
        n_low=n_low,
        n_high=n_high,
        n_class=n_class,
        k=k,
        beta=beta,
        n_rounds=n_rounds,
        n_examples=n_examples,
        p=p,
        class_bias=class_bias,
        class_passes=class_passes,
        class_beta=class_beta,
        absence_exposure_prob=absence_exposure_prob,
    )
    prototypes = _build_prototypes_from_outputs(high_outputs, k)
    generative_prototypes = None
    if enable_generative_head:
        from neural_assemblies.programs.colt_mnist_attractor import (
            apply_forward_generative_curriculum,
            capture_generative_prototypes,
        )

        apply_forward_generative_curriculum(
            brain,
            high_bias,
            examples,
            n_examples,
            seed=seed,
            digit3_center_passes=digit3_center_passes,
            absence_curriculum_prob=absence_curriculum_prob,
            absence_exposure_passes=absence_exposure_passes,
            enable_top_down_low=False,
        )
        generative_prototypes = capture_generative_prototypes(
            brain, high_bias, examples, n_examples, k, n_high,
        )

    params = {
        "seed": seed,
        "n_examples": n_examples,
        "k": k,
        "base": "visual_advanced_recurrent",
        "absence_exposure_prob": absence_exposure_prob,
        "enable_generative_head": enable_generative_head,
        **kwargs,
    }
    if generative_prototypes is not None:
        params["generative_prototypes"] = generative_prototypes

    bundle = VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        parameters=params,
    )
    if use_cache:
        _RECURRENT_CACHE[key] = bundle
    return bundle


def load_attractor_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    use_cache: bool = True,
    **kwargs,
) -> VentralBundle:
    """Train convergent attractor recurrent cortex (generative completeness path)."""
    key = (seed, n_examples, k, frozenset(kwargs.items()))
    if use_cache and key in _ATTRACTOR_CACHE:
        return _ATTRACTOR_CACHE[key]

    from neural_assemblies.programs.colt_mnist_attractor import train_attractor_brain

    brain, high_outputs, prototypes, high_bias, k, examples, generative_prototypes = train_attractor_brain(
        seed=seed, n_examples=n_examples, k=k, **kwargs,
    )
    from neural_assemblies.programs.colt_mnist_data import find_mnist_dir

    bundle = VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        parameters={
            "seed": seed,
            "n_examples": n_examples,
            "k": k,
            "base": "attractor_recurrent",
            "generative_prototypes": generative_prototypes,
            **kwargs,
        },
    )
    if use_cache:
        _ATTRACTOR_CACHE[key] = bundle
    return bundle


def load_spatial_ventral_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    use_cache: bool = True,
    **kwargs,
) -> VentralBundle:
    """Train spatial RF LOW→MID→HIGH ventral stream (CNN-hierarchy analogue)."""
    key = (seed, n_examples, k, frozenset(kwargs.items()))
    if use_cache and key in _SPATIAL_CACHE:
        return _SPATIAL_CACHE[key]

    from neural_assemblies.programs.colt_mnist_spatial_ventral import train_spatial_ventral_brain

    brain, high_outputs, prototypes, high_bias, k, examples, stats = train_spatial_ventral_brain(
        seed=seed, n_examples=n_examples, k=k, **kwargs,
    )
    bundle = VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        parameters={
            "seed": seed,
            "n_examples": n_examples,
            "k": k,
            "base": "spatial_local_rf_ventral",
            "spatial_stats": stats,
            **kwargs,
        },
    )
    if use_cache:
        _SPATIAL_CACHE[key] = bundle
    return bundle


def load_multiscale_spatial_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    use_cache: bool = True,
    enable_generative_head: bool = True,
    radii: tuple[int, ...] = (1, 3),
    **kwargs,
) -> VentralBundle:
    """Multi-scale spatial RF pyramid with optional generative dual-head."""
    key = (seed, n_examples, k, enable_generative_head, radii, frozenset(kwargs.items()))
    if use_cache and key in _MULTISCALE_CACHE:
        return _MULTISCALE_CACHE[key]

    from neural_assemblies.programs.colt_mnist_attractor import (
        apply_forward_generative_curriculum,
        capture_generative_prototypes,
    )
    from neural_assemblies.programs.colt_mnist_spatial_ventral import train_spatial_ventral_brain
    from neural_assemblies.programs.patch_graph import build_grid_patch_graph

    patch_graph = build_grid_patch_graph(grid=3, radii=radii, seed=seed)
    patch_passes = kwargs.pop("patch_curriculum_passes", 2)
    brain, high_outputs, prototypes, high_bias, k, examples, stats = train_spatial_ventral_brain(
        seed=seed,
        n_examples=n_examples,
        k=k,
        radii=radii,
        digit3_center_passes=0,
        patch_curriculum=("center_patch", "top_half_patches"),
        patch_graph=patch_graph,
        patch_curriculum_passes=patch_passes,
        **kwargs,
    )
    generative_prototypes = None
    if enable_generative_head:
        apply_forward_generative_curriculum(
            brain, high_bias, examples, n_examples, seed=seed, enable_top_down_low=False,
            patch_graph=patch_graph,
            absence_protocols=tuple(patch_graph.absence_protocols.keys())[:3]
            if patch_graph is not None
            else None,
        )
        from neural_assemblies.core.brain import Brain
        typed_brain = cast(Brain, brain)
        generative_prototypes = capture_generative_prototypes(
            typed_brain, high_bias, examples, n_examples, k, typed_brain.areas["HIGH"].n,
        )

    params = {
        "seed": seed,
        "n_examples": n_examples,
        "k": k,
        "base": "spatial_multiscale_pyramid",
        "spatial_stats": stats,
        "patch_graph": "grid_multiscale",
        "radii": radii,
        "enable_generative_head": enable_generative_head,
        **kwargs,
    }
    if generative_prototypes is not None:
        params["generative_prototypes"] = generative_prototypes

    bundle = VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        parameters=params,
    )
    if use_cache:
        _MULTISCALE_CACHE[key] = bundle
    return bundle


def load_fashion_spatial_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    use_cache: bool = True,
    **kwargs,
) -> VentralBundle:
    """Train Fashion-MNIST multi-scale spatial ventral with patch curriculum."""
    key = (seed, n_examples, k, frozenset(kwargs.items()))
    if use_cache and key in _FASHION_CACHE:
        return _FASHION_CACHE[key]

    from neural_assemblies.programs.fashion_ventral import train_fashion_spatial_bundle

    bundle = train_fashion_spatial_bundle(
        seed=seed, n_examples=n_examples, k=k, **kwargs,
    )
    if use_cache:
        _FASHION_CACHE[key] = bundle
    return bundle


def clear_ventral_bundle_cache() -> None:
    """Drop cached bundles (e.g. between tests with different seeds)."""
    _BUNDLE_CACHE.clear()
    _RECURRENT_CACHE.clear()
    _ATTRACTOR_CACHE.clear()
    _SPATIAL_CACHE.clear()
    _MULTISCALE_CACHE.clear()
    _FASHION_CACHE.clear()


def connectome_predict(
    high_vec: np.ndarray,
    brain,
    *,
    k: int,
    src: str = "HIGH",
    dst: str = "CLASS",
) -> int:
    """Connectome sum readout (matches advanced ventral stream)."""
    from neural_assemblies.programs.colt_mnist_advanced_util import (
        read_class_connectome_scores,
    )

    scores = read_class_connectome_scores(high_vec, brain, src, dst, k, NUM_DIGITS)
    return int(np.argmax(scores))


def eval_high_outputs(
    high_outputs: np.ndarray,
    predict_fn,
    *,
    n_examples: int,
) -> np.ndarray:
    """Per-class accuracy from a predict(high_vec)->digit callable."""
    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            predict_fn(high_outputs[digit, j]) == digit
            for j in range(n_examples)
        )
        correct[digit] = hits / n_examples
    return correct
