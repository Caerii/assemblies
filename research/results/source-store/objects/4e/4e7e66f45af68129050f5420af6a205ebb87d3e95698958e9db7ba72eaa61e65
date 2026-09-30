"""
Fashion-MNIST spatial ventral stream with patch-graph generative curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes
from neural_assemblies.programs.colt_mnist_attractor import (
    apply_forward_generative_curriculum,
    capture_generative_prototypes,
)
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
)
from neural_assemblies.programs.colt_mnist_spatial_ventral import (
    forward_high_spatial,
    train_spatial_ventral_brain,
)
from neural_assemblies.programs.colt_mnist_tier_util import VentralBundle, connectome_predict
from neural_assemblies.programs.patch_graph import build_fashion_salient_patch_graph
from neural_assemblies.programs.vision_data import (
    FASHION_CONFUSED_LABELS,
    VisionDataset,
    load_vision_examples,
)


FASHION_PATCH_CURRICULUM: tuple[str, ...] = (
    "collar_occlude", "torso_occlude", "hem_occlude", "upper_body",
)


@dataclass
class FashionVentralResult(ColtMnistHierarchicalBrainResult):
    tier: str = "R"
    method: str = "fashion_spatial_patch"
    extra: dict = field(default_factory=dict)


def train_fashion_spatial_bundle(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    radii: tuple[int, ...] = (1, 3, 5),
    patch_curriculum_passes: int = 3,
    train_iters: int | None = None,
    enable_generative_head: bool = True,
) -> VentralBundle:
    """Train multi-scale spatial ventral on Fashion-MNIST with patch curriculum."""
    examples, info = load_vision_examples(
        VisionDataset.FASHION_MNIST, n_examples=n_examples, cap_size=k,
    )
    graph = build_fashion_salient_patch_graph()
    if train_iters is None:
        train_iters = n_examples * 3

    brain, high_outputs, prototypes, high_bias, k, _, stats = train_spatial_ventral_brain(
        seed=seed,
        n_examples=n_examples,
        k=k,
        radii=radii,
        train_iters=train_iters,
        digit3_center_passes=0,
        examples=examples,
        patch_curriculum=FASHION_PATCH_CURRICULUM,
        patch_graph=graph,
        patch_curriculum_passes=patch_curriculum_passes,
    )
    apply_forward_generative_curriculum(
        brain, high_bias, examples, n_examples, seed=seed, enable_top_down_low=False,
        patch_graph=graph,
        absence_protocols=tuple(graph.absence_protocols.keys())[:3],
    )
    generative_prototypes = None
    typed_brain = cast(Brain, brain)
    if enable_generative_head:
        generative_prototypes = capture_generative_prototypes(
            typed_brain, high_bias, examples, n_examples, k, typed_brain.areas[HIGH].n,
        )

    typed_brain.disable_plasticity = True
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = forward_high_spatial(
                brain, examples[digit, j], high_bias,
            )

    params = {
        "seed": seed,
        "n_examples": n_examples,
        "k": k,
        "base": "fashion_spatial_multiscale",
        "patch_graph": "fashion_salient",
        "dataset": "fashion_mnist",
        "radii": radii,
        "spatial_stats": stats,
    }
    if generative_prototypes is not None:
        params["generative_prototypes"] = generative_prototypes

    return VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source=info.data_source,
        parameters=params,
    )


def run_fashion_spatial_mnist(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    **kwargs,
) -> FashionVentralResult:
    """Evaluate Fashion-MNIST spatial + patch curriculum bundle."""
    bundle = train_fashion_spatial_bundle(
        seed=seed, n_examples=n_examples, k=k, **kwargs,
    )
    wire_class_from_prototypes(bundle.brain, bundle.prototypes, HIGH, CLASS, k)
    graph = build_fashion_salient_patch_graph()

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], bundle.brain, k=k) == digit
            for j in range(n_examples)
        )
        correct[digit] = hits / n_examples

    from neural_assemblies.programs.colt_mnist_forward_completion import encode_and_predict

    patch_hits = patch_total = 0
    for proto in ("collar_occlude", "torso_occlude"):
        for digit in sorted(FASHION_CONFUSED_LABELS)[:4]:
            for j in range(n_examples):
                pat = graph.apply_absence(bundle.examples[digit, j], proto)
                pred, _, _ = encode_and_predict(bundle, pat, head="auto")
                patch_total += 1
                patch_hits += pred == digit
    patch_abs_acc = patch_hits / max(patch_total, 1)
    confused = float(np.mean([correct[d] for d in FASHION_CONFUSED_LABELS]))

    return FashionVentralResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=bundle.data_source,
        parameters=bundle.parameters,
        backend="fashion_spatial_patch",
        extra={
            "confused_digit_accuracy": confused,
            "patch_absence_accuracy": patch_abs_acc,
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fashion-MNIST spatial ventral")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    args = parser.parse_args()

    r = run_fashion_spatial_mnist(seed=args.seed, n_examples=args.n_examples)
    print(f"accuracy={r.mean_accuracy:.1%}  confused={r.extra.get('confused_digit_accuracy'):.1%}")
    print(f"patch_absence={r.extra.get('patch_absence_accuracy'):.1%}")


if __name__ == "__main__":
    main()
