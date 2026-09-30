"""
Vision generalization panel — MNIST vs Fashion-MNIST under matched spatial training.

Isolates whether the spatial multiscale ventral stack generalizes beyond stroke-local
MNIST structure or is dataset-specific.

Run::

    python -m neural_assemblies.programs.colt_mnist_vision_generalization --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_assemblies.programs.colt_mnist_geometry_panel import prototype_overlap_matrix
from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_tier_util import (
    VentralBundle,
    clear_ventral_bundle_cache,
    connectome_predict,
    load_multiscale_spatial_bundle,
)


@dataclass
class GeneralizationRow:
    name: str
    dataset: str
    mean_accuracy: float
    confused_digit_accuracy: float
    mean_prototype_overlap: float
    radii: tuple[int, ...]
    train_iters: int | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionGeneralizationPanel:
    rows: list[GeneralizationRow]
    narrative: str


def _evaluate_bundle(
    bundle: VentralBundle,
    confused_digits: set[int],
) -> tuple[float, float, float, dict[str, float]]:
    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes

    wire_class_from_prototypes(bundle.brain, bundle.prototypes, HIGH, CLASS, bundle.k)
    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], bundle.brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        )
        correct[digit] = hits / bundle.n_examples
    mat = prototype_overlap_matrix(bundle.prototypes, bundle.k)
    triu = mat[np.triu_indices(NUM_DIGITS, k=1)]
    per_digit = {str(d): float(correct[d]) for d in range(NUM_DIGITS)}
    confused_acc = float(np.mean([correct[d] for d in confused_digits]))
    return float(correct.mean()), confused_acc, float(np.mean(triu)), per_digit


def run_vision_generalization_panel(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
) -> VisionGeneralizationPanel:
    from neural_assemblies.programs.fashion_ventral import train_fashion_spatial_bundle
    from neural_assemblies.programs.vision_data import (
        FASHION_CONFUSED_LABELS,
        VisionDataset,
        vision_data_info,
    )

    clear_ventral_bundle_cache()
    kw = dict(seed=seed, n_examples=n_examples, k=k, use_cache=False)
    train_kw = {key: val for key, val in kw.items() if key != "use_cache"}
    mnist_info = vision_data_info(VisionDataset.MNIST)
    fashion_info = vision_data_info(VisionDataset.FASHION_MNIST)
    fashion_confused = set(FASHION_CONFUSED_LABELS)

    configs: list[tuple[str, str, dict[str, Any], set[int], str]] = [
        (
            "mnist_multiscale_r13",
            "mnist",
            {"radii": (1, 3), "patch_curriculum_passes": 2},
            {d for pair in mnist_info.confused_pairs for d in pair},
            mnist_info.data_source,
        ),
        (
            "fashion_multiscale_r13",
            "fashion",
            {"radii": (1, 3), "patch_curriculum_passes": 2, "train_iters": n_examples},
            fashion_confused,
            fashion_info.data_source,
        ),
        (
            "fashion_multiscale_r135",
            "fashion",
            {"radii": (1, 3, 5), "patch_curriculum_passes": 2, "train_iters": n_examples},
            fashion_confused,
            fashion_info.data_source,
        ),
        (
            "fashion_multiscale_r13_3x_iters",
            "fashion",
            {"radii": (1, 3), "patch_curriculum_passes": 3, "train_iters": n_examples * 3},
            fashion_confused,
            fashion_info.data_source,
        ),
    ]

    rows: list[GeneralizationRow] = []
    for name, dataset, params, confused_digits, source in configs:
        if dataset == "mnist":
            bundle = load_multiscale_spatial_bundle(**kw, **params)
        else:
            bundle = train_fashion_spatial_bundle(**train_kw, **params)
        mean_acc, conf_acc, mean_ov, per_digit = _evaluate_bundle(bundle, confused_digits)
        rows.append(GeneralizationRow(
            name=name,
            dataset=dataset,
            mean_accuracy=mean_acc,
            confused_digit_accuracy=conf_acc,
            mean_prototype_overlap=mean_ov,
            radii=params["radii"],
            train_iters=params.get("train_iters"),
            details={"per_digit_accuracy": per_digit, "data_source": source},
        ))

    narrative = _format_narrative(rows, mnist_info.data_source, fashion_info.data_source)
    return VisionGeneralizationPanel(rows=rows, narrative=narrative)


def _format_narrative(
    rows: list[GeneralizationRow],
    mnist_source: str,
    fashion_source: str,
) -> str:
    lines = [
        "Vision generalization panel",
        "=" * 40,
        f"MNIST source: {mnist_source}",
        f"Fashion source: {fashion_source}",
        "",
    ]
    for r in rows:
        iters_s = f"  iters={r.train_iters}" if r.train_iters else ""
        lines.append(
            f"  {r.name}: acc={r.mean_accuracy:.1%}  "
            f"confused={r.confused_digit_accuracy:.1%}  "
            f"mean_pair_ov={r.mean_prototype_overlap:.3f}  "
            f"radii={r.radii}{iters_s}"
        )
        worst = min(r.details.get("per_digit_accuracy", {}).items(), key=lambda x: x[1], default=("?", 0))
        best = max(r.details.get("per_digit_accuracy", {}).items(), key=lambda x: x[1], default=("?", 0))
        lines.append(f"    per_digit worst={worst[0]}:{float(worst[1]):.0%}  best={best[0]}:{float(best[1]):.0%}")
    mnist_acc = next((r.mean_accuracy for r in rows if r.dataset == "mnist"), 0.0)
    fashion_best = max((r.mean_accuracy for r in rows if r.dataset == "fashion"), default=0.0)
    gap = mnist_acc - fashion_best
    lines.extend([
        "",
        f"MNIST vs best Fashion gap: {gap:.1%}",
        "Gap >15pp suggests spatial RF is MNIST-biased; <15pp suggests partial transfer.",
    ])
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MNIST vs Fashion spatial generalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()

    panel = run_vision_generalization_panel(
        seed=args.seed, n_examples=args.n_examples, k=args.k,
    )
    print(panel.narrative)


if __name__ == "__main__":
    main()
