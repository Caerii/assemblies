"""
Patch binding empirical panel — multi-scale spatial, merge routing, synthesis.

Run::

    python -m neural_assemblies.programs.patch_binding_panel --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_assemblies.programs.colt_mnist_synthesis import audit_bundle
from neural_assemblies.programs.patch_graph import (
    PatchGraph,
    build_fashion_salient_patch_graph,
    build_grid_patch_graph,
    build_halves_patch_graph,
)


@dataclass
class PatchBindingRow:
    name: str
    mean_accuracy: float | None
    routing_fidelity: float | None
    generative_viable: bool | None
    assembly_stable: bool | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchBindingPanel:
    rows: list[PatchBindingRow]
    patch_graphs: dict[str, dict[str, Any]]
    narrative: str


def _audit_row(name: str, bundle, *, seed: int = 42) -> PatchBindingRow:
    card = audit_bundle(bundle, label=name, seed=seed)
    return PatchBindingRow(
        name=name,
        mean_accuracy=card.discriminative_accuracy,
        routing_fidelity=None,
        generative_viable=card.generative_viable,
        assembly_stable=card.assembly_stable,
        details={
            "digit3_absence": card.digit3_absence_accuracy,
            "forward_center_band": card.forward_center_band_overlap,
            "pattern_complete": card.pattern_complete_recovery,
            "confused_pair_overlap": card.mean_confused_pair_overlap,
        },
    )


def run_patch_binding_panel(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
) -> PatchBindingPanel:
    from neural_assemblies.programs.colt_mnist_tier_a import run_merge_halves_mnist
    from neural_assemblies.programs.patch_merge import run_grid_patch_merge_mnist
    from neural_assemblies.programs.fashion_ventral import run_fashion_spatial_mnist
    from neural_assemblies.programs.colt_mnist_tier_util import (
        clear_ventral_bundle_cache,
        load_multiscale_spatial_bundle,
        load_recurrent_bundle,
        load_spatial_ventral_bundle,
        load_ventral_bundle,
    )
    from neural_assemblies.programs.vision_data import ensure_fashion_mnist_csv

    clear_ventral_bundle_cache()
    ensure_fashion_mnist_csv()
    kw = dict(seed=seed, n_examples=n_examples, k=k, use_cache=False)

    graphs = {
        "halves": _graph_summary(build_halves_patch_graph()),
        "grid_4x4": _graph_summary(build_grid_patch_graph(grid=4, radii=(2,))),
        "grid_multiscale": _graph_summary(build_grid_patch_graph(grid=3, radii=(1, 3))),
        "fashion_salient": _graph_summary(build_fashion_salient_patch_graph()),
    }

    rows: list[PatchBindingRow] = []

    rec = load_recurrent_bundle(**kw)
    rows.append(_audit_row("recurrent", rec, seed=seed))

    spatial = load_spatial_ventral_bundle(**kw)
    rows.append(_audit_row("spatial_single_rf", spatial, seed=seed))

    ms = load_multiscale_spatial_bundle(**kw)
    rows.append(_audit_row("spatial_multiscale_gen", ms, seed=seed))

    ventral_ref = load_ventral_bundle(**kw)
    grid_merge = run_grid_patch_merge_mnist(
        bundle=ventral_ref, seed=seed, n_examples=n_examples, k=k, grid=2,
        merge_mode="chain",
    )
    rows.append(PatchBindingRow(
        name="grid_patch_merge_2x2",
        mean_accuracy=grid_merge.mean_accuracy,
        routing_fidelity=grid_merge.extra.get("routing_fidelity"),
        generative_viable=None,
        assembly_stable=None,
        details={
            "n_patches": grid_merge.extra.get("n_patches"),
            "grid": 2,
            "merge_mode": "chain",
        },
    ))

    merge = run_merge_halves_mnist(
        bundle=ventral_ref, seed=seed, n_examples=n_examples, k=k,
    )
    rows.append(PatchBindingRow(
        name="merge_halves_spatial_teacher",
        mean_accuracy=merge.mean_accuracy,
        routing_fidelity=merge.extra.get("routing_fidelity"),
        generative_viable=None,
        assembly_stable=None,
        details={
            "use_spatial_halves": merge.parameters.get("use_spatial_halves"),
            "teacher_align": merge.parameters.get("teacher_align"),
        },
    ))

    fashion = run_fashion_spatial_mnist(seed=seed, n_examples=n_examples, k=k)
    rows.append(PatchBindingRow(
        name="fashion_spatial_patch",
        mean_accuracy=fashion.mean_accuracy,
        routing_fidelity=None,
        generative_viable=None,
        assembly_stable=None,
        details={
            "data_source": fashion.data_source,
            "confused_digit_accuracy": fashion.extra.get("confused_digit_accuracy"),
            "patch_absence_accuracy": fashion.extra.get("patch_absence_accuracy"),
        },
    ))

    narrative = _format_narrative(rows, graphs)
    narrative += (
        "\n\nNote: H9 falsified — part-merge streams increase confused-pair overlap "
        "(~0.94 vs ~0.23 ventral). Merge rows kept for audit only; not on critical path."
    )
    return PatchBindingPanel(rows=rows, patch_graphs=graphs, narrative=narrative)


def _graph_summary(g: PatchGraph) -> dict[str, Any]:
    return {
        "n_patches": len(g.patches),
        "n_bind_pairs": len(g.bind_pairs),
        "scales": sorted({p.scale for p in g.patches}),
        "absence_protocols": list(g.absence_protocols.keys()),
        "patches": [p.label for p in g.patches],
    }


def _format_narrative(rows: list[PatchBindingRow], graphs: dict[str, dict]) -> str:
    lines = [
        "Patch binding panel",
        "=" * 40,
        "",
        "Patch graphs:",
    ]
    for name, g in graphs.items():
        lines.append(
            f"  {name}: {g['n_patches']} patches, {g['n_bind_pairs']} bind pairs, "
            f"scales={g['scales']}, protocols={g['absence_protocols']}"
        )
    lines.extend(["", "Empirical rows:"])
    for r in rows:
        extra = ""
        if r.routing_fidelity is not None:
            extra = f"  routing={r.routing_fidelity:.1%}"
        if r.generative_viable is not None:
            extra += f"  gen={'YES' if r.generative_viable else 'no'}"
        lines.append(f"  {r.name}: acc={r.mean_accuracy:.1%}{extra}  {r.details}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Patch binding empirical panel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()

    panel = run_patch_binding_panel(
        seed=args.seed, n_examples=args.n_examples, k=args.k,
    )
    print(panel.narrative)


if __name__ == "__main__":
    main()
