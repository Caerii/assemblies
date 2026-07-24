"""Patch graph registry and bundle helpers."""

from __future__ import annotations

from neural_assemblies.programs.patch_graph import (
    PatchGraph,
    build_fashion_salient_patch_graph,
    build_grid_patch_graph,
    build_halves_patch_graph,
)

from typing import Callable

PATCH_GRAPH_BUILDERS: dict[str, Callable[[], PatchGraph]] = {
    "halves": build_halves_patch_graph,
    "grid_2": lambda: build_grid_patch_graph(grid=2, radii=(3,)),
    "grid_4": lambda: build_grid_patch_graph(grid=4, radii=(2,)),
    "grid_multiscale": lambda: build_grid_patch_graph(grid=3, radii=(1, 3)),
    "fashion_salient": build_fashion_salient_patch_graph,
}


def resolve_patch_graph(bundle) -> PatchGraph | None:
    """Return ``PatchGraph`` for a bundle when ``parameters['patch_graph']`` is set."""
    key = bundle.parameters.get("patch_graph") if bundle is not None else None
    if not key:
        return None
    builder = PATCH_GRAPH_BUILDERS.get(key)
    if builder is None:
        return None
    seed = bundle.parameters.get("patch_graph_seed", bundle.parameters.get("seed", 42))
    if key == "grid_multiscale":
        radii = bundle.parameters.get("patch_radii", bundle.parameters.get("radii", (1, 3)))
        return build_grid_patch_graph(grid=3, radii=tuple(radii), seed=seed)
    if key.startswith("grid_"):
        parts = key.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            grid = int(parts[1])
            radii = bundle.parameters.get("patch_radii", (2,))
            return build_grid_patch_graph(grid=grid, radii=tuple(radii), seed=seed)
    return builder()


def patch_absence_protocols(bundle) -> tuple[str, ...]:
    """Named absence protocols available for this bundle."""
    graph = resolve_patch_graph(bundle)
    if graph is not None:
        legacy = ("top_half", "bottom_half", "center_band")
        return tuple(dict.fromkeys(list(graph.absence_protocols.keys()) + list(legacy)))
    from neural_assemblies.programs.colt_mnist_absence import STRUCTURED_PROTOCOLS
    return STRUCTURED_PROTOCOLS
