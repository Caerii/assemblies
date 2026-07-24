"""
CNN-inspired local receptive-field connectomes for MNIST ventral encoding.

Replaces dense random LOW→MID with spatially local, translation-tiled fibers
(CNN first-layer analogue).  Used by ``colt_mnist_spatial_ventral``.
"""

from __future__ import annotations

import numpy as np

IMG_SIDE = 28
N_PIXELS = IMG_SIDE * IMG_SIDE


def pixel_index(row: int, col: int) -> int:
    return row * IMG_SIDE + col


def local_patch_indices(
    center_row: int,
    center_col: int,
    *,
    radius: int = 2,
) -> np.ndarray:
    """Flat indices of a (2*radius+1)^2 neighborhood, clipped to image bounds."""
    indices: list[int] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = center_row + dr, center_col + dc
            if 0 <= r < IMG_SIDE and 0 <= c < IMG_SIDE:
                indices.append(pixel_index(r, c))
    return np.asarray(indices, dtype=np.int32)


def init_spatial_low_mid_weights(
    rng: np.random.Generator,
    *,
    n_low: int = N_PIXELS,
    n_mid: int = 2000,
    rf_radius: int = 2,
    p_within_rf: float = 1.0,
) -> np.ndarray:
    """
    Local RF LOW→MID: each MID column samples a tiled 5×5 neighborhood.

    Centers are placed on a grid over 28×28 (CNN stride analogue); columns
    beyond grid count wrap with jitter.
    """
    a = np.zeros((n_low, n_mid), dtype=np.float64)
    grid_side = int(np.ceil(np.sqrt(n_mid)))
    col_idx = 0
    for gr in range(grid_side):
        for gc in range(grid_side):
            if col_idx >= n_mid:
                break
            cy = int((gr + 0.5) * IMG_SIDE / grid_side) % IMG_SIDE
            cx = int((gc + 0.5) * IMG_SIDE / grid_side) % IMG_SIDE
            if rng.random() < 0.3:
                cy = int(rng.integers(0, IMG_SIDE))
                cx = int(rng.integers(0, IMG_SIDE))
            patch = local_patch_indices(cy, cx, radius=rf_radius)
            if p_within_rf < 1.0:
                mask = rng.random(patch.size) < p_within_rf
                patch = patch[mask]
            if patch.size == 0:
                patch = local_patch_indices(cy, cx, radius=rf_radius)
            a[patch, col_idx] = 1.0
            col_idx += 1
    while col_idx < n_mid:
        cy = int(rng.integers(0, IMG_SIDE))
        cx = int(rng.integers(0, IMG_SIDE))
        patch = local_patch_indices(cy, cx, radius=rf_radius)
        a[patch, col_idx] = 1.0
        col_idx += 1
    col_sums = np.maximum(a.sum(axis=0, keepdims=True), 1e-12)
    a /= col_sums
    return a


def init_spatial_half_connectome(
    rng: np.random.Generator,
    *,
    n_low: int = N_PIXELS,
    n_mid: int = 2000,
    rf_radius: int = 2,
    half: str = "top",
) -> np.ndarray:
    """Local RF connectome restricted to top or bottom image half."""
    a = init_spatial_low_mid_weights(
        rng, n_low=n_low, n_mid=n_mid, rf_radius=rf_radius,
    )
    mid_pixel = n_low // 2
    if half == "top":
        a[mid_pixel:, :] = 0.0
    else:
        a[:mid_pixel, :] = 0.0
    col_sums = np.maximum(a.sum(axis=0, keepdims=True), 1e-12)
    a /= col_sums
    return a


def init_spatial_patch_connectome(
    rng: np.random.Generator,
    patch_indices: np.ndarray,
    *,
    n_low: int = N_PIXELS,
    n_mid: int = 400,
    rf_radius: int = 2,
) -> np.ndarray:
    """LOW→MID connectome reading only from ``patch_indices`` support."""
    a = init_spatial_low_mid_weights(
        rng, n_low=n_low, n_mid=n_mid, rf_radius=rf_radius,
    )
    allowed = np.zeros(n_low, dtype=bool)
    allowed[np.asarray(patch_indices, dtype=int)] = True
    a[~allowed, :] = 0.0
    col_sums = np.maximum(a.sum(axis=0, keepdims=True), 1e-12)
    a /= col_sums
    return a


def init_multiscale_low_mid_weights(
    rng: np.random.Generator,
    *,
    n_low: int = N_PIXELS,
    n_mid: int = 2000,
    radii: tuple[int, ...] = (1, 3),
    grid: int = 5,
    share_across_scales: bool = True,
) -> np.ndarray:
    """
    Pyramid LOW→MID: concatenate local RFs at multiple radii into one MID area.

    Each scale occupies a contiguous column block; optional weight sharing
    reuses the same relative RF offsets across scales (translation prior).
    """
    n_scales = len(radii)
    cols_per_scale = n_mid // n_scales
    a = np.zeros((n_low, n_mid), dtype=np.float64)
    base_offsets: dict[int, np.ndarray] | None = {} if share_across_scales else None

    for si, radius in enumerate(radii):
        col_start = si * cols_per_scale
        col_end = col_start + cols_per_scale if si < n_scales - 1 else n_mid
        n_cols = col_end - col_start
        sub = init_spatial_low_mid_weights(
            rng, n_low=n_low, n_mid=n_cols, rf_radius=radius,
        )
        if base_offsets is not None and radius not in base_offsets:
            for c in range(n_cols):
                idx = np.flatnonzero(sub[:, c] > 0)
                if idx.size:
                    cy = int(idx.mean()) // IMG_SIDE
                    cx = int(idx.mean()) % IMG_SIDE
                    base_offsets[radius] = idx - (cy * IMG_SIDE + cx)
        a[:, col_start:col_end] = sub

    col_sums = np.maximum(a.sum(axis=0, keepdims=True), 1e-12)
    a /= col_sums
    return a


def spatial_connectome_stats(a_lm: np.ndarray) -> dict[str, float]:
    """Diagnostic: mean RF fan-in per MID column."""
    fan_in = (a_lm > 0).sum(axis=0)
    return {
        "mean_rf_fan_in": float(fan_in.mean()),
        "max_rf_fan_in": float(fan_in.max()),
        "fraction_local": float((fan_in <= 25).mean()),
    }
