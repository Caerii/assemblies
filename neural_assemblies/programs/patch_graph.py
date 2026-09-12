"""
Patch graph — dataset-agnostic spatial patch lattice for assembly-calculus vision.

A **patch** is a local receptive field (center, radius, scale) over a fixed
image grid.  A **PatchGraph** declares:

* patch specs at multiple scales (fine → coarse pyramid)
* **bind pairs** — which patches ``merge`` / cross-project into a bind hub
* **absence protocols** — structured occlusion per patch (generative curriculum)

Used by spatial ventral, merge-halves, and Fashion-MNIST adapters.  MNIST
row-band masks are one legacy specialization of patch absence.

Run empirical panel::

    python -m neural_assemblies.programs.patch_binding_panel --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

IMG_SIDE = 28
N_PIXELS = IMG_SIDE * IMG_SIDE


class AbsenceMode(str, Enum):
    ZERO = "zero"
    NOISE = "noise"


@dataclass(frozen=True)
class PatchSpec:
    """One local receptive field on the image grid."""

    patch_id: int
    center_row: int
    center_col: int
    radius: int
    scale: int = 0
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or f"p{self.patch_id}_s{self.scale}_r{self.center_row}c{self.center_col}"


@dataclass
class PatchGraph:
    """
    Spatial patch lattice + bind topology for one image size.

    ``bind_pairs`` lists undirected patch pairs that should participate in
    cross-patch ``merge`` (or simultaneous project) during binding.
    """

    img_side: int = IMG_SIDE
    patches: tuple[PatchSpec, ...] = ()
    bind_pairs: tuple[tuple[int, int], ...] = ()
    absence_protocols: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def n_pixels(self) -> int:
        return self.img_side * self.img_side

    def patch_indices(self, spec: PatchSpec) -> np.ndarray:
        """Flat pixel indices in the RF neighborhood."""
        indices: list[int] = []
        for dr in range(-spec.radius, spec.radius + 1):
            for dc in range(-spec.radius, spec.radius + 1):
                r, c = spec.center_row + dr, spec.center_col + dc
                if 0 <= r < self.img_side and 0 <= c < self.img_side:
                    indices.append(r * self.img_side + c)
        return np.asarray(indices, dtype=np.int32)

    def patch_mask(self, spec: PatchSpec, *, dtype=np.float32) -> np.ndarray:
        """Binary 784-d mask for one patch."""
        mask = np.zeros(self.n_pixels, dtype=dtype)
        mask[self.patch_indices(spec)] = 1.0
        return mask

    def apply_absence(
        self,
        pattern: np.ndarray,
        protocol: str,
        *,
        rng: np.random.Generator | None = None,
        mode: AbsenceMode = AbsenceMode.ZERO,
    ) -> np.ndarray:
        """Zero (or noise) pixels in the patches named by ``protocol``."""
        if protocol in ("top_half", "bottom_half"):
            return _apply_row_half_absence(pattern, protocol, img_side=self.img_side)
        patch_ids = self.absence_protocols.get(protocol)
        if patch_ids is None:
            raise ValueError(f"Unknown patch absence protocol: {protocol!r}")
        out = np.asarray(pattern, dtype=pattern.dtype).copy()
        rng = rng or np.random.default_rng(0)
        for pid in patch_ids:
            spec = self.patches[pid]
            idx = self.patch_indices(spec)
            if mode == AbsenceMode.ZERO:
                out[idx] = 0
            else:
                out[idx] = rng.random(idx.size)
        return out

    def split_patch_fields(
        self,
        pattern: np.ndarray,
        patch_ids: tuple[int, ...],
    ) -> list[np.ndarray]:
        """
        Extract patch-local fields embedded in full LOW vectors.

        Each field is a length-``n_pixels`` vector with nonzeros only in the
        patch support (merge-halves uses two fields; grid binding uses K).
        """
        pat = np.asarray(pattern).reshape(-1)
        fields: list[np.ndarray] = []
        for pid in patch_ids:
            spec = self.patches[pid]
            field = np.zeros(self.n_pixels, dtype=pat.dtype)
            idx = self.patch_indices(spec)
            field[idx] = pat[idx]
            fields.append(field)
        return fields


def _apply_row_half_absence(
    pattern: np.ndarray,
    protocol: str,
    *,
    img_side: int = IMG_SIDE,
) -> np.ndarray:
    """Legacy row-half masks (merge-halves field layout)."""
    img = np.asarray(pattern).reshape(img_side, img_side).copy()
    mid = img_side // 2
    if protocol == "top_half":
        img[:mid, :] = 0
    elif protocol == "bottom_half":
        img[mid:, :] = 0
    else:
        raise ValueError(protocol)
    return img.reshape(-1)


def pixel_index(row: int, col: int, *, img_side: int = IMG_SIDE) -> int:
    return row * img_side + col


def build_grid_patch_graph(
    *,
    img_side: int = IMG_SIDE,
    grid: int = 4,
    radii: tuple[int, ...] = (2,),
    seed: int = 42,
) -> PatchGraph:
    """
    Tiled patch lattice at one or more scales (radius per scale level).

    ``scale=0`` uses ``radii[0]``, ``scale=1`` uses ``radii[1]``, etc.
    Centers are placed on a ``grid × grid`` lattice over the image.
    """
    rng = np.random.default_rng(seed)
    patches: list[PatchSpec] = []
    pid = 0
    for scale, radius in enumerate(radii):
        for gr in range(grid):
            for gc in range(grid):
                cy = int((gr + 0.5) * img_side / grid) % img_side
                cx = int((gc + 0.5) * img_side / grid) % img_side
                if rng.random() < 0.15:
                    cy = int(rng.integers(0, img_side))
                    cx = int(rng.integers(0, img_side))
                patches.append(PatchSpec(
                    patch_id=pid,
                    center_row=cy,
                    center_col=cx,
                    radius=radius,
                    scale=scale,
                    name=f"g{grid}_s{scale}_r{cy}c{cx}",
                ))
                pid += 1

    bind_pairs: list[tuple[int, int]] = []
    by_scale: dict[int, list[int]] = {}
    for p in patches:
        by_scale.setdefault(p.scale, []).append(p.patch_id)
    for _scale, ids in by_scale.items():
        for i in range(len(ids) - 1):
            bind_pairs.append((ids[i], ids[i + 1]))
        if len(ids) >= 2:
            bind_pairs.append((ids[0], ids[-1]))

    absence_protocols: dict[str, tuple[int, ...]] = {}
    if patches:
        mid = len(patches) // 2
        absence_protocols["center_patch"] = (mid,)
        absence_protocols["left_half_patches"] = tuple(
            p.patch_id for p in patches if p.center_col < img_side // 2
        )
        absence_protocols["top_half_patches"] = tuple(
            p.patch_id for p in patches if p.center_row < img_side // 2
        )

    return PatchGraph(
        img_side=img_side,
        patches=tuple(patches),
        bind_pairs=tuple(bind_pairs),
        absence_protocols=absence_protocols,
    )


def build_halves_patch_graph(*, img_side: int = IMG_SIDE) -> PatchGraph:
    """TOP / BOT part areas — merge-halves specialization."""
    mid_row = img_side // 2
    top = PatchSpec(
        0, center_row=mid_row // 2, center_col=img_side // 2,
        radius=mid_row // 2 + 1, name="TOP",
    )
    bot = PatchSpec(
        1, center_row=mid_row + mid_row // 2, center_col=img_side // 2,
        radius=mid_row // 2 + 1, name="BOT",
    )
    return PatchGraph(
        img_side=img_side,
        patches=(top, bot),
        bind_pairs=((0, 1),),
        absence_protocols={
            "top_half": (0,),
            "bottom_half": (1,),
            "center_row_band": tuple(),
        },
    )


def build_fashion_salient_patch_graph(*, img_side: int = IMG_SIDE) -> PatchGraph:
    """
    Hand-specified salient regions for Fashion-MNIST (28×28).

    Collar/chest, torso, hem, and footwear zones — diagnostic for
    shirt / coat / trouser / shoe confusions.
    """
    patches = (
        PatchSpec(0, 6, 14, 4, scale=1, name="collar_chest"),
        PatchSpec(1, 14, 14, 5, scale=1, name="torso"),
        PatchSpec(2, 22, 14, 4, scale=1, name="hem"),
        PatchSpec(3, 24, 10, 3, scale=0, name="foot_left"),
        PatchSpec(4, 24, 18, 3, scale=0, name="foot_right"),
        PatchSpec(5, 10, 14, 2, scale=0, name="texture_fine"),
    )
    return PatchGraph(
        img_side=img_side,
        patches=patches,
        bind_pairs=((0, 1), (1, 2), (3, 4), (0, 5)),
        absence_protocols={
            "collar_occlude": (0,),
            "torso_occlude": (1,),
            "hem_occlude": (2,),
            "feet_occlude": (3, 4),
            "upper_body": (0, 1),
            "lower_body": (2, 3, 4),
        },
    )


def halves_fields_from_pattern(
    pattern: np.ndarray,
    *,
    img_side: int = IMG_SIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split 784-d pattern into TOP/BOT fields (merge-halves layout)."""
    img = np.asarray(pattern).reshape(img_side, img_side)
    n = img_side * img_side
    top = np.zeros(n, dtype=pattern.dtype)
    bot = np.zeros(n, dtype=pattern.dtype)
    top[: img_side // 2 * img_side] = img[: img_side // 2].ravel()
    bot[img_side // 2 * img_side :] = img[img_side // 2 :].ravel()
    return top, bot


def split_halves(pattern: np.ndarray, *, img_side: int = IMG_SIDE) -> tuple[np.ndarray, np.ndarray]:
    """Alias for merge-halves field split."""
    return halves_fields_from_pattern(pattern, img_side=img_side)
