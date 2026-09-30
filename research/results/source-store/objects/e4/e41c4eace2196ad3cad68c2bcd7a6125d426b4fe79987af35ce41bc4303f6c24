"""
Structured absence masks for generative-completeness training and evaluation.

Ember diagnostic: "drop out the 3s" — remove digit-distinctive structure, not
random pixel noise, and test whether the circuit can still recover identity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Protocols used in regeneration panel and attractor curriculum.
ABSENCE_PROTOCOLS: dict[str, str] = {
    "top_half": "Zero top 14 rows (loops / upper strokes)",
    "bottom_half": "Zero bottom 14 rows (base / lower strokes)",
    "center_band": "Zero rows 10–18 (middle segment — critical for digit 3)",
    "left_half": "Zero left 14 columns",
    "right_half": "Zero right 14 columns",
    "random_50": "Random 50% pixel dropout (weak baseline)",
}

# Digits that receive structured-absence curriculum during attractor training.
ABSENCE_CURRICULUM_DIGITS: frozenset[int] = frozenset({2, 3, 5, 8})


def _reshape784(pattern: np.ndarray) -> np.ndarray:
    return np.asarray(pattern, dtype=pattern.dtype).reshape(28, 28)


def _flatten(img: np.ndarray) -> np.ndarray:
    return img.reshape(784)


def mask_top_half(pattern: np.ndarray) -> np.ndarray:
    img = _reshape784(pattern).copy()
    img[:14, :] = 0
    return _flatten(img)


def mask_bottom_half(pattern: np.ndarray) -> np.ndarray:
    img = _reshape784(pattern).copy()
    img[14:, :] = 0
    return _flatten(img)


def mask_center_band(pattern: np.ndarray) -> np.ndarray:
    img = _reshape784(pattern).copy()
    img[10:18, :] = 0
    return _flatten(img)


def mask_left_half(pattern: np.ndarray) -> np.ndarray:
    img = _reshape784(pattern).copy()
    img[:, :14] = 0
    return _flatten(img)


def mask_right_half(pattern: np.ndarray) -> np.ndarray:
    img = _reshape784(pattern).copy()
    img[:, 14:] = 0
    return _flatten(img)


def mask_random_fraction(
    pattern: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(pattern, dtype=pattern.dtype).copy()
    keep = rng.random(784) > fraction
    return out * keep


_MASK_FNS: dict[str, Callable[..., np.ndarray]] = {
    "top_half": mask_top_half,
    "bottom_half": mask_bottom_half,
    "center_band": mask_center_band,
    "left_half": mask_left_half,
    "right_half": mask_right_half,
}

# Structured masks used for emergent training exposure (not random dropout).
STRUCTURED_PROTOCOLS: tuple[str, ...] = (
    "top_half", "bottom_half", "center_band", "left_half", "right_half",
)


def apply_absence_mask(
    pattern: np.ndarray,
    protocol: str,
    *,
    rng: np.random.Generator | None = None,
    random_fraction: float = 0.5,
    patch_graph=None,
) -> np.ndarray:
    """Apply a named structured-absence mask to a 784-d pattern."""
    if patch_graph is not None:
        try:
            return patch_graph.apply_absence(pattern, protocol, rng=rng)
        except ValueError:
            pass
    if protocol == "random_50":
        if rng is None:
            rng = np.random.default_rng(0)
        return mask_random_fraction(pattern, random_fraction, rng)
    fn = _MASK_FNS.get(protocol)
    if fn is None:
        raise ValueError(f"Unknown absence protocol: {protocol!r}")
    return fn(pattern)


def apply_bundle_absence(
    bundle,
    pattern: np.ndarray,
    protocol: str,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply absence using bundle patch graph when available."""
    from neural_assemblies.programs.patch_registry import resolve_patch_graph

    graph = resolve_patch_graph(bundle)
    return apply_absence_mask(pattern, protocol, rng=rng, patch_graph=graph)


# Digit-3 focused curriculum (Ember: drop the middle stroke).
DIGIT3_CURRICULUM_PROTOCOLS: tuple[str, ...] = ("center_band", "top_half", "bottom_half")


def digit3_curriculum_masks(pattern: np.ndarray, *, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Structured absence variants for digit-3 training."""
    generator = rng if rng is not None else np.random.default_rng(0)
    protocols = list(DIGIT3_CURRICULUM_PROTOCOLS)
    generator.shuffle(protocols)
    return [apply_absence_mask(pattern, p, rng=generator) for p in protocols]


def digit3_absence_battery() -> tuple[str, ...]:
    """Protocols most diagnostic for digit-3 generative completeness."""
    return DIGIT3_CURRICULUM_PROTOCOLS
