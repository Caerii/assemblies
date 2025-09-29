# utils.py

"""
Shared utilities for math_primitives modules.

Includes input validation and index normalization helpers used across
statistics, sparse simulation, and plasticity components.
"""

import numpy as np
from typing import List


def validate_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def validate_finite_scalar(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def normalize_index_list(indices: List[int], *, allow_negative: bool = False) -> List[int]:
    """Return de-duplicated integer indices; reject float-like input.

    Only Python ints and numpy integer types are accepted. Floats are rejected to
    prevent silent truncation.
    """
    normalized: List[int] = []
    seen = set()
    for idx in indices:
        if isinstance(idx, (int, np.integer)):
            val = int(idx)
        else:
            raise ValueError("Indices must be integers (int or np.integer)")
        if not allow_negative and val < 0:
            continue
        if val not in seen:
            seen.add(val)
            normalized.append(val)
    return normalized


