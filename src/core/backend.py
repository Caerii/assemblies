"""
Backend abstraction for NumPy / CuPy interchangeability.

All array-producing code should call ``get_xp()`` to obtain the current
array module instead of importing ``numpy`` directly.  RNG and scipy
remain on CPU (numpy) -- only array operations are dispatched.

Usage::

    from .backend import get_xp, to_cpu, to_xp

    def some_function():
        xp = get_xp()
        arr = xp.zeros(100, dtype=xp.float32)
"""

import numpy as np

_xp = np
_HAS_CUPY = None


def _detect_cupy():
    global _HAS_CUPY
    if _HAS_CUPY is None:
        try:
            import cupy
            cupy.array([1.0])  # verify GPU is usable
            _HAS_CUPY = True
        except Exception:
            _HAS_CUPY = False
    return _HAS_CUPY


def set_backend(name="auto"):
    """Select the array backend.

    Args:
        name: ``"numpy"``, ``"cupy"``, or ``"auto"`` (cupy if available,
              else numpy).
    """
    global _xp
    if name == "numpy":
        _xp = np
    elif name == "cupy":
        import cupy
        _xp = cupy
    elif name == "auto":
        _xp = __import__("cupy") if _detect_cupy() else np
    else:
        raise ValueError(f"Unknown backend: {name!r}")


def get_xp():
    """Return the current array module (numpy or cupy)."""
    return _xp


def get_backend_name():
    """Return ``"cupy"`` or ``"numpy"``."""
    return "cupy" if _xp.__name__ == "cupy" else "numpy"


def to_cpu(arr):
    """Move an array to CPU (no-op for numpy arrays)."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def to_xp(arr):
    """Move a CPU array to the current backend."""
    return _xp.asarray(arr)


_HAS_TORCH_CUDA = None


def _detect_torch_cuda():
    """Check for PyTorch with CUDA support (cached)."""
    global _HAS_TORCH_CUDA
    if _HAS_TORCH_CUDA is None:
        try:
            import torch
            _HAS_TORCH_CUDA = torch.cuda.is_available()
        except Exception:
            _HAS_TORCH_CUDA = False
    return _HAS_TORCH_CUDA


# Crossover point from benchmarking (CSR torch_sparse vs numpy_sparse):
#   n < 1M:  numpy_sparse faster (lower dispatch overhead)
#   n >= 1M: torch_sparse 1.5-54x faster for area->area projections
_TORCH_SPARSE_THRESHOLD = 1_000_000


def detect_best_engine(n_hint: int = 0) -> str:
    """Return the name of the best available compute engine.

    Uses *n_hint* (expected neuron count per area) to select the optimal
    backend.  When ``n_hint >= 1_000_000`` and PyTorch+CUDA is available,
    returns ``"torch_sparse"`` which uses CSR connectivity and GPU
    acceleration for 1.5-54x speedup over CPU at scale.

    Otherwise returns ``"numpy_sparse"`` — the fastest engine for
    typical assembly sizes (n < 1M, k < 5000) due to lower per-op
    dispatch overhead.

    Args:
        n_hint: Expected neuron count per area.  Pass 0 (default) to
                always get ``"numpy_sparse"``.
    """
    if n_hint >= _TORCH_SPARSE_THRESHOLD and _detect_torch_cuda():
        return "torch_sparse"
    return "numpy_sparse"
