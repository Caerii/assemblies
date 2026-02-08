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


def detect_best_engine() -> str:
    """Return the name of the best available compute engine.

    Returns ``"cuda_implicit"`` if CuPy + GPU are available,
    otherwise ``"numpy_sparse"``.
    """
    if _detect_cupy():
        return "cuda_implicit"
    return "numpy_sparse"
