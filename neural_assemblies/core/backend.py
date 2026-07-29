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


def _torch_first():
    """Load PyTorch's CUDA libraries BEFORE CuPy's, if torch is installed.

    On Windows the two ship their own copies of the CUDA runtime and cuBLAS,
    and whichever imports first wins DLL resolution process-wide. CuPy winning
    leaves torch bound to CuPy's cuBLAS, which dies on larger matmuls with a
    bare `Windows fatal exception: access violation` -- no traceback, no
    Python-level error, and the crash surfaces in whatever unrelated code
    happens to run the matmul.

    Measured with cupy 14.1.1 (CUDA 13.02) and torch 2.12.1+cu130: importing
    CuPy first crashes a torch training loop even when NO CuPy operation is
    ever executed; importing torch first is fine. So this is called at every
    site that genuinely imports CuPy.

    Silently does nothing when torch is absent -- torch is an optional
    dependency and CuPy-only installations must keep working.
    """
    try:
        import torch  # noqa: F401
    except Exception:                                        # noqa: BLE001
        pass


def _detect_cupy():
    global _HAS_CUPY
    if _HAS_CUPY is None:
        try:
            _torch_first()
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
        _torch_first()
        import cupy
        _xp = cupy
    elif name == "auto":
        # _detect_cupy already ran _torch_first before importing CuPy.
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


# Heuristic crossover from local profiling (CSR torch_sparse vs numpy_sparse):
#   n < 1M:  numpy_sparse often wins due to lower dispatch overhead
#   n >= 1M: torch_sparse is often the better choice when CUDA is available
_TORCH_SPARSE_THRESHOLD = 1_000_000


def detect_best_engine(n_hint: int = 0) -> str:
    """Return the name of the best available compute engine.

    Uses *n_hint* (expected neuron count per area) to choose a practical
    default backend. When ``n_hint >= 1_000_000`` and PyTorch+CUDA is
    available, returns ``"torch_sparse"`` which uses CSR connectivity on
    GPU. Otherwise returns ``"numpy_sparse"``.

    This is a heuristic based on local profiling, not a universal proof that
    one engine dominates another for every workload.

    Args:
        n_hint: Expected neuron count per area.  Pass 0 (default) to
                always get ``"numpy_sparse"``.
    """
    if n_hint >= _TORCH_SPARSE_THRESHOLD and _detect_torch_cuda():
        return "torch_sparse"
    return "numpy_sparse"


def detect_fastest_engine() -> str:
    """Return the fastest available sparse engine (GPU when CUDA is present)."""
    if _detect_torch_cuda():
        return "torch_sparse"
    return "numpy_sparse"


def resolve_mixed_engine(engine: str) -> str:
    """Engine for brains that mix explicit and sparse areas (e.g. TACL LEX + grammar).

    Dense explicit↔sparse bridges require ``Connectome(sparse=False)`` on the
    sparse engine via ``set_dense_area_conn``.  Auto-selection therefore prefers
    ``numpy_sparse``, which is the validated path for literature parser parity.
    """
    if engine in ("auto", ""):
        return "numpy_sparse"
    return engine
