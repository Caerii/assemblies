"""NumPy-based compute engines for assembly calculus.

This package provides two CPU engines:

- ``NumpySparseEngine``:   Statistical sparse simulation (default, scales to large n)
- ``NumpyExplicitEngine``: Dense matrix simulation (faithful, for small n)

Sub-modules:

- ``_state``:    Per-area state containers (dataclasses)
- ``_sparse``:   NumpySparseEngine implementation
- ``_explicit``: NumpyExplicitEngine implementation
"""

from ._sparse import NumpySparseEngine
from ._explicit import NumpyExplicitEngine

__all__ = ["NumpySparseEngine", "NumpyExplicitEngine"]

from ..engine import register_engine
register_engine("numpy_sparse", NumpySparseEngine)
register_engine("numpy_explicit", NumpyExplicitEngine)
