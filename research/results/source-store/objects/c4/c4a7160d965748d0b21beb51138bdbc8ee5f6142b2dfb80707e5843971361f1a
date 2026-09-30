"""NumPy-based compute engines for assembly calculus.

This package provides three CPU engines, which differ in WHERE the substrate
lives and therefore in what they can afford:

- ``NumpySparseEngine``:   Statistical sparse simulation (default, scales to
  large n). Stores the synapses it has seen and INVENTS a drive for neurons
  that have never fired (`sample_new_winner_inputs`) -- the only approximation
  of the three, and the one `arbitrate` exists to isolate.
- ``NumpyExplicitEngine``: Dense n x n matrices. Exact, bounded by O(n^2).
- ``NumpyExactEngine``:    Exact drive with NO stored substrate -- the random
  factor is recomputed from a content-addressed hash and only potentiation is
  kept, so it is exact at large n. See ``_exact`` and
  ``research/notes/substrate/exact_drive_equivalences.md``.

Sub-modules:

- ``_state``:    Per-area state containers (dataclasses)
- ``_sparse``:   NumpySparseEngine implementation
- ``_explicit``: NumpyExplicitEngine implementation
- ``_exact``:    NumpyExactEngine implementation
"""

from ._sparse import NumpySparseEngine
from ._explicit import NumpyExplicitEngine
from ._exact import NumpyExactEngine

__all__ = ["NumpySparseEngine", "NumpyExplicitEngine", "NumpyExactEngine"]

from ..engine import register_engine
register_engine("numpy_sparse", NumpySparseEngine)
register_engine("numpy_explicit", NumpyExplicitEngine)
register_engine("numpy_exact", NumpyExactEngine)
