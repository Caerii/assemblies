"""An engine's array backend belongs to THAT ENGINE, not to the process.

THE BUG THIS PINS.  `cupy_engine.__init__` and `cuda_engine.__init__` call
`set_backend("cupy")` and never restore it. The backend was process-global and
every hot path re-read it, so constructing a GPU engine ANYWHERE in the process
retroactively changed what an already-running numpy engine did: it started
handing itself CuPy arrays mid-projection and died on "Implicit conversion to a
NumPy array is not allowed".

It took out 11 tests that pass in isolation, and because it only reproduces
where CuPy is actually installed, CI never saw it. A conftest fixture now
restores the global around every test, but that is containment for the SUITE --
a library user constructing a CuPy engine and then touching their existing
numpy Brain got no such protection.

The fix is that `NumpySparseEngine` captures `get_xp()` once at construction
into `self._xp`, and passes it down to `SparseSimulationEngine`, so a later
global flip cannot reach backwards. `CupySparseEngine` still works because it
sets the global BEFORE `super().__init__`, so it captures CuPy.

These tests deliberately flip the global themselves rather than constructing a
CuPy engine, so they run everywhere -- including CI, where the bug was
invisible.
"""

import numpy as np
import pytest

from neural_assemblies.core import backend
from neural_assemblies.core.brain import Brain

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def restore_backend():
    """Put the global back however this test leaves it."""
    before = backend._xp
    yield
    backend._xp = before


def _grow(seed=3, n=400, k=20, rounds=3):
    b = Brain(p=0.05, save_winners=True, seed=seed, engine="numpy_sparse")
    b.add_stimulus("s", k)
    b.add_area("A", n, k, 0.1)
    for _ in range(rounds):
        b.project({"s": ["A"]}, {})
    return b


def test_engine_captures_its_backend_at_construction():
    b = _grow()
    assert b._engine._xp is np
    assert b._engine._sparse_sim._xp is np, "the simulator did not inherit it"


def test_a_later_global_flip_cannot_reach_backwards(restore_backend):
    """The actual user-visible bug: build a Brain, flip the global, keep going.

    Before the fix this raised "Implicit conversion to a NumPy array is not
    allowed" from inside `project_into`.
    """
    cupy = pytest.importorskip("cupy", reason="the leak needs a real CuPy")
    b = _grow()
    backend.set_backend("cupy")
    assert backend.get_xp() is cupy, "precondition: the global really moved"

    b.project({"s": ["A"]}, {})          # used to explode
    assert b._engine._xp is np
    assert isinstance(b.areas["A"].winners, np.ndarray)


def test_results_are_unchanged_by_a_global_flip(restore_backend):
    """Not crashing is not enough -- the numbers must be identical too.

    A backend that silently switched mid-run could still produce *a* result,
    and it would be a different brain.
    """
    pytest.importorskip("cupy", reason="the leak needs a real CuPy")
    poisoned = _grow(rounds=3)
    backend.set_backend("cupy")
    poisoned.project({"s": ["A"]}, {})
    poisoned.project({"s": ["A"]}, {})
    poisoned.project({"s": ["A"]}, {})
    got = sorted(int(x) for x in poisoned.areas["A"].winners)

    backend.set_backend("numpy")
    clean = _grow(rounds=6)
    want = sorted(int(x) for x in clean.areas["A"].winners)

    assert got == want, "the global flip changed the trajectory"


def test_flipping_the_global_before_construction_is_still_honoured(restore_backend):
    """The capture must not become a hardcoded numpy.

    `CupySparseEngine` works precisely by setting the global before
    `super().__init__`, so an engine built while the global says CuPy has to
    pick CuPy up. Breaking that would silently turn the GPU engine into a CPU
    one -- correct results, no error, and none of the speed.
    """
    cupy = pytest.importorskip("cupy")
    backend.set_backend("cupy")
    try:
        from neural_assemblies.core.numpy_engine._sparse import NumpySparseEngine
        eng = NumpySparseEngine(p=0.05, seed=0)
        assert eng._xp is cupy
        assert eng._sparse_sim._xp is cupy
    finally:
        backend.set_backend("numpy")


def test_no_hot_path_re_reads_the_global():
    """`get_xp()` must not reappear inside the engine's per-call code.

    This is the property that actually prevents the bug; it is easy to undo by
    adding one convenient `xp = get_xp()` line, and nothing would fail until
    someone ran a GPU engine in the same process.
    """
    import inspect

    from neural_assemblies.core.numpy_engine import _sparse

    src = inspect.getsource(_sparse.NumpySparseEngine)
    offenders = [ln.strip() for ln in src.splitlines()
                 if "get_xp()" in ln
                 and "self._xp = get_xp()" not in ln
                 and not ln.strip().startswith("#")]
    assert not offenders, (
        "these lines re-read the process-global backend inside the engine:\n  "
        + "\n  ".join(offenders)
        + "\nUse `self._xp` -- see this module's docstring for what that costs.")


def test_backend_module_to_xp_is_not_used_by_the_engine():
    """`backend.to_xp` reads the same global, one level down.

    Fixing only `get_xp()` left `to_xp` converting into whatever the process
    most recently selected, which reproduced the bug through a different door.
    """
    import inspect

    from neural_assemblies.core.numpy_engine import _sparse

    src = inspect.getsource(_sparse.NumpySparseEngine)
    bad = [ln.strip() for ln in src.splitlines()
           if "to_xp(" in ln
           and "self._to_xp(" not in ln
           and not ln.strip().startswith("#")
           and not ln.strip().startswith("def _to_xp(")]   # the definition
    assert not bad, "engine calls the global to_xp:\n  " + "\n  ".join(bad)
