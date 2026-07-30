"""`fiber_census` must tell a dead fiber from a live one on BOTH engines.

It could not. It read `conn.weights` directly, and the torch engine's `CSRConn`
has no such attribute, so every torch fiber measured as shape (0,0) / nnz 0 --
`dead`, and `silently_ignored` whenever the target was live. On a two-area torch
brain it flagged 4 of 4 fibers, including a healthy one at nnz=1314.

That is the worst failure mode for a diagnostic: not silence, but confident
false positives. This function is what the project reaches for when a result
looks like "mechanism X has surprisingly little effect", and its docstring gets
cited as grounds to trust a negative.

So these tests construct the TRUE NEGATIVE rather than asserting a threshold:
one brain carrying a fiber that is genuinely dead beside one that is genuinely
live, and the census must separate them.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import fiber_census


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


ENGINES = ["numpy_sparse"] + (["torch_sparse"] if _has_torch_cuda() else [])

N, K, P, BETA = 1000, 50, 0.05, 0.1


def _brain_with_one_live_and_one_dead(engine):
    """A -> B is driven and live; A -> C is wired but never driven.

    Returns the brain plus the two pair keys, so the assertions name what they
    expect rather than trusting an index.
    """
    b = Brain(p=P, save_winners=True, seed=1, engine=engine)
    b.add_stimulus("s", K)
    for area in ("A", "B", "C"):
        b.add_area(area, N, K, BETA)
    for _ in range(5):
        b.project({"s": ["A"]}, {})
    for _ in range(5):
        b.project({"s": ["A"]}, {"A": ["B"]})
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_census_separates_a_live_fiber_from_a_dead_one(engine):
    b = _brain_with_one_live_and_one_dead(engine)
    by_pair = {(f.src, f.dst): f for f in fiber_census(b)}

    live = by_pair.get(("A", "B"))
    assert live is not None, f"{engine}: no A->B fiber in the census at all"
    assert not live.dead, (
        f"{engine}: A->B was driven for 5 rounds and B has "
        f"w={live.dst_w}, but the census reports it dead "
        f"(shape {live.rows}x{live.cols}, nnz {live.nnz}). This is the torch "
        f"blindness -- the census is reading a storage attribute the backend "
        f"does not have.")
    assert live.nnz > 0


@pytest.mark.parametrize("engine", ENGINES)
def test_census_reports_nonzero_shape_for_a_materialized_fiber(engine):
    """The shape itself must be real, not just the dead/alive verdict.

    A census that returned `dead=False` while still reporting (0,0) would pass
    the test above and still be lying about everything a caller reads next.
    """
    b = _brain_with_one_live_and_one_dead(engine)
    live = {(f.src, f.dst): f for f in fiber_census(b)}[("A", "B")]
    assert live.rows > 0 and live.cols > 0, (
        f"{engine}: A->B reports shape {live.rows}x{live.cols}")


@pytest.mark.parametrize("engine", ENGINES)
def test_census_does_not_flag_every_fiber(engine):
    """The blindness signature: EVERY fiber reads dead.

    This is what the bug actually looked like in practice -- 4 of 4 flagged --
    so assert against that shape directly, not just against one fiber.
    """
    b = _brain_with_one_live_and_one_dead(engine)
    fibers = fiber_census(b)
    assert fibers, f"{engine}: census returned nothing"
    dead = [f for f in fibers if f.dead]
    assert len(dead) < len(fibers), (
        f"{engine}: all {len(fibers)} fibers report dead -- the census cannot "
        f"be distinguishing anything")
