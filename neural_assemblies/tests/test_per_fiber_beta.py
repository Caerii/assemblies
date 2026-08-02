"""Per-fiber plasticity must reach the engine, and the dead route must say so.

Task #88. `Area.update_beta_by_area` wrote `Area.beta_by_area`; every engine
except the legacy dense `compute.explicit_projection` path reads its own
`AreaState.beta_by_source`. So the setter returned happily and changed nothing.

That cost a real experiment: a gated-recurrence arm was run against an ungated
one for 10 seeds and the two were bit-identical, because the beta override never
arrived. `diagnostics.compare_arms` caught it only because it refuses arms that
never differ.

These tests pin BOTH halves -- that the working route changes the dynamics, and
that the dead route refuses instead of no-opping. A test for the first alone has
no power against the bug that actually happened.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain

N, K, P, BETA = 2000, 50, 0.05, 0.10
SEED_ROUNDS, TRAIN_ROUNDS = 3, 12


def _brain() -> Brain:
    b = Brain(p=P, seed=17)
    b.add_stimulus("s", K)
    b.add_area("A", N, K, BETA)
    for _ in range(SEED_ROUNDS):
        b.project({"s": ["A"]}, {})
    return b


def _run(b: Brain) -> np.ndarray:
    """Drive A with its own recurrence open, then report the assembly.

    The self-fiber is named EXPLICITLY: `recurrent_projection` defaults False,
    so `project({"s": ["A"]}, {})` alone builds no self-recurrence and there
    would be nothing for a self-fiber beta to act on.
    """
    for _ in range(TRAIN_ROUNDS):
        b.project({"s": ["A"]}, {"A": ["A"]})
    return np.array(b.areas["A"].winners, dtype=np.uint64, copy=True)


def test_working_route_changes_the_assembly():
    """`Brain.update_plasticity` on a self-fiber must alter the dynamics.

    Arms are built by deepcopy, not by rebuilding: dense `Connectome` init is
    draw-order dependent (#81), so two fresh Brains are not the same substrate
    and a difference would prove nothing.
    """
    base = _brain()
    plastic, frozen = copy.deepcopy(base), copy.deepcopy(base)
    frozen.update_plasticity("A", "A", 0.0)

    assert not np.array_equal(_run(plastic), _run(frozen)), (
        "beta=0 on the A->A fiber produced the SAME assembly as beta=0.10. "
        "The override did not reach the engine -- this is #88 regressing."
    )


def test_working_route_is_visible_to_the_engine_and_the_dense_path():
    """Both readers must agree, or the two engines silently diverge."""
    b = _brain()
    b.update_plasticity("A", "A", 0.0)
    assert b._engine.get_beta("A", "A") == 0.0          # sparse/torch/cuda read this
    assert b.areas["A"].beta_by_area["A"] == 0.0        # compute.explicit_projection reads this


def test_dead_route_refuses_instead_of_no_opping():
    """The exact call that silently did nothing must now raise."""
    b = _brain()
    with pytest.raises(NotImplementedError, match="update_plasticity"):
        b.areas["A"].update_beta_by_area("A", 0.0)

    with pytest.raises(NotImplementedError, match="update_plasticities"):
        b.areas["A"].update_beta_by_stimulus("s", 0.0)


def test_writing_the_bookkeeping_dict_alone_still_does_nothing():
    """The TRUE-NEGATIVE case, constructed on purpose.

    Raising from the setter does not stop anyone writing the dict directly --
    `trainer.py` does exactly that, and is correct only because it also calls
    `engine.set_beta`. This asserts the dict alone is inert, so the day someone
    "simplifies" that pair into a single dict write, this test fails rather
    than the experiment silently reporting a null.
    """
    base = _brain()
    untouched, dict_only = copy.deepcopy(base), copy.deepcopy(base)
    dict_only.areas["A"].beta_by_area["A"] = 0.0        # deliberately the dead half

    assert np.array_equal(_run(untouched), _run(dict_only)), (
        "writing beta_by_area alone changed the sparse dynamics -- if that is "
        "now a real route, the docstring on Area.update_beta_by_area is wrong"
    )
