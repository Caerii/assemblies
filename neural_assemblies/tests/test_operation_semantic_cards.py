"""Regression obligations from docs/reviews/whole-codebase/SEMANTIC_CARDS.md."""

import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import ops


@pytest.fixture
def brain():
    b = Brain(engine="numpy_exact", norm_init=False)
    for name in ("A", "B", "T"):
        b.add_area(name, 100, 10)
        ops.activate_assembly(b, ops.Assembly(name, list(range(10))))
    b.add_stimulus("s", 10)
    return b


@pytest.mark.parametrize("name", ["reciprocal_project", "associate", "merge"])
@pytest.mark.parametrize("pre_fixed", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_r1_m2_restore_callers_source_clamp(brain, monkeypatch, name, pre_fixed, raises):
    if pre_fixed:
        brain.areas["A"].fix_assembly()
        brain._engine.fix_assembly("A")
    def step(*args, **kwargs):
        if raises:
            raise RuntimeError("constructed projection failure")
    monkeypatch.setattr(brain, "project", step)
    args = ("A", "T") if name == "reciprocal_project" else ("A", "B", "T")
    if raises:
        with pytest.raises(RuntimeError, match="constructed"):
            getattr(ops, name)(brain, *args, rounds=2)
    else:
        getattr(ops, name)(brain, *args, rounds=2)
    assert brain.areas["A"].fixed_assembly is pre_fixed
    assert brain._engine.is_fixed("A") is pre_fixed
    assert not brain.areas["B"].fixed_assembly


def test_p2_zero_rounds_is_rejected_before_projection(brain, monkeypatch):
    calls = []
    monkeypatch.setattr(brain, "project", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(ValueError, match="rounds"):
        ops.project(brain, "s", "T", rounds=0)
    assert not calls


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("global_recurrence", [False, True])
@pytest.mark.parametrize("norm_init", [False, True])
def test_p3_operation_owns_recurrence_schedule(monkeypatch, recurrent,
                                               global_recurrence, norm_init):
    """Observe actual backend calls and learning, not just returned winners."""
    b = Brain(engine="numpy_exact", seed=17, norm_init=norm_init,
              recurrent_projection=global_recurrence)
    b.add_area("T", 100, 10, beta=0.1)
    b.add_stimulus("s", 10)
    calls = []
    original = b._engine.project_into

    def observed(target, from_stimuli, from_areas, *args, **kwargs):
        calls.append((target, tuple(from_stimuli), tuple(from_areas)))
        return original(target, from_stimuli, from_areas, *args, **kwargs)

    monkeypatch.setattr(b._engine, "project_into", observed)
    ops.project(b, "s", "T", rounds=3, recurrent=recurrent)
    tail_sources = ("T",) if recurrent else ()
    assert calls == [("T", ("s",), ()),
                     ("T", ("s",), tail_sources),
                     ("T", ("s",), tail_sources)]
    # A source-edge trace must correspond to learned state, not a dead probe.
    assert bool(b._engine._area_pot.get(("T", "T"))) is recurrent
