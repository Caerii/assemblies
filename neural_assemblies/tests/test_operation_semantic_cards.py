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
