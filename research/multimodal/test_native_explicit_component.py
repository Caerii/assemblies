"""Test actual native NumpyExplicitEngine.project_into source in isolation.

This extracts the CURRENT method's AST directly from the repo's source.
Surrounding engine services are controlled test doubles; the numerical
method itself is NOT reimplemented. Passing this test is NOT a full Brain
integration result. The native source must be present.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, cast
import numpy as np
import pytest
from research.multimodal.exact_synapse_law import verify_exact_hebbian_write


@dataclass
class ProjectionResult:
    winners: np.ndarray
    num_first_winners: int
    num_ever_fired: int
    total_activation: float = 0.
    pre_kwta_inputs: np.ndarray | None = None
    pre_kwta_prev_only: np.ndarray | None = None
    pre_kwta_total: float = 0.
    pre_kwta_count: int = 0


def _to_cpu(x):
    return x


def _no_slots(*_):
    raise AssertionError("slot selection outside this contract")


def _real_method():
    path = Path(__file__).resolve().parents[2] / (
        "neural_assemblies/core/numpy_engine/_explicit.py")
    if not path.is_file():
        pytest.fail("missing full native checkout: " + str(path))
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        f for cls in tree.body if isinstance(cls, ast.ClassDef)
        and cls.name == "NumpyExplicitEngine"
        for f in cls.body if isinstance(f, ast.FunctionDef)
        and f.name == "project_into"
    ]
    if len(matches) != 1:
        pytest.fail("native projection method missing or ambiguous")
    if matches[0].decorator_list:
        pytest.fail("native projection has decorators; inspect before testing")
    wrapper = ast.fix_missing_locations(ast.Module(body=[
        ast.ClassDef(name="ExtractedMethod", bases=[], keywords=[],
                     body=[matches[0]], decorator_list=[])
    ], type_ignores=[]))
    symbols = dict(np=np, List=List, Any=Any, cast=cast,
                   ProjectionResult=ProjectionResult,
                   to_cpu=_to_cpu, select_slot_winners=_no_slots)
    exec(compile(wrapper, str(path), "exec"), symbols)
    return symbols["ExtractedMethod"].project_into


class ControlledEngine:
    """Only the neighboring engine services are faked."""
    def __init__(self, method, *, seed, n=48, k=6, beta=.1, ceiling=20.):
        rng = np.random.default_rng(seed)
        self.n, self.k, self.w_max = n, k, ceiling
        self._plasticity_enabled_global = True
        self._areas = {
            "TARGET": SimpleNamespace(
                n=n, k=k, beta=beta, beta_by_source={},
                fixed_assembly=False, slot_count=0,
                winners=np.array([], dtype=np.uint32),
                ever_fired=np.zeros(n, dtype=bool), num_ever_fired=0, w=0)}
        self._stim_conns = {}
        w = (rng.random((n, n)) < .3).astype(np.float32)
        self._area_conns = {"SOURCE": {"TARGET": SimpleNamespace(weights=w)}}
        self.source = np.sort(rng.choice(n, k, replace=False)).astype(np.uint32)
        self.project_into = method.__get__(self, type(self))

    def validate_projection_inputs(self, target, stims, sources, drive):
        assert target == "TARGET" and not stims and sources == ["SOURCE"]
        if drive is not None:
            drive = np.asarray(drive, dtype=np.float32)
            if drive.shape != (self.n,) or not np.isfinite(drive).all():
                raise ValueError("invalid teacher drive")
        return {"SOURCE": self.source}, drive

    def _select_winners(self, scores, area):
        return np.lexsort((np.arange(area.n), -scores))[:area.k]

    def _validated_winners(self, target, values):
        assert target == "TARGET" and len(values) == self.k
        assert len(np.unique(values)) == self.k
        return np.asarray(values, dtype=np.uint32)

    def fiber_learning_allowed(self, source, target):
        return True


@pytest.mark.parametrize("seed", [0, 1, 2, 17, 42, 111])
@pytest.mark.parametrize("beta", [0., .1, .5])
def test_real_method_matches_exact_independent_oracle(seed, beta):
    engine = ControlledEngine(_real_method(), seed=seed, beta=beta)
    before = {
        ("SOURCE", "TARGET"):
            engine._area_conns["SOURCE"]["TARGET"].weights.copy(),
        ("TARGET", "TARGET"):
            np.arange(engine.n * engine.n, dtype=np.float32)
            .reshape(engine.n, engine.n).copy()
    }
    teacher = np.sort(np.random.default_rng(seed + 999)
                      .choice(engine.n, engine.k, replace=False)).astype(np.uint32)
    drive = np.zeros(engine.n, dtype=np.float32)
    drive[teacher] = 100.
    observed = []
    for _ in range(3):
        result = engine.project_into("TARGET", [], ["SOURCE"], True, False, drive)
        np.testing.assert_array_equal(np.sort(result.winners), teacher)
        observed.append(result.winners.copy())
    after = {
        ("SOURCE", "TARGET"):
            engine._area_conns["SOURCE"]["TARGET"].weights.copy(),
        ("TARGET", "TARGET"): before["TARGET", "TARGET"].copy()
    }
    evidence = verify_exact_hebbian_write(
        before, after, source="SOURCE", target="TARGET",
        cue=engine.source, selected_by_round=observed, beta=beta,
        ceiling=engine.w_max)
    assert evidence["rounds"] == 3
    frozen = engine._area_conns["SOURCE"]["TARGET"].weights.copy()
    for _ in range(2):
        engine.project_into("TARGET", [], ["SOURCE"], False, True, None)
    np.testing.assert_array_equal(
        frozen, engine._area_conns["SOURCE"]["TARGET"].weights)


def test_real_method_clips_and_nonplastic_read_does_not_write():
    engine = ControlledEngine(_real_method(), seed=7, beta=.5, ceiling=1.2)
    drive = np.zeros(engine.n, dtype=np.float32)
    drive[[1, 2, 3, 4, 5, 6]] = 100.
    engine.project_into("TARGET", [], ["SOURCE"], True, True, drive)
    weights = engine._area_conns["SOURCE"]["TARGET"].weights
    assert np.max(weights) <= 1.2
    frozen = weights.copy()
    engine.project_into("TARGET", [], ["SOURCE"], False, True, None)
    np.testing.assert_array_equal(weights, frozen)
