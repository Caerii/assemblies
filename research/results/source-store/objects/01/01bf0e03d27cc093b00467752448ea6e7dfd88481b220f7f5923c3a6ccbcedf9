"""Deterministic allocation is a typed engine capability, not a broad promise."""

import pytest

from neural_assemblies import Brain
from neural_assemblies.core.engine import create_engine


@pytest.mark.parametrize("value", [0, 1, None, "false", "true"])
def test_deterministic_request_requires_a_bool(value):
    with pytest.raises(ValueError, match="deterministic must be a bool"):
        Brain(engine="numpy_sparse", deterministic=value)


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_explicit"])
def test_incapable_engine_rejects_enabled_deterministic_policy(engine):
    with pytest.raises(ValueError, match="does not support deterministic allocation"):
        Brain(engine=engine, norm_init=False, deterministic=True)


def test_disabled_policy_remains_universal_default():
    exact = Brain(engine="numpy_exact", norm_init=False, deterministic=False)
    dense = Brain(engine="numpy_explicit", norm_init=False, deterministic=False)
    assert exact.deterministic is dense.deterministic is False


def test_capable_engine_stores_canonical_execution_policy():
    brain = Brain(engine="numpy_sparse", norm_init=False, deterministic=True)
    assert brain.deterministic is True
    assert brain._engine._deterministic is True


def test_factory_rejects_before_incapable_constructor(monkeypatch):
    import neural_assemblies.core.engine as engine_module

    called = False
    original = engine_module.engine_type("numpy_explicit")

    class RecordingExplicit(original):
        def __init__(self, *args, **kwargs):
            nonlocal called
            called = True
            super().__init__(*args, **kwargs)

    monkeypatch.setitem(
        engine_module._ENGINE_REGISTRY, "numpy_explicit", RecordingExplicit
    )
    with pytest.raises(ValueError, match="does not support deterministic allocation"):
        create_engine("numpy_explicit", p=0.1, deterministic=True)
    assert called is False


def test_supplied_engine_policy_must_match_brain_request():
    engine = create_engine("numpy_sparse", p=0.1, deterministic=True)

    with pytest.raises(ValueError, match="deterministic setting conflicts"):
        Brain(engine=engine, p=0.1, norm_init=False, deterministic=False)
    brain = Brain(engine=engine, p=0.1, norm_init=False, deterministic=True)
    assert brain._engine is engine


def test_direct_engine_constructors_apply_the_same_boundary():
    from neural_assemblies.core.numpy_engine import (
        NumpyExactEngine,
        NumpyExplicitEngine,
        NumpySparseEngine,
    )

    with pytest.raises(ValueError, match="deterministic must be a bool"):
        NumpySparseEngine(p=0.1, deterministic="false")
    with pytest.raises(ValueError, match="does not support deterministic allocation"):
        NumpyExactEngine(p=0.1, deterministic=True)
    with pytest.raises(ValueError, match="does not support deterministic allocation"):
        NumpyExplicitEngine(p=0.1, deterministic=True)


def test_direct_torch_constructor_requires_a_boolean_before_device_setup():
    pytest.importorskip("torch")
    from neural_assemblies.core.torch_engine import TorchSparseEngine

    with pytest.raises(ValueError, match="deterministic must be a bool"):
        TorchSparseEngine(p=0.1, deterministic="true")


@pytest.mark.parametrize("option", ["gpu_sampling", "dense_drive"])
def test_torch_only_options_reject_explicit_values_on_other_engines(option):
    kwargs = {option: False}
    with pytest.raises(ValueError, match=f"does not support {option}"):
        Brain(engine="numpy_exact", norm_init=False, **kwargs)
