"""Compiled projection is a declared topology, never an ignored selector."""

import pytest

from neural_assemblies import Brain
from neural_assemblies.core.engine import create_engine


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_explicit"])
def test_compiled_projection_rejects_on_incapable_brain(engine):
    with pytest.raises(ValueError, match="does not support compiled projection"):
        Brain(engine=engine, norm_init=False, projection_fidelity="compiled")


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
    with pytest.raises(ValueError, match="does not support compiled projection"):
        create_engine("numpy_explicit", p=0.1, projection_fidelity="compiled")
    assert called is False


def test_direct_incapable_engine_setter_rejects_compiled():
    engine = create_engine("numpy_exact", p=0.1)

    with pytest.raises(ValueError, match="does not support compiled projection"):
        engine.set_projection_fidelity("compiled")
    assert engine.get_projection_fidelity() == "exact"


def test_direct_exact_constructor_rejects_compiled_but_accepts_exact_alias():
    from neural_assemblies.core.numpy_engine import NumpyExactEngine

    with pytest.raises(ValueError, match="does not support compiled projection"):
        NumpyExactEngine(p=0.1, projection_fidelity="compiled")
    engine = NumpyExactEngine(p=0.1, projection_fidelity="full")
    assert engine.get_projection_fidelity() == "exact"


def test_direct_torch_constructor_rejects_compiled_before_device_setup():
    pytest.importorskip("torch")
    from neural_assemblies.core.torch_engine import TorchSparseEngine

    with pytest.raises(ValueError, match="does not support compiled projection"):
        TorchSparseEngine(p=0.1, projection_fidelity="compiled")


def test_supplied_engine_fidelity_must_match_brain_request():
    engine = create_engine("numpy_sparse", p=0.1, projection_fidelity="compiled")

    with pytest.raises(ValueError, match="projection fidelity conflicts"):
        Brain(engine=engine, p=0.1, norm_init=False)
    brain = Brain(
        engine=engine,
        p=0.1,
        norm_init=False,
        projection_fidelity="fuzzy",
    )
    assert brain.projection_fidelity == "compiled"


def test_runtime_setter_normalizes_alias_and_rejects_unknown_without_mutation():
    brain = Brain(engine="numpy_sparse", norm_init=False)
    brain.projection_fidelity = "fuzzy"
    assert brain.projection_fidelity == "compiled"

    with pytest.raises(ValueError, match="Unknown projection fidelity"):
        brain.projection_fidelity = "approximately-exact"
    assert brain.projection_fidelity == "compiled"
