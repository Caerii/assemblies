"""Permissive engine constructors reject every unconsumed model option."""

import pytest

from neural_assemblies.core.engine import create_engine
from neural_assemblies.core.numpy_engine import NumpyExactEngine


@pytest.mark.parametrize("surface", ["direct", "factory"])
def test_exact_engine_rejects_unknown_constructor_option(surface):
    with pytest.raises(TypeError, match="unsupported constructor options: norm_innit"):
        if surface == "direct":
            NumpyExactEngine(p=0.1, norm_innit=True)
        else:
            create_engine("numpy_exact", p=0.1, norm_innit=True)


def test_exact_area_rejects_unknown_option_before_registration():
    engine = NumpyExactEngine(p=0.1)

    with pytest.raises(TypeError, match="unsupported constructor options: slot_counts"):
        engine.add_area("A", 100, 10, 0.1, slot_counts=2)
    assert "A" not in engine._areas


def test_torch_engine_rejects_unknown_option_before_device_setup():
    pytest.importorskip("torch")
    from neural_assemblies.core.torch_engine import TorchSparseEngine

    with pytest.raises(TypeError, match="unsupported constructor options: dense_driv"):
        TorchSparseEngine(p=0.1, dense_driv=True)


def test_torch_recognized_options_are_removed_from_remainder(monkeypatch):
    pytest.importorskip("torch")
    from neural_assemblies.core.torch_engine import TorchSparseEngine

    class StopAfterAdmission(RuntimeError):
        pass

    import neural_assemblies.core.torch_engine._engine as torch_module

    monkeypatch.setattr(
        torch_module.torch,
        "device",
        lambda *_: (_ for _ in ()).throw(StopAfterAdmission()),
    )
    with pytest.raises(StopAfterAdmission):
        TorchSparseEngine(
            p=0.1,
            norm_init=True,
            synaptic_scaling={"A"},
            dense_drive=True,
            readonly=True,
        )
