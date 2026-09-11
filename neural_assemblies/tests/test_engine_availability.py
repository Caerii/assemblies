"""Engine admission distinguishes invalid names from unavailable backends."""

import importlib

import pytest

from neural_assemblies.core import engine as engine_module
from neural_assemblies.core.engine import EngineUnavailableError, create_engine


def test_known_engine_import_failure_preserves_the_cause(monkeypatch):
    """A missing optional dependency must not be reported as an unknown engine."""
    name = "known_but_unavailable_test_engine"
    missing = ModuleNotFoundError("No module named 'missing_engine_dependency'")
    monkeypatch.setitem(engine_module._ENGINE_MODULES, name, "missing_backend")

    def fail_import(module_name, package):
        assert module_name == ".missing_backend"
        assert package == "neural_assemblies.core"
        raise missing

    monkeypatch.setattr(importlib, "import_module", fail_import)
    try:
        with pytest.raises(
            EngineUnavailableError,
            match=(
                "known_but_unavailable_test_engine.*known but unavailable.*"
                "missing_engine_dependency"
            ),
        ) as caught:
            create_engine(name)
        assert caught.value.__cause__ is missing
    finally:
        engine_module._ENGINE_LOAD_ERRORS.pop(name, None)


def test_unknown_engine_remains_a_name_error(monkeypatch):
    """An invalid engine name has no backend import cause to diagnose."""
    monkeypatch.setattr(engine_module, "_ensure_engines_loaded", lambda: None)
    with pytest.raises(ValueError, match="Unknown engine '__not_an_engine__'"):
        create_engine("__not_an_engine__")


def test_provider_that_forgets_registration_fails_at_admission(monkeypatch):
    """A stale module map must fail before a later registry lookup obscures it."""
    name = "unregistered_test_engine"
    monkeypatch.setitem(engine_module._ENGINE_MODULES, name, "empty_backend")
    monkeypatch.setattr(importlib, "import_module", lambda *_args: object())

    with pytest.raises(
        EngineUnavailableError,
        match="provider module loaded without registering the engine",
    ) as caught:
        create_engine(name)
    assert caught.value.__cause__ is None
