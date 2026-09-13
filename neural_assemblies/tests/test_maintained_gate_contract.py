"""Construction contracts for the maintained-source verification gate."""

import importlib.util
from pathlib import Path

import pytest


def _gate_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "verify_maintained", root / "scripts" / "verify_maintained.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maintained_sources_are_unique_and_include_contract_surfaces():
    gate = _gate_module()
    sources = gate.maintained_sources()

    assert sources == sorted(set(sources))
    normalized = {source.replace("\\", "/") for source in sources}
    assert "neural_assemblies/assembly_calculus/assembly.py" in normalized
    assert "research/runner.py" in normalized
    assert "research/experiments/seq_a1_horizon_hashed.py" in normalized


def test_missing_maintained_scope_fails_before_static_analysis(tmp_path):
    gate = _gate_module()

    with pytest.raises(FileNotFoundError, match="scope does not exist"):
        gate.maintained_sources(tmp_path)
