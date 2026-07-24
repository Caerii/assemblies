"""Tests for Assembly IR v1 and cross-language parity runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pytest

from neural_assemblies.ir.protocol import (
    export_protocol_document,
    load_protocol_document,
    validate_protocol_document,
)


def _golden(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "research" / "literature" / "parity" / "golden" / name


def test_protocol_ir_validates_cross_lang_golden():
    doc = load_protocol_document(_golden("cross_lang_pnas_scaling.json"))
    assert doc["ir_version"] == "1"
    assert doc["protocol"] == "cross_lang.pnas_scaling"
    assert "ci_parity" in doc["regimes"]


def test_export_protocol_document_roundtrip():
    doc = export_protocol_document(
        protocol_id="test.proto",
        metrics={"x": 1.0},
        backend="python",
    )
    assert validate_protocol_document(doc) == []


def test_cross_lang_runner_pnas_scaling():
    from research.literature.cross_lang.runner import run_protocol

    ok, diffs = run_protocol("cross_lang.pnas_scaling")
    assert ok, diffs


def test_julia_pnas_scaling_when_available():
    import shutil

    if shutil.which("julia") is None:
        pytest.skip("julia not installed")
    from research.literature.cross_lang.runner import (
        _golden_path,
        load_protocol_document,
        verify_julia_against_golden,
    )

    golden = load_protocol_document(_golden_path("cross_lang_pnas_scaling.json"))
    ok, diffs = verify_julia_against_golden(golden)
    assert ok, diffs


def test_brain_reinforce_connectome():
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=0.1, seed=0, engine="numpy_sparse", w_max=1e9)
    brain.add_area("A", 100, 10, 1.0, explicit=True)
    brain.add_area("B", 100, 10, 1.0, explicit=True)
    brain.areas["A"].winners = np.array([1, 2, 3], dtype=np.uint32)
    brain._explicit_engine.set_winners("A", brain.areas["A"].winners)
    post = np.arange(20, 30, dtype=np.uint32)
    w_before = brain.connectomes["A"]["B"].weights[1, 20]
    brain.reinforce_connectome("A", "B", post)
    w_after = brain.connectomes["A"]["B"].weights[1, 20]
    assert w_after > w_before


@pytest.mark.parametrize("sentence", ["мальчик дал девочке мяч"])
def test_russian_cyrillic_sentence_in_lexicon(sentence):
    from neural_assemblies.language.grammar_rules import RUSSIAN_LEXEME_DICT

    for word in sentence.split():
        assert word in RUSSIAN_LEXEME_DICT, f"missing lexeme: {word}"
