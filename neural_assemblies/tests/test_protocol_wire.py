"""The same wire documents must survive Python and Rust without semantic loss."""
import json
from pathlib import Path

import pytest

from neural_assemblies.ir.protocol import (
    load_protocol_document, schema_path, validate_protocol_document, write_json_document,
    write_protocol_document,
)

CASES = json.loads(schema_path("protocol.cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_wire_corpus(case, tmp_path):
    if "raw_json" in case:
        path = tmp_path / "raw.json"
        path.write_text(case["raw_json"], encoding="utf-8")
        with pytest.raises(ValueError):
            load_protocol_document(path)
        return
    document = case["document"]
    assert (not validate_protocol_document(document)) == case["valid"]
    path = tmp_path / "document.json"
    if case["valid"]:
        write_protocol_document(path, document)
        assert load_protocol_document(path) == document
    else:
        with pytest.raises(ValueError):
            write_protocol_document(path, document)
        assert not path.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), (1, 2), {1: 2}])
def test_non_json_python_values_cannot_enter_the_ir(value):
    assert validate_protocol_document({"ir_version": "1", "protocol": "x", "metrics": {"value": value}})


def test_old_writer_cannot_overwrite_committed_evidence(tmp_path):
    path = tmp_path / "result.json"
    original = {"ir_version": "1", "protocol": "x", "metrics": {"x": 1}}
    write_protocol_document(path, original)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_protocol_document(path, {**original, "metrics": {"x": 2}})
    assert path.read_bytes() == before


def test_protocol_writer_uses_canonical_json_bytes(tmp_path):
    path = tmp_path / "protocol.json"
    write_protocol_document(path, {"ir_version": "1", "protocol": "x",
                                   "metrics": {"b": 1, "a": "λ"}})
    assert path.read_text(encoding="utf-8") == (
        '{\n  "ir_version": "1",\n  "metrics": {\n'
        '    "a": "λ",\n    "b": 1\n  },\n  "protocol": "x"\n}\n'
    )


def test_generic_json_writer_is_canonical_finite_and_create_only(tmp_path):
    path = tmp_path / "report.json"
    write_json_document(path, {"b": 1, "a": "\u03bb"})
    assert path.read_text(encoding="utf-8") == '{\n  "a": "\u03bb",\n  "b": 1\n}\n'
    with pytest.raises(FileExistsError):
        write_json_document(path, {"a": 2})
    assert path.read_text(encoding="utf-8").endswith('"b": 1\n}\n')

    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="finite JSON"):
        write_json_document(invalid, {"value": float("nan")})
    assert not invalid.exists()


def test_committed_parity_documents_still_validate():
    root = Path(__file__).resolve().parents[2]
    checked = 0
    for path in (root / "research/literature/parity/golden").glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "ir_version" in document:
            assert not validate_protocol_document(document), path
            checked += 1
    assert checked > 0


HOMEOSTASIS_CASES = json.loads(schema_path("homeostasis.cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", HOMEOSTASIS_CASES, ids=lambda case: case["name"])
def test_shared_homeostasis_wire(case):
    from neural_assemblies import HomeostasisConfig
    if not case["valid"]:
        with pytest.raises(ValueError):
            HomeostasisConfig.from_document(case["document"])
        return
    config = HomeostasisConfig.from_document(case["document"])
    document = config.to_document()
    assert HomeostasisConfig.from_document(json.loads(json.dumps(document))) == config
    expected = dict(case["document"])
    if isinstance(expected['synaptic_scaling'], list):
        expected['synaptic_scaling'] = sorted(expected['synaptic_scaling'])
    assert document == expected


def test_homeostasis_document_controls_the_actual_engine():
    from neural_assemblies import HomeostasisConfig, Brain
    document = next(c['document'] for c in HOMEOSTASIS_CASES if c['name'] == 'deferred')
    config = HomeostasisConfig.from_document(document)
    brain = Brain(p=.1, engine='numpy_sparse', **config.as_kwargs())
    assert HomeostasisConfig.from_engine(brain._engine).to_document() == config.to_document()
    assert brain._engine.synaptic_scaling == frozenset({'A'})
    assert brain._engine.synaptic_scaling_deferred is True
