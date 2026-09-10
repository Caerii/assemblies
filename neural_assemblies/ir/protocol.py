"""Specification: neural_assemblies/ir/VERIFICATION.md#contract-protocol-wire

Protocol document validation and export for cross-language parity.
"""

from __future__ import annotations

from functools import lru_cache

from jsonschema import Draft202012Validator

import json
from pathlib import Path
from typing import Any

IR_VERSION = "1"


def schema_path(name: str) -> Path:
    """A v1 schema file; the schemas are package data under ``ir/v1/``."""
    return Path(__file__).resolve().parent / "v1" / name


@lru_cache(maxsize=None)
def _validator(schema_name):
    schema = json.loads(schema_path(schema_name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    # `format` remains an annotation in v1, consistently with the Rust bridge.
    return Draft202012Validator(schema)


def validate_schema_document(doc: dict[str, Any], schema_name: str) -> list[str]:
    """Validate JSON representability and the authoritative packaged v1 schema."""
    try:
        encoded = json.dumps(doc, allow_nan=False)
        if json.loads(encoded) != doc:
            return ["document contains non-JSON containers or object keys"]
    except (TypeError, ValueError, OverflowError) as exc:
        return [f"document is not finite JSON: {exc}"]
    return [f"{error.json_path}: {error.message}"
            for error in _validator(schema_name).iter_errors(doc)]


def validate_protocol_document(doc: dict[str, Any]) -> list[str]:
    return validate_schema_document(doc, "protocol.schema.json")


def load_protocol_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    errors = validate_protocol_document(doc)
    if errors:
        raise ValueError(f"invalid protocol IR at {path}: {errors}")
    return doc


def export_protocol_document(
    *,
    protocol_id: str,
    metrics: dict[str, Any],
    backend: str = "python",
    parameters: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
    regimes: dict[str, Any] | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build an IR v1 protocol document from runtime metrics."""
    doc: dict[str, Any] = {
        "ir_version": IR_VERSION,
        "protocol": protocol_id,
        "backend": backend,
        "metrics": metrics,
    }
    if parameters is not None:
        doc["parameters"] = parameters
    if thresholds is not None:
        doc["thresholds"] = thresholds
    if regimes is not None:
        doc["regimes"] = regimes
    if source is not None:
        doc["source"] = source
    if notes is not None:
        doc["notes"] = notes
    errors = validate_protocol_document(doc)
    if errors:
        raise ValueError(errors)
    return doc


def write_protocol_document(path: str | Path, doc: dict[str, Any]) -> None:
    errors = validate_protocol_document(doc)
    if errors:
        raise ValueError(errors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
