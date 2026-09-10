"""Specification: neural_assemblies/ir/VERIFICATION.md#contract-ir-verification

Protocol document validation and export for cross-language parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IR_VERSION = "1"
_REQUIRED_ROOT = frozenset({"ir_version", "protocol", "metrics"})


def schema_path(name: str) -> Path:
    """A v1 schema file; the schemas are package data under ``ir/v1/``."""
    return Path(__file__).resolve().parent / "v1" / name


def validate_protocol_document(doc: dict[str, Any]) -> list[str]:
    """Lightweight validation without jsonschema dependency."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    missing = _REQUIRED_ROOT - doc.keys()
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")
    if doc.get("ir_version") != IR_VERSION:
        errors.append(f"ir_version must be {IR_VERSION!r}")
    if "metrics" in doc and not isinstance(doc["metrics"], dict):
        errors.append("metrics must be an object")
    if "regimes" in doc and not isinstance(doc["regimes"], dict):
        errors.append("regimes must be an object")
    return errors


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
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
