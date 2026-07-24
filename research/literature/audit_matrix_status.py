#!/usr/bin/env python3
"""Audit reproduction_matrix claim statuses against registry + repro commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LIT = Path(__file__).resolve().parent
MATRIX = LIT / "reproduction_matrix.json"
SUPPLEMENT = LIT / "reproduction_matrix_supplement.json"
REGISTRY = LIT / "parity" / "registry.json"

# Claims promoted when registry protocol exists and repro_command is set
PIN_WHEN_REGISTRY = True

# Manual overrides: claim_id -> status (after registry sync)
STATUS_OVERRIDES: dict[str, str] = {
    "COLT22-E02": "pinned",
    "COLT22-E04": "pinned",
    "COLT22-E06": "pinned",
    "COIN24-E02": "pinned",
    "COIN24-E04": "pinned",
    "EPWTA26-E03": "pinned",
    "PNAS20-E02": "pinned",
    "PNAS20-M04": "pinned",
    "PNAS20-M05": "pinned",
    "SEQ25-E06": "pinned",
    "TACL21-E05": "pinned",
    "XINF-E02": "pinned",
}


def _load_registry_claims() -> set[str]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims: set[str] = set()
    for proto in data.get("protocols", []):
        claims.update(proto.get("claim_ids", []))
    return claims


def _audit_file(path: Path, registry_claims: set[str]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = {"partial": 0, "missing": 0, "pinned": 0, "updated": 0, "no_repro": 0}
    for row in data.get("claims", []):
        cid = row.get("claim_id", "")
        status = row.get("status", "missing")
        if status in stats:
            stats[status] += 1
        if not row.get("repro_command"):
            stats["no_repro"] += 1
        new_status = STATUS_OVERRIDES.get(cid)
        if new_status is None and PIN_WHEN_REGISTRY and cid in registry_claims:
            if row.get("repro_command") and status in ("missing", "partial"):
                new_status = "pinned"
        if new_status and row.get("status") != new_status:
            row["status"] = new_status
            stats["updated"] += 1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    if not MATRIX.is_file():
        print(f"missing {MATRIX}", file=sys.stderr)
        return 1
    registry_claims = _load_registry_claims() if REGISTRY.is_file() else set()
    s1 = _audit_file(MATRIX, registry_claims)
    s2 = {}
    if SUPPLEMENT.is_file():
        s2 = _audit_file(SUPPLEMENT, registry_claims)
    print(f"{MATRIX.name}: {s1}")
    if s2:
        print(f"{SUPPLEMENT.name}: {s2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
