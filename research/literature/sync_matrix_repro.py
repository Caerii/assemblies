#!/usr/bin/env python3
"""Backfill reproduction_matrix repro_command + status from parity/registry.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LIT = Path(__file__).resolve().parent
MATRIX = LIT / "reproduction_matrix.json"
SUPPLEMENT = LIT / "reproduction_matrix_supplement.json"
REGISTRY = LIT / "parity" / "registry.json"

STATUS_OVERRIDES = {
    "coin2024_softmax": "pinned",
    "colt2022_mnist_brain": "pinned",
    "colt2022_mnist_hierarchical": "pinned",
    "colt2022_mnist_notebook": "pinned",
    "coin2024_markov_arc": "pinned",
    "nemo2025_fsm_mod3_numpy": "pinned",
}


def _load_registry_index() -> dict[str, dict]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    idx: dict[str, dict] = {}
    for proto in data.get("protocols", []):
        for cid in proto.get("claim_ids", []):
            idx[cid] = proto
    return idx


def _sync_file(path: Path, idx: dict[str, dict]) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    repro_updates = 0
    status_updates = 0
    for row in data.get("claims", []):
        cid = row.get("claim_id")
        if cid not in idx:
            continue
        proto = idx[cid]
        cmd = proto.get("repro_command")
        if cmd and row.get("repro_command") != cmd:
            row["repro_command"] = cmd
            repro_updates += 1
        if row.get("protocol_id") in (None, "") and proto.get("protocol_id"):
            row["protocol_id"] = proto["protocol_id"]
        pid = proto["protocol_id"]
        if pid in STATUS_OVERRIDES and row.get("status") in ("missing", None):
            row["status"] = STATUS_OVERRIDES[pid]
            status_updates += 1
        elif cmd and row.get("status") == "missing" and pid not in STATUS_OVERRIDES:
            row["status"] = "pinned"
            status_updates += 1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return repro_updates, status_updates


def main() -> int:
    if not REGISTRY.is_file():
        print(f"missing registry: {REGISTRY}", file=sys.stderr)
        return 1
    idx = _load_registry_index()
    r1, s1 = _sync_file(MATRIX, idx)
    r2, s2 = 0, 0
    if SUPPLEMENT.is_file():
        r2, s2 = _sync_file(SUPPLEMENT, idx)
    print(f"synced {MATRIX.name}: repro={r1} status={s1}")
    if SUPPLEMENT.is_file():
        print(f"synced {SUPPLEMENT.name}: repro={r2} status={s2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
