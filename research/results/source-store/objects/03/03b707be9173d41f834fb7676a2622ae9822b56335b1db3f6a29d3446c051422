"""Load ``parity/registry.json`` protocol specifications."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

from .paths import golden_dir, parity_root, registry_path
from .protocol import Backend, Protocol


def _parse_backend(value: str | None) -> Backend:
    if not value:
        return Backend.BRAIN
    return Backend(value)


@lru_cache(maxsize=1)
def load_registry() -> dict:
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(f"parity registry not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_protocols() -> List[Protocol]:
    data = load_registry()
    out: List[Protocol] = []
    for row in data.get("protocols", []):
        out.append(protocol_from_row(row))
    return out


def protocol_from_row(row: dict) -> Protocol:
    claim_ids = row.get("claim_ids") or ([row["claim_id"]] if row.get("claim_id") else [])
    return Protocol(
        protocol_id=row["protocol_id"],
        paper_id=row.get("paper_id", ""),
        claim_ids=tuple(claim_ids),
        golden_path=row["golden"],
        backend=_parse_backend(row.get("backend")),
        config_file=row.get("config"),
        regime=row.get("regime"),
        repro_command=row.get("repro_command"),
        slow=bool(row.get("slow")),
        description=row.get("description", ""),
    )


def get_protocol(protocol_id: str) -> Protocol:
    for proto in list_protocols():
        if proto.protocol_id == protocol_id:
            return proto
    raise KeyError(protocol_id)


def get_protocol_by_claim(claim_id: str) -> Protocol | None:
    for proto in list_protocols():
        if claim_id in proto.claim_ids:
            return proto
    return None


def resolve_golden_path(proto: Protocol) -> str:
    rel = proto.golden_path
    if rel.startswith("golden/"):
        return str(golden_dir() / rel.removeprefix("golden/"))
    path = parity_root() / rel
    return str(path)


def claim_index() -> Dict[str, Protocol]:
    idx: Dict[str, Protocol] = {}
    for proto in list_protocols():
        for cid in proto.claim_ids:
            idx[cid] = proto
    return idx
