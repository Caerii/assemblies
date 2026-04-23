"""Validate the research claims index."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "research" / "claims" / "index.json"
ALLOWED_STATUSES = {"formalized_claim", "evidence_summary"}


def main() -> int:
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise SystemExit("claims index must contain a non-empty 'entries' list")

    seen_ids: set[str] = set()
    for entry in entries:
        claim_id = entry.get("id")
        if not claim_id or not isinstance(claim_id, str):
            raise SystemExit("each entry must have a string 'id'")
        if claim_id in seen_ids:
            raise SystemExit(f"duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)

        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            raise SystemExit(f"{claim_id}: invalid status {status!r}")

        for key in ("title", "category", "primary_artifact", "notes"):
            value = entry.get(key)
            if not value or not isinstance(value, str):
                raise SystemExit(f"{claim_id}: missing string field {key!r}")

        for key in ("experiment_scripts", "result_artifacts"):
            value = entry.get(key)
            if not isinstance(value, list) or not value:
                raise SystemExit(f"{claim_id}: field {key!r} must be a non-empty list")

        for path_key in ("primary_artifact",):
            path = ROOT / entry[path_key]
            if not path.exists():
                raise SystemExit(f"{claim_id}: missing referenced file {entry[path_key]}")

        for list_key in ("experiment_scripts", "result_artifacts"):
            for rel_path in entry[list_key]:
                path = ROOT / rel_path
                if not path.exists():
                    raise SystemExit(f"{claim_id}: missing referenced file {rel_path}")

    formalized = sum(1 for e in entries if e["status"] == "formalized_claim")
    evidence = sum(1 for e in entries if e["status"] == "evidence_summary")
    print(
        f"Claims index valid: {len(entries)} entries "
        f"({formalized} formalized, {evidence} evidence summaries)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
