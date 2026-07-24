#!/usr/bin/env python3
"""Validate research/literature/index.json structure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_ENTRY_KEYS = {
    "id",
    "tier",
    "year",
    "title",
    "authors",
    "venue",
    "bibkey",
    "urls",
    "mechanisms",
    "implementation_status",
    "package_modules",
    "gaps",
}

VALID_STATUSES = {"implemented", "partial", "legacy", "research", "not_started"}
VALID_TIERS = {"foundations", "nemo_language", "extensions"}


def main() -> int:
    index_path = Path(__file__).with_name("index.json")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for i, entry in enumerate(data.get("entries", [])):
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            errors.append(f"entries[{i}] ({entry.get('id', '?')}): missing {sorted(missing)}")
        if entry.get("implementation_status") not in VALID_STATUSES:
            errors.append(
                f"entries[{i}] ({entry.get('id')}): bad status {entry.get('implementation_status')!r}"
            )
        if entry.get("tier") not in VALID_TIERS:
            errors.append(f"entries[{i}] ({entry.get('id')}): bad tier {entry.get('tier')!r}")

    roadmap = data.get("implementation_roadmap_priority", [])
    ids = {e["id"] for e in data.get("entries", [])}
    for pid in roadmap:
        if pid not in ids:
            errors.append(f"roadmap references unknown id: {pid}")

    if errors:
        print("Literature index validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Literature index OK ({len(data.get('entries', []))} papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
