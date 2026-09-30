"""Validate the machine-readable research suite registry."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "research" / "registry.json"


def main() -> int:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    suites = data.get("suites", [])

    errors = []
    for suite in suites:
        name = suite["name"]
        experiments_dir = REPO_ROOT / suite["experiments_dir"]
        results_dir = REPO_ROOT / suite["results_dir"]

        if not experiments_dir.is_dir():
            errors.append(f"{name}: missing experiments dir {experiments_dir}")
        if not results_dir.exists():
            errors.append(f"{name}: missing results dir {results_dir}")

    if errors:
        print("Registry validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Registry valid: {len(suites)} suites checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
