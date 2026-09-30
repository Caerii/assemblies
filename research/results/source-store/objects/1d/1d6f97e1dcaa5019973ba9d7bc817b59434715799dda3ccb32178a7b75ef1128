#!/usr/bin/env python3
"""Clone or update upstream reference repositories for literature parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "reference" / "manifest.json"


def ref_root() -> Path:
    return Path(os.environ.get("REF_ROOT", ROOT / ".reference"))


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sync_repo(repo: dict, root: Path) -> str:
    dest = root / repo["clone_dir"]
    url = repo["url"]
    if dest.is_dir() and (dest / ".git").is_dir():
        _run(["git", "fetch", "--depth", "1", "origin"], cwd=dest)
        branch = repo.get("branch", "main")
        _run(["git", "checkout", branch], cwd=dest)
        _run(["git", "pull", "--ff-only", "origin", branch], cwd=dest)
        return "updated"
    root.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--branch", repo.get("branch", "main"), url, str(dest)])
    return "cloned"


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = ref_root()
    print(f"Reference root: {root}")
    results: list[tuple[str, str]] = []
    for repo in data["repos"]:
        try:
            status = sync_repo(repo, root)
            results.append((repo["id"], status))
            print(f"  OK {repo['id']}: {status}")
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL {repo['id']}: {exc}", file=sys.stderr)
            return 1
    print(f"\nSynced {len(results)} repos into {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
