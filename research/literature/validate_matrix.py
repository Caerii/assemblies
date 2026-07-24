#!/usr/bin/env python3
"""Validate reproduction matrix + supplement; recompute merged summary."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LIT_DIR = Path(__file__).resolve().parent
INDEX_PATH = LIT_DIR / "index.json"
MATRIX_PATH = LIT_DIR / "reproduction_matrix.json"
SUPPLEMENT_PATH = LIT_DIR / "reproduction_matrix_supplement.json"

REQUIRED_CLAIM_KEYS = {
    "claim_id",
    "paper_id",
    "category",
    "claim",
    "status",
    "module",
    "test",
    "protocol_id",
    "golden",
    "gap",
    "priority",
    "repro_command",
}
OPTIONAL_CLAIM_KEYS = {"config_ref"}
VALID_STATUSES = {"pinned", "partial", "missing", "theorem", "research", "legacy"}
VALID_CATEGORIES = {"theorem", "mechanism", "empirical", "demo", "config"}
VALID_PRIORITIES = {"P0", "P1", "P2", "P3", "P4"}
NO_TEST_STATUSES = {"theorem", "config", "missing", "legacy"}


def load_claims() -> list[dict]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    claims = list(matrix.get("claims", []))
    if SUPPLEMENT_PATH.exists():
        sup = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
        seen = {c["claim_id"] for c in claims}
        for row in sup.get("claims", []):
            if row["claim_id"] in seen:
                raise ValueError(f"duplicate claim_id across base+supplement: {row['claim_id']}")
            claims.append(row)
            seen.add(row["claim_id"])
    return claims


def main() -> int:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    claims = load_claims()
    errors: list[str] = []
    claim_ids: set[str] = set()
    paper_ids = {e["id"] for e in index.get("entries", [])}

    for i, row in enumerate(claims):
        cid = row.get("claim_id", f"#{i}")
        missing = REQUIRED_CLAIM_KEYS - set(row)
        if missing:
            errors.append(f"{cid}: missing keys {sorted(missing)}")
        if row.get("status") not in VALID_STATUSES:
            errors.append(f"{cid}: bad status {row.get('status')!r}")
        if row.get("category") not in VALID_CATEGORIES:
            errors.append(f"{cid}: bad category {row.get('category')!r}")
        if row.get("priority") not in VALID_PRIORITIES:
            errors.append(f"{cid}: bad priority {row.get('priority')!r}")
        pid = row.get("paper_id")
        if pid is not None and pid not in paper_ids and not str(cid).startswith("XINF"):
            errors.append(f"{cid}: unknown paper_id {pid!r}")
        if cid in claim_ids:
            errors.append(f"duplicate claim_id {cid!r}")
        claim_ids.add(cid)
        if (
            row.get("status") == "pinned"
            and row.get("category") not in NO_TEST_STATUSES
            and not row.get("test")
            and not row.get("repro_command")
        ):
            errors.append(f"{cid}: pinned empirical/mechanism but no test or repro_command")

    counts = Counter(row["status"] for row in claims)
    cat_counts = Counter(row["category"] for row in claims)
    papers = {r["paper_id"] for r in claims if r.get("paper_id")}

    if errors:
        print("reproduction matrix validation FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(claims)} merged claims across {len(papers)} papers\n"
        f"  status: pinned={counts['pinned']} partial={counts['partial']} "
        f"missing={counts['missing']} theorem={counts['theorem']} "
        f"research={counts['research']} legacy={counts.get('legacy', 0)}\n"
        f"  category: config={cat_counts['config']} empirical={cat_counts['empirical']} "
        f"mechanism={cat_counts['mechanism']} demo={cat_counts['demo']} "
        f"theorem={cat_counts['theorem']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
