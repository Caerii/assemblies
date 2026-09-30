#!/usr/bin/env python3
"""Run a single literature claim reproduction by claim_id."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LIT_DIR = Path(__file__).resolve().parent
MATRIX_PATH = LIT_DIR / "reproduction_matrix.json"
SUPPLEMENT_PATH = LIT_DIR / "reproduction_matrix_supplement.json"
REPO_ROOT = LIT_DIR.parents[1]


def _all_claims() -> list[dict]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    rows = list(matrix.get("claims", []))
    if SUPPLEMENT_PATH.exists():
        sup = json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))
        rows.extend(sup.get("claims", []))
    return rows


def load_claim(claim_id: str) -> dict:
    for row in _all_claims():
        if row["claim_id"] == claim_id:
            return row
    raise KeyError(claim_id)


def list_claims(status: str | None = None) -> list[dict]:
    rows = _all_claims()
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return rows


def _run_via_parity(claim_id: str, *, prefer: str = "repro") -> int | None:
    """Return exit code if claim is in parity registry, else None."""
    try:
        from neural_assemblies.parity.registry import get_protocol_by_claim
        from neural_assemblies.parity.runner import run_protocol
    except ImportError:
        return None

    proto = get_protocol_by_claim(claim_id)
    if proto is None:
        return None

    result = run_protocol(proto.protocol_id, prefer=prefer)
    if not result.passed:
        if result.message:
            print(result.message, file=sys.stderr)
        if result.diffs:
            print(json.dumps(result.diffs, indent=2), file=sys.stderr)
    return 0 if result.passed else (result.exit_code or 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", help="claim_id from reproduction_matrix.json")
    parser.add_argument("--list", action="store_true", help="list all claims")
    parser.add_argument("--status", help="filter --list by status")
    parser.add_argument("--paper", help="filter --list by paper_id")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="in-process golden check (parity registry only)",
    )
    args = parser.parse_args(argv)

    if args.list:
        for row in list_claims(args.status):
            if args.paper and row.get("paper_id") != args.paper:
                continue
            print(f"{row['claim_id']}\t{row['status']}\t{row.get('paper_id') or '-'}\t{row['claim'][:60]}")
        return 0

    if not args.claim:
        parser.error("provide --claim <id> or --list")

    try:
        row = load_claim(args.claim)
    except KeyError:
        print(f"unknown claim_id: {args.claim}", file=sys.stderr)
        return 2

    print(f"# {row['claim_id']}: {row['claim']}")
    print(f"# status={row['status']} paper={row.get('paper_id')}")

    if args.verify:
        parity_code = _run_via_parity(args.claim, prefer="verify")
        if parity_code is not None:
            return parity_code

    cmd = row.get("repro_command")
    if cmd:
        result = subprocess.run(cmd, shell=True, cwd=REPO_ROOT)
        return result.returncode

    parity_code = _run_via_parity(args.claim, prefer="repro")
    if parity_code is not None:
        return parity_code

    print(f"{args.claim}: status={row['status']} — no repro_command yet", file=sys.stderr)
    if row.get("gap"):
        print(f"  gap: {row['gap']}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
