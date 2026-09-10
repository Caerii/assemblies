"""CLI: ``python -m neural_assemblies.parity.cli verify coin2024_demo``."""

from __future__ import annotations

import argparse
import json

from .registry import list_protocols
from .runner import run_claim, run_protocol, verify_protocol, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Literature parity protocol runner")
    sub = parser.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="list registered protocols")
    ls.add_argument("--json", action="store_true")

    run_p = sub.add_parser("run", help="run repro_command (pytest)")
    run_p.add_argument("protocol_id")

    verify_p = sub.add_parser("verify", help="in-process golden verification")
    verify_p.add_argument("protocol_id")
    verify_p.add_argument("--manifest", help="write JSON manifest path")

    claim_p = sub.add_parser("claim", help="run by matrix claim_id")
    claim_p.add_argument("claim_id")
    claim_p.add_argument("--verify", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "list":
        rows = [
            {
                "protocol_id": p.protocol_id,
                "claim_ids": list(p.claim_ids),
                "backend": p.backend.value,
                "slow": p.slow,
            }
            for p in list_protocols()
        ]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                claims = ",".join(r["claim_ids"])
                print(f"{r['protocol_id']}\t{r['backend']}\t{claims}")
        return 0

    if args.command == "run":
        result = run_protocol(args.protocol_id, prefer="repro")
        print(json.dumps(result.to_manifest(), indent=2))
        return 0 if result.passed else 1

    if args.command == "verify":
        from neural_assemblies.programs.colt_mnist_data import DatasetUnavailable
        try:
            result = verify_protocol(args.protocol_id)
        except DatasetUnavailable as exc:
            print(json.dumps({'protocol_id': args.protocol_id, 'status': 'unavailable',
                              'passed': False, 'message': str(exc)}))
            return 2
        print(json.dumps(result.to_manifest(), indent=2))
        if args.manifest:
            write_manifest(result, args.manifest)
        return 0 if result.passed else 1

    if args.command == "claim":
        result = run_claim(
            args.claim_id,
            prefer="verify" if args.verify else "repro",
        )
        print(json.dumps(result.to_manifest(), indent=2))
        return 0 if result.passed else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
