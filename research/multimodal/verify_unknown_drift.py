"""Validate compressed R1n evidence and optionally reproduce every trial."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path

from .analyze_unknown_drift import _index
from .unknown_drift import Protocol, Scenario, run

EVIDENCE_SHA256 = 'ebfc57c6c6c0580d3933707b6445110ebbeacf86c1cc1b364491e685eeafbd06'


def read_evidence(path: Path, *, expected_sha256: str | None = EVIDENCE_SHA256):
    try:
        decoded = base64.b64decode(path.read_text(encoding='ascii').strip(), validate=True)
        raw = lzma.decompress(decoded)
    except (ValueError, lzma.LZMAError, OSError) as exc:
        raise ValueError('invalid R1n compressed evidence') from exc
    found = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and found != expected_sha256:
        raise ValueError(f'evidence checksum mismatch: {found}')
    report = json.loads(raw)
    _index(report)  # fails on a missing or unpaired scenario/seed cell
    return report, found


def verify_replay(report):
    config = report['protocol']
    p = Protocol(n=config['n'], k=config['k'], rounds=config['rounds'],
                 probes_per_role=config['probes_per_role'],
                 seeds=tuple(config['seeds']),
                 scenarios=tuple(Scenario(s['name'], tuple(s['changes']), tuple(s['corrupt']))
                                 for s in config['scenarios']))
    repeated = run(p)
    if json.loads(json.dumps(repeated)) != report:
        raise AssertionError('R1n recomputation disagrees with committed evidence')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=Path(__file__).parent / 'evidence' / 'r1n-20seed.json.xz.base64')
    ap.add_argument('--replay', action='store_true')
    args = ap.parse_args()
    report, sha = read_evidence(args.archive)
    replay_ok = verify_replay(report) if args.replay else None
    print(json.dumps({'sha256': sha, 'trials': len(report['trials']),
                      'replay': replay_ok}, sort_keys=True))


if __name__ == '__main__':
    main()
