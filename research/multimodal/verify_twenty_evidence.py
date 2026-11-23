"""Verify archived R1l experiment evidence without GitHub Actions.

Default: decode/re-hash the archived 20-stage summary and check schema.
--recompute: rerun all 8,000 synthetic learning updates to ensure exact
source-to-evidence agreement. No native Brain numerical claim.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path

from research.multimodal.campaign_twenty import campaign
from research.multimodal.continual_twenty import Protocol

SHA256_JSON = '35b42065dfff39a16133ad6857302c6bc9e616ea2ab54e5e24d5c694a44980d1'
SHA256_XZ = '7bb159ead79ba342cbe49d9773ee09480ee271bdf199e6234fbb848c8e929270'
ARCHIVE = Path(__file__).parent / 'evidence/r1l-20stage-manifest.json.xz.base64'


def verify_archive(path: Path = ARCHIVE, *, recompute=False) -> dict:
    encoded = path.read_text(encoding='ascii').strip()
    xz = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(xz).hexdigest() != SHA256_XZ:
        raise ValueError('compressed evidence SHA-256 mismatch')
    raw = lzma.decompress(xz)
    if hashlib.sha256(raw).hexdigest() != SHA256_JSON:
        raise ValueError('decoded evidence SHA-256 mismatch')
    result = json.loads(raw)
    if (result['status'] != '20_stage_synthetic_ablation_not_native_brain'
            or result['stage_count'] != 20 or result['total_learning_updates'] != 8000
            or len(result['stages']) != 20):
        raise ValueError('invalid 20-stage experiment ledger')
    if len({x['name'] for x in result['stages']}) != 20:
        raise ValueError('duplicate stage in evidence')
    if any(len(x['raw_sha256']) != 64 for x in result['stages']):
        raise ValueError('missing stage-level evidence hashes')
    if recompute:
        fresh = campaign(Protocol(), draws=500)
        # JSON restores tuples to lists: compare canonical serialized form,
        # never Python dict equality between pre/post JSON-roundtrip objects.
        canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'))
        if canonical(fresh) != canonical(result):
            raise AssertionError('current code no longer reproduces archived evidence')
    return {'status': 'verified', 'artifact_sha256': SHA256_JSON,
            'stage_count': 20, 'learning_updates': 8000,
            'recomputed': bool(recompute)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recompute',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify_archive(recompute=args.recompute),indent=2))

if __name__=='__main__':main()
