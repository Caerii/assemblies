"""Integrity and optional exact deterministic replay of the R1o evidence."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path

from .noncommutative_sweep import Sweep, run

EVIDENCE = Path(__file__).with_name('evidence') / 'r1o-20seed.json.xz.base64'
EXPECTED_RAW_SHA256 = '868f6a45295ebf703fc8c690063242059a717d3de605a564cf955807bccf725b'
EXPECTED_ROWS_SHA256 = '3bd3d1cb5f3dba7f62ed39b08ed2885a57c678c7833028cd2848c57da84547db'


def verify(path=EVIDENCE, *, replay=False):
    try:
        enc = Path(path).read_text(encoding='ascii').strip()
        raw = lzma.decompress(base64.b64decode(enc, validate=True))
    except (OSError, ValueError, lzma.LZMAError) as e:
        raise AssertionError('cannot decode evidence') from e
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_RAW_SHA256:
        raise AssertionError(f'raw evidence hash mismatch: {sha}')
    evidence = json.loads(raw)
    if evidence['rows_sha256'] != EXPECTED_ROWS_SHA256:
        raise AssertionError('cell provenance hash mismatch')
    if evidence['status'] != 'synthetic_affine_composition_not_native_brain':
        raise AssertionError('incorrect experiment type')
    if len(evidence['cells']) != 60 * 6:
        raise AssertionError('unexpected number of trials/models')
    if replay:
        parameters = evidence['spec']
        spec = Sweep(geometries=tuple(tuple(x) for x in parameters['geometries']),
                     seeds=tuple(parameters['seeds']),
                     model_seed=parameters['model_seed'],
                     atomic_examples=parameters['atomic_examples'],
                     test_cues=parameters['test_cues'])
        regenerated = (json.dumps(run(spec), indent=2, sort_keys=True) + '\n').encode('utf-8')
        if raw != regenerated:
            raise AssertionError('replay produced different bytes')
    return {'raw_sha256': sha, 'rows_sha256': evidence['rows_sha256'],
            'cells': len(evidence['cells']), 'replayed': bool(replay)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=Path, default=EVIDENCE)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    print(json.dumps(verify(args.path, replay=args.replay), indent=2))

if __name__ == '__main__': main()
