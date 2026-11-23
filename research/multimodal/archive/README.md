# Archived v6 prototype snapshot

The complete runnable candidate is being migrated into native `Caerii/assemblies` modules incrementally. The snapshot below preserves the **source** of the companion workbench, including Rust files, Python code, tests, READMEs, and benchmark scripts.

## Decode

```bash
# From the repository root, with Python 3 available:
python - <<'PY'
from pathlib import Path
import base64, hashlib, lzma, tarfile
source = Path('research/multimodal/archive/v6-companion-sources.tar.xz.base64')
blob = base64.b64decode(source.read_text().strip(), validate=True)
expected = '95d327681b87d74fdfcfd2c8609d521159d2e0c90cbc82f288622248a09b6f70'
assert hashlib.sha256(blob).hexdigest() == expected
target = Path('research/multimodal/archive/restore-v6')
target.mkdir(parents=True, exist_ok=True)
archive = target.parent / 'v6-companion-sources.tar.xz'
archive.write_bytes(blob)
with tarfile.open(archive, 'r:xz') as tar:
    for member in tar.getmembers():
        dest = (target / member.name).resolve()
        if not dest.is_relative_to(target.resolve()):
            raise ValueError('unsafe archive member')
    tar.extractall(target, filter='data')
print('Verified, restored to', target)
PY
```

The archival snapshot intentionally excludes `experiments_v1.json` and other generated `*.json` data, as well as `__pycache__`. The canonical results will be re-run and placed under `research/notes/multimodal/` with source and machine provenance. The contemporaneous ChatGPT companion ZIP includes the original JSON diagnostics.

## Science status

Archiving source **does not** mean that the archived Rust modules compile or that native `Brain` parity has been established. Keep the scientific evidence gates in `../RESEARCH_PROGRAM.md` intact.

## Workflow

Develop new, reviewed modules in `research/multimodal/` or native package directories, commit to the single branch `research/multimodal-continual-assemblies`, and update `../CONTINUE_HERE.md` after each integration/validation batch.
