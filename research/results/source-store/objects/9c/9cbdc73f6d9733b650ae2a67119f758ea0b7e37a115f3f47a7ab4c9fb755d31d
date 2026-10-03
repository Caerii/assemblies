"""Bring a finished run from its pinned worktree into this checkout.

Source-linked specification: research/README.md#recoverable-source

A registered run executes in a worktree pinned at its registration commit
([[pinned-runs-worktree]]), and its record, its log and the source-store
objects it wrote are untracked files THERE. Recording a result starts by
bringing them here, validating them, and clearing them from the worktree so
the next registration can check it out. Done by hand, that step once deleted
committed source objects (an over-broad clean); here every move is checked:

* the run directories are copied, and refused if a different copy is
  already here;
* exactly the source-store objects the runs' manifests name are copied, each
  verified against its SHA-256 name, never a whole directory;
* the log is copied to ``research/results/logs/<name>``;
* every record is validated (``evidence.validate_artifact``);
* with ``--clean``, the worktree's copies are removed only if they are
  untracked there and byte-identical here.

    python -m research.collect --worktree PATH --log a5.log:word-capacity-optimum-20261001.log \\
        research/results/runs/aligner.word-capacity-synapses/word-capacity-optimum-p005-20261001 ...
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = Path("research") / "results" / "source-store" / "objects"
LOGS = Path("research") / "results" / "logs"


def _same_tree(a: Path, b: Path) -> bool:
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    return files_a == files_b and all(
        filecmp.cmp(a / f, b / f, shallow=False) for f in files_a)


def _objects(run: Path) -> list[str]:
    from research.source_store import read_manifest
    return sorted({digest for _name, digest in read_manifest(run)["members"]})


def _untracked(worktree: Path, path: Path) -> bool:
    tracked = subprocess.run(["git", "-C", str(worktree), "ls-files", "--error-unmatch",
                              str(path)], capture_output=True)
    return tracked.returncode != 0


def collect(worktree: Path, runs, logs=(), *, root: Path = ROOT, clean=False,
            validate=None) -> dict:
    """Copy ``runs`` (repository-relative directories) and ``logs``
    ((worktree file, name under the logs directory)) from ``worktree``;
    returns what was copied, validated and cleaned."""
    if validate is None:
        from research.evidence import validate_artifact

        def _validate(path):
            return validate_artifact(path, root=root)
        validate = _validate
    report = {"runs": [], "objects": [], "logs": [], "errors": {}, "cleaned": []}
    for rel in map(Path, runs):
        src, dst = worktree / rel, root / rel
        if not (src / "results.json").is_file():
            raise FileNotFoundError(f"{src} holds no results.json")
        if dst.exists():
            if not _same_tree(src, dst):
                raise FileExistsError(f"{dst} exists and differs from the worktree's copy")
        else:
            shutil.copytree(src, dst)
        report["runs"].append(rel.as_posix())
        for digest in _objects(src):
            obj = STORE / digest[:2] / digest
            data = (worktree / obj).read_bytes() if (worktree / obj).exists() \
                else (root / obj).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"source object {digest} does not match its name")
            if not (root / obj).exists():
                (root / obj).parent.mkdir(parents=True, exist_ok=True)
                (root / obj).write_bytes(data)
                report["objects"].append(digest)
        errors = validate(dst / "results.json")
        if errors:
            report["errors"][rel.as_posix()] = errors
    for src_name, name in logs:
        src, dst = worktree / src_name, root / LOGS / name
        if dst.exists() and not filecmp.cmp(src, dst, shallow=False):
            raise FileExistsError(f"{dst} exists and differs from {src}")
        shutil.copyfile(src, dst)
        report["logs"].append((LOGS / name).as_posix())
    if clean and not report["errors"]:
        for rel in map(Path, report["runs"]):
            if _untracked(worktree, rel) and _same_tree(worktree / rel, root / rel):
                shutil.rmtree(worktree / rel)
                report["cleaned"].append(rel.as_posix())
            for digest in _objects(root / rel):
                obj = STORE / digest[:2] / digest
                if ((worktree / obj).exists() and _untracked(worktree, obj)
                        and filecmp.cmp(worktree / obj, root / obj, shallow=False)):
                    (worktree / obj).unlink()
                    report["cleaned"].append(obj.as_posix())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--log", action="append", default=[],
                        help="WORKTREE_FILE:NAME, copied to research/results/logs/NAME")
    parser.add_argument("--clean", action="store_true",
                        help="remove the worktree's untracked copies once verified here")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    logs = [tuple(item.split(":", 1)) for item in args.log]
    if any(len(item) != 2 for item in logs):
        parser.error("--log takes WORKTREE_FILE:NAME")
    report = collect(args.worktree, args.runs, logs, clean=args.clean)
    print(json.dumps(report, indent=1))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
