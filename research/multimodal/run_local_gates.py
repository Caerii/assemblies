"""Reproducible, fail-closed local research gate runner (no GitHub Actions).

Run from repo root:
  python -m research.multimodal.run_local_gates --output /tmp/assembly-gates
Outputs per-step stdout/stderr plus SHA-256 manifests, preserving failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def specs(python=sys.executable):
    # Smoke tests API connectivity; their numerical values are NOT publication evidence.
    return {
        # Discover ALL tests in this research subtree, including newly added
        # audit and provenance regression tests. Enumerating a few files here
        # silently excluded most of R1b–R1i from the previous gate.
        "protocol-tests": [python, "-m", "pytest", "-q", "research/multimodal"],
        "r1b-native": [python, "-m", "research.multimodal.native_retention",
                        "--n", "256", "--k", "12", "--loads", "2", "4",
                        "--seeds", "1"],
        "r1c-refraction": [python, "-m", "research.multimodal.refraction_paired",
                           "--n", "240", "--k", "20", "--loads", "2", "4",
                           "--recall-sample", "2", "--seeds", "42"],
        "r1d-bindings": [python, "-m", "research.multimodal.role_recombination",
                         "--contents", "4", "--seeds", "11"],
        "r1g-unsupervised": [python, "-m", "research.multimodal.native_zero_shot",
                             "--n", "128", "--k", "8", "--contents", "4",
                             "--seed", "42"],
        "r1h-supervised": [python, "-m", "research.multimodal.native_teacher_drive",
                           "--n", "128", "--k", "8", "--contents", "4",
                           "--seed", "42"],
        "r1i-controls": [python, "-m", "research.multimodal.native_teacher_controls",
                         "--n", "128", "--k", "8", "--contents", "4",
                         "--seed", "42"],
    }


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_revision(repo: Path):
    cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                        capture_output=True, text=True, check=False)
    return cp.stdout.strip() if cp.returncode == 0 else None


def run(root: Path, output: Path, *, names=None, timeout=180, dry_run=False,
        plan=None):
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    available = specs() if plan is None else plan
    chosen = tuple(names or available)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("nonempty unique gates required")
    for name in chosen:
        if name not in available or "/" in name or ".." in name:
            raise ValueError(f"invalid gate: {name}")
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    manifest = {"schema": 1, "git_head": git_revision(root),
                "python": sys.version.split()[0], "root": str(root),
                "dry_run": dry_run, "gates": []}
    for name in chosen:
        command = available[name]
        record = {"name": name, "argv": command, "status": "not-executed"}
        if not dry_run:
            start = time.perf_counter()
            try:
                proc = subprocess.run(command, cwd=root, capture_output=True,
                                      timeout=timeout, env=env, check=False)
                stdout, stderr = proc.stdout, proc.stderr
                record["exit_code"] = proc.returncode
                record["status"] = "passed" if proc.returncode == 0 else "failed"
            except subprocess.TimeoutExpired as error:
                stdout, stderr = error.stdout or b"", error.stderr or b""
                record["exit_code"] = None
                record["status"] = "timeout"
            record["elapsed_seconds"] = round(time.perf_counter() - start, 4)
            for label, data in (("stdout", stdout), ("stderr", stderr)):
                if isinstance(data, str):
                    data = data.encode()
                (output / f"{name}.{label}.txt").write_bytes(data)
                record[f"{label}_sha256"] = sha(data)
        manifest["gates"].append(record)
    manifest["all_passed"] = (not dry_run and all(
        record["status"] == "passed" for record in manifest["gates"]))
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (output / "run.json").write_bytes(encoded)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gates", nargs="+")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    manifest = run(args.root, args.output, names=args.gates,
                   timeout=args.timeout, dry_run=args.dry_run)
    print(json.dumps({"all_passed": manifest["all_passed"],
                      "status": {v["name"]: v["status"] for v in manifest["gates"]},
                      "run_file": str(args.output / "run.json")}, indent=2))
    return 0 if manifest["all_passed"] or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
