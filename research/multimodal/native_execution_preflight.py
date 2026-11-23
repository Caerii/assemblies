"""R1q: fail-closed preflight and numeric smoke for the CURRENT native Brain.

Only run from a real checkout of research/multimodal-continual-assemblies.
Preflight refuses PyPI wheels, an imported package outside that checkout,
wrong branches and uncommitted edits in scientific modules.

    python -m research.multimodal.native_execution_preflight --output /tmp/r1q.json

A successful preflight proves APIs and explicit synapse behavior execute. It
DOES NOT prove compositional generalization, parity or native acceleration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

EXPECTED_BRANCH = "research/multimodal-continual-assemblies"


class NativePreflightError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True, check=False,
                           timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativePreflightError("Git checkout inspection unavailable") from exc
    if p.returncode:
        raise NativePreflightError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def inspect_checkout(root: Path, *, require_clean: bool = True) -> dict:
    root = root.resolve()
    actual_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != root:
        raise NativePreflightError(f"root mismatch: {actual_root} != {root}")
    branch = _git(root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise NativePreflightError(f"wrong branch: {branch!r}")
    dirty = bool(_git(root, "status", "--porcelain", "--", "research/multimodal", "neural_assemblies"))
    if require_clean and dirty:
        raise NativePreflightError("research or native package contains uncommitted changes")
    expected = root / "neural_assemblies" / "core" / "brain.py"
    if not expected.is_file():
        raise NativePreflightError("missing current native Brain source")
    brain_mod = importlib.import_module("neural_assemblies.core.brain")
    source = Path(inspect.getfile(brain_mod.Brain)).resolve()
    if source != expected:
        raise NativePreflightError(f"installed/stale Brain loaded: {source} != {expected}")
    native_sha = hashlib.sha256(expected.read_bytes()).hexdigest()
    from neural_assemblies.core.brain import Brain
    if "external_drive" not in inspect.signature(Brain.project).parameters:
        raise NativePreflightError("Brain.project lacks explicit teacher-drive API")
    if not all(callable(getattr(Brain, member, None)) for member in
               ("engine_for", "read_only", "frozen", "add_area", "clear_activity")):
        raise NativePreflightError("required native Brain APIs absent")
    return {
        "git_head": _git(root, "rev-parse", "HEAD"),
        "branch": branch, "checkout_root": str(root), "dirty": dirty,
        "brain_source": str(source), "brain_sha256": native_sha,
        "python": sys.version.split()[0], "numpy": np.__version__,
    }


def numeric_smoke(seed: int = 37, n: int = 64, k: int = 8) -> dict:
    """Two native models, same graph seed: beta>0 versus beta=0.

    Teacher clamp must win target neurons, change eligible synapses only if
    beta>0, then run repeated no-teacher reads without any matrix mutation.
    """
    if not (0 < k < n) or type(seed) is not int or seed < 0:
        raise ValueError("invalid native smoke geometry/seed")
    from .zero_shot_gate import Config
    from .native_teacher_drive import NativeTeacherDriven
    cfg = Config(n=n, k=k, contents=4, seed=seed,
                 model_seed=seed + 100000, roles=("SUBJ", "OBJ"),
                 transform_family="cyclic_shift")
    rng = np.random.default_rng(seed + 1)
    cue = np.sort(rng.choice(n, k, replace=False)).astype(np.uint32)
    target = np.sort(rng.choice(n, k, replace=False)).astype(np.uint32)
    output = {}
    for beta, name in ((.1, "plastic"), (0., "nonplastic")):
        model = NativeTeacherDriven(cfg, beta=beta, presentations=3,
                                    teacher_gain=100.)
        before = model.state_fingerprint()
        model.train(0, "SUBJ", cue.copy(), target.copy())
        after = model.state_fingerprint()
        diag = model.clamp_diagnostics[-1]
        if not diag["teacher_target_overlap"] == 1.:
            raise NativePreflightError("teacher clamp did not select exact target")
        if beta == 0 and before != after:
            raise NativePreflightError("nonplastic training mutated explicit graph")
        if beta > 0 and diag["active_edge_changed"] == 0:
            raise NativePreflightError("plastic training did not potentiate target edges")
        read1 = model.predict(0, "SUBJ", cue.copy())
        if model.state_fingerprint() != after:
            raise NativePreflightError("first read mutated explicit graph")
        read2 = model.predict(0, "SUBJ", cue.copy())
        if model.state_fingerprint() != after:
            raise NativePreflightError("second read mutated explicit graph")
        if not np.array_equal(read1, read2):
            raise NativePreflightError("repeated native read not deterministic")
        output[name] = {
            "teacher_target_overlap": diag["teacher_target_overlap"],
            "changed_edges": diag["active_edge_changed"],
            "synapse_delta": diag["active_edge_delta_sum"],
            "trained_graph_changed": before != after,
            "unprompted_overlap": int(np.intersect1d(read1, target).size) / k,
            "read_repeatable": True,
            "full_graph_immutable_on_read": True,
            "graph_sha256_after_training": after,
        }
    return output


def run(root: Path, *, smoke: bool = True, require_clean: bool = True) -> dict:
    provenance = inspect_checkout(root, require_clean=require_clean)
    result = {"status": "native_api_smoke_only", "provenance": provenance,
              "smoke": numeric_smoke() if smoke else "not-executed",
              "claim_limits": [
                  "A native smoke checks only teacher clamp, Hebbian writes and read isolation.",
                  "It is not evidence of affine zero-shot generalization or native speedup.",
                  "A full R1b–R1p research gate run is separately required.",
              ]}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    try:
        result = run(args.root, smoke=not args.no_smoke,
                     require_clean=not args.allow_dirty)
    except (NativePreflightError, ModuleNotFoundError, ImportError,
            AssertionError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
