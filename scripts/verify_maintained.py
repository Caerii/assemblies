"""Run the repository's maintained-library verification gate.

The scope is intentionally explicit: shipped runtime packages are checked by
Pyright, while the package test suite is run without slow GPU studies. Legacy
research tests remain a separate, visible workload.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_SCOPES = (
    "neural_assemblies/assembly_calculus",
    "neural_assemblies/core",
    "neural_assemblies/compute",
    "neural_assemblies/nemo",
    "neural_assemblies/lexicon",
    "neural_assemblies/text_generation",
    "neural_assemblies/visualization",
    "neural_assemblies/language",
    "neural_assemblies/programs",
    "neural_assemblies/benchmarks",
    "neural_assemblies/reference/nemo_numpy",
)
# Research infrastructure is part of the supported developer surface.  Keep
# this list explicit while the historical experiment tree is being migrated;
# adding the whole tree would turn unresolved legacy studies into a noisy,
# non-actionable gate.
MAINTAINED_FILES = (
    "research/runner.py",
    "research/evidence.py",
    "research/harness.py",
    "research/json_documents.py",
    "research/source_archive.py",
    "research/compare_migration.py",
    "research/experiments/_historical.py",
    # Registered sequence entry point: kept in the gate after its dynamic
    # torch boundary was migrated to the typed torch_ops namespace.
    "research/experiments/seq_a1_horizon_hashed.py",
)


def default_test_workers() -> str:
    """Return a bounded default that avoids oversubscribing numerical tests.

    ``pytest -n auto`` starts one worker per logical CPU.  That is often slower
    for this NumPy/Torch-heavy suite and can leave worker teardown contending
    for native thread pools. Twelve workers are the measured fastest default on
    the 16-logical-core development machine;
    callers can override it with ``ASSEMBLIES_TEST_WORKERS`` or ``--workers``.
    """
    configured = os.environ.get("ASSEMBLIES_TEST_WORKERS")
    if configured:
        return configured
    return str(min(12, os.cpu_count() or 1))


def run(
    command: list[str],
    *,
    root: Path = ROOT,
    capture_output: bool = False,
    display_command: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", display_command or " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, check=False, text=True,
                          capture_output=capture_output)


def maintained_sources(root: Path = ROOT) -> list[str]:
    """Return the exact maintained Python sources, failing closed on drift."""
    files: list[str] = []
    for scope in MAINTAINED_SCOPES:
        scope_path = root / scope
        if not scope_path.is_dir():
            raise FileNotFoundError(
                f"maintained verification scope does not exist: {scope}"
            )
        files.extend(
            str(path.relative_to(root))
            for path in scope_path.rglob("*.py")
            if "\\tests\\" not in str(path).lower()
            and "\\archive\\" not in str(path).lower()
        )
    for relative in MAINTAINED_FILES:
        if not (root / relative).is_file():
            raise FileNotFoundError(
                f"maintained verification file does not exist: {relative}"
            )
    files.extend(MAINTAINED_FILES)
    return sorted(set(files))


def check_pyright(root: Path = ROOT) -> bool:
    files = maintained_sources(root)
    result = run(
        ["uv", "run", "pyright", *files, "--outputjson"],
        root=root,
        capture_output=True,
        display_command=(
            f"uv run pyright <{len(files)} maintained sources> --outputjson"
        ),
    )
    if result.stdout:
        report = json.loads(result.stdout)
        print(json.dumps(report["summary"], sort_keys=True))
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode == 0


def check_evidence_graph(root: Path = ROOT) -> bool:
    """Validate registered evidence and source-linked specifications.

    This is deliberately a separate phase from Pyright: the evidence checker
    validates repository relationships and serialized run records, while
    Pyright validates executable source. Keeping both in the maintained gate
    makes a green gate meaningful for scientific as well as code integrity.
    """
    result = run(
        ["uv", "run", "python", "-m", "research.evidence", "check"],
        root=root,
        capture_output=True,
        display_command="uv run python -m research.evidence check",
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true",
                        help="only run the maintained-source static gate")
    parser.add_argument(
        "--workers", default=default_test_workers(),
        help="pytest-xdist worker count (default: ASSEMBLIES_TEST_WORKERS or min(12, CPUs))",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="disable xdist and run the test suite in one process",
    )
    args = parser.parse_args()

    ok = check_pyright(ROOT)
    ok = check_evidence_graph(ROOT) and ok
    if not args.skip_tests:
        test_command = ["uv", "run", "pytest", "neural_assemblies/tests", "-q", "-m", "not slow"]
        if not args.serial:
            test_command[3:3] = ["-n", args.workers, "--dist", "load"]
        tests = run(test_command)
        ok = ok and tests.returncode == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
