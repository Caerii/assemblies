"""Run the repository's maintained-library verification gate.

The scope is intentionally explicit: shipped runtime packages are checked by
Pyright, while the package test suite is run without slow GPU studies. Legacy
research tests remain a separate, visible workload.
"""

from __future__ import annotations

import argparse
import json
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
)


def run(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False, text=True,
                          capture_output=capture_output)


def check_pyright() -> bool:
    files = [
        str(path.relative_to(ROOT))
        for scope in MAINTAINED_SCOPES
        for path in (ROOT / scope).rglob("*.py")
        if "\\tests\\" not in str(path).lower()
        and "\\archive\\" not in str(path).lower()
    ]
    result = run(["uv", "run", "pyright", *files, "--outputjson"], capture_output=True)
    if result.stdout:
        report = json.loads(result.stdout)
        print(json.dumps(report["summary"], sort_keys=True))
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true",
                        help="only run the maintained-source static gate")
    args = parser.parse_args()

    ok = check_pyright()
    if not args.skip_tests:
        tests = run(["uv", "run", "pytest", "neural_assemblies/tests", "-q", "-m", "not slow"])
        ok = ok and tests.returncode == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
