"""Cross-language parity runner (Python baseline + IR export)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_assemblies.ir.protocol import export_protocol_document, load_protocol_document


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _golden_path(name: str) -> Path:
    return _repo_root() / "research" / "literature" / "parity" / "golden" / name


def run_python_pnas_scaling() -> dict[str, Any]:
    from neural_assemblies.parity.executors import execute_pnas2020_scaling

    return execute_pnas2020_scaling()


def run_julia_pnas_scaling() -> dict[str, Any] | None:
    """Run Julia reference script if ``julia`` is on PATH."""
    import shutil
    import subprocess

    if shutil.which("julia") is None:
        return None
    script = _repo_root() / "research" / "literature" / "cross_lang" / "julia" / "pnas_scaling.jl"
    if not script.is_file():
        return None
    proc = subprocess.run(
        ["julia", str(script)],
        capture_output=True,
        text=True,
        cwd=script.parent,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"julia pnas_scaling failed: {proc.stderr}")
    return json.loads(proc.stdout)


def verify_julia_against_golden(
    golden: dict[str, Any],
    *,
    tol: float = 0.05,
) -> tuple[bool, dict[str, Any]]:
    julia_doc = run_julia_pnas_scaling()
    if julia_doc is None:
        return False, {"julia": "julia executable or script not found"}
    regimes = julia_doc.get("regimes", julia_doc.get("metrics", {}))
    return verify_pnas_scaling(regimes, golden, tol=tol)


def export_pnas_scaling_ir(path: Path | None = None) -> dict[str, Any]:
    metrics = run_python_pnas_scaling()
    doc = export_protocol_document(
        protocol_id="cross_lang.pnas_scaling",
        backend="python",
        metrics=metrics,
        regimes=metrics,
        source="neural_assemblies parity executor",
        notes="Cross-language golden for PNAS 2020 scaling (ci_parity regime).",
    )
    if path is not None:
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _metric_close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def verify_pnas_scaling(
    actual: dict[str, Any],
    golden: dict[str, Any],
    *,
    tol: float = 0.05,
) -> tuple[bool, dict[str, Any]]:
    diffs: dict[str, Any] = {}
    regimes = golden.get("regimes", golden.get("metrics", {}))
    for regime, expected in regimes.items():
        got = actual.get(regime, {})
        for key in ("project_persistence", "separate_overlap", "chance_overlap"):
            if key not in expected:
                continue
            a = got.get(key)
            e = expected[key]
            if a is None or not _metric_close(float(a), float(e), tol):
                diffs[f"{regime}.{key}"] = {"actual": a, "expected": e}
    return len(diffs) == 0, diffs


def run_protocol(protocol_id: str) -> tuple[bool, dict[str, Any]]:
    if protocol_id == "cross_lang.pnas_scaling":
        golden = load_protocol_document(_golden_path("cross_lang_pnas_scaling.json"))
        actual = run_python_pnas_scaling()
        return verify_pnas_scaling(actual, golden)
    raise KeyError(f"unknown cross-lang protocol: {protocol_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-language parity runner")
    parser.add_argument(
        "--protocol",
        default="cross_lang.pnas_scaling",
        help="Protocol id (default: cross_lang.pnas_scaling)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Write Python IR metrics JSON to this path",
    )
    parser.add_argument(
        "--julia",
        action="store_true",
        help="Also verify Julia reference script against golden",
    )
    args = parser.parse_args()

    if args.export is not None:
        export_pnas_scaling_ir(args.export)
        print(f"wrote {args.export}")
        return

    ok, diffs = run_protocol(args.protocol)
    if args.julia:
        golden = load_protocol_document(_golden_path("cross_lang_pnas_scaling.json"))
        j_ok, j_diffs = verify_julia_against_golden(golden)
        ok = ok and j_ok
        if not j_ok:
            diffs = {**diffs, **j_diffs}

    if ok:
        print(f"PASS {args.protocol}")
    else:
        print(f"FAIL {args.protocol}: {json.dumps(diffs, indent=2)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
