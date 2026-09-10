"""Run and verify literature parity protocols."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .executors import EXECUTORS, verify_against_golden
from .protocol import Protocol, ProtocolResult
from .registry import get_protocol, get_protocol_by_claim, resolve_golden_path
from .paths import repo_root


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_golden(proto: Protocol) -> dict:
    path = Path(resolve_golden_path(proto))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_protocol(protocol_id: str) -> ProtocolResult:
    """Run in-process executor and compare to golden JSON."""
    proto = get_protocol(protocol_id)
    t0 = time.perf_counter()
    result = ProtocolResult(
        protocol_id=protocol_id,
        claim_id=proto.primary_claim_id,
        passed=False,
        backend=proto.backend.value,
        git_sha=_git_sha(),
    )

    if protocol_id not in EXECUTORS:
        result.message = f"no in-process executor for {protocol_id}"
        result.duration_s = time.perf_counter() - t0
        return result

    golden = load_golden(proto)
    if golden.get('metrics', {}).get('data_source') == 'mnist_csv':
        from neural_assemblies.programs.colt_mnist_data import require_mnist_dir
        require_mnist_dir()  # refuse before invoking a potentially expensive executor
    metrics = EXECUTORS[protocol_id]()
    expected_source = golden.get('metrics', {}).get('data_source')
    if expected_source is not None and metrics.get('data_source') != expected_source:
        raise ValueError(f'dataset mismatch: expected {expected_source!r}, got {metrics.get("data_source")!r}')
    result.metrics = metrics
    result.golden_metrics = golden.get("metrics", golden.get("regimes", {}))

    if protocol_id in ("pnas2020_scaling", "cross_lang.pnas_scaling"):
        passed, diffs = _verify_pnas_scaling(metrics, golden)
    elif protocol_id == "hoff2026_size_dist":
        passed, diffs = _verify_hoff_smoke(metrics, golden)
    elif protocol_id == "direct2026_pearl":
        passed, diffs = _verify_direct_pearl(metrics, golden)
    elif protocol_id == "colt2022_mnist":
        passed, diffs = _verify_colt_mnist(metrics, golden)
    elif protocol_id in ("coin2024_demo", "coin2024_compete", "coin2024_softmax"):
        passed, diffs = _verify_coin(metrics, golden)
    elif protocol_id == "colt2022_mnist_brain":
        passed, diffs = _verify_colt_mnist_brain(metrics, golden)
    else:
        passed, diffs = verify_against_golden(protocol_id, metrics, golden)

    result.passed = passed
    result.diffs = diffs
    result.duration_s = time.perf_counter() - t0
    result.message = "ok" if passed else f"metric mismatch: {diffs}"
    return result


def _verify_pnas_scaling(metrics: dict, golden: dict) -> tuple[bool, dict]:
    diffs = {}
    checked = 0
    for regime, expected in golden.get("regimes", {}).items():
        actual = metrics.get(regime, {})
        required = {key: expected[key] for key in
                    ("project_persistence", "separate_overlap", "chance_overlap")
                    if key in expected}
        _, mismatches = verify_against_golden(
            regime, actual, {"expected": required,
                             "tolerance": golden.get("thresholds", {}).get("metric_tolerance", 0.05)})
        checked += len(required)
        diffs.update({f"{regime}.{key}": value for key, value in mismatches.items()})
    if not checked:
        diffs["criteria"] = {"error": "no scaling acceptance criteria"}
    return len(diffs) == 0, diffs


def _verify_hoff_smoke(metrics: dict, golden: dict) -> tuple[bool, dict]:
    size = metrics.get("assembly_size", 0)
    paper_median = golden["table1_epwta_with_feedforward_inhibition"]["beta_0.1"]["size_median_iqr"][0]
    k = 80
    ok = 1 <= size <= max(paper_median * 3, k)
    diffs = {} if ok else {"assembly_size": {"actual": size, "max": max(paper_median * 3, k)}}
    return ok, diffs


def _verify_direct_pearl(metrics: dict, golden: dict) -> tuple[bool, dict]:
    diffs = {}
    th = golden["thresholds"]
    gm = golden["metrics"]
    fwd = metrics["forward_overlap"]
    if fwd < th["forward_overlap_min"]:
        diffs["forward_overlap_min"] = {"actual": fwd, "min": th["forward_overlap_min"]}
    if abs(fwd - gm["forward_overlap"]) > 0.05:
        diffs["forward_overlap"] = {"actual": fwd, "golden": gm["forward_overlap"]}
    if abs(metrics["reverse_overlap"] - gm["reverse_overlap"]) > 0.05:
        diffs["reverse_overlap"] = {"actual": metrics["reverse_overlap"], "golden": gm["reverse_overlap"]}
    do_fwd = metrics["do_effect_forward_overlap"]
    min_do = fwd * th["do_effect_forward_min_fraction_of_forward"]
    if do_fwd < min_do:
        diffs["do_effect"] = {"actual": do_fwd, "min": min_do}
    return len(diffs) == 0, diffs


def _verify_colt_mnist_brain(metrics: dict, golden: dict) -> tuple[bool, dict]:
    diffs = {}
    th = golden["thresholds"]
    if metrics["mean_accuracy"] < th["mean_accuracy_min"]:
        diffs["mean_accuracy"] = {
            "actual": metrics["mean_accuracy"],
            "min": th["mean_accuracy_min"],
        }
    gm = golden["metrics"]
    tol = th.get("metrics_match_tolerance", 0.02)
    if abs(metrics["mean_accuracy"] - gm["mean_accuracy"]) > tol:
        diffs["mean_accuracy_golden"] = {
            "actual": metrics["mean_accuracy"],
            "golden": gm["mean_accuracy"],
        }
    return len(diffs) == 0, diffs


def _verify_colt_mnist(metrics: dict, golden: dict) -> tuple[bool, dict]:
    diffs = {}
    th = golden["thresholds"]
    if metrics["train_classify_accuracy"] < th["train_classify_accuracy_min"]:
        diffs["train_classify_accuracy"] = {
            "actual": metrics["train_classify_accuracy"],
            "min": th["train_classify_accuracy_min"],
        }
    if metrics["min_pairwise_overlap"] > th["min_pairwise_overlap_max"]:
        diffs["min_pairwise_overlap"] = {
            "actual": metrics["min_pairwise_overlap"],
            "max": th["min_pairwise_overlap_max"],
        }
    return len(diffs) == 0, diffs


def _verify_coin(metrics: dict, golden: dict) -> tuple[bool, dict]:
    """Coin protocols: assert qualitative ``expected`` fields, not flip counts."""
    diffs = {}
    exp = golden.get("expected", {})
    tol = golden.get("tolerance", 0.02)
    default_tol = 0.02 if not isinstance(tol, dict) else 0.02

    for key, expected in exp.items():
        if key not in metrics:
            continue
        actual = metrics[key]
        if isinstance(expected, bool):
            if actual != expected:
                diffs[key] = {"actual": actual, "expected": expected}
        elif isinstance(expected, (int, float)):
            t = tol.get(key, default_tol) if isinstance(tol, dict) else default_tol
            if abs(actual - expected) > t:
                diffs[key] = {"actual": actual, "expected": expected}
    return len(diffs) == 0, diffs


def run_repro_command(proto: Protocol) -> ProtocolResult:
    """Execute ``repro_command`` via subprocess (typically pytest)."""
    t0 = time.perf_counter()
    result = ProtocolResult(
        protocol_id=proto.protocol_id,
        claim_id=proto.primary_claim_id,
        passed=False,
        backend=proto.backend.value,
        repro_command=proto.repro_command,
        git_sha=_git_sha(),
    )
    if not proto.repro_command:
        result.message = "no repro_command configured"
        result.duration_s = time.perf_counter() - t0
        return result

    cmd = proto.repro_command
    if cmd.startswith("uv run "):
        cmd = cmd[len("uv run ") :]

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=repo_root(),
    )
    result.exit_code = proc.returncode
    result.passed = proc.returncode == 0
    result.duration_s = time.perf_counter() - t0
    result.message = "pytest ok" if result.passed else f"exit {proc.returncode}"
    return result


def run_protocol(protocol_id: str, *, prefer: str = "repro") -> ProtocolResult:
    """Run a protocol: ``repro`` (subprocess) or ``verify`` (in-process golden check)."""
    proto = get_protocol(protocol_id)
    if prefer == "verify":
        return verify_protocol(protocol_id)
    if proto.repro_command:
        return run_repro_command(proto)
    return verify_protocol(protocol_id)


def run_claim(claim_id: str, **kwargs: Any) -> ProtocolResult:
    proto = get_protocol_by_claim(claim_id)
    if proto is None:
        raise KeyError(f"no protocol registered for claim {claim_id}")
    return run_protocol(proto.protocol_id, **kwargs)


def write_manifest(result: ProtocolResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_manifest(), indent=2) + "\n", encoding="utf-8")
