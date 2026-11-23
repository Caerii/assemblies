import json
import sys
from pathlib import Path
import pytest
from research.multimodal.run_local_gates import run, specs


def test_has_native_smoke_cases_and_protocol_tests():
    cases = specs("python")
    assert {"protocol-tests", "r1b-native", "r1c-refraction", "r1d-bindings",
            "r1g-unsupervised", "r1h-supervised", "r1i-controls",
            "r1n-evidence", "r1o-evidence", "r1o-native-composition",
            "native-preflight"} == set(cases)
    assert all(cmd[0] == "python" for cmd in cases.values())
    assert cases["protocol-tests"] == ["python", "-m", "pytest", "-q", "research/multimodal"]
    assert cases["r1n-evidence"] == ["python", "-m", "research.multimodal.verify_unknown_drift", "--replay"]
    assert cases["r1o-evidence"] == ["python", "-m", "research.multimodal.verify_noncommutative", "--replay"]
    assert cases["r1o-native-composition"][2] == "research.multimodal.native_affine_chain"
    assert cases["native-preflight"][2] == "research.multimodal.native_execution_preflight"


def test_dry_run_distinct_and_not_passed(tmp_path):
    result = run(tmp_path, tmp_path / "out", dry_run=True,
                 plan={"good": [sys.executable, "-c", "print(1)"]})
    assert not result["all_passed"]
    assert result["gates"][0]["status"] == "not-executed"
    assert not (tmp_path / "out" / "good.stdout.txt").exists()


def test_logs_exit_code_and_hashes_success_and_failure(tmp_path):
    plan = {"pass": [sys.executable, "-c", "print('hello')"],
            "fail": [sys.executable, "-c", "import sys;sys.stderr.write('ouch');sys.exit(3)"]}
    out = tmp_path / "out"
    result = run(tmp_path, out, plan=plan, timeout=10)
    assert not result["all_passed"]
    assert [r["status"] for r in result["gates"]] == ["passed", "failed"]
    assert result["gates"][1]["exit_code"] == 3
    assert (out / "fail.stderr.txt").read_text() == "ouch"
    assert (out / "pass.stdout.txt").read_text() == "hello\n"
    assert json.loads((out / "run.json").read_text())["gates"] == result["gates"]


def test_fail_closed_on_invalid_cases(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "bad", names=["../escape"], plan={"ok": ["true"]})
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "bad2", names=["ok", "ok"],
            plan={"ok": ["true"]})


def test_success_and_timeout(tmp_path):
    good = run(tmp_path, tmp_path / "good", plan={"yes": [sys.executable, "-c", "pass"]})
    assert good["all_passed"]
    timeout = run(tmp_path, tmp_path / "timeout", timeout=.05,
                  plan={"slow": [sys.executable, "-c", "import time;time.sleep(10)"]})
    assert not timeout["all_passed"]
    assert timeout["gates"][0]["status"] == "timeout"


def test_native_gate_blocked_when_preflight_fails(tmp_path):
    plan = {
        "native-preflight": [sys.executable, "-c", "import sys;sys.exit(2)"],
        "r1b-native": [sys.executable, "-c", "print('MUST_NOT_RUN')"],
        "r1o-evidence": [sys.executable, "-c", "print('SYNTHETIC_OK')"],
    }
    result = run(tmp_path, tmp_path / "blocked", plan=plan)
    assert not result["all_passed"]
    statuses = {r["name"]: r["status"] for r in result["gates"]}
    assert statuses == {
        "native-preflight": "failed",
        "r1b-native": "blocked",
        "r1o-evidence": "passed",
    }
    assert result["gates"][1]["blocked_by"] == "native-preflight"
    assert not (tmp_path / "blocked/r1b-native.stdout.txt").exists()


def test_native_gate_blocked_without_preflight(tmp_path):
    plan = {"r1g-unsupervised": [sys.executable, "-c", "print('MUST_NOT_RUN')"]}
    result = run(tmp_path, tmp_path / "no_preflight", plan=plan)
    assert not result["all_passed"]
    assert result["gates"][0]["status"] == "blocked"


def test_native_gate_allowed_after_successful_preflight(tmp_path):
    plan = {
        "native-preflight": [sys.executable, "-c", "pass"],
        "r1o-native-composition": [sys.executable, "-c", "print('NATIVE_ALLOWED')"],
    }
    result = run(tmp_path, tmp_path / "allowed", plan=plan)
    assert result["all_passed"]
    assert [r["status"] for r in result["gates"]] == ["passed", "passed"]
