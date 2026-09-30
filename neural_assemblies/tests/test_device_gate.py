"""The device gate (tests/_devices.py) skips on an ordinary machine and FAILS
on one that is supposed to have the capability.

A gate that can only skip cannot catch the failure it exists for: a fused
kernel that stops building turns every parity test into a skip, and a GPU run
reads green. These tests construct that case with a fake missing level.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from _pytest.outcomes import Failed, Skipped

from neural_assemblies.tests import _devices

TESTS = Path(__file__).resolve().parent


@pytest.fixture
def fused_missing(monkeypatch):
    """Pretend the kernels did not build, whatever this machine can do."""
    real = _devices.why_unavailable

    def fake(level):
        return "fake build failure" if level == "fused" else real(level)
    monkeypatch.setattr(_devices, "why_unavailable", fake)


def test_a_missing_level_skips_with_its_reason(fused_missing, monkeypatch):
    monkeypatch.delenv(_devices.STRICT_ENV, raising=False)
    with pytest.raises(Skipped, match="needs fused: fake build failure"):
        _devices.require("fused")


@pytest.mark.parametrize("value", ["fused", "all", "1", "torch,fused"])
def test_the_strict_gate_turns_the_skip_into_a_failure(fused_missing, monkeypatch, value):
    monkeypatch.setenv(_devices.STRICT_ENV, value)
    with pytest.raises(Failed, match="needs fused: fake build failure"):
        _devices.require("fused")


def test_a_level_implies_the_levels_below_it(monkeypatch):
    monkeypatch.setenv(_devices.STRICT_ENV, "fused")
    assert _devices.required_levels() == {"torch", "cuda", "fused"}
    monkeypatch.setenv(_devices.STRICT_ENV, "cuda")
    assert _devices.required_levels() == {"torch", "cuda"}
    monkeypatch.setenv(_devices.STRICT_ENV, "")
    assert _devices.required_levels() == frozenset()


def test_an_unknown_level_is_refused(monkeypatch):
    monkeypatch.setenv(_devices.STRICT_ENV, "gpu")
    with pytest.raises(ValueError, match="unknown levels"):
        _devices.required_levels()
    with pytest.raises(ValueError, match="unknown device level"):
        _devices.why_unavailable("gpu")


def test_strict_mode_only_fails_the_levels_it_names(fused_missing, monkeypatch):
    """Requiring torch on a machine without fused kernels must still let a
    fused test skip: the CPU gate asks for torch, not for a GPU."""
    monkeypatch.setenv(_devices.STRICT_ENV, "torch")
    with pytest.raises(Skipped):
        _devices.require("fused")


# The old spellings of "is there a GPU?" -- each one either skipped silently
# under the gate or dropped a test arm without a trace. Guard, not advice.
_BYPASSES = {
    "importorskip of torch": re.compile(r"importorskip\((['\"])torch\1"),
    "a direct torch.cuda.is_available() check": re.compile(r"torch\.cuda\.is_available\(\)"),
    "a direct fused-kernel availability check": re.compile(
        r"_fused_cuda\.(available\(\)|load\(\) is None)"),
    "a local _has_torch_cuda helper": re.compile(r"def _has_torch_cuda"),
}


def test_no_test_module_asks_the_device_question_itself():
    offenders = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name in ("_devices.py", "test_device_gate.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for what, pattern in _BYPASSES.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.name}:{line}: {what}")
    assert not offenders, (
        "device requirements go through tests/_devices.py (markers requires_torch, "
        "requires_cuda, requires_fused, requires_cupy, or the fused_kernels fixture):\n  "
        + "\n  ".join(offenders))


def test_ci_installs_the_locked_torch_version():
    """CI installs the CPU wheel by hand; the version must be the locked one,
    or the CPU job tests a torch no developer runs."""
    root = TESTS.parents[1]
    workflow = (root / ".github" / "workflows" / "research-contracts.yml").read_text(encoding="utf-8")
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    pinned = re.findall(r'torch==([0-9][0-9.]*)"', workflow)
    locked = set(re.findall(r'name = "torch", version = "([0-9][0-9.]*)\+cu', lock))
    assert pinned, "the CPU job no longer installs torch; update this test with the job"
    assert len(locked) == 1, f"expected one locked CUDA torch version, found {sorted(locked)}"
    assert set(pinned) == locked, f"CI installs torch {pinned}, uv.lock pins {sorted(locked)}"
