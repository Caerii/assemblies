"""Compiler discovery must not silently prefer an unsupported newer VS."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import check_cuda_toolchain as toolchain


@pytest.mark.parametrize("override,expected", [(None, "[16.0,18.0)"), ("[17.0,18.0)", "[17.0,18.0)")])
def test_compiler_discovery_filters_version_before_selecting_latest(monkeypatch, override, expected):
    if override is None:
        monkeypatch.delenv("ASSEMBLIES_VS_VERSION", raising=False)
    else:
        monkeypatch.setenv("ASSEMBLIES_VS_VERSION", override)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    def discover(argv, **kwargs):
        assert argv[argv.index("-version") + 1] == expected
        assert "-latest" in argv
        return SimpleNamespace(returncode=0, stdout="C:/supported-vs")
    monkeypatch.setattr(toolchain.subprocess, "run", discover)
    assert toolchain.find_vcvars64() == Path("C:/supported-vs/VC/Auxiliary/Build/vcvars64.bat")


def test_no_matching_compiler_does_not_fall_back_to_latest(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    calls = []
    def discover(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(toolchain.subprocess, "run", discover)
    assert toolchain.find_vcvars64() is None
    assert len(calls) == 1
    assert "-version" in calls[0]
