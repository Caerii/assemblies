from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_paths(*paths: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_local_agent_state_is_not_tracked():
    assert _tracked_paths(".claude") == []


def test_stale_requirements_file_is_not_tracked():
    assert _tracked_paths("requirements.txt") == []


def test_uv_lock_is_tracked_for_reproducible_dev_environment():
    assert _tracked_paths("uv.lock") == ["uv.lock"]


def test_root_holds_only_metadata():
    """Code lives in neural_assemblies/, research/, legacy/, cpp/, crates/,
    examples/, scripts/ and tests/. The root has project metadata only: no
    Python module, no schema directory, no benchmark directory."""
    tracked_root_files = sorted(
        p for p in _tracked_paths(".") if "/" not in p
    )
    assert tracked_root_files == [
        ".gitignore", ".python-version", "CITATION.cff", "LICENSE",
        "MANIFEST.in", "README.md", "pyproject.toml", "uv.lock",
    ], tracked_root_files
    assert _tracked_paths("assembly_ir") == []
    assert _tracked_paths("benchmarks") == []
