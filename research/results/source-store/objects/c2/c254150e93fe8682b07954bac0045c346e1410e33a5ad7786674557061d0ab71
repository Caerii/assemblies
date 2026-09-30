#!/usr/bin/env python3
"""Optional environment probe for the Windows CUDA build toolchain."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest


VS_SCRIPTS = [
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
]
CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"


def _find_vs_script() -> str | None:
    for script in VS_SCRIPTS:
        if os.path.exists(script):
            return script
    return None


def _build_batch(vs_script: str) -> str:
    cuda_bin = os.path.join(CUDA_HOME, "bin")
    cuda_lib = os.path.join(CUDA_HOME, "lib", "x64")
    return "\r\n".join(
        [
            "@echo off",
            f'call "{vs_script}"',
            f"set PATH={cuda_bin};{cuda_lib};%PATH%",
            f"set CUDA_HOME={CUDA_HOME}",
            "",
            "echo Testing cl.exe...",
            'cl 2>&1 | findstr "Microsoft"',
            "if errorlevel 1 exit /b 1",
            "",
            "echo Testing nvcc...",
            "nvcc --version",
            "if errorlevel 1 exit /b 1",
            "",
            "echo Toolchain ready.",
            "",
        ]
    )


def test_vs_cuda_environment() -> None:
    vs_script = _find_vs_script()
    if vs_script is None:
        pytest.skip("Visual Studio build tools environment script not found")
    if not os.path.isdir(CUDA_HOME):
        pytest.skip("CUDA toolkit v13.0 not found")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".bat",
        delete=False,
        encoding="ascii",
        newline="",
    ) as handle:
        handle.write(_build_batch(vs_script))
        batch_file = handle.name

    try:
        result = subprocess.run(
            [batch_file],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            "Visual Studio/CUDA environment probe failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    finally:
        try:
            os.unlink(batch_file)
        except OSError:
            pass
