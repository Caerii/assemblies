"""Check extension-build prerequisites without importing torch or building code.

Run through the environment that will compile: uv run python scripts/check_cuda_toolchain.py.
This checks tools, not device availability, compiler compatibility or kernel parity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_vcvars64() -> Path | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    result = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode or not result.stdout.strip():
        return None
    candidate = Path(result.stdout.strip()) / "VC/Auxiliary/Build/vcvars64.bat"
    return candidate if candidate.is_file() else None


def check() -> dict:
    issues = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = Path(cuda_home) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc") if cuda_home else None
    if not cuda_home:
        issues.append("CUDA_HOME (or CUDA_PATH) is unset; set it to the CUDA toolkit root.")
    elif not nvcc.is_file():
        issues.append(f"CUDA_HOME has no CUDA compiler at {nvcc}.")
    ninja = shutil.which("ninja")
    if ninja is None:
        issues.append("ninja is missing from PATH; install it in the build environment.")
    compiler = shutil.which("cl" if os.name == "nt" else "c++")
    vcvars = find_vcvars64() if os.name == "nt" else None
    if os.name == "nt" and vcvars is None:
        issues.append("vcvars64.bat was not found through vswhere; install Visual Studio C++ build tools.")
    if compiler is None:
        issues.append(
            "cl.exe is missing from PATH; run scripts\\cuda-dev.cmd in cmd.exe first."
            if os.name == "nt" else "c++ is missing from PATH; install a supported C++ compiler."
        )
    if importlib.util.find_spec("torch") is None:
        issues.append("torch is missing from this Python environment; run uv sync --extra gpu.")
    return {
        "ready_to_attempt_build": not issues,
        "python": sys.executable,
        "cuda_home": cuda_home,
        "nvcc": str(nvcc) if nvcc else None,
        "ninja": ninja,
        "compiler": compiler,
        "vcvars64": str(vcvars) if vcvars else None,
        "issues": issues,
        "scope": "Tools only. No CUDA import, build, device probe, or parity run performed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key in ("python", "cuda_home", "nvcc", "ninja", "compiler", "vcvars64"):
            print(f"{key}: {result[key] or 'MISSING / not applicable'}")
        for issue in result["issues"]:
            print(f"MISSING: {issue}")
        print(result["scope"])
    return 0 if result["ready_to_attempt_build"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
