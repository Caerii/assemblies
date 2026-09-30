"""Which device capabilities a test needs, decided in one place.

Every test that needs more than NumPy says so with a marker, and this module
answers whether the machine has it:

    requires_torch   PyTorch imports (a CPU build is enough)
    requires_cuda    PyTorch sees a CUDA device
    requires_fused   the fused CUDA kernels build and load
    requires_cupy    CuPy imports

The levels nest: ``fused`` implies ``cuda``, which implies ``torch``.

On an ordinary machine a missing level SKIPS the test, with the reason in the
skip message (the compiler error, the missing import). That keeps the package
suite runnable on a laptop and in CPU-only CI.

On a machine that is supposed to have the capability, a skip is a lie: a
fused kernel that stops building would turn every parity gate into a skip and
the gate would still read green. Setting

    ASSEMBLIES_REQUIRE_DEVICE=fused        (or torch, cuda, cupy, all)

turns a missing level, and every level it implies, into a FAILURE. The GPU
gate (``scripts/gpu-gate.cmd``) sets it. Before this module the suite used
seven different spellings of the same question, and a ``gpu`` marker that
gated nothing.
"""
from __future__ import annotations

import functools
import importlib
import os

import pytest

LEVELS = ("torch", "cuda", "fused", "cupy")
STRICT_ENV = "ASSEMBLIES_REQUIRE_DEVICE"

#: what each level implies; requiring a level requires these too
_IMPLIES = {"torch": ("torch",), "cuda": ("torch", "cuda"),
            "fused": ("torch", "cuda", "fused"), "cupy": ("cupy",)}


@functools.lru_cache(maxsize=None)
def why_unavailable(level: str) -> str | None:
    """None when ``level`` is available here, else the reason it is not."""
    if level not in LEVELS:
        raise ValueError(f"unknown device level {level!r}; one of {LEVELS}")
    if level == "torch":
        try:
            importlib.import_module("torch")
        except Exception as exc:  # an import can fail for more than absence
            return f"PyTorch does not import ({type(exc).__name__}: {exc})"
        return None
    if level == "cuda":
        reason = why_unavailable("torch")
        if reason:
            return reason
        import torch
        if not torch.cuda.is_available():
            return "PyTorch sees no CUDA device"
        return None
    if level == "fused":
        reason = why_unavailable("cuda")
        if reason:
            return reason
        from neural_assemblies.core.torch_engine import _fused_cuda
        if _fused_cuda.load() is None:
            return ("the fused CUDA kernels did not build: "
                    f"{_fused_cuda.last_error()} (run scripts\\cuda-dev.cmd first)")
        return None
    try:
        importlib.import_module("cupy")
    except Exception as exc:
        return f"CuPy does not import ({type(exc).__name__}: {exc})"
    return None


def required_levels() -> frozenset:
    """The levels ``ASSEMBLIES_REQUIRE_DEVICE`` makes mandatory, with their implications."""
    raw = os.environ.get(STRICT_ENV, "").strip().lower()
    if not raw or raw in ("0", "none", "false"):
        return frozenset()
    named = LEVELS if raw in ("1", "all", "true") else tuple(
        part.strip() for part in raw.split(",") if part.strip())
    unknown = [n for n in named if n not in LEVELS]
    if unknown:
        raise ValueError(f"{STRICT_ENV}={raw!r} names unknown levels {unknown}; "
                         f"use a comma list of {LEVELS} or 'all'")
    return frozenset(lvl for n in named for lvl in _IMPLIES[n])


def require(level: str, *, allow_module_level: bool = False) -> None:
    """Skip, or under ``ASSEMBLIES_REQUIRE_DEVICE`` fail, unless ``level`` is available."""
    reason = why_unavailable(level)
    if reason is None:
        return
    message = f"needs {level}: {reason}"
    if level in required_levels():
        pytest.fail(f"{message} [{STRICT_ENV}={os.environ[STRICT_ENV]} requires it]",
                    pytrace=False)
    pytest.skip(message, allow_module_level=allow_module_level)


def import_torch():
    """``import torch`` for a test module whose top level needs it."""
    require("torch", allow_module_level=True)
    import torch
    return torch


def fused_kernels():
    """The loaded fused-kernel module, or a skip/fail with the build error."""
    require("fused")
    from neural_assemblies.core.torch_engine import _fused_cuda
    return _fused_cuda.load()
