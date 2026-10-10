"""Fused CUDA kernels: generate the connectome, and select without sorting.

Two kernels, both optional -- everything here degrades to ``None`` if nvcc, a
host compiler or ninja is missing, and callers must check :func:`available`.

WHY THESE EXIST. See ``research/notes/substrate/DESIGN_gpu_hashed_drive.md``. A cell's
weight is a pure function of its position, and on a card with a ~100:1
arithmetic-to-bandwidth ratio, computing it is cheaper than reading it -- but
only if the intermediates never reach memory, which is why this is a fused
kernel and not a torch expression (the torch spelling measured 0.7-0.9x against
the stored SpMM it was meant to beat, because every elementwise op writes DRAM).

THE HASH IS THE ENGINE'S, NOT A RE-DERIVATION. `drive` must agree cell-for-cell
with ``_hash.hash_bernoulli_2d``, including murmur3's fmix32 finalizer and the
INTEGER threshold convention ``(h & 0xFFFFFF) < int(p * 2**24)``. A float
comparison ``(h & 0xFFFFFF) / 2**24 < p`` is NOT the same predicate -- it
differs on the boundary cell whenever ``p * 2**24`` is not an integer. A kernel
that agrees with a plausible transcription and disagrees with the engine is a
fast wrong answer, which is the only outcome worse than a slow right one, so
``tests/test_fused_cuda.py`` pins it against the engine's own function.

SELECTION IS A COUNTING PROBLEM. The drive is a Bernoulli sum -- an integer in
[0, k] -- so it is bounded, concentrated and massively tied. ``topk_select``
histograms the top 12 bits of the key, which narrows n=20000 to a few hundred
candidates in one pass, then sorts those in shared memory.

    ** THE TIE ORDER IS DIFFERENT FROM torch.topk, ON PURPOSE. **

The key is ``(float_bits << 16) | (65535 - j)``, so keys are unique and
"largest key" means "largest value, ties to the smallest index" -- stable
argsort, by construction. ``torch.topk``'s tie order is unspecified and the
engine's CPU selector (``heapq_select_top_k``) is argpartition+argsort, both
unstable. With 5-18 columns tied at the bar in practice, that changes WHICH
neurons fire, not merely their order. ``_kwta_prune`` states the rule for this
project: making the selector's tie-break canonical is a SCIENCE-AFFECTING
change that needs its own registration and must not be smuggled in as an
optimisation. So nothing here is used unless a caller asks for it explicitly.
"""
from __future__ import annotations

import hashlib
import os
import threading

_LOCK = threading.Lock()
_MODULE = None
_TRIED = False
_ERROR: str | None = None

# THE KERNEL SOURCE lives in kernels/ as one compilation unit cut by topic: the files are
# concatenated in name order (their numeric prefixes) into the string `load_inline` compiles,
# and the host-side declarations are kernels/bindings.cpp. Read in text mode, so a checkout's
# line endings never change the source, and `build_name` -- a hash of it -- with it.
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def _read_kernel(name: str) -> str:
    with open(os.path.join(_KERNELS, name), encoding="utf-8") as fh:
        return fh.read()


#: the .cu files, in concatenation order
KERNEL_FILES = sorted(n for n in os.listdir(_KERNELS) if n.endswith(".cu"))
_CUDA_SRC = "".join(_read_kernel(n) for n in KERNEL_FILES)
_CPP = _read_kernel("bindings.cpp")


# Visual Studio ships ninja, but only puts it on PATH inside a developer shell.
# torch's `load_inline` needs it importable OR on PATH; adding the standard
# locations is a best effort that costs nothing when they are absent.
_VS_NINJA = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE"
    r"\CommonExtensions\Microsoft\CMake\Ninja",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE"
    r"\CommonExtensions\Microsoft\CMake\Ninja",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE"
    r"\CommonExtensions\Microsoft\CMake\Ninja",
]


def _augment_path() -> None:
    for d in _VS_NINJA:
        if os.path.isdir(d) and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def load() -> object | None:
    """Build (once) and return the fused module, or ``None`` if unavailable.

    Never raises: a missing toolchain is a capability question, not an error.
    The first failure's message is kept in :func:`last_error` for diagnosis --
    silently returning ``None`` with no way to ask why is how a GPU path
    quietly stops being used.
    """
    global _MODULE, _TRIED, _ERROR
    with _LOCK:
        if _TRIED:
            return _MODULE
        _TRIED = True
        try:
            import torch
            if not torch.cuda.is_available():
                _ERROR = "no CUDA device"
                return None
            from torch.utils.cpp_extension import load_inline
            _augment_path()
            _MODULE = load_inline(
                name=build_name(), cpp_sources=[_CPP],
                cuda_sources=[_CUDA_SRC],
                functions=["hashed_drive", "hashed_indegree", "dev_correct",
                           "dev_correct_csr", "dev_correct_exact",
                           "column_mass_exact", "dev_correct_rel",
                           "column_mass_rel", "hashed_presence",
                           "organ_drive", "organ_write", "stim_add", "charge",
                           "present_degree", "present_fill", "present_drive",
                           "present_write", "present_train", "present_probe",
                           "column_mass", "topk_select"],
                verbose=False, extra_cuda_cflags=["-O3"])
        except Exception as exc:                       # noqa: BLE001
            _MODULE = None
            _ERROR = f"{type(exc).__name__}: {exc}"
        return _MODULE


def build_name() -> str:
    """The extension's build name, content-addressed.

    torch builds an inline extension into a cache directory named after it,
    shared by every checkout on the machine. One fixed name meant a pinned
    run worktree and an edited checkout rebuilt each other's kernels in
    turn, and on Windows a rebuild cannot replace a module a running study
    has loaded. A name per source keeps every version's build apart.
    """
    digest = hashlib.sha256((_CPP + _CUDA_SRC + "-O3").encode()).hexdigest()
    return f"na_fused_cuda_{digest[:12]}"


def available() -> bool:
    """True if the fused kernels built and can be called."""
    return load() is not None


def last_error() -> str | None:
    """Why :func:`load` returned ``None``, or ``None`` if it did not."""
    load()
    return _ERROR


def threshold_for(p: float) -> int:
    """The engine's integer Bernoulli threshold.

    ``_hash.hash_bernoulli_2d`` tests ``(h & 0xFFFFFF) < int(p * 2**24)``. This
    must be spelled the same way here; comparing ``(h & 0xFFFFFF) / 2**24 < p``
    instead flips the boundary cell whenever ``p * 2**24`` is not an integer.
    """
    return int(p * 16777216.0)
