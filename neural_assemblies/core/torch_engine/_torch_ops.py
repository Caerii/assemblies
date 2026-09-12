"""Typed runtime namespace for PyTorch generated operators.

PyTorch's tensor and generator classes are typed by its stubs, but many
factory/operator names are generated dynamically and are invisible to static
checkers.  Keeping those values behind this narrow protocol makes the runtime
surface explicit without replacing ``torch`` as the type namespace.
"""
from typing import Any, Callable, Protocol, cast
import torch as _torch

class _SparseNamespace(Protocol):
    coo: Any
    csr: Any
    mm: Callable[..., Any]

class TorchOps(Protocol):
    sparse: _SparseNamespace
    sparse_coo: Any
    int64: Any
    int32: Any
    bfloat16: Any
    float32: Any
    float64: Any
    bool: Any
    cat: Callable[..., Any]
    stack: Callable[..., Any]
    zeros: Callable[..., Any]
    zeros_like: Callable[..., Any]
    arange: Callable[..., Any]
    topk: Callable[..., Any]
    sparse_csr_tensor: Callable[..., Any]
    sparse_coo_tensor: Callable[..., Any]
    tensor: Callable[..., Any]
    meshgrid: Callable[..., Any]
    empty: Callable[..., Any]
    ones: Callable[..., Any]
    zeros_like: Callable[..., Any]
    repeat_interleave: Callable[..., Any]
    isin: Callable[..., Any]

torch_ops = cast(TorchOps, _torch)
__all__ = ["TorchOps", "torch_ops"]
