"""Typed runtime namespace for PyTorch's generated operator surface.

PyTorch wheels expose many factory/operator names dynamically, while tensor
and generator classes remain useful type anchors.  This protocol makes the
runtime boundary explicit without replacing ``torch`` as the type namespace.
The cast is erased at runtime: ``torch_ops`` is the imported torch module.
"""
from typing import Any, Callable, Protocol, cast

import torch as _torch

TensorCall = Callable[..., _torch.Tensor]


class _SparseNamespace(Protocol):
    coo: Any
    csr: Any
    mm: TensorCall


class TorchOps(Protocol):
    sparse: _SparseNamespace
    sparse_coo: Any
    cuda: Any
    Generator: Any
    device: Any

    # Dtypes and layout sentinels are dynamically exposed values.
    bool: Any
    bfloat16: Any
    float32: Any
    float64: Any
    int8: Any
    int16: Any
    int32: Any
    int64: Any
    long: Any

    # Tensor-returning factories/operators.
    arange: TensorCall
    as_tensor: TensorCall
    bmm: TensorCall
    cat: TensorCall
    cumsum: TensorCall
    empty: TensorCall
    empty_like: TensorCall
    einsum: TensorCall
    erfinv: Callable[..., Any]
    full: TensorCall
    full_like: TensorCall
    from_numpy: TensorCall
    gather: TensorCall
    isin: TensorCall
    maximum: TensorCall
    normal: TensorCall
    ones: TensorCall
    ones_like: TensorCall
    rand: Callable[..., Any]
    randperm: TensorCall
    unique: TensorCall
    clamp: TensorCall
    repeat_interleave: TensorCall
    sparse_coo_tensor: TensorCall
    sparse_csr_tensor: TensorCall
    stack: TensorCall
    tensor: TensorCall
    where: TensorCall
    zeros: TensorCall
    zeros_like: TensorCall

    # Structured-return operations are intentionally left open until their
    # per-call result contracts are modeled (indices/values/inverse maps).
    argsort: Callable[..., Any]
    meshgrid: Callable[..., Any]
    sort: Callable[..., Any]
    topk: Callable[..., Any]
    unique_consecutive: Callable[..., Any]


torch_ops = cast(TorchOps, _torch)
__all__ = ["TensorCall", "TorchOps", "torch_ops"]
