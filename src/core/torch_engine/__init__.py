"""TorchSparseEngine: PyTorch-native GPU engine for assembly calculus.

This package provides a GPU-accelerated engine using PyTorch CUDA tensors
for all state and computation.  Sub-modules:

- ``_hash``:   Deterministic hash-based connectivity utilities
- ``_csr``:    CSR-format sparse connectivity matrix
- ``_state``:  Per-area and per-stimulus state containers
- ``_engine``: TorchSparseEngine class (the public entry point)
"""

from ._engine import TorchSparseEngine

__all__ = ["TorchSparseEngine"]

# Register engine (only succeeds if torch+CUDA available)
import torch
if torch.cuda.is_available():
    from ..engine import register_engine
    register_engine("torch_sparse", TorchSparseEngine)
