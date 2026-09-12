"""Compatibility import for the shared lazy Torch operator boundary.

The canonical boundary lives in :mod:`neural_assemblies.core._torch_ops` so
CPU-only callers do not execute the CUDA engine package initializer.
"""
from .._torch_ops import TensorCall, TorchOps, torch_ops

__all__ = ["TensorCall", "TorchOps", "torch_ops"]
