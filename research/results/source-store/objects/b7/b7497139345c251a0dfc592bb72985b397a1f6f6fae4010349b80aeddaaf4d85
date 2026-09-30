"""PyTorch CUDA paths for assembly calculus.

Two things live here.

The ``Brain`` engine, ``torch_sparse``:

- ``_engine``: ``TorchSparseEngine``, the public entry point
- ``_state``:  per-area and per-stimulus state containers
- ``_csr``:    CSR sparse connectivity
- ``_hash``:   deterministic hash-based connectivity utilities

The hashed substrate, for research at width (many independent brains per
launch, connectomes regenerated from a hash inside the kernel; gated
against the numpy engine on the drive):

- ``_fused_cuda``:        the kernels (presence, drive, selection, write-back)
- ``_hashed``:            ``HashedArea`` and the fibers, one per density regime
- ``_memory``:            ``AssemblyMemory``, the refracted associative memory
- ``_arc_core``:          ``HashedArcCore``, the refracted arc-and-state core
- ``_hashed_fsm``:        ``HashedArcFSM``, the assigned-state machine
- ``_hashed_transducer``: ``HashedTransducer``, the induced-state transducer
- ``_scheduled_aligner``, ``_hashed_aligner``: the word learner
- ``_batched``:           batched projection helpers and the older wrapper

See ``docs/architecture.md`` and ``research/notes/README.md``.
"""

from ._engine import TorchSparseEngine

__all__ = ["TorchSparseEngine"]

# Register engine (only succeeds if torch+CUDA available)
from .._torch_ops import torch_ops
if torch_ops.cuda.is_available():
    from ..engine import register_engine
    register_engine("torch_sparse", TorchSparseEngine)
