"""Fast paths for the memory studies -- each one BIT-IDENTICAL to the loop it replaces, and tested
so (neural_assemblies/tests/test_memory_fast.py).

Profiled 2026-10-10 at (10000, 75, 0.48), B = 20: every phase of a sleep or lifetime study was
LAUNCH-bound (a frozen projection moves 15 MB in 0.40 ms, 37 GB/s of the card's ~760), and the
replay check paid three further taxes: a host sync per step (the overflow read of a frozen
`project`), a read-out that scans all L stored tokens (L k work per brain per step), and a fresh
CUDA generator per cue (1.2 s of a 3.8 s check). What replaces them, and why each is exact:

  CUES        `memory_load_drift.cue` draws torch.rand(k) from a fresh generator seeded per
              (brain, sequence). For k < 256 PyTorch's uniform kernel gives element i the first
              word of Philox4x32-10 at key = seed, counter = (0, 0, i, 0), scaled x 2^-32 + 2^-33:
              a pure function, computed here for every cue at once. `philox_matches_torch` checks
              it against torch.rand before use; the slow path is the fallback. Rows with an exact
              tie are re-sorted by the original one-dimensional argsort, so the order is the same.
  READ-OUT    a token index (neuron -> the stored tokens it is in): the overlaps of a state with
              every token are the sum over its k neurons of their token lists, k * (L k / n) work
              instead of L k -- n/k ~ 133x less. Integer counts, then the same first-maximum
              argmax.
  REPLAY      every sequence of every brain in one pass, as VIRTUAL BRAINS (the organ fiber's
              brain map, as AssemblyMemory.recall_many): the same drive kernel per row, the same
              top-k, no bias (the masked read subtracts a zero bias, which changes no bit).
  VRAM        `vram_guard` refuses to start when the card lacks the memory a study needs: the
              2026-10-10 lifetime arms ran ~90x slower than the same code alone, with the card
              full.
"""
from __future__ import annotations

from ._guard import vram_guard
from ._cues import cue_index, philox_matches_torch, philox_uniform
from ._token_index import TokenIndex
from ._replay import MAX_VIRTUAL, frozen_step, reliability
from ._writer import SequenceWriter
from ._sleep import DreamGraph, _acc, _dream, _graph_for, calibrate, sleep

__all__ = ["vram_guard", "cue_index", "philox_matches_torch", "philox_uniform", "TokenIndex", "MAX_VIRTUAL",
           "frozen_step", "reliability", "SequenceWriter", "DreamGraph", "calibrate", "sleep",
           "_acc", "_dream", "_graph_for"]
