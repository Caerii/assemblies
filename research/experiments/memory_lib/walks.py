"""The sequences the reuse studies write: walks on a per-brain word grammar.

    LENGTH   elements per sequence (16)
    RHO      the load each study writes at, rho = L / unit(n, k, p) (0.05)

Owned here since 2026-10-10. Before, both lived in memory_reuse_grammar (Amendment 44), which
re-exports them.
"""
from __future__ import annotations

LENGTH, RHO = 16, 0.05
