"""The memory guard: refuse a study on a card another process fills.

Part of research.experiments.memory_fast (see its __init__ for why each path is exact)."""
from __future__ import annotations


def vram_guard(need_gb, device="cuda", margin_gb=0.5):
    """Raise unless `need_gb` (+ margin) of device memory is free right now -- whoever holds the
    rest. Returns the free GB."""
    import torch
    free, total = torch.cuda.mem_get_info(torch.device(device))
    free_gb, total_gb = free / 2**30, total / 2**30
    if free_gb < need_gb + margin_gb:
        raise RuntimeError(f"only {free_gb:.1f} of {total_gb:.1f} GB free on {device}; this needs "
                           f"{need_gb:.1f} + {margin_gb:.1f}. Another process holds the card: a run "
                           "in shared memory is ~90x slower.")
    return free_gb
