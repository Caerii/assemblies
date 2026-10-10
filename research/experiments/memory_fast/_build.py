"""Building a memory: memory_write_separation.build, with the count width a choice.

Part of research.experiments.memory_fast. The registered build stays as recorded; this is the same
construction with ``count_dtype`` passed through (None: the fiber's default, int8 here; "int4":
packed counts -- see DenseOrganFiber for what that changes under unlearning)."""
from __future__ import annotations

import math


def build_memory(n, k, p, tau, seeds, device, count_dtype=None):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments import memory_load_law as ml
    from research.experiments import memory_pattern_efficiency as pe
    from research.experiments import memory_threshold_law as tl
    from research.experiments.seq_capacity_scaling import seeds_for
    return AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX,
                          norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=device,
                          bias_decay=math.exp(-1.0 / tau), count_dtype=count_dtype)
