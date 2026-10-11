"""The memory model every refraction-memory study runs, and how a run describes it.

    W_MAX          the weight ceiling: a synapse's weight is clipped at 20
    STRENGTH       the refraction's strength, as a fraction of the learning rate: a winner's
                   recurrent drive is charged STRENGTH x beta (the adopted 0.5 beta,
                   REFRACTION-ANTI-MERGING)
    profile(beta)  the organ semantics a run records for learning rate beta: the hashed
                   substrate with norm_init, no synaptic scaling, no convergence gate
    profile_name   the name a run records that profile under ("beta-0.05")

Owned here since 2026-10-10. Before, W_MAX lived in memory_pattern_efficiency (Amendment 9),
STRENGTH in three studies at once (memory_pattern_efficiency, memory_learning_rate,
memory_load_law, all 0.5), and profile in memory_learning_rate (Amendment 12); those modules
re-export these names, so every registered study computes what it ran.
"""
from __future__ import annotations

from neural_assemblies import describe_assembly_memory

W_MAX = 20.0
STRENGTH = 0.5


def profile_name(beta):
    return f"beta-{beta:g}"


def profile(beta):
    return describe_assembly_memory(w_max=W_MAX, beta=beta, strength=STRENGTH,
                                    gate=False, norm_init=True, synaptic_scaling=False)
