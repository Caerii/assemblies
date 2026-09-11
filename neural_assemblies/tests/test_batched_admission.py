"""Batched inference rejects engines before backend-specific setup."""

import pytest

from neural_assemblies import Brain


def test_batched_lm_rejects_cpu_engine_by_capability():
    from neural_assemblies.assembly_calculus.batched_next_token import BatchedLM

    brain = Brain(engine="numpy_exact", norm_init=False)
    brain.add_area("LEX", 20, 2, 0.1)
    with pytest.raises(TypeError, match="batched next-token support"):
        BatchedLM(brain, "LEX", [], {}, {})
