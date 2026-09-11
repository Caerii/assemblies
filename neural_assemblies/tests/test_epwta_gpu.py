"""GPU / torch E%-WTA policy tests."""

import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.compute import EPercentPolicy


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TestEPercentNumpy:
    def test_variable_winner_count_epwta(self):
        b = Brain(p=0.05, seed=42, engine="numpy_sparse")
        policy = EPercentPolicy(fraction_of_max=0.35, min_winners=5, e_fraction=0.02)
        b.add_area("A", 5000, 80, 0.1, winner_policy=policy)
        b.add_stimulus("s", 80)
        from neural_assemblies.assembly_calculus import project
        asm = project(b, "s", "A", rounds=8)
        assert 5 <= len(asm) <= 80


@pytest.mark.skipif(not _has_torch_cuda(), reason="torch_sparse requires CUDA")
class TestEPercentTorchGPU:
    def test_torch_epwta_policy_wiring(self):
        policy = EPercentPolicy(fraction_of_max=0.4, min_winners=8)
        b = Brain(p=0.05, seed=42, engine="torch_sparse")
        b.add_area("A", 10000, 100, 0.1, winner_policy=policy)
        b.add_stimulus("s", 100)
        from neural_assemblies.assembly_calculus import project
        asm = project(b, "s", "A", rounds=8)
        assert len(asm) >= 8

    def test_torch_topk_default_when_no_policy(self):
        b = Brain(p=0.05, seed=42, engine="torch_sparse")
        b.add_area("A", 10000, 100, 0.1)
        b.add_stimulus("s", 100)
        from neural_assemblies.assembly_calculus import project
        asm = project(b, "s", "A", rounds=8)
        assert len(asm) == 100
