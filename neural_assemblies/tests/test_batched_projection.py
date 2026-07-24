"""Tests for batched projection through a shared connectome (Phase 2, Lever B).

See neural_assemblies/core/torch_engine/_batched.py and
docs/gpu_scale_design.md.
"""
import pytest


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_torch_cuda(), reason="batched projection requires PyTorch + CUDA")


def _rand_csr(n, p, seed):
    import torch
    torch.manual_seed(seed)
    nnz = max(int(n * n * p), n)
    rows = torch.randint(0, n, (nnz,), device="cuda")
    cols = torch.randint(0, n, (nnz,), device="cuda")
    vals = torch.ones(nnz, device="cuda")
    W = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n, n)).coalesce()
    return W.to_sparse_csr()


def _project_one(W, winners_row, k, rounds):
    """Reference: project a single [1, k] winner-set, no batching."""
    import torch
    n = W.shape[0]
    act = torch.zeros(1, n, device="cuda").scatter_(
        1, winners_row.view(1, -1), 1.0)
    Wt = W.t()
    idx = winners_row.view(1, -1)
    for _ in range(rounds):
        drive = torch.sparse.mm(Wt, act.t()).t()
        idx = torch.topk(drive, k, dim=1).indices
        act = torch.zeros_like(act).scatter_(1, idx, 1.0)
    return idx.view(-1)


class TestBatchedProjection:
    def test_batched_equals_sequential(self):
        import torch
        from neural_assemblies.core.torch_engine._batched import batched_project

        n, p, k, rounds, B = 4000, 0.01, 60, 4, 16
        W = _rand_csr(n, p, seed=1)
        torch.manual_seed(2)
        winners = torch.randint(0, n, (B, k), device="cuda")

        batched = batched_project(W, winners, k, rounds)  # [B, k]

        for b in range(B):
            ref = _project_one(W, winners[b], k, rounds)
            # topk indices are sorted by drive; compare as sets (ties aside)
            assert set(batched[b].tolist()) == set(ref.tolist()), (
                f"item {b}: batched != sequential")

    def test_batched_assembly_size_k_and_deterministic(self):
        import torch
        from neural_assemblies.core.torch_engine._batched import batched_project

        n, p, k, B = 4000, 0.01, 60, 8
        W = _rand_csr(n, p, seed=3)
        torch.manual_seed(4)
        winners = torch.randint(0, n, (B, k), device="cuda")
        out = batched_project(W, winners, k, rounds=10)
        assert out.shape == (B, k)
        # each item's assembly is exactly k distinct neurons
        for b in range(B):
            assert len(set(out[b].tolist())) == k
        # deterministic: same inputs -> same outputs (no hidden RNG in the path)
        out_again = batched_project(W, winners, k, rounds=10)
        assert set(out[0].tolist()) == set(out_again[0].tolist())
        # items are independent: shuffling the batch permutes results, not mixes
        perm = torch.tensor([B - 1 - i for i in range(B)], device="cuda")
        out_perm = batched_project(W, winners[perm], k, rounds=10)
        for b in range(B):
            assert set(out_perm[b].tolist()) == set(out[perm[b]].tolist())

    def test_csrconn_conversion_roundtrip(self):
        import torch
        from neural_assemblies.core.brain import Brain
        from neural_assemblies.assembly_calculus.ops import project
        from neural_assemblies.core.torch_engine._batched import (
            csrconn_to_torch_csr,
        )

        # train a tiny recurrent connectome on the torch engine, then batch-
        # project through its extracted CSR.
        b = Brain(p=0.05, seed=5, engine="torch_sparse", norm_init=False)
        b.add_stimulus("stim", 40)
        b.add_area("A", 2000, 40, 0.1)
        project(b, "stim", "A", rounds=10)
        for _ in range(5):
            b.project({}, {"A": ["A"]})

        eng = b._engine
        csr = eng._area_conns["A"]["A"]
        W = csrconn_to_torch_csr(csr, 2000)
        assert W.shape == (2000, 2000)

        from neural_assemblies.core.torch_engine._batched import batched_project
        winners = torch.randint(0, 2000, (4, 40), device="cuda")
        out = batched_project(W, winners, 40, rounds=3)
        assert out.shape == (4, 40)


class TestBlockDiagonalIndependent:
    """Batching INDEPENDENT connectomes (Phase 3): each item has its own weights,
    stacked block-diagonal so one SpMM projects all B. Must match looping B
    independent projections."""

    def test_block_diagonal_matches_looped(self):
        import torch
        from neural_assemblies.core.torch_engine._batched import (
            block_diagonal, batched_project_independent,
        )

        n, p, k, rounds, B = 3000, 0.004, 80, 4, 8
        mats = [_rand_csr(n, p, seed=200 + b) for b in range(B)]
        W_block = block_diagonal(mats, n)
        assert W_block.shape == (B * n, B * n)

        torch.manual_seed(9)
        winners = torch.randint(0, n, (B, k), device="cuda")
        batched = batched_project_independent(W_block, winners, B, n, k, rounds)

        for b in range(B):
            ref = _project_one(mats[b], winners[b], k, rounds)
            assert set(batched[b].tolist()) == set(ref.tolist()), (
                f"item {b}: block-diagonal != independent looped")
