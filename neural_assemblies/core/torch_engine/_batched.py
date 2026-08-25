"""Batched projection through a shared connectome (Phase 2, Lever B).

See docs/gpu_scale_design.md. The single-area sparse loop is CPU-favorable
because each projection is sub-millisecond of work; the GPU win comes from
running MANY independent projections at once so the fixed per-round overhead
(kernel launches, selection) is amortized across the batch.

This module implements the cleanest batching case: ``B`` independent activity
states projected through ONE shared connectome (batch inference / parse -- e.g.
classifying or parsing B inputs in parallel through a trained brain). Because
the connectome is shared there is no ragged-materialization problem: the drive
is a single ``[B, n]`` tensor, computed by one batched sparse-dense matmul, and
selection is one ``torch.topk(dim=1)`` over the batch.

Measured (RTX 3080, n=50k, k=224): identical results to sequential per item, and
7x (B=8) / 21x (B=32) faster; efficiency peaks near B=32 before the batched SpMM
saturates memory bandwidth.

Batching across INDEPENDENT connectomes (data-parallel training of different
brains) is Phase 3 -- it needs a block-diagonal CSR and is scoped separately.
"""

import torch


def csrconn_to_torch_csr(csr, n):
    """Convert an engine ``CSRConn`` (crow/col/val) to a torch sparse CSR [n, n].

    Rows/cols are the compact materialized indices; the result is padded to the
    full ``n x n`` shape so batched drive vectors are length ``n``.
    """
    nrows = int(csr._nrows)
    crow = csr._crow[: nrows + 1].to(torch.int64)
    # pad crow out to n+1 rows (extra rows are empty: repeat the last offset)
    if nrows < n:
        pad = crow[-1].repeat(n - nrows)
        crow = torch.cat([crow, pad])
    col = csr._col.to(torch.int64)
    val = csr._val.to(torch.float32)
    return torch.sparse_csr_tensor(crow, col, val, size=(n, n))


def batched_project(
    W, winners, k, rounds, *, stim_drive=None, return_activity=False,
):
    """Project ``B`` independent winner-sets through shared connectome ``W``.

    Args:
        W: shared connectome as a torch sparse CSR tensor ``[n, n]`` (row i ->
           col j weight is the i->j synapse). Build one with
           :func:`csrconn_to_torch_csr`.
        winners: ``[B, k]`` int64 indices of each item's initial active set.
        k: winners kept each round.
        rounds: recurrent projection rounds.
        stim_drive: optional ``[n]`` or ``[B, n]`` additive drive applied every
           round (a fixed stimulus shared or per-item).
        return_activity: if True also return the final ``[B, n]`` activity.

    Returns:
        ``[B, k]`` int64 winner indices (sorted by drive), and optionally the
        ``[B, n]`` activity. Bit-identical to projecting each item on its own.
    """
    n = W.shape[0]
    B = winners.shape[0]
    device = W.device
    act = torch.zeros(B, n, dtype=torch.float32, device=device)
    act.scatter_(1, winners.to(torch.int64), 1.0)
    Wt = W.t()
    idx = winners.to(torch.int64)
    for _ in range(rounds):
        # drive[b, j] = sum_i act[b, i] * W[i, j]  ==  act @ W
        drive = torch.sparse.mm(Wt, act.t()).t()          # [B, n]
        if stim_drive is not None:
            drive = drive + stim_drive
        idx = torch.topk(drive, min(k, n), dim=1).indices  # [B, k]
        act = torch.zeros_like(act)
        act.scatter_(1, idx, 1.0)
    if return_activity:
        return idx, act
    return idx


def block_diagonal(mats, n):
    """Stack B independent [n, n] sparse connectomes into one block-diagonal
    ``[B*n, B*n]`` sparse COO, so a single SpMM projects all B at once.

    ``mats`` is a list of B torch sparse tensors (COO or CSR), each [n, n].
    Item b occupies rows/cols ``[b*n, (b+1)*n)``.
    """
    import torch
    rows, cols, vals = [], [], []
    for b, W in enumerate(mats):
        Wc = W.coalesce() if W.layout == torch.sparse_coo else W.to_sparse_coo()
        ij = Wc.indices()
        rows.append(ij[0] + b * n)
        cols.append(ij[1] + b * n)
        vals.append(Wc.values())
    B = len(mats)
    idx = torch.stack([torch.cat(rows), torch.cat(cols)])
    return torch.sparse_coo_tensor(
        idx, torch.cat(vals), (B * n, B * n)).coalesce()


def batched_project_independent(
    W_block, winners, B, n, k, rounds, *, beta=0.0, w_max=None,
    return_weights=False,
):
    """Project (and optionally Hebbian-train) B items through their OWN
    connectomes via a block-diagonal matrix -- the data-parallel case.

    Build ``W_block`` with :func:`block_diagonal`. Each round: one SpMM +
    reshape to ``[B, n]`` + one batched topk drives all B; with ``beta > 0`` a
    single masked scatter potentiates every within-item winner-pair edge
    (``w *= 1+beta``, clamped at ``w_max``). Because block-diagonal edges never
    cross items, that mask is automatically per-item correct -- so batched
    training is **bit-identical to training the B brains independently**
    (validated), at ~3.5-5x throughput.

    Returns ``[B, k]`` int64 local winner indices; if ``return_weights`` also the
    updated ``[nnz]`` value tensor (aligned to ``W_block.coalesce().values()``).
    """
    import torch
    device = W_block.device
    W = W_block.coalesce()
    r, c = W.indices()[0], W.indices()[1]
    vals = W.values().clone()
    offs = torch.arange(B, device=device).view(B, 1) * n
    idx_local = winners.to(torch.int64)

    # THE TRANSPOSED PATTERN IS BUILT ONCE, NOT PER ROUND. Hebbian learning
    # changes weight VALUES; it never adds or removes an edge, so the sparsity
    # pattern of W^T is invariant across rounds. The previous form rebuilt it
    # every round -- `sparse_coo_tensor(...).coalesce().to_sparse_csr()` is
    # O(nnz log nnz) in a loop whose actual work is one O(nnz) SpMM -- and that
    # rebuild, not the arithmetic, was what the batched path spent its time on.
    #
    # `perm` carries the mapping: coalescing the transpose of a UNIQUE index
    # set cannot sum entries, so putting `arange` in the value slot recovers
    # exactly where each original edge landed.
    _t0 = torch.sparse_coo_tensor(
        torch.stack([c, r]),
        torch.arange(vals.numel(), dtype=torch.float64, device=device),
        (B * n, B * n)).coalesce()
    perm = _t0.values().to(torch.int64)
    _tcsr = _t0.to_sparse_csr()
    crow, tcol = _tcsr.crow_indices(), _tcsr.col_indices()

    for _ in range(rounds):
        # transpose (swap r,c) so drive = act @ W; values only, pattern reused
        Wt = torch.sparse_csr_tensor(crow, tcol, vals[perm],
                                     size=(B * n, B * n))
        act = torch.zeros(B * n, 1, device=device)
        act[(idx_local + offs).reshape(-1)] = 1.0
        drive = torch.sparse.mm(Wt, act).view(B, n)
        idx_local = torch.topk(drive, min(k, n), dim=1).indices
        if beta:
            mask = torch.zeros(B * n, dtype=torch.bool, device=device)
            mask[(idx_local + offs).reshape(-1)] = True
            pot = mask[r] & mask[c]
            vals = vals.clone()
            vals[pot] = vals[pot] * (1.0 + beta)
            if w_max is not None:
                vals.clamp_(max=w_max)
    if return_weights:
        return idx_local, vals
    return idx_local


def batched_project_hashed(
    n, k, p, seeds, winners, rounds, *, stim_drive=None,
    return_drive=False,
):
    """B INDEPENDENT connectomes, GENERATED rather than stored.

    This is the case the module docstring calls Phase 3 and defers -- "it needs
    a block-diagonal CSR and is scoped separately". It turns out not to need a
    CSR at all. :func:`batched_project_independent` must materialise a
    ``[B*n, B*n]`` sparse matrix, which at B=64, n=20000, p=0.05 is about 1.3
    BILLION edges; the independent-connectome case simply cannot be run at
    organ scale that way. Here the connectome is regenerated from its hash
    inside the drive kernel, so the only memory is the ``[B, n]`` drive.

    Each brain gets its own ``pair_seed``, which is what makes the connectomes
    independent -- the same role the block offset plays in the block-diagonal
    form.

    Args:
        n, k, p: area size, winners per round, connection probability.
        seeds: ``[B]`` int32 pair seeds, one per brain (see
            ``_hash.fnv1a_pair_seed``).
        winners: ``[B, k]`` initial active set.
        rounds: recurrent projection rounds.
        stim_drive: optional ``[n]`` or ``[B, n]`` additive drive each round.
        return_drive: also return the final ``[B, n]`` drive.

    Returns:
        ``[B, k]`` int64 winner indices.

    Raises:
        RuntimeError: if the fused kernels are unavailable, or if a brain's
            candidate set overflows the selector's buffer. Neither is allowed
            to degrade quietly into a wrong winner set.

    NOTE ON TIES -- this does NOT reproduce :func:`batched_project_independent`
    cell for cell. The selector here breaks ties to the smallest index (stable
    argsort, by construction); ``torch.topk`` leaves tie order unspecified, and
    the drive is an integer Bernoulli sum, so ties at the bar are the common
    case rather than an edge case. That is a science-affecting difference and
    is why this is a separate entry point rather than a faster path inside the
    existing one. See ``_fused_cuda`` and ``_kwta_prune``.
    """
    from . import _fused_cuda

    mod = _fused_cuda.load()
    if mod is None:
        raise RuntimeError(
            "batched_project_hashed needs the fused CUDA kernels: "
            f"{_fused_cuda.last_error()}")

    device = winners.device
    seeds_t = torch.as_tensor(seeds, dtype=torch.int32, device=device)
    idx = winners.to(torch.int32)
    threshold = _fused_cuda.threshold_for(p)
    drive = None
    for _ in range(rounds):
        drive = mod.hashed_drive(idx.contiguous(), seeds_t, n, threshold)
        if stim_drive is not None:
            drive = drive + stim_drive
        sel, ovf = mod.topk_select(drive, min(k, n))
        bad = int(ovf.max())
        if bad:
            raise RuntimeError(
                f"k-WTA candidate set overflowed ({bad} candidates) -- the "
                "drive is too flat for the histogram to narrow. Refusing to "
                "return a truncated winner set.")
        idx = sel
    out = idx.to(torch.int64)
    return (out, drive) if return_drive else out
