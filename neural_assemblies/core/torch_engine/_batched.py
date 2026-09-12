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
selection is one ``torch_ops.topk(dim=1)`` over the batch.

Measured (RTX 3080, n=50k, k=224): identical results to sequential per item, and
7x (B=8) / 21x (B=32) faster; efficiency peaks near B=32 before the batched SpMM
saturates memory bandwidth.

Batching across INDEPENDENT connectomes (data-parallel training of different
brains) is Phase 3 -- it needs a block-diagonal CSR and is scoped separately.
"""

from ._torch_ops import torch_ops


def csrconn_to_torch_csr(csr, n):
    """Convert an engine ``CSRConn`` (crow/col/val) to a torch sparse CSR [n, n].

    Rows/cols are the compact materialized indices; the result is padded to the
    full ``n x n`` shape so batched drive vectors are length ``n``.
    """
    nrows = int(csr._nrows)
    crow = csr._crow[: nrows + 1].to(torch_ops.int64)
    # pad crow out to n+1 rows (extra rows are empty: repeat the last offset)
    if nrows < n:
        pad = crow[-1].repeat(n - nrows)
        crow = torch_ops.cat([crow, pad])
    col = csr._col.to(torch_ops.int64)
    val = csr._val.to(torch_ops.float32)
    return torch_ops.sparse_csr_tensor(crow, col, val, size=(n, n))


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
    act = torch_ops.zeros(B, n, dtype=torch_ops.float32, device=device)
    act.scatter_(1, winners.to(torch_ops.int64), 1.0)
    Wt = W.t()
    idx = winners.to(torch_ops.int64)
    for _ in range(rounds):
        # drive[b, j] = sum_i act[b, i] * W[i, j]  ==  act @ W
        drive = torch_ops.sparse.mm(Wt, act.t()).t()          # [B, n]
        if stim_drive is not None:
            drive = drive + stim_drive
        idx = torch_ops.topk(drive, min(k, n), dim=1).indices  # [B, k]
        act = torch_ops.zeros_like(act)
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
    rows, cols, vals = [], [], []
    for b, W in enumerate(mats):
        Wc = W.coalesce() if W.layout == torch_ops.sparse_coo else W.to_sparse_coo()
        ij = Wc.indices()
        rows.append(ij[0] + b * n)
        cols.append(ij[1] + b * n)
        vals.append(Wc.values())
    B = len(mats)
    idx = torch_ops.stack([torch_ops.cat(rows), torch_ops.cat(cols)])
    return torch_ops.sparse_coo_tensor(
        idx, torch_ops.cat(vals), (B * n, B * n)).coalesce()


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
    device = W_block.device
    W = W_block.coalesce()
    r, c = W.indices()[0], W.indices()[1]
    vals = W.values().clone()
    offs = torch_ops.arange(B, device=device).view(B, 1) * n
    idx_local = winners.to(torch_ops.int64)

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
    _t0 = torch_ops.sparse_coo_tensor(
        torch_ops.stack([c, r]),
        torch_ops.arange(vals.numel(), dtype=torch_ops.float64, device=device),
        (B * n, B * n)).coalesce()
    perm = _t0.values().to(torch_ops.int64)
    _tcsr = _t0.to_sparse_csr()
    crow, tcol = _tcsr.crow_indices(), _tcsr.col_indices()

    for _ in range(rounds):
        # transpose (swap r,c) so drive = act @ W; values only, pattern reused
        Wt = torch_ops.sparse_csr_tensor(crow, tcol, vals[perm],
                                     size=(B * n, B * n))
        act = torch_ops.zeros(B * n, 1, device=device)
        act[(idx_local + offs).reshape(-1)] = 1.0
        drive = torch_ops.sparse.mm(Wt, act).view(B, n)
        idx_local = torch_ops.topk(drive, min(k, n), dim=1).indices
        if beta:
            mask = torch_ops.zeros(B * n, dtype=torch_ops.bool, device=device)
            mask[(idx_local + offs).reshape(-1)] = True
            pot = mask[r] & mask[c]
            vals = vals.clone()
            vals[pot] = vals[pot] * (1.0 + beta)
            if w_max is not None:
                vals.clamp_(max=w_max)
    if return_weights:
        return idx_local, vals
    return idx_local


def _gain_table(beta, rounds):
    """``(1 + beta)**c`` for c = 0..rounds, by repeated float32 multiply.

    Separate from `_chain_table` because a stimulus weight does NOT start at
    1.0 -- it starts at the pre-summed input count -- so the clip applies to
    `base * gain` and cannot be folded into the table.
    """
    import numpy as np
    g = np.float32(1.0 + beta)
    out = np.ones(rounds + 1, dtype=np.float32)
    v = np.float32(1.0)
    for c in range(1, rounds + 1):
        v = np.float32(v * g)
        out[c] = v
    return out


def batched_project_hashed(
    n, k, p, seeds, winners, rounds, *, beta=0.0, w_max=None,
    norm_init=False, synaptic_scaling=False, stim_drive=None,
    stim_seeds=None, stim_size=None, return_drive=False, state=None,
    max_rounds=None, return_state=False, freeze=False,
    refracted_strength=0.0, mask_bias=False, stop_when_stable=False,
):
    """B INDEPENDENT connectomes, GENERATED rather than stored.

    A thin convenience wrapper over :mod:`._hashed`, which holds the pieces:
    an :class:`~._hashed.AreaFiber` per afferent area, a
    :class:`~._hashed.StimulusFiber` per stimulus, and a
    :class:`~._hashed.HashedArea` that sums them and runs the k-WTA. Reach for
    those directly when composing more than one input; this function exists for
    the single-recurrent-fiber case and to keep existing callers working.

    This is the case `batched_project` calls Phase 3 and defers -- "it needs a
    block-diagonal CSR and is scoped separately". It needs no CSR at all.
    :func:`batched_project_independent` must materialise a ``[B*n, B*n]``
    matrix, about 1.3 BILLION edges at B=64, n=20000, p=0.05, so the
    independent-connectome case could not run at organ scale. Here the
    connectome is regenerated from its hash inside the drive kernel.

    ** `batched_project_independent` USES A DIFFERENT RULE. ** It potentiates
    ``new x new`` where the engine (and this path) potentiate ``prev x new``;
    measured, it reproduces a new x new reference exactly and differs from
    prev x new in 41 of 64x64 cells ([[HEBB-OUTER-PRODUCT]]). The two are NOT
    interchangeable.

    NOTE ON TIES -- the selector breaks ties to the smallest index by
    construction, while ``torch_ops.topk`` leaves tie order unspecified and the
    drive is an integer Bernoulli sum, so ties at the bar are the common case
    ([[KWTA-TIE-FRAGILE]]). That is why this is a separate entry point.

    Args:
        n, k, p: area size, winners per round, connection probability.
        seeds: ``[B]`` int32 pair seeds (see ``_hash.fnv1a_pair_seed``).
        winners: ``[B, k]`` initial active set; ``[B, 0]`` for an inhibited
            area, whose first round is then driven by afferents alone.
        rounds: rounds in THIS episode; at most 64, the round-mask width.
        beta, w_max: Hebbian gain and weight clip.
        norm_init: substrate B. synaptic_scaling: substrate C.
        stim_seeds, stim_size: a hash-generated stimulus firing every round.
        stim_drive: a RAW additive drive, taken as already post-norm.
        state: carry from a previous call to CONTINUE training, which is what
            makes assemblies compete for one connectome.
        freeze: read the learned state without writing -- `brain.probe()`.
        refracted_strength: the engine's `refracted` mode on the area (a
            per-neuron bias, see `HashedArea`); only read when `state` is
            created, since the bias is state.
        mask_bias: with `freeze`, read the synaptic memory with the bias
            neither subtracted nor charged (PREREG_refraction_capacity P1).
        stop_when_stable: gate the rounds per brain on convergence, `rounds`
            becoming the ceiling; see `HashedArea.project`. The rounds each
            brain spent are `state["area"].rounds_used`. Organ fiber only.
    """
    from ._hashed import DenseOrganFiber, HashedArea, StimulusFiber
    from ._memory import recurrent_fiber

    device = str(winners.device)
    total = int(max_rounds or rounds)
    if freeze and state is None:
        raise ValueError("freeze=True reads a learned state; none was given")
    if state is None:
        # the fiber rule lives with AssemblyMemory, the unit this wrapper
        # predates; both must pick the same fiber for the same regime
        fiber = recurrent_fiber(seeds, n, p, beta=beta, w_max=w_max,
                                norm_init=norm_init,
                                synaptic_scaling=synaptic_scaling,
                                max_rounds=total, device=device)
        state = {
            "area": HashedArea(n, k, seeds, device=device,
                               refracted_strength=refracted_strength),
            "fiber": fiber,
        }
    area, fiber = state["area"], state["fiber"]
    area.winners = winners.to(torch_ops.int64)

    fibers = [fiber]
    if stim_seeds is not None:
        if stim_size is None:
            raise ValueError("stim_seeds needs stim_size")
        fibers.append(StimulusFiber(
            stim_seeds, stim_size, n, p, beta=beta, w_max=w_max,
            norm_init=norm_init, max_rounds=rounds, device=device))

    if stop_when_stable and not isinstance(fiber, DenseOrganFiber):
        raise ValueError("stop_when_stable needs the organ fiber (-1 rows); "
                         "this cell built the store fiber")
    res = area.project(rounds, fibers, freeze=freeze, stim_drive=stim_drive,
                       return_drive=return_drive, mask_bias=mask_bias,
                       stop_when_stable=stop_when_stable)
    out, drive = res if return_drive else (res, None)
    if return_drive and return_state:
        return out, drive, state
    if return_state:
        return out, state
    return (out, drive) if return_drive else out
