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


def _chain_table(beta, w_max, rounds):
    """``chain(1.0, c)`` for c = 0..rounds, by the ENGINE's own arithmetic.

    The engine potentiates with ``w *= (1 + beta)`` and clamps at ``w_max``
    every round, so a cell potentiated c times is a per-step
    multiply-and-clip -- NOT ``min((1+beta)**c, w_max)``, which differs once
    the clip binds. Replaying it in float32 on the host makes the kernel's job
    a table lookup rather than a re-derivation.
    """
    import numpy as np
    g = np.float32(1.0 + beta)
    out = np.ones(rounds + 1, dtype=np.float32)
    v = np.float32(1.0)
    for c in range(1, rounds + 1):
        v = np.float32(v * g)
        if w_max is not None:
            v = min(v, np.float32(w_max))
        out[c] = v
    return out


MAX_LEARNING_ROUNDS = 64


def batched_project_hashed(
    n, k, p, seeds, winners, rounds, *, beta=0.0, w_max=None,
    norm_init=False, synaptic_scaling=False, stim_drive=None,
    stim_seeds=None, stim_size=None,
    return_drive=False, state=None, max_rounds=None, return_state=False,
    freeze=False,
):
    """B INDEPENDENT connectomes, GENERATED rather than stored.

    This is the case the module docstring calls Phase 3 and defers -- "it needs
    a block-diagonal CSR and is scoped separately". It turns out not to need a
    CSR at all. :func:`batched_project_independent` must materialise a
    ``[B*n, B*n]`` sparse matrix, which at B=64, n=20000, p=0.05 is about 1.3
    BILLION edges; the independent-connectome case simply cannot be run at
    organ scale that way. Here the connectome is regenerated from its hash
    inside the drive kernel, so the only memory is the ``[B, n]`` drive plus,
    when learning, a ``[B, n]`` int64 round-mask.

    Each brain gets its own ``pair_seed``, which is what makes the connectomes
    independent -- the role the block offset plays in the block-diagonal form.

    PLASTICITY, AND WHY NOTHING IS STORED. With ``beta > 0`` this potentiates
    ``w *= (1 + beta)`` on the pairs the ENGINE potentiates: source winners x
    target winners, which for a recurrent fiber is ``prev x new`` (see
    ``_engine._apply_plasticity``, where ``tgt.winners`` is assigned AFTER the
    update). The learned state is never materialised as a weight matrix,
    because

        count = SUM_t x_{t-1} x_t^T   =>   count[i,j] = popcount(rm[i] & cm[j])

    with bit t of ``rm[i]`` / ``cm[j]`` recording that i fired at t-1 / j fired
    at t. The entire learned connectome is TWO BIT-MASKS, and its size does not
    grow with the number of rounds.

    ** `batched_project_independent` USES A DIFFERENT RULE. ** It builds one
    mask from the NEW winners and applies it to both endpoints, i.e.
    ``new x new``. Measured, not inferred: it reproduces a ``new x new``
    reference exactly and differs from ``prev x new`` in 41 of 64x64 cells.
    This function follows the ENGINE, so the two are NOT interchangeable and
    results from them are not comparable. Which rule is right for that function
    is a separate question, deliberately not decided here.

    Args:
        n, k, p: area size, winners per round, connection probability.
        seeds: ``[B]`` int32 pair seeds (see ``_hash.fnv1a_pair_seed``).
        winners: ``[B, k]`` initial active set.
        rounds: recurrent projection rounds.
        beta: Hebbian gain; 0 disables learning entirely.
        w_max: weight clip, or None for unbounded.
        norm_init: substrate B -- divide each column's drive by its in-degree.
            Here that divisor is EXACT rather than estimated; see below.
        synaptic_scaling: substrate C -- renormalise each winner column's
            mass to `n * p` every round.
        stim_seeds: ``[B]`` pair seeds for a hash-generated STIMULUS fiber,
            fired every round -- the anchor the registered training
            protocol relies on. Priced exactly as the engine prices it.
        stim_size: the stimulus's neuron count. Required with stim_seeds.
        stim_drive: optional ``[n]`` or ``[B, n]`` RAW additive drive each
            round, taken as already on the post-norm scale. Prefer
            ``stim_seeds``, which is priced.
        return_drive: also return the final ``[B, n]`` drive.
        state: learned state from a previous call, to CONTINUE training. A
            capacity protocol stores many assemblies in ONE area, so the
            episodes must share a connectome; passing the returned state back
            is what makes them compete instead of starting fresh.
        max_rounds: total rounds this state will ever see. Sets the round-mask
            width (one 64-bit word per 64 rounds); defaults to ``rounds``.
        return_state: also return the state, for the next episode.
        freeze: READ the learned state without writing to it -- the
            equivalent of `brain.probe()`. A retrieval that quietly
            trained on its own cue would score itself
            ([[probe-isolation-required]]).

    Returns:
        ``[B, k]`` int64 winner indices.

    Raises:
        RuntimeError: fused kernels unavailable, or a brain's candidate set
            overflowed the selector. Neither degrades quietly into a wrong
            winner set.
        ValueError: more than 64 learning rounds -- the round-mask is 64 bits.

    NOTE ON TIES -- this does NOT reproduce
    :func:`batched_project_independent` cell for cell. The selector breaks ties
    to the smallest index (stable argsort, by construction) while
    ``torch.topk`` leaves tie order unspecified, and the drive is an integer
    Bernoulli sum, so ties at the bar are the common case. That is a
    science-affecting difference and is why this is a separate entry point.
    """
    from . import _fused_cuda

    mod = _fused_cuda.load()
    if mod is None:
        raise RuntimeError(
            "batched_project_hashed needs the fused CUDA kernels: "
            f"{_fused_cuda.last_error()}")

    learn = bool(beta)
    if learn and rounds > MAX_LEARNING_ROUNDS:
        raise ValueError(
            f"rounds={rounds} exceeds {MAX_LEARNING_ROUNDS}: the round-mask "
            "that stands in for the weight matrix is 64 bits wide")

    device = winners.device
    B = winners.shape[0]
    seeds_t = torch.as_tensor(seeds, dtype=torch.int32, device=device)
    idx = winners.to(torch.int32)
    threshold = _fused_cuda.threshold_for(p)

    track = learn or synaptic_scaling
    if freeze and state is None:
        raise ValueError("freeze=True reads a learned state; none was given")
    total = int(max_rounds or rounds)
    if state is None:
        state = {"t": 0, "hist": [], "W": max(1, (total + 63) // 64)}
        W = state["W"]
        if track:
            # Masks are [B, W, n]: W 64-bit words, so the history is not capped
            # at 64 rounds. The word dimension is OUTER so that threads walking
            # consecutive rows read coalesced memory for a fixed word.
            state["rowmask"] = torch.zeros(B, W, n, dtype=torch.int64,
                                           device=device)
            state["colmask"] = torch.zeros(B, W, n, dtype=torch.int64,
                                           device=device)
            state["tab"] = torch.from_numpy(
                _chain_table(beta, w_max, total)).to(device)
            state["colids"] = torch.arange(n, dtype=torch.int32,
                                           device=device).expand(B, n)
        if synaptic_scaling:
            # Substrate C. `_scale_columns_now` renormalises each WINNER column
            # to `setpoint = rows * p_fiber` -- the fiber's p, not the brain's,
            # which is the third member of the pricing-law defect class. Every
            # row exists here, so `rows` is n.
            state["scale"] = torch.ones(B, n, dtype=torch.float32,
                                        device=device)
    rowmask = state.get("rowmask")
    colmask_d = state.get("colmask")
    tab = state.get("tab")
    scale = state.get("scale")
    hist = state["hist"]
    setpoint = max(float(n) * float(p), 1e-12)
    if track and not freeze and state["t"] + rounds > state["W"] * 64:
        raise ValueError(
            f"round {state['t'] + rounds} exceeds the mask width "
            f"({state['W']} words = {state['W'] * 64} rounds). Pass "
            "max_rounds covering every episode this state will see.")

    # Substrate B. `_pricing.inverse_indegree` computes
    # `d_j = deg_j + p * (n_pre - rows_known)` because engines materialise
    # neurons lazily, so a column's full in-degree does not exist yet and the
    # second term estimates the missing rows. HERE THAT TERM IS ZERO: a
    # generated connectome has every row from the start, so `d_j` is the true
    # in-degree. Computed once -- the connectome does not change -- and the
    # divisor applies to the WHOLE drive, potentiation included, which is exact
    # because plasticity is multiplicative:
    #     (w_0 / d_j) * prod_t (1 + beta_t) == (w_0 * prod_t (1 + beta_t)) / d_j
    dj = None
    if norm_init:
        dj = mod.hashed_indegree(seeds_t, n, threshold, 1.0)

    # ---- the stimulus fiber -------------------------------------------
    # A stimulus connectome stores PRE-SUMMED input: one weight per TARGET
    # neuron, equal to the number of stimulus neurons wired to it. So its base
    # is `hashed_drive` over the stimulus's own rows, and potentiation
    # multiplies that scalar.
    #
    # TWO PRICES THAT ARE EASY TO GET WRONG, both taken from the engine:
    #
    #  * norm_init divides by `d_j = deg_j + p * (tgt.n - stim_size)` -- note
    #    `tgt.n`, NOT the stimulus size. That is deliberate: it puts the
    #    stimulus on the SAME ~n*p divisor as the area fiber so the two are
    #    commensurable. Dividing by the stimulus's own in-degree would make an
    #    untrained stimulus contribute exactly 1.0 per touched neuron against
    #    an area contribution of ~0.06, and the stimulus would decide every
    #    winner -- the documented failure mode of getting `_pricing` wrong.
    #  * w_max means "multiples of the INITIAL weight", and a stimulus weight
    #    starts near `stim_size * p`, not at 1. The cap is therefore
    #    `w_max * max(1, stim_size * p)`. Capping at a raw `w_max` clipped
    #    every winner on the first update and pinned it there -- measured as a
    #    flat 0.30 recovery for beta = 0.001, 0.01 and 0.1 alike.
    stim_base = stim_pot = stim_dj = gpow = None
    stim_hi = float("inf")
    if stim_seeds is not None:
        if stim_size is None:
            raise ValueError("stim_seeds needs stim_size")
        st = torch.as_tensor(stim_seeds, dtype=torch.int32, device=device)
        srows = torch.arange(stim_size, dtype=torch.int32,
                             device=device).expand(B, stim_size).contiguous()
        stim_base = mod.hashed_drive(srows, st, n, threshold)
        stim_pot = torch.zeros(B, n, dtype=torch.int64, device=device)
        if norm_init:
            stim_dj = (stim_base + float(p) * (n - stim_size)).clamp_min(1.0)
        if w_max is not None:
            stim_hi = float(w_max) * max(1.0, float(stim_size) * float(p))
        gpow = torch.from_numpy(_gain_table(beta, rounds)).to(device)

    drive = None
    for t in range(rounds):
        drive = mod.hashed_drive(idx.contiguous(), seeds_t, n, threshold)
        if tab is not None and (state["t"] + t) > 0:
            # The dense colmask IS the column index; `colids` is the identity.
            mod.dev_correct(idx.contiguous(), rowmask, state["colids"],
                            colmask_d, tab, seeds_t, threshold, drive)
        if scale is not None:
            drive = drive * scale
        if dj is not None:
            drive = drive / dj
        if stim_base is not None:
            sd = stim_base * gpow[stim_pot.clamp_max(gpow.numel() - 1)]
            if stim_hi != float("inf"):
                sd = sd.clamp_max(stim_hi)
            if stim_dj is not None:
                sd = sd / stim_dj
            drive = drive + sd
        if stim_drive is not None:
            drive = drive + stim_drive
        sel, ovf = mod.topk_select(drive, min(k, n))
        bad = int(ovf.max())
        if bad:
            raise RuntimeError(
                f"k-WTA candidate set overflowed ({bad} candidates) -- the "
                "drive is too flat for the histogram to narrow. Refusing to "
                "return a truncated winner set.")
        if track and not freeze:
            gt = state["t"] + t
            wrd, bit = gt // 64, 1 << (gt % 64)
            pidx, sidx = idx.long(), sel.long()
            rw, cw = rowmask[:, wrd], colmask_d[:, wrd]
            rw.scatter_(1, pidx, rw.gather(1, pidx) | bit)
            cw.scatter_(1, sidx, cw.gather(1, sidx) | bit)
            if learn:
                hist.append(sel)
        if stim_pot is not None and not freeze and beta:
            stim_pot.scatter_add_(1, sel.long(),
                                  torch.ones_like(sel, dtype=torch.int64))
        if scale is not None and not freeze:
            # Masks are updated FIRST: the engine scales after applying this
            # round's potentiation, so the mass must include it.
            mass = mod.column_mass(sel.contiguous(), rowmask, colmask_d,
                                   tab, seeds_t, threshold)
            scale.scatter_(1, sel.long(), setpoint / mass.clamp_min(1e-12))
            if w_max is not None:
                # The factorisation assumes min() never fires. Conservative
                # bound on any cell: deepest potentiation times the largest
                # accumulated scale. Refuse rather than diverge quietly.
                # Bound on the DEEPEST POSSIBLE count so far, not on the
                # worst the table can hold: no cell can have been potentiated
                # more times than rounds have elapsed. Using tab[-1] made the
                # guard fire whenever the table itself saturated, which is
                # always once max_rounds is large.
                elapsed = min(state["t"] + t + 1, int(tab.numel()) - 1)
                bound = float(tab[elapsed].item()) * float(scale.max().item())
                if bound >= float(w_max):
                    raise RuntimeError(
                        f"synaptic_scaling would cross w_max={w_max} (bound "
                        f"{bound:.4g} at round {t}): column scaling and the "
                        "clip do not commute, so the factored form is no "
                        "longer exact. Re-run with w_max=None or fewer rounds.")
        idx = sel

    if track and not freeze:
        state["t"] += rounds
    out = idx.to(torch.int64)
    if return_drive and return_state:
        return out, drive, state
    if return_state:
        return out, state
    return (out, drive) if return_drive else out
