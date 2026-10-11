"""The replay check: every sequence of every brain at once, as virtual brains.

Part of research.experiments.memory_fast (see its __init__ for why each path is exact)."""
from __future__ import annotations

from research.experiments.memory_lib.walks import LENGTH
from ._token_index import TokenIndex
from ._cues import cue_index


#: a pass reads at most this many virtual brains (bounds the [V, n] drive and [V, L] overlaps)
MAX_VIRTUAL = 2048


def frozen_step(mem, winners, brains):
    """one masked frozen read of V virtual brains: winners [V, k] -> next winners [V, k] and the
    k-WTA overflow flag (on the device)"""
    import torch
    raw = torch.zeros(winners.shape[0], mem.n, dtype=torch.float32, device=winners.device)
    mem.fiber.contribute(raw, winners, brains)
    sel, ovf = mem.area.mod.topk_select(raw, min(mem.k, mem.n))
    return sel.to(torch.int64), ovf


def reliability(store, device, index=None, qs=None):
    """memory_sleep.reliability, every sequence at once: per brain the fraction of sequences whose
    every step reads the right word. Leaves the area as memory_sleep.reliability does (bias zero,
    winners the last sequence's last read). ``index``: a TokenIndex over store["allst"] (built
    when omitted). ``qs``: only these sequences, in this order (each read exactly as there: the
    cue is drawn from the WHOLE store's length L and the sequence's own index)."""
    import torch
    mem, seqs, allst, wordof, M, L, seeds = (store[x] for x in ("mem", "seqs", "allst", "wordof", "M", "L", "seeds"))
    B, n, k = mem.B, mem.n, mem.k
    LEN = LENGTH
    mem.area.check_overflow()
    index = index or TokenIndex.build(allst, n, device)
    qs = list(range(M)) if qs is None else [int(q) for q in qs]
    Q = len(qs)
    # cues: sequence-major virtual brains v = i * B + b for the i-th sequence read
    gseeds = [int(sd) * 1_000_003 + int(L * 100_000 + q) for q in qs for sd in seeds]
    keep = cue_index(gseeds, k, device)                                  # [Q B, k/2]
    first = torch.stack([seqs[q][0] for q in qs]).long().view(Q * B, k)
    cues = torch.gather(first, 1, keep)
    brains = torch.arange(B, dtype=torch.int32, device=device).repeat(Q)
    qi = torch.as_tensor(qs, device=device)
    target = wordof.view(-1, LEN, B)[qi].permute(0, 2, 1).reshape(Q * B, LEN)  # true word per step
    M = Q
    alive = torch.ones(M * B, dtype=torch.bool, device=device)
    last = None
    ovf_acc = torch.zeros((), dtype=torch.int32, device=device)
    per = max(1, MAX_VIRTUAL)
    for v0 in range(0, M * B, per):
        sl_ = slice(v0, min(M * B, v0 + per))
        w, br = cues[sl_], brains[sl_]
        a = alive[sl_]
        for j in range(1, LEN):
            w, ovf = frozen_step(mem, w, br)
            ovf_acc = torch.maximum(ovf_acc, ovf.max())
            best = index.overlaps(w, br).argmax(dim=1)                     # [V]
            a &= wordof[best, br.long()] == target[sl_, j]
        alive[sl_] = a
        last = w
    bad = int(ovf_acc)
    if bad:
        raise RuntimeError(f"k-WTA candidate set overflowed ({bad} candidates)")
    mem.area.bias = torch.zeros(B, n, device=device)
    mem.area.winners = last[-B:].clone()
    rel = alive.view(M, B).float().sum(dim=0)
    return (rel / M).tolist()
