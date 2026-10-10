"""Sleep: memory_sleep.dream's episode without host syncs, recorded as a CUDA graph.

Part of research.experiments.memory_fast (see its __init__ for why each path is exact)."""
from __future__ import annotations


def _dream(mem, gen, threshold, acc, contrasts):
    """memory_sleep.dream's episode without a host sync: the same operations in the same order
    (the frozen projection written out: drive = raw - bias, top-k; the decrement applied
    unconditionally, which writes back unchanged counts where the gate is shut). Totals and the
    overflow flag accumulate in ``acc`` on the device."""
    import torch
    from research.experiments import memory_sleep as sl
    area, fib = mem.area, mem.fiber
    C = fib.C
    B, n, k = mem.B, mem.n, mem.k
    ar = torch.arange(B, device=C.device)
    area.bias = torch.zeros(B, n, device=C.device)
    area.winners = torch.argsort(torch.rand(B, n, device=C.device, generator=gen), dim=1)[:, :k]
    bi = ar.view(B, 1, 1)
    for s in range(sl.STEPS):
        prev = area.winners.clone()
        raw = torch.zeros(B, n, dtype=torch.float32, device=C.device)
        fib.contribute(raw, area.winners)
        drive = area.apply_bias(raw)
        sel, ovf = area.mod.topk_select(drive, min(k, n))
        torch.maximum(acc["overflow"], ovf.max(), out=acc["overflow"])
        x = sel.to(torch.int64)
        area.winners = x
        if s < sl.SETTLE:
            continue
        con = torch.gather(drive, 1, x).mean(1) / drive.mean(1).clamp_min(1e-9)
        if contrasts is not None:
            contrasts.append(con)
        acc["judged"] += B
        if threshold is None:
            continue
        g = con >= threshold
        ri, ci = prev.view(B, k, 1), x.view(B, 1, k)
        old = C[bi, ri, ci]
        dec = ((old > 0) & g.view(B, 1, 1)).to(old.dtype)
        C[bi, ri, ci] = old - dec
        acc["removed"] += dec.sum(dtype=torch.int64)
        acc["gated"] += g.sum(dtype=torch.int64)


def _acc(device):
    import torch
    return {"removed": torch.zeros((), dtype=torch.int64, device=device),
            "gated": torch.zeros((), dtype=torch.int64, device=device),
            "judged": 0, "overflow": torch.zeros((), dtype=torch.int32, device=device)}


class DreamGraph:
    """One sleep episode recorded as a CUDA GRAPH and replayed: ~100 small launches an episode were
    the whole cost (each ~80 us on this Windows driver against ~1 ms of GPU work). Capture runs
    nothing -- the counts and the generator are untouched until a replay -- and the generator is
    registered with the graph, so replay i draws exactly what eager call i would. Everything the
    episode writes is in place (the counts) or a graph-owned buffer the area then points at."""

    def __init__(self, mem, gen, threshold, device, record=False):
        import torch
        # the generator and a tensor gate are held: the graph reads them through their pointers
        self.mem, self.gen, self.threshold = mem, gen, threshold
        self.acc = _acc(device)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(gen)
        cons = [] if record else None
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            _dream(mem, gen, threshold, self.acc, cons)
            self.cons = torch.stack(cons) if record else None
        self.winners, self.bias = mem.area.winners, mem.area.bias
        self.judged = self.acc["judged"]                     # per episode (a host count)

    def replay(self):
        area = self.mem.area
        self.graph.replay()
        area.winners, area.bias = self.winners, self.bias


def _graph_for(mem, gen, threshold, device, record=False):
    key = (id(gen), threshold if not hasattr(threshold, "data_ptr") else ("t", threshold.data_ptr()),
           record, id(mem.fiber.C))
    cache = mem.__dict__.setdefault("_dream_graphs", {})
    if key not in cache:
        while len(cache) >= 4:                               # a graph holds its pool: keep a few
            cache.pop(next(iter(cache)))
        cache[key] = DreamGraph(mem, gen, threshold, device, record)
    return cache[key]


def sleep(mem, gen, threshold, episodes, device, graph=True):
    """``episodes`` calls of memory_sleep.dream: (counts removed, steps gated, steps judged), the
    sums of what those calls return, with one host read at the end. ``graph``: replay a recorded
    episode (identical, and ~10x fewer launches)."""
    if not graph:
        acc = _acc(device)
        for _ in range(episodes):
            _dream(mem, gen, threshold, acc, None)
        judged = acc["judged"]
    else:
        g = _graph_for(mem, gen, threshold, device)
        g.acc["removed"].zero_()
        g.acc["gated"].zero_()
        g.acc["overflow"].zero_()
        for _ in range(episodes):
            g.replay()
        acc, judged = g.acc, g.judged * episodes
    if int(acc["overflow"]):
        raise RuntimeError(f"k-WTA candidate set overflowed ({int(acc['overflow'])} candidates)")
    return int(acc["removed"]), int(acc["gated"]), judged


def calibrate(mem, gen, episodes, device, graph=True):
    """the contrasts memory_sleep.dream(mem, gen, None, ...) appends over ``episodes`` dreams,
    concatenated: [episodes * judged steps * B] float32 on the device (the order of
    torch.cat(contrasts) there)"""
    import torch
    if not graph:
        acc, cons = _acc(device), []
        for _ in range(episodes):
            _dream(mem, gen, None, acc, cons)
        out = torch.cat(cons)
    else:
        g = _graph_for(mem, gen, None, device, record=True)
        g.acc["overflow"].zero_()
        parts = []
        for _ in range(episodes):
            g.replay()
            parts.append(g.cons.reshape(-1).clone())
        acc, out = g.acc, torch.cat(parts)
    if int(acc["overflow"]):
        raise RuntimeError(f"k-WTA candidate set overflowed ({int(acc['overflow'])} candidates)")
    return out
