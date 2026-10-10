"""Fast paths for the memory studies -- each one BIT-IDENTICAL to the loop it replaces, and tested
so (neural_assemblies/tests/test_memory_fast.py).

Profiled 2026-10-10 at (10000, 75, 0.48), B = 20: every phase of a sleep or lifetime study was
LAUNCH-bound (a frozen projection moves 15 MB in 0.40 ms, 37 GB/s of the card's ~760), and the
replay check paid three further taxes: a host sync per step (the overflow read of a frozen
`project`), a read-out that scans all L stored tokens (L k work per brain per step), and a fresh
CUDA generator per cue (1.2 s of a 3.8 s check). What replaces them, and why each is exact:

  CUES        `memory_load_drift.cue` draws torch.rand(k) from a fresh generator seeded per
              (brain, sequence). For k < 256 PyTorch's uniform kernel gives element i the first
              word of Philox4x32-10 at key = seed, counter = (0, 0, i, 0), scaled x 2^-32 + 2^-33:
              a pure function, computed here for every cue at once. `philox_matches_torch` checks
              it against torch.rand before use; the slow path is the fallback. Rows with an exact
              tie are re-sorted by the original one-dimensional argsort, so the order is the same.
  READ-OUT    a token index (neuron -> the stored tokens it is in): the overlaps of a state with
              every token are the sum over its k neurons of their token lists, k * (L k / n) work
              instead of L k -- n/k ~ 133x less. Integer counts, then the same first-maximum
              argmax.
  REPLAY      every sequence of every brain in one pass, as VIRTUAL BRAINS (the organ fiber's
              brain map, as AssemblyMemory.recall_many): the same drive kernel per row, the same
              top-k, no bias (the masked read subtracts a zero bias, which changes no bit).
  VRAM        `vram_guard` refuses to start when the card lacks the memory a study needs: the
              2026-10-10 lifetime arms ran ~90x slower than the same code alone, with the card
              full.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_reuse_grammar as rg            # noqa: E402

#: a pass reads at most this many virtual brains (bounds the [V, n] drive and [V, L] overlaps)
MAX_VIRTUAL = 2048


# ----------------------------------------------------------------- memory guard
def vram_guard(need_gb, device="cuda", margin_gb=0.5):
    """Raise unless `need_gb` (+ margin) of device memory is free right now -- whoever holds the
    rest. Returns the free GB."""
    import torch
    free, total = torch.cuda.mem_get_info(torch.device(device))
    free_gb, total_gb = free / 2**30, total / 2**30
    if free_gb < need_gb + margin_gb:
        raise RuntimeError(f"only {free_gb:.1f} of {total_gb:.1f} GB free on {device}; this needs "
                           f"{need_gb:.1f} + {margin_gb:.1f}. Another process holds the card: a run "
                           "in shared memory is ~90x slower.")
    return free_gb


# ----------------------------------------------------------------- cues
_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


def _mulhilo(a, b):
    p = a * b                                              # < 2^64: wraps in int64, bits intact
    return (p >> 32) & _MASK, p & _MASK


def philox_uniform(seeds, k, device):
    """[N, k] float32: torch.rand(k, generator=a fresh CUDA generator seeded seeds[i]), row i."""
    import torch
    s = torch.as_tensor(seeds, dtype=torch.int64, device=device).view(-1, 1)
    N = s.shape[0]
    k0 = (s & _MASK).expand(N, k).clone()
    k1 = ((s >> 32) & _MASK).expand(N, k).clone()
    c0 = torch.zeros(N, k, dtype=torch.int64, device=device)
    c1 = torch.zeros_like(c0)
    c2 = torch.arange(k, dtype=torch.int64, device=device).view(1, -1).expand(N, k).clone()
    c3 = torch.zeros_like(c0)
    for r in range(10):
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if r < 9:
            k0 = (k0 + _W0) & _MASK
            k1 = (k1 + _W1) & _MASK
    inv = torch.tensor(2.0 ** -32, dtype=torch.float32, device=device)
    u = c0.to(torch.float32) * inv + inv / 2.0
    return torch.where(u == 1.0, torch.zeros_like(u), u)


_PHILOX_OK = None


def philox_matches_torch(device, trials=64):
    """Does philox_uniform reproduce torch.rand on this torch build? Checked once per process."""
    global _PHILOX_OK
    if _PHILOX_OK is None:
        import torch
        seeds = [i * 1_000_003 + 3472 * 100_000 + i * 7 for i in range(trials)] + [2**40 + 12345, 1]
        mine = philox_uniform(seeds, 75, device)
        ref = torch.stack([torch.rand(75, device=device, generator=torch.Generator(device=device).manual_seed(s))
                           for s in seeds])
        _PHILOX_OK = bool(torch.equal(mine, ref))
    return _PHILOX_OK


def cue_index(seeds, k, device):
    """[N, k // 2] int64: the positions memory_load_drift.cue keeps, for generator seeds `seeds`."""
    import torch
    if k >= 256 or not philox_matches_torch(device):
        out = []
        for s in seeds:
            gen = torch.Generator(device=device).manual_seed(int(s))
            out.append(torch.argsort(torch.rand(k, device=device, generator=gen))[:k // 2])
        return torch.stack(out)
    vals = philox_uniform(seeds, k, device)
    perm = torch.argsort(vals, dim=1)
    srt = torch.gather(vals, 1, perm)
    tied = (srt[:, 1:] == srt[:, :-1]).any(dim=1).nonzero().view(-1).tolist()
    for r in tied:                                        # the original 1-D sort decides tie order
        perm[r] = torch.argsort(vals[r])
    return perm[:, :k // 2]


# ----------------------------------------------------------------- token index
class TokenIndex:
    """neuron -> the stored tokens it belongs to, per brain: ``table`` [B, n, cap] (token ids,
    -1 padded) and ``fill`` [B, n]. Appending a token is O(k); the overlaps of V states with
    every token cost V k cap."""

    def __init__(self, B, n, device, cap=64):
        import torch
        self.B, self.n, self.device = B, n, device
        self.table = torch.full((B, n, cap), -1, dtype=torch.int32, device=device)
        self.fill = torch.zeros(B, n, dtype=torch.int64, device=device)
        self.L = 0
        self._b = torch.arange(B, device=device).view(-1, 1)

    @classmethod
    def build(cls, allst, n, device):
        idx = cls(allst.shape[1], n, device)
        idx.extend(allst)
        return idx

    def _grow(self, need):
        import torch
        cap = self.table.shape[2]
        if need <= cap:
            return
        new = max(need, 2 * cap)
        t = torch.full((self.B, self.n, new), -1, dtype=torch.int32, device=self.device)
        t[:, :, :cap] = self.table
        self.table = t

    def extend(self, tokens):
        """append tokens [T, B, k] (ids L, L+1, ...)"""
        import torch
        T = tokens.shape[0]
        if T == 0:
            return
        k = tokens.shape[2]
        # every (brain, neuron, token) entry, ordered by brain and neuron, tokens ascending within
        neuron = tokens.long().permute(1, 0, 2).reshape(self.B, T * k)          # [B, T k]
        tok = (torch.arange(T, device=self.device).repeat_interleave(k) + self.L).view(1, -1).expand(self.B, -1)
        key = (self._b * self.n + neuron).reshape(-1)
        order = torch.sort(key, stable=True).indices
        key_s = key[order]
        tok_s = tok.reshape(-1)[order]
        start = torch.ones_like(key_s, dtype=torch.bool)
        start[1:] = key_s[1:] != key_s[:-1]
        i = torch.arange(key_s.numel(), device=self.device)
        first = torch.cummax(torch.where(start, i, torch.zeros_like(i)), dim=0).values
        rank = i - first                                   # position among this call's entries
        b_s, n_s = key_s // self.n, key_s % self.n
        pos = self.fill[b_s, n_s] + rank
        self._grow(int(pos.max()) + 1)
        self.table[b_s, n_s, pos] = tok_s.to(torch.int32)
        self.fill.view(-1).scatter_add_(0, key_s, torch.ones_like(key_s))
        self.L += T

    def overlaps(self, states, brains):
        """[V, L] float32 overlap counts of states [V, k] (brain ``brains`` [V]) with every token"""
        import torch
        V = states.shape[0]
        ids = self.table[brains.long().view(-1, 1), states.long()]          # [V, k, cap]
        ids = ids.reshape(V, -1).long()
        ids = torch.where(ids < 0, torch.full_like(ids, self.L), ids)
        out = torch.zeros(V, self.L + 1, dtype=torch.float32, device=self.device)
        out.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        return out[:, :self.L]


# ----------------------------------------------------------------- replay
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
    LEN = rg.LENGTH
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


# ----------------------------------------------------------------- sleep
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


class SequenceWriter:
    """Writes sequences as ``memory_write_separation.store`` (compare=False: store_sequence's loop)
    or ``memory_comparator.store`` without its oracle bookkeeping (compare=True), each WHOLE
    SEQUENCE replayed from one recorded CUDA graph (its seeds uploaded once, [LEN, B]).

    Exact by construction: a replay runs the eager sequence's kernels. The comparator's host branch
    (``if bool(flag.any())``) becomes an unconditional scatter of INHIBIT x flag -- zero rows where
    no brain is flagged, and adding then removing zero changes no bit of a non-negative bias -- and
    its previews defer the overflow check (checked at ``check()``). The first sequence is written
    eagerly: it builds the stimulus tables a capture must not copy from the host. The bias is
    copied into the graph's input buffer before each replay, so other code may replace it in
    between (a sequence starts from an inhibited area, so the winners are not an input)."""

    def __init__(self, mem, device, compare=False):
        """``compare``: a bool, or a bool tensor [B] -- the comparator per brain (a brain with it
        off writes exactly the plain store: its previews change nothing and its flag is forced
        off), so arms with and without it can share one launch"""
        import torch
        self.mem, self.device = mem, device
        self.mask = compare if torch.is_tensor(compare) else None
        self.compare = bool(compare.any()) if torch.is_tensor(compare) else bool(compare)
        self.graph = None
        self.warm = False
        self.flags = torch.zeros((), dtype=torch.int64, device=device)
        self.judged = 0

    def _stim(self, seeds):
        from neural_assemblies.core.torch_engine._hashed import StimulusFiber
        m = self.mem
        return StimulusFiber(seeds, m.k, m.n, m.p, beta=m.beta, w_max=m.w_max,
                             norm_init=m.norm_init, max_rounds=1, device=self.device)

    def _element(self, seeds, first):
        """one element: memory_comparator.store's body (previews from the second element on), or
        the plain write"""
        import torch
        from research.experiments import memory_comparator as mc
        m = self.mem
        area, fib = m.area, m.fiber
        B, n, k = m.B, m.n, m.k
        stim = self._stim(seeds)
        added = None
        if self.compare and not first:
            keep_b, keep_w = area.bias.clone(), area.winners.clone()
            area.bias.mul_(area.bias_decay)
            P = area.project(1, [fib, stim], freeze=True, mask_bias=False, defer_overflow=True)
            area.winners = keep_w.clone()
            R = area.project(1, [fib], freeze=True, mask_bias=False, defer_overflow=True)
            area.bias, area.winners = keep_b, keep_w
            hP = torch.zeros(B, n, device=self.device)
            hP.scatter_(1, P.long(), 1.0)
            flag = (torch.gather(hP, 1, R.long()).sum(1) / k) >= mc.THRESHOLD
            if self.mask is not None:
                flag = flag & self.mask
            self.flags += flag.sum(dtype=torch.int64)
            added = torch.zeros(B, n, device=self.device)
            added.scatter_(1, R.long(), mc.INHIBIT * flag.float().view(-1, 1).expand(-1, k).contiguous())
            area.bias += added
        win = area.project(1, [fib, stim], defer_overflow=True)
        if added is not None:
            area.bias -= added * area.bias_decay
        return win

    def _sequence(self, seeds):
        import torch
        self.mem.area.inhibit()
        return torch.stack([self._element(seeds[e], e == 0) for e in range(seeds.shape[0])])

    def write(self, elem_seeds, stored=None):
        """one sequence: elem_seeds [LEN][B] (or an int32 tensor); returns its states [LEN, B, k]
        (appended to ``stored`` one by one, as the store loops do)"""
        import torch
        m = self.mem
        area = m.area
        seeds = torch.as_tensor(elem_seeds, dtype=torch.int32).to(self.device)
        LEN = seeds.shape[0]
        if not self.warm:
            states = self._sequence(seeds)
            self.warm = True
            m.items += 1
        else:
            if self.graph is None:
                self.seed_buf = seeds.clone()
                self.bias_buf = area.bias.clone()
                area.bias = self.bias_buf
                seen, items = area.rounds_seen, m.items
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.out = self._sequence(self.seed_buf)
                    self.out_bias, self.out_win = area.bias, area.winners
                area.rounds_seen, m.items = seen, items                    # capture ran nothing
                self.flags_per = None
            else:
                self.seed_buf.copy_(seeds)
                if area.bias is not self.bias_buf:
                    self.bias_buf.copy_(area.bias)
            area.bias = self.bias_buf
            self.graph.replay()
            area.bias, area.winners = self.out_bias, self.out_win
            area.rounds_seen += LEN
            m.items += 1
            states = self.out.clone()
        if self.compare:
            self.judged += m.B * (LEN - 1)
        if stored is not None:
            stored.extend(states.clone().unbind(0))
        return states

    def check(self):
        self.mem.area.check_overflow()


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
