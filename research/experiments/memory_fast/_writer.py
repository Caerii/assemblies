"""The sequence writer: a whole sequence (plain or comparator) as one recorded CUDA graph.

Part of research.experiments.memory_fast (see its __init__ for why each path is exact)."""
from __future__ import annotations


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
