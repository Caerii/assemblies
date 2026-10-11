"""What a memory study writes and the store it writes into -- the setup every registration
repeated by hand (fourteen copies of the store loop alone by Amendment 55).

    Plan           the sequences: per-brain random walks on a grammar of V words (b successors
                   per word, or random), M sequences of LENGTH elements, and their element seeds.
                   Plan.of(cell, seeds, uses, ...) sizes them as the registrations did (M from the
                   load rho, V from the uses per word); Plan.lifetime(...) a life of days.
    Store          a memory being written to: memory_fast's builder and graphed writer (plain, the
                   comparator, or the comparator per brain), a token index kept as it grows, and
                   the replay check, sleep, downscaling and count clamping on it.
    reference_median(...), birth_setpoint(...)
                   the two sleep gates registered so far (Amendments 54 and 55).

EXACT. A plain Store writes what memory_sleep.build_store writes, a comparator Store what
memory_lifecycle.comparator_store writes, and their checks read what memory_sleep.reliability reads
(test_memory_lib.py), because each piece is the registered convention -- the walk salt
L * 1000 + 7 b + U, the seed fnv1a_pair_seed(brain, "w<word>", "A") -- run on the fast paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .laws import unit
from .seeding import to_i32
from .walks import LENGTH, RHO


@dataclass
class Plan:
    cell: object
    seeds: tuple
    M: int
    V: int
    successors: object = None          # b successors per word, or None (random)
    salt: int = 0
    words: list = field(default_factory=list, repr=False)

    @classmethod
    def of(cls, cell, seeds, uses, rho=None, successors=None, M=None):
        """the registrations' sizing: M = rho * unit / LENGTH sequences (rho by default the
        programme's 0.05), V = L / uses words, salt L * 1000 + 7 b + uses"""
        rho = RHO if rho is None else rho
        if M is None:
            M = max(1, round(rho * unit(cell.n, cell.k, cell.p) / LENGTH))
        L = M * LENGTH
        V = max(8, round(L / uses))
        return cls(cell, tuple(seeds), M, V, successors, L * 1000 + (successors or 0) * 7 + uses)._walk()

    @classmethod
    def lifetime(cls, cell, seeds, days, per_day, uses_end, salt):
        """a life: days x per_day sequences on a fixed vocabulary sized so each word reaches
        ``uses_end`` uses on the last day (probe_lifetime's plan)"""
        M = days * per_day
        V = max(8, round(M * LENGTH / uses_end))
        return cls(cell, tuple(seeds), M, V, None, salt)._walk()

    def _walk(self):
        from research.experiments import memory_reuse_grammar as rg
        self.words = [rg.walks(sd, self.M, self.V, self.successors, self.salt) for sd in self.seeds]
        return self

    @property
    def L(self):
        return self.M * LENGTH

    def wordof(self, device):
        """[L, B] the word of every element of every sequence"""
        import numpy as np
        import torch
        return torch.tensor(np.stack([w.reshape(-1) for w in self.words], 1), device=device).long()

    def element_seeds(self, q):
        """[LENGTH][B] the stimulus seeds of sequence q"""
        from neural_assemblies.core.numpy_engine import _seeding
        return [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(self.words[i][q, e])}", "A"))
                 for i, sd in enumerate(self.seeds)] for e in range(LENGTH)]


class Store:
    """a memory being written with a Plan's sequences, on the fast paths"""

    def __init__(self, plan, device, compare=False, count_dtype=None):
        from research.experiments import memory_fast as mf
        self.plan, self.device = plan, device
        c = plan.cell
        self.mem = mf.build_memory(c.n, c.k, c.p, c.tau, plan.seeds, device, count_dtype=count_dtype)
        self.writer = mf.SequenceWriter(self.mem, device, compare=compare)
        self.index = mf.TokenIndex(self.mem.B, c.n, device)
        self.seqs, self.tokens = [], []
        self._wordof = plan.wordof(device)

    @property
    def written(self):
        return len(self.seqs)

    def write(self, count=None):
        """write the next ``count`` sequences of the plan (default: all that remain)"""
        import torch
        end = self.plan.M if count is None else min(self.plan.M, self.written + int(count))
        start = len(self.tokens)
        for q in range(self.written, end):
            st = self.writer.write(self.plan.element_seeds(q))
            self.seqs.append(st)
            self.tokens.extend(st.unbind(0))
        self.writer.check()
        if len(self.tokens) > start:
            self.index.extend(torch.stack(self.tokens[start:]))
        return self

    def record(self):
        """the dict memory_sleep's functions take"""
        import torch
        L = len(self.tokens)
        return {"mem": self.mem, "seqs": self.seqs, "allst": torch.stack(self.tokens).long(),
                "wordof": self._wordof[:L], "M": self.written, "L": L, "seeds": list(self.plan.seeds)}

    def replay(self, qs=None):
        """per brain, the fraction of the sequences ``qs`` (default all written) replayed whole"""
        from research.experiments import memory_fast as mf
        return mf.reliability(self.record(), self.device, self.index, qs)

    def sleep(self, gate, episodes, gen):
        """(counts removed, steps gated, steps judged) over ``episodes`` gated dreams"""
        from research.experiments import memory_fast as mf
        return mf.sleep(self.mem, gen, gate, episodes, self.device)

    def downscale(self, rates, gen):
        return self.mem.fiber.downscale(rates, gen)

    def clamp(self, top):
        self.mem.fiber.clamp_counts(top)

    @property
    def flags(self):
        """the comparator's flag rate so far"""
        return int(self.writer.flags) / max(1, self.writer.judged)


def _brain_maxima(mem, device):
    import torch
    from research.experiments import memory_fast as mf
    from research.experiments import memory_sleep as sl
    con = mf.calibrate(mem, torch.Generator(device=device).manual_seed(sl.CAL_SEED), sl.CALIBRATION, device)
    return con.view(-1, mem.B).max(0).values


def reference_median(cell, reference_seeds, device):
    """Amendment 54's gate: 1.02 x the median over reference brains of each brain's largest
    contrast in 300 dreams of a healthy (10 uses per word) store"""
    from research.experiments import memory_sleep as sl
    ref = Store(Plan.of(cell, reference_seeds, 10), device).write()
    return float(_brain_maxima(ref.mem, device).median()) * sl.MARGIN


def birth_setpoint(store, device):
    """Amendment 55's gate: per brain, 1.02 x the largest contrast its own EMPTY network's dreams
    reach -- call before the store's first write"""
    from research.experiments import memory_sleep as sl
    if store.written:
        raise ValueError("a set point is fixed at birth: before any write")
    return _brain_maxima(store.mem, device) * sl.MARGIN
