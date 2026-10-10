"""Exploratory (not registered): SYSTEMS CONSOLIDATION over a simulated lifetime. probe_lifetime
showed every single-store life dies of LOAD (the reuse edge yields to the lifecycle; the load edge
does not), and nightly downscaling slow enough to keep a useful span forgets too slowly to stop it.
Here a small fast store forgets on purpose and a large slow store keeps what it was told:

  H ("hippocampus")  (10000, 75, 0.48), tau 67: comparator writes by day; each night contrast-gated
                     sleep at its birth set point and DOWNSCALING at rate R_H (a span of days)
  C ("cortex")       (20000, 150, 0.48), tau 67 (n/k = 133, tau = n/2k as H; ~1100 sequences before its load edge,
                     a whole life of 1032): each night, BEFORE H downscales, every sequence H wrote
                     that day is REPLAYED in H from its cue, read out word by word, and the words read
                     are written into C as a new sequence (comparator writes) -- consolidation copies
                     what H can replay, errors included; then C sleeps at its own birth set point.
                     C never downscales.
                     Counts in both stores SATURATE at their clip each night (probe_lifetime's sat arms).

The life of probe_lifetime: 24 days x 43 sequences, vocabulary 165 words (100 uses per word by the
end). Replay is frozen, so consolidation does not change H: H here lives exactly the life of an
H-only brain. Every 4 days, replay by AGE from H and from C (C cued from its own stored first token;
a C replay is correct only if it reproduces the TRUE words), and from EITHER (a memory alive in one
store or the other). TEN brains (seeds 980-989), not twenty: two stores on one 10 GB GPU."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_comparator as mc
from research.experiments import memory_load_law as ml
from research.experiments import memory_reuse_grammar as rg
from research.experiments import memory_setpoint_sleep as sp
from research.experiments import memory_sleep as sl
from research.experiments import memory_write_separation as ws
from research.experiments.seq_capacity_scaling import to_i32
from neural_assemblies.core.numpy_engine import _seeding

dev = "cuda"
seeds = list(range(980, 990))
B = len(seeds)
LEN = rg.LENGTH
HC = (10000, 75, 0.48, 67)
CC = (20000, 150, 0.48, 67)
R_H = float(os.environ.get("R_H", "0.25"))
DAYS, RHO_DAY, U_END, EPISODES, EVERY, SAMPLE = 24, 0.01, 100, 100, 4, 40
D = round(RHO_DAY * ml.unit(*HC[:3]) / LEN)
M_END = D * DAYS
V = max(8, round(M_END * LEN / U_END))
WORDS = [rg.walks(sd, M_END, V, None, 777_000 + M_END) for sd in seeds]          # as probe_lifetime
TRUE = torch.tensor(np.stack([w.reshape(-1) for w in WORDS], 1), device=dev).long()  # [L_END, B]
print(f"life: {DAYS} days x {D} sequences, vocabulary {V}; H {HC}, downscaling {R_H}/night; C {CC}; "
      f"{B} brains", flush=True)


def stim_seeds(words_by_brain):
    """element seeds [LEN][B] for one sequence given each brain's LEN words"""
    return [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words_by_brain[i][e])}", "A")) for i, sd in enumerate(seeds)]
            for e in range(LEN)]


def comparator_store(mem, elem_seeds, stored, stats):
    """memory_comparator.store with the comparator on, without the oracle bookkeeping"""
    from neural_assemblies.core.torch_engine._hashed import StimulusFiber
    area, fib = mem.area, mem.fiber
    n, k = mem.n, mem.k
    area.inhibit()
    states = []
    for e, sds in enumerate(elem_seeds):
        stim = StimulusFiber(sds, k, n, mem.p, beta=mem.beta, w_max=mem.w_max,
                             norm_init=mem.norm_init, max_rounds=1, device=dev)
        added = None
        if e > 0:
            keep_b, keep_w = area.bias.clone(), area.winners.clone()
            area.bias.mul_(area.bias_decay)
            P = area.project(1, [fib, stim], freeze=True, mask_bias=False)
            area.winners = keep_w.clone()
            R = area.project(1, [fib], freeze=True, mask_bias=False)
            area.bias, area.winners = keep_b, keep_w
            hP = torch.zeros(B, n, device=dev)
            hP.scatter_(1, P.long(), 1.0)
            flag = (torch.gather(hP, 1, R.long()).sum(1) / k) >= mc.THRESHOLD
            stats["judged"] += B
            stats["flag"] += int(flag.sum())
            if bool(flag.any()):
                added = torch.zeros(B, n, device=dev)
                added.scatter_(1, R.long(), mc.INHIBIT * flag.float().view(-1, 1).expand(-1, k).contiguous())
                area.bias += added
        win = area.project(1, [fib, stim], defer_overflow=True)
        if added is not None:
            area.bias -= added * area.bias_decay
        states.append(win.clone())
        stored.append(win.clone())
    mem.items += 1
    area.check_overflow()
    return torch.stack(states)


class Store:
    def __init__(self, cell):
        n, k, p, tau = cell
        self.n, self.k = n, k
        self.mem = ws.build(n, k, p, tau, seeds, dev)
        self.setpoint = (sp.brain_maxima(self.mem, B, dev) * sl.MARGIN).to(dev)   # before any learning
        self.stored, self.seqs = [], []
        self.words = []          # per stored token: the word written [B] (what the store believes)
        self.stats = {"judged": 0, "flag": 0}
        from neural_assemblies.core.torch_engine._hashed import clip_count
        self.sat = clip_count(self.mem.beta, 20.0, 1)

    def saturate(self):
        """clamp counts at the clip: no weight changes, but decrements reach saturated synapses"""
        self.mem.fiber.C.clamp_(max=self.sat)

    def write(self, words_by_brain):
        st = comparator_store(self.mem, stim_seeds(words_by_brain), self.stored, self.stats)
        self.seqs.append(st)
        for e in range(LEN):
            self.words.append(torch.tensor([int(words_by_brain[i][e]) for i in range(B)], device=dev))
        return len(self.seqs) - 1

    def replay(self, q, steps=LEN - 1):
        """masked replay of stored sequence q: the words read at steps 1..LEN-1, [LEN-1, B]"""
        area, fib = self.mem.area, self.mem.fiber
        n, k = self.n, self.k
        allst = torch.stack(self.stored).long()
        L = allst.shape[0]
        believed = torch.stack(self.words)                           # [L, B]
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.stack([sl.md.cue(self.seqs[q][0, i], sd, L * 100_000 + q, k, dev)
                                    for i, sd in enumerate(seeds)]).long()
        ar = torch.arange(B, device=dev)
        out = []
        for _ in range(steps):
            x = area.project(1, [fib], freeze=True, mask_bias=False)
            hot = torch.zeros(B, n, device=dev)
            hot.scatter_(1, x.long(), 1.0)
            ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
            out.append(believed[ov.argmax(0), ar])
        return torch.stack(out)

    def sleep(self, gen):
        removed = gated = judged = 0
        for _ in range(EPISODES):
            r, g, j = sl.dream(self.mem, gen, self.setpoint, dev)
            removed += r
            gated += g
            judged += j
        return removed, gated / max(1, judged)

    def downscale(self, r, gen):
        C = self.mem.fiber.C
        for b in range(B):
            cut = (C[b] > 0) & (torch.rand(C.shape[1], C.shape[2], device=dev, generator=gen) < r)
            C[b] -= cut.to(C.dtype)


def correct(read, q):
    """per brain: did the replay reproduce the true words of life-sequence q (steps 1..LEN-1)?"""
    return (read == TRUE[q * LEN + 1:(q + 1) * LEN]).all(0)


def bins(day):
    rng = np.random.default_rng(day)
    spans = {"today": (day, day), "1-3 days": (day - 3, day - 1), "4-7 days": (day - 7, day - 4),
             "8+ days": (0, day - 8)}
    out = {}
    for name, (a, b) in spans.items():
        a = max(a, 0)
        if b < a:
            continue
        qs = np.arange(a * D, (b + 1) * D)
        out[name] = sorted(rng.choice(qs, size=min(SAMPLE, len(qs)), replace=False).tolist())
    return out


H = Store(HC)
C = Store(CC)
print(f"set points: H {float(H.setpoint.min()):.3f}-{float(H.setpoint.max()):.3f}, "
      f"C {float(C.setpoint.min()):.3f}-{float(C.setpoint.max()):.3f}", flush=True)
c_index = {}                     # life-sequence q -> its index in C
gH = torch.Generator(device=dev).manual_seed(sl.NOISE_SEED)
gC = torch.Generator(device=dev).manual_seed(sl.NOISE_SEED + 1)
gS = torch.Generator(device=dev).manual_seed(99)
fidelity = []
for day in range(DAYS):
    for q in range(day * D, (day + 1) * D):
        H.write([WORDS[i][q] for i in range(B)])
    # night: consolidate today's sequences as H replays them (the first word from the cue)
    ok_tonight = []
    for q in range(day * D, (day + 1) * D):
        read = H.replay(q)
        ok_tonight.append(float(correct(read, q).float().mean()))
        words = torch.cat([TRUE[q * LEN].view(1, B), read], 0).T.cpu().numpy()   # [B, LEN]
        c_index[q] = C.write(words)
    fidelity.append(np.mean(ok_tonight))
    H.saturate()
    C.saturate()
    rH, gateH = H.sleep(gH)
    H.downscale(R_H, gS)
    rC, gateC = C.sleep(gC)
    if (day + 1) % EVERY == 0:
        parts = []
        for name, qs in bins(day).items():
            h = torch.stack([correct(H.replay(q), q) for q in qs]).float()
            c = torch.stack([correct(C.replay(c_index[q]), q) for q in qs]).float()
            parts.append(f"{name} H {float(h.mean()):.3f} C {float(c.mean()):.3f} either {float(torch.maximum(h, c).mean()):.3f}")
        rho_c = len(C.stored) / ml.unit(*CC[:3])
        print(f"day {day + 1:2d}: " + "; ".join(parts) + f" | consolidated tonight {fidelity[-1]:.3f} correct; "
              f"gates H {gateH:.3f} C {gateC:.3f}; flags H {H.stats['flag'] / max(1, H.stats['judged']):.4f} "
              f"C {C.stats['flag'] / max(1, C.stats['judged']):.4f}; C rho {rho_c:.3f}", flush=True)
