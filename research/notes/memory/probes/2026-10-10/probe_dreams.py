"""Exploratory (not registered): READ THE DREAMS. Sleep (Amendments 51-55) runs the network freely
from noise for 8 steps and unlearns the transitions of abnormally deep dreams. What does it dream?
Each dream step is decoded with the replay read-out: the stored token its winners overlap most,
which names a word, a sequence and a position. A step SETTLES if that overlap is >= 0.5; a
transition is FAITHFUL if two settled steps are consecutive tokens of one stored sequence (a
fragment of real replay), a LOOP if a settled step returns to a token the dream already visited,
a SPLICE if two settled steps jump to the successor of ANOTHER occurrence of the current word (the
fan of the drive decomposition, acting freely: the dream leaves one memory for another at a shared
word).

Three stores on the same brains, (10000, 75, 0.48), tau 67, subjects 980-999:
    healthy   U = 10 (replays 1.000)
    captured  U = 50 (collapsed: replays 0.025)
    slept     the same U = 50 store after 300 episodes of contrast-gated sleep (Amendment 54's
              median rule, reference brains 960-979)
For each: statistics over 200 dreams x 20 brains, and transcripts of a few dreams of the brain the
captured store hurt most."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_lifecycle as lc
from research.experiments import memory_reuse_grammar as rg
from research.experiments import memory_sleep as sl
from research.experiments import memory_threshold_law as tl

dev = "cuda"
seeds = list(range(980, 1000))
ref_seeds = list(range(960, 980))
n, k, p, tau = 10000, 75, 0.48, 67
B = len(seeds)
LEN = rg.LENGTH
spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5)}
DREAMS, SHOW, STEPS = 200, 4, 12
SETTLED = 0.5


def dream_states(mem, gen):
    """one free-running dream per brain from noise (sleep's dynamics, nothing unlearned):
    winners [STEPS, B, k] and contrasts [STEPS, B]"""
    area, fib = mem.area, mem.fiber
    area.bias = torch.zeros(B, n, device=dev)
    area.winners = torch.argsort(torch.rand(B, n, device=dev, generator=gen), dim=1)[:, :k]
    ws, cs = [], []
    for _ in range(STEPS):
        x, drive = area.project(1, [fib], freeze=True, mask_bias=False, return_drive=True)
        x = x.long()
        ws.append(x)
        cs.append(torch.gather(drive, 1, x).mean(1) / drive.mean(1).clamp_min(1e-9))
    return torch.stack(ws), torch.stack(cs)


def decode(st, winners):
    """best stored token per step and brain: index [STEPS, B], overlap / k [STEPS, B]"""
    allst = st["allst"]                                  # [L, B, k]
    L = allst.shape[0]
    idx, ov = [], []
    for s in range(winners.shape[0]):
        hot = torch.zeros(B, n, device=dev)
        hot.scatter_(1, winners[s], 1.0)
        o = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)   # [L, B]
        v, i = o.max(0)
        idx.append(i)
        ov.append(v / k)
    return torch.stack(idx), torch.stack(ov)


def word(st, t, b):
    return int(st["wordof"][t, b])


def read(label, st, show_brain):
    gen = torch.Generator(device=dev).manual_seed(4242)
    settled = faithful = loops = trans = splices = 0
    wo = st["wordof"].cpu().numpy()
    runs, distinct, conts = [], [], []
    shown = []
    for d in range(DREAMS):
        winners, con = dream_states(st["mem"], gen)
        idx, ov = decode(st, winners)
        idx, ov, con = idx.cpu().numpy(), ov.cpu().numpy(), con.cpu().numpy()
        conts.append(con[2:].ravel())
        for b in range(B):
            seen, run, best = set(), 0, 0
            for s in range(STEPS):
                ok = ov[s, b] >= SETTLED
                settled += ok
                if ok and int(idx[s, b]) in seen:
                    loops += 1
                if s > 0:
                    trans += 1
                    prev_ok = ov[s - 1, b] >= SETTLED
                    t0, t1 = int(idx[s - 1, b]), int(idx[s, b])
                    f = prev_ok and ok and t1 == t0 + 1 and t1 // LEN == t0 // LEN
                    faithful += f
                    splices += (prev_ok and ok and not f and t1 % LEN != 0
                                and wo[t1 - 1, b] == wo[t0, b] and t1 - 1 != t0)
                    run = run + 1 if f else 0
                    best = max(best, run)
                if ok:
                    seen.add(int(idx[s, b]))
            runs.append(best)
            distinct.append(len(seen))
        if d < SHOW:
            shown.append((idx[:, show_brain], ov[:, show_brain], con[:, show_brain]))
    steps = DREAMS * B * STEPS
    c = np.concatenate(conts)
    print(f"\n{label}: replay {np.mean(sl.reliability(st, dev)):.3f}", flush=True)
    print(f"  settled steps {settled / steps:.3f}; faithful transitions {faithful / trans:.3f}; "
          f"splices at a shared word {splices / trans:.3f}; "
          f"loops (a settled step revisiting a token) {loops / max(1, settled):.3f} of settled", flush=True)
    print(f"  per dream: distinct settled tokens {np.mean(distinct):.2f}; longest faithful run "
          f"{np.mean(runs):.2f} (max {max(runs)}); contrast mean {c.mean():.3f}, p99 {np.quantile(c, 0.99):.3f}",
          flush=True)
    print(f"  transcripts, brain {seeds[show_brain]} (word / sequence:position / overlap / contrast):", flush=True)
    for j, (i_, o_, c_) in enumerate(shown):
        parts = []
        for s in range(STEPS):
            t = int(i_[s])
            if o_[s] >= SETTLED:
                parts.append(f"w{word(st, t, show_brain)}/{t // LEN}:{t % LEN}/{o_[s]:.2f}/{c_[s]:.2f}")
            else:
                parts.append(f"(~w{word(st, t, show_brain)} {o_[s]:.2f})")
        print(f"    dream {j + 1}: " + " -> ".join(parts), flush=True)


ref = sl.build_store(spec, 10, ref_seeds, dev)
cal = []
g0 = torch.Generator(device=dev).manual_seed(sl.CAL_SEED)
for _ in range(sl.CALIBRATION):
    sl.dream(ref["mem"], g0, None, dev, cal)
del ref
thr = float(torch.cat(cal).float().view(-1, len(ref_seeds)).max(0).values.median()) * sl.MARGIN
print(f"sleep threshold (median rule) {thr:.3f}", flush=True)
torch.cuda.empty_cache()

st = sl.build_store(spec, 50, seeds, dev)
worst = int(np.argmin(sl.reliability(st, dev)))
del st
torch.cuda.empty_cache()
h = sl.build_store(spec, 10, seeds, dev)
read("HEALTHY U = 10", h, worst)
del h
torch.cuda.empty_cache()
st = sl.build_store(spec, 50, seeds, dev)
read("CAPTURED U = 50", st, worst)
removed = lc.slept(st, 300, thr, dev)
print(f"\n(300 episodes of sleep removed {removed:.4f} of counts)", flush=True)
read("SLEPT U = 50", st, worst)
