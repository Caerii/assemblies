"""Exploratory (not registered): a SIMULATED LIFETIME of days and nights. Every day the brain writes
D new 16-word sequences (random walks on its own grammar, random successors, a FIXED vocabulary, so
reuse per word grows linearly with age); every night it sleeps E episodes. Over 24 days the store
passes the reuse edge (~40 uses per word, Amendment 45) near day 10 and the load edge (rho ~ 0.14,
Amendment 37) near day 14, ending at rho = 0.24 and 100 uses per word.

Arms, same brains and the same life:
  awake        no sleep, standard writes
  reference    nightly sleep gated at Amendment 54's median rule (reference brains, healthy U = 10)
  set point    nightly sleep gated at each brain's own birth set point (Amendment 55)
  lifecycle    comparator writes (Amendment 50) + nightly set-point sleep
  + downscale  the lifecycle plus nightly DOWNSCALING (synaptic homeostasis, Tononi & Cirelli): each
               potentiated synapse loses one count with probability r per night, r = 0.05 and 0.1
               -- a forgetting rate, so the store might hold a span of recent days instead of
               overflowing (the load edge no write rule moves)
  sat          counts SATURATE: each night every count is clamped at the count where the weight
               clip binds (clip_count: 10 here). Weights are unchanged (the chain table is flat above
               the clip), but a count can no longer run on to 127 above it, so one-count decrements
               -- downscaling and sleep's unlearning -- reach the saturated hub synapses at all
               (probe_lifetime_rest.log: without it, downscaling at 0.05 and 0.1 left today's
               memories at day 12 where no downscaling did, 0.73-0.75)
(A first run, probe_lifetime_first.log, stopped at day 16 of the awake arm: its store loop also ran
the comparator's previews and oracle bookkeeping, quadratic in the store; the loops here are
store_sequence's own, and the comparator's without the oracle.)
Every 4 days: replay of memories by AGE (written today, 1-3 days ago, 4-7, 8+), up to 40 sampled
sequences per age bin, and 20 dreams decoded (share of steps settled on a stored token).
(10000, 75, 0.48), tau 67, subjects 980-999, reference brains 960-979."""
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
from research.experiments import memory_threshold_law as tl
from research.experiments import memory_write_separation as ws
from research.experiments.seq_capacity_scaling import to_i32
from neural_assemblies.core.numpy_engine import _seeding

dev = "cuda"
seeds = list(range(980, 1000))
ref_seeds = list(range(960, 980))
n, k, p, tau = 10000, 75, 0.48, 67
B = len(seeds)
LEN = rg.LENGTH
spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5)}
DAYS, RHO_DAY, U_END, EPISODES, EVERY, SAMPLE = 24, 0.01, 100, 100, 4, 40
D = round(RHO_DAY * ml.unit(n, k, p) / LEN)
M_END = D * DAYS
V = max(8, round(M_END * LEN / U_END))
WORDS = [rg.walks(sd, M_END, V, None, 777_000 + M_END) for sd in seeds]
WORDOF = torch.tensor(np.stack([w.reshape(-1) for w in WORDS], 1), device=dev).long()
print(f"life: {DAYS} days x {D} sequences (rho {RHO_DAY}/day), vocabulary {V} words "
      f"({D * LEN / V:.1f} uses per word per day), {EPISODES} sleep episodes per night", flush=True)


def comparator_store(mem, elem_seeds, stored, stats):
    """memory_comparator.store with the comparator on, without the oracle bookkeeping"""
    from neural_assemblies.core.torch_engine._hashed import StimulusFiber
    area, fib = mem.area, mem.fiber
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
    return torch.stack(states)


def write_day(mem, stored, seqs, day, compare, stats):
    for q in range(day * D, (day + 1) * D):
        es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(WORDS[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
              for e in range(LEN)]
        seqs.append(comparator_store(mem, es, stored, stats) if compare
                    else ws.store(mem, es, stored, False, dev)[0])
    mem.area.check_overflow()


def downscale(mem, r, gen):
    """each potentiated synapse loses one count with probability r; returns the counts removed"""
    C = mem.fiber.C
    removed = 0
    for b in range(B):
        cut = (C[b] > 0) & (torch.rand(C.shape[1], C.shape[2], device=dev, generator=gen) < r)
        C[b] -= cut.to(C.dtype)
        removed += int(cut.sum())
    return removed


def replay(mem, seqs, stored, qs):
    """masked replay of sequences qs, word-level, as memory_sleep.reliability: per sequence, the
    fraction of brains whose whole sequence is read correctly"""
    area, fib = mem.area, mem.fiber
    allst = torch.stack(stored).long()
    L = allst.shape[0]
    wordof = WORDOF[:L]
    ar = torch.arange(B, device=dev)
    out = []
    for q in qs:
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.stack([sl.md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev)
                                    for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, LEN):
            x = area.project(1, [fib], freeze=True, mask_bias=False)
            hot = torch.zeros(B, n, device=dev)
            hot.scatter_(1, x.long(), 1.0)
            ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
            alive &= wordof[ov.argmax(0), ar] == wordof[q * LEN + j]
        out.append(float(alive.float().mean()))
    return out


def dreams_settled(mem, stored, gen, dreams=20, steps=12):
    area, fib = mem.area, mem.fiber
    allst = torch.stack(stored).long()
    L = allst.shape[0]
    settled = total = 0
    for _ in range(dreams):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.argsort(torch.rand(B, n, device=dev, generator=gen), dim=1)[:, :k]
        for _ in range(steps):
            x = area.project(1, [fib], freeze=True, mask_bias=False).long()
            hot = torch.zeros(B, n, device=dev)
            hot.scatter_(1, x, 1.0)
            o = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).max(0).values / k
            settled += int((o >= 0.5).sum())
            total += B
    return settled / total


def bins(day):
    """age bins of sequence indices at the end of `day` (0-based)"""
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


# gates: the reference median (Amendment 54) and each subject's birth set point (Amendment 55)
ref = sl.build_store(spec, 10, ref_seeds, dev)
median_thr = float(sp.brain_maxima(ref["mem"], len(ref_seeds), dev).median()) * sl.MARGIN
del ref
empty = ws.build(n, k, p, tau, seeds, dev)
setpoint = (sp.brain_maxima(empty, B, dev) * sl.MARGIN).to(dev)
del empty
torch.cuda.empty_cache()
print(f"gates: reference median {median_thr:.3f}; set points {float(setpoint.min()):.3f}-{float(setpoint.max()):.3f}",
      flush=True)

from neural_assemblies.core.torch_engine._hashed import clip_count
SAT = clip_count(spec["beta"], 20.0, 1)


def saturate(mem):
    mem.fiber.C.clamp_(max=SAT)


# clamping at the clip changes no weight: replay of a small store is identical before and after
_chk = sl.build_store({"n": 2000, "k": 60, "p": 0.5, "tau": 17, "rho": rg.RHO,
                       "beta": round(tl.theta(2000, 60, 0.5), 5)}, 40, seeds[:4], dev)
_a = sl.reliability(_chk, dev)
_c_max = int(_chk["mem"].fiber.C.max())
_chk["mem"].fiber.C.clamp_(max=clip_count(round(tl.theta(2000, 60, 0.5), 5), 20.0, 1))
assert sl.reliability(_chk, dev) == _a, "clamping at the clip changed replay"
print(f"saturation check: clamping counts (max {_c_max}) at the clip leaves replay identical; SAT = {SAT}", flush=True)
del _chk
torch.cuda.empty_cache()

ARMS = (("awake", False, None, 0, False), ("reference", False, median_thr, 0, False),
        ("set point", False, setpoint, 0, False), ("lifecycle", True, setpoint, 0, False),
        ("lifecycle + downscale 0.05", True, setpoint, 0.05, False),
        ("lifecycle + downscale 0.1", True, setpoint, 0.1, False),
        ("sat lifecycle", True, setpoint, 0, True), ("sat lifecycle + downscale 0.05", True, setpoint, 0.05, True),
        ("sat lifecycle + downscale 0.1", True, setpoint, 0.1, True))
ONLY = [x.strip() for x in os.environ.get("LIFETIME_ARMS", "").split(",") if x.strip()]
for name, compare, gate, rate, sat in [arm for arm in ARMS if not ONLY or arm[0] in ONLY]:
    print(f"\n=== {name} ===", flush=True)
    mem = ws.build(n, k, p, tau, seeds, dev)
    stored, seqs = [], []
    stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
    noise = torch.Generator(device=dev).manual_seed(sl.NOISE_SEED)
    probe_gen = torch.Generator(device=dev).manual_seed(4242)
    removed_life = 0
    scale_gen = torch.Generator(device=dev).manual_seed(99)
    for day in range(DAYS):
        write_day(mem, stored, seqs, day, compare, stats)
        if sat:
            saturate(mem)
        night = gated = judged = 0
        if gate is not None:
            held = int(mem.fiber.C.sum())
            for _ in range(EPISODES):
                r, g, j = sl.dream(mem, noise, gate, dev)
                night += r
                gated += g
                judged += j
            removed_life += night
        faded = downscale(mem, rate, scale_gen) if rate else 0
        if (day + 1) % EVERY == 0:
            uses = (day + 1) * D * LEN / V
            rho = (day + 1) * RHO_DAY
            line = []
            for b, qs in bins(day).items():
                line.append(f"{b} {np.mean(replay(mem, seqs, stored, qs)):.3f}")
            set_ = dreams_settled(mem, stored, probe_gen)
            extra = (f"; tonight gate open {gated / max(1, judged):.4f}, removed {night / max(1, held):.4f}"
                     if gate is not None else "")
            flags = f"; flags {stats['flag'] / max(1, stats['judged']):.4f}" if compare else ""
            if sat:
                flags += f"; counts held {int(mem.fiber.C.sum())}"
            if rate:
                flags += f"; downscaled {faded / max(1, held):.4f} tonight, {int(mem.fiber.C.sum())} counts held"
            print(f"day {day + 1:2d} (rho {rho:.2f}, {uses:5.1f} uses/word): " + ", ".join(line)
                  + f"; dreams settled {set_:.3f}{extra}{flags}", flush=True)
    del mem, stored, seqs
    torch.cuda.empty_cache()
