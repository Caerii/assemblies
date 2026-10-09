"""Exploratory (not registered): the WRITE-SLEEP LIFECYCLE -- how far does reuse go when the local
comparator (Amendment 50) guards writes and the contrast-gated sleep (Amendment 51's rule)
repairs what forms anyway? Random-successor reuse, (10000, 75, 0.48), tau = 67 (judged cell),
subject seeds 980-999, reference brains 960-979 for the sleep threshold (300 dreams of a U = 10
store). U = 60, 80, 100. Arms per U, same memories: standard; comparator; standard + sleep;
comparator + sleep (300 and 1000 episodes). Masked replay of every sequence, word-level."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_comparator as mc
from research.experiments import memory_load_law as ml
from research.experiments import memory_reuse_grammar as rg
from research.experiments import memory_sleep as sl
from research.experiments import memory_threshold_law as tl
from research.experiments import memory_write_separation as ws
from research.experiments.seq_capacity_scaling import to_i32
from neural_assemblies.core.numpy_engine import _seeding

seeds = list(range(980, 1000))
ref_seeds = list(range(960, 980))
n, k, p, tau = 10000, 75, 0.48, 67
dev = "cuda"
spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5)}
LEN = rg.LENGTH

ref = sl.build_store(spec, 10, ref_seeds, dev)
cal = []
g0 = torch.Generator(device=dev).manual_seed(sl.CAL_SEED)
for _ in range(sl.CALIBRATION):
    sl.dream(ref["mem"], g0, None, dev, cal)
thr = float(torch.cat(cal).max()) * sl.MARGIN
print(f"sleep threshold {thr:.3f}", flush=True)
del ref
torch.cuda.empty_cache()


def comparator_store(U):
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / LEN)); L = M * LEN; V = max(8, round(L / U))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    mem = ws.build(n, k, p, tau, seeds, dev)
    stored, seqs = [], []
    stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
    for q in range(M):
        es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
              for e in range(LEN)]
        seqs.append(mc.store(mem, es, stored, True, dev, stats))
    mem.area.check_overflow()
    return {"mem": mem, "seqs": seqs, "allst": torch.stack(stored).long(), "wordof": wordof, "M": M, "L": L,
            "seeds": seeds}, stats


def sleep(st, episodes, gen):
    removed = 0
    for _ in range(episodes):
        removed += sl.dream(st["mem"], gen, thr, dev)[0]
    return removed


def show(name, rel, extra=""):
    rel = np.array(rel)
    print(f"  {name:28s} {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}; min {rel.min():.2f}){extra}", flush=True)


for U in (60, 80, 100):
    print(f"\nU = {U}", flush=True)
    st = sl.build_store(spec, U, seeds, dev)
    held = int(st["mem"].fiber.C.sum())
    show("standard", sl.reliability(st, dev))
    gen = torch.Generator(device=dev).manual_seed(sl.NOISE_SEED)
    r = sleep(st, 300, gen)
    show("standard + sleep 300", sl.reliability(st, dev), f"; removed {r / held:.4f}")
    del st; torch.cuda.empty_cache()
    st, stats = comparator_store(U)
    held = int(st["mem"].fiber.C.sum())
    show("comparator", sl.reliability(st, dev), f"; flags {stats['flag'] / max(1, stats['judged']):.4f}")
    gen = torch.Generator(device=dev).manual_seed(sl.NOISE_SEED)
    r = sleep(st, 300, gen)
    show("comparator + sleep 300", sl.reliability(st, dev), f"; removed {r / held:.4f}")
    r += sleep(st, 700, gen)
    show("comparator + sleep 1000", sl.reliability(st, dev), f"; removed {r / held:.4f}")
    del st; torch.cuda.empty_cache()
