"""Exploratory (not registered): does the write-sleep lifecycle reach the REPETITION edge? Amendment
44-45: repeating transitions (each word followed by one of b successors) breaks replay at ~2.8
repeats per transition -- tokens of a repeated bigram merge (overlap to 0.42). Two readings:
  merging   the failure is capture of one token by another of the same bigram; the comparator
            (Amendment 50) and contrast-gated sleep (Amendment 51) should rescue it.
  ambiguity with b successors per word, a word-level read has no way to know which successor a
            given occurrence had except through its token's context; if the context does not
            survive, no write rule helps, and a TYPE store is what is missing.
Arms, same memories: standard; sleep 300; comparator; comparator + sleep 300. Reuse U = 10 with
b = 3 and b = 5 successors per word (repetition R = U / b per transition ~ 3.3 and 2), and U = 10
random successors (healthy control). (10000, 75, 0.48), tau = 67 (judged cell), seeds 980-999;
sleep threshold from reference brains 960-979."""
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


def store(U, b, compare):
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / LEN)); L = M * LEN; V = max(8, round(L / U))
    words = [rg.walks(sd, M, V, b, L * 1000 + (b or 0) * 7 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    mem = ws.build(n, k, p, tau, seeds, dev)
    stored, seqs = [], []
    stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
    for q in range(M):
        es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
              for e in range(LEN)]
        seqs.append(mc.store(mem, es, stored, compare, dev, stats))
    mem.area.check_overflow()
    return {"mem": mem, "seqs": seqs, "allst": torch.stack(stored).long(), "wordof": wordof, "M": M, "L": L,
            "seeds": seeds}, stats


def sleep(st, episodes):
    gen = torch.Generator(device=dev).manual_seed(sl.NOISE_SEED)
    held = int(st["mem"].fiber.C.sum())
    return sum(sl.dream(st["mem"], gen, thr, dev)[0] for _ in range(episodes)) / held


def bigram_merge(st):
    """mean overlap of the two tokens of a repeated bigram's SUCCESSOR (tokens following equal
    word pairs), and of unrelated tokens of the same word -- the merging the readings disagree on"""
    allst, wordof, L = st["allst"], st["wordof"], st["L"]
    out = []
    for i in range(len(seeds)):
        w = wordof[:, i].cpu().numpy()
        pairs = {}
        for t in range(1, L):
            if t % LEN == 0:
                continue
            pairs.setdefault((int(w[t - 1]), int(w[t])), []).append(t)
        H = torch.zeros(L, n, device=dev, dtype=torch.float16); H.scatter_(1, allst[:, i], 1.0)
        ovs = []
        for ts in pairs.values():
            if len(ts) >= 2:
                a, c = ts[0], ts[1]
                ovs.append(float((H[a] * H[c]).sum()) / k)
        out.append(np.mean(ovs) if ovs else float("nan"))
        del H
    return float(np.nanmean(out))


def show(name, rel, extra=""):
    rel = np.array(rel)
    print(f"  {name:24s} {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}){extra}", flush=True)


for U, b in ((10, 3), (10, 5), (10, None)):
    print(f"\nU = {U}, b = {b or 'random'}", flush=True)
    st, _ = store(U, b, False)
    show("standard", sl.reliability(st, dev), f"; repeated-bigram token overlap {bigram_merge(st):.3f}")
    r = sleep(st, 300)
    show("standard + sleep 300", sl.reliability(st, dev), f"; removed {r:.4f}")
    del st; torch.cuda.empty_cache()
    st, stats = store(U, b, True)
    show("comparator", sl.reliability(st, dev),
         f"; flags {stats['flag'] / max(1, stats['judged']):.4f}; repeated-bigram token overlap {bigram_merge(st):.3f}")
    r = sleep(st, 300)
    show("comparator + sleep 300", sl.reliability(st, dev), f"; removed {r:.4f}")
    del st; torch.cuda.empty_cache()
