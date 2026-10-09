"""Exploratory (not registered): PATTERN SEPARATION AT WRITE -- does refusing to lay a new token onto
another word's assembly stop the capture cascade, and the recurrence collapse with it?
(Conjecture conj:separation in the notebook.) Random-successor reuse, (10000, 75, 0.48), tau = 67,
rho = 0.05 (judged cell; seeds 980-999), U = 40 and 50.

The store loop is AssemblyMemory.store_sequence's, written out: per sequence the area is
inhibited; per element a stimulus fiber is built and one round projects [recurrence, stimulus]
with the write. SEPARATED: before each element's write, the same round is PREVIEWED frozen (the
refraction bias decayed as the write would decay it); if the previewed state shares >= 0.5 of its
neurons with any stored token of ANOTHER word, those tokens' neurons are inhibited (bias + 1e4)
for this write only, and the real write runs. An equivalence check first: with separation off,
the written loop reproduces store_sequence bit for bit.

Read: interventions (share of writes), residual captures in the store, largest cluster, and
masked replay reliability per brain, standard store vs separated store, same brains.
CAPTURE=0.3 (environment) separates at the Hebbian limit's overlap instead (probe_separate_03.log);
LABEL_FREE=1 refuses overlap with ANY stored token, knowing no word labels (probe_separate_lf.log)."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_threshold_law as tl, memory_load_law as ml
from research.experiments import memory_load_drift as md, memory_pattern_efficiency as pe
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from neural_assemblies.core.torch_engine._hashed import StimulusFiber

seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
USES = (40, 50)
if len(sys.argv) > 1:
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
dev = "cuda"
LEN = rg.LENGTH
CAPTURE, BIG = float(os.environ.get("CAPTURE", 0.5)), 1e4
LABEL_FREE = os.environ.get("LABEL_FREE") == "1"   # refuse ANY stored token, not only another word's
print(f"capture threshold {CAPTURE}; label-free {LABEL_FREE}", flush=True)


def build(nn, kk, pp, tt, sds):
    return AssemblyMemory(seeds_for(sds), nn, kk, pp, beta=round(tl.theta(nn, kk, pp), 5), w_max=pe.W_MAX,
                          norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=dev,
                          bias_decay=math.exp(-1 / tt))


def store(mem, elem_seeds, word_ids, stored, stored_words, separate):
    """One sequence by the written-out loop. elem_seeds [LEN][B]; word_ids [LEN][B] (tensor);
    stored: list of [B, k] tokens so far; returns (states, interventions)."""
    area, fib = mem.area, mem.fiber
    B = mem.B if hasattr(mem, "B") else len(elem_seeds[0])
    area.inhibit()
    states, hits = [], 0
    for e, sds in enumerate(elem_seeds):
        stim = StimulusFiber(sds, mem.k, mem.n, mem.p, beta=mem.beta, w_max=mem.w_max,
                             norm_init=mem.norm_init, max_rounds=1, device=dev)
        added = None
        if separate and stored:
            keep_b, keep_w = area.bias.clone(), area.winners.clone()
            area.bias.mul_(area.bias_decay)
            x = area.project(1, [fib, stim], freeze=True, mask_bias=False)
            area.bias, area.winners = keep_b, keep_w
            hot = torch.zeros(B, mem.n, device=dev); hot.scatter_(1, x.long(), 1.0)
            T = torch.stack(stored)                                         # [T, B, k]
            ov = torch.gather(hot.unsqueeze(0).expand(T.shape[0], B, mem.n), 2, T).sum(2) / mem.k
            other = (torch.ones_like(ov, dtype=torch.bool) if LABEL_FREE
                     else torch.stack(stored_words) != word_ids[e].view(1, -1))  # [T, B]
            bad = (ov >= CAPTURE) & other
            if bool(bad.any()):
                added = torch.zeros(B, mem.n, device=dev)
                for t, b in bad.nonzero().tolist():
                    added[b, T[t, b]] = BIG
                hits += int(bad.any(0).sum())
                area.bias += added
        win = area.project(1, [fib, stim], defer_overflow=True)
        if added is not None:
            area.bias -= added * area.bias_decay
        states.append(win.clone())
        stored.append(win.clone()); stored_words.append(word_ids[e].clone())
    mem.items += 1
    return torch.stack(states), hits


# equivalence: the written loop with separation off == store_sequence
eq_seeds = list(range(980, 984))
a, b = build(2000, 60, 0.5, 17, eq_seeds), build(2000, 60, 0.5, 17, eq_seeds)
for q in range(3):
    es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{q}-{e}", "A")) for sd in eq_seeds] for e in range(LEN)]
    sa = a.store_sequence(es)[0]
    sb, _ = store(b, es, [torch.zeros(len(eq_seeds), device=dev)] * LEN, [], [], False)
    assert torch.equal(torch.sort(sa, -1).values, torch.sort(sb, -1).values), "written loop != store_sequence"
print("equivalence: the written loop reproduces store_sequence (3 sequences, 4 brains)", flush=True)
del a, b

B = len(seeds)
ar = torch.arange(B, device=dev)
for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    res = {}
    for arm in ("standard", "separated"):
        mem = build(n, k, p, tau, seeds)
        stored, stored_words, seqs, hits = [], [], [], 0
        for q in range(M):
            es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
                  for e in range(LEN)]
            st, h = store(mem, es, [wordof[q * LEN + e] for e in range(LEN)], stored, stored_words, arm == "separated")
            seqs.append(st); hits += h
        allst = torch.stack(stored).long()                                   # [L, B, k]
        area = mem.area
        saved = area.bias.clone()
        rel = torch.zeros(B, device=dev)
        for q in range(M):
            area.bias = torch.zeros(B, n, device=dev)
            area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
            alive = torch.ones(B, dtype=torch.bool, device=dev)
            for j in range(1, LEN):
                x = area.project(1, [mem.fiber], freeze=True, mask_bias=False)
                hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
                alive &= wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
            rel += alive.float()
        rel /= M
        area.bias = saved
        dup, clus = [], []
        for i in range(B):
            H = torch.zeros(L, n, device=dev, dtype=torch.float16); H.scatter_(1, allst[:, i], 1.0)
            O = (H @ H.T).float() / k; O.fill_diagonal_(0.0)
            diffw = wordof[:, i].view(-1, 1) != wordof[:, i].view(1, -1)
            dup.append(float(((O >= 0.5) & diffw).any(1).float().mean()))
            clus.append(int((O >= 0.5).sum(1).max()))
            del H, O, diffw
        res[arm] = (rel.cpu().numpy(), np.array(dup), np.array(clus), hits)
        del mem, seqs, allst, stored
        torch.cuda.empty_cache()
    print(f"\nU = {U} (V {V}):", flush=True)
    for arm, (rel, dup, clus, hits) in res.items():
        print(f"  {arm:9s}: reliability {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum()}); captured tokens "
              f"{dup.mean():.4f}; largest cluster {clus.max()}; interventions {hits} of {L * B} writes", flush=True)
    r0, r1 = res["standard"][0], res["separated"][0]
    print("  per brain standard -> separated: " + "  ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(r0, r1)), flush=True)
