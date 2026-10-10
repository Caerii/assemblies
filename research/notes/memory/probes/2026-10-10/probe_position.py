"""Exploratory (not registered): why do captured dreams jump between sequences KEEPING the position
(probe_dreams: 155:8 -> 163:8, 163:9 -> 169:9, 212:13 -> 213:13)? Two readings:
  CODE    tokens at the same position of different sequences share neurons in a HEALTHY store (a
          position signature refraction or the sequence-start schedule writes), and captures follow it.
  ZIPPER  captures propagate: once token (q, e) is written onto (q', e'), recurrence drives the next
          write (q, e+1) toward (q', e'+1), so captures run in diagonals of constant offset e' - e,
          the offset set by where the first capture landed; position 0 is special only if first
          captures land there.
Measured on the same brains, (10000, 75, 0.48), tau 67, subjects 980-999:
  healthy U = 10   overlap of different-word, different-sequence token pairs at offset 0 vs others
  captured U = 50  captured pairs (different words, overlap >= 0.3): the offset distribution against
                   all different-word cross-sequence pairs; the ZIPPER ratio P(next pair captured |
                   pair captured) / P(pair captured); which way captures point in write order
  dreams           in the captured store, the offset of every settled jump between sequences"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg
from research.experiments import memory_sleep as sl
from research.experiments import memory_threshold_law as tl

dev = "cuda"
seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
B = len(seeds)
LEN = rg.LENGTH
CAPTURE, SETTLED = 0.3, 0.5
spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5)}


def overlaps(st, b):
    """token-by-token overlap / k [L, L] of brain b, and per token its word, sequence, position"""
    allst = st["allst"]
    L = allst.shape[0]
    H = torch.zeros(L, n, device=dev)
    H.scatter_(1, allst[:, b], 1.0)
    O = (H @ H.T) / k
    t = torch.arange(L, device=dev)
    return O, st["wordof"][:, b], t // LEN, t % LEN


def pair_masks(w, q, e):
    diffw = w.view(-1, 1) != w.view(1, -1)
    diffq = q.view(-1, 1) != q.view(1, -1)
    off = e.view(1, -1) - e.view(-1, 1)            # e' - e for pair (row t, column t')
    return diffw & diffq, off


def healthy(st):
    same, other = [], []
    by_dq = {d: [] for d in (1, 2, 4, 8)}
    for b in range(B):
        O, w, q, e = overlaps(st, b)
        m, off = pair_masks(w, q, e)
        same.append(float(O[m & (off == 0)].mean()))
        other.append(float(O[m & (off != 0)].mean()))
        dq = q.view(1, -1) - q.view(-1, 1)
        for d in by_dq:
            by_dq[d].append(float(O[m & (off == 0) & (dq == d)].mean()))
    print(f"\nHEALTHY U = 10: overlap of different-word cross-sequence pairs at offset 0 {np.mean(same):.4f} "
          f"vs other offsets {np.mean(other):.4f} (ratio {np.mean(same) / np.mean(other):.2f})", flush=True)
    print("  at offset 0, by sequence distance: " + "  ".join(f"{d}: {np.mean(v):.4f}" for d, v in by_dq.items()),
          flush=True)


def captured(st):
    hist_cap = np.zeros(2 * LEN - 1)
    hist_all = np.zeros(2 * LEN - 1)
    z_num = z_den = base_num = base_den = 0
    for b in range(B):
        O, w, q, e = overlaps(st, b)
        m, off = pair_masks(w, q, e)
        cap = (O >= CAPTURE) & m
        upper = torch.triu(torch.ones_like(cap), 1)
        cu = cap & upper
        mu = m & upper
        hist_cap += np.bincount((off[cu] + LEN - 1).cpu().numpy(), minlength=2 * LEN - 1)
        hist_all += np.bincount((off[mu] + LEN - 1).cpu().numpy(), minlength=2 * LEN - 1)
        # zipper: pair (t, t') captured -> is (t+1, t'+1) captured? (both successors inside their sequences)
        nxt = torch.zeros_like(cap)
        nxt[:-1, :-1] = cap[1:, 1:]
        ok = torch.zeros_like(cap)
        ok[:-1, :-1] = ((e[:-1] < LEN - 1).view(-1, 1) & (e[:-1] < LEN - 1).view(1, -1))
        ok &= m & upper
        valid_next = torch.zeros_like(cap)
        valid_next[:-1, :-1] = m[1:, 1:]
        ok &= valid_next
        z_num += int((cap & ok & nxt).sum())
        z_den += int((cap & ok).sum())
        base_num += int((cap[1:, 1:] & m[1:, 1:] & upper[1:, 1:]).sum())
        base_den += int((m[1:, 1:] & upper[1:, 1:]).sum())
    rate = hist_cap / np.maximum(hist_all, 1)
    base = hist_cap.sum() / hist_all.sum()
    print(f"\nCAPTURED U = 50: {int(hist_cap.sum())} captured pairs over {B} brains "
          f"(rate {base:.5f} of different-word cross-sequence pairs)", flush=True)
    print("  capture rate by offset e' - e, relative to the mean (offset: x):", flush=True)
    print("    " + "  ".join(f"{o}: {rate[o + LEN - 1] / base:.2f}" for o in range(-6, 7)), flush=True)
    print(f"  share of captures at offset 0: {hist_cap[LEN - 1] / hist_cap.sum():.3f} "
          f"(share of pairs at offset 0: {hist_all[LEN - 1] / hist_all.sum():.3f})", flush=True)
    zp, bp = z_num / max(1, z_den), base_num / max(1, base_den)
    print(f"  ZIPPER: P(next pair captured | pair captured) {zp:.3f} vs P(pair captured) {bp:.5f} "
          f"-> ratio {zp / max(bp, 1e-9):.0f}", flush=True)


def dream_offsets(st, dreams=100, steps=12):
    allst = st["allst"]
    L = allst.shape[0]
    gen = torch.Generator(device=dev).manual_seed(4242)
    offs = []
    area, fib = st["mem"].area, st["mem"].fiber
    for _ in range(dreams):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.argsort(torch.rand(B, n, device=dev, generator=gen), dim=1)[:, :k]
        prev = None
        for _ in range(steps):
            x = area.project(1, [fib], freeze=True, mask_bias=False).long()
            hot = torch.zeros(B, n, device=dev)
            hot.scatter_(1, x, 1.0)
            o = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2) / k
            v, i = o.max(0)
            cur = [(int(i[b]), float(v[b]) >= SETTLED) for b in range(B)]
            if prev is not None:
                for (t0, s0), (t1, s1) in zip(prev, cur):
                    if s0 and s1 and t0 // LEN != t1 // LEN:
                        offs.append(t1 % LEN - t0 % LEN)
            prev = cur
    offs = np.array(offs)
    print(f"\nDREAMS in the captured store: {len(offs)} settled jumps between sequences; offset "
          f"(position after - before): 0: {np.mean(offs == 0):.3f}, +1: {np.mean(offs == 1):.3f}, "
          f"-1: {np.mean(offs == -1):.3f}, |offset| >= 2: {np.mean(np.abs(offs) >= 2):.3f}", flush=True)
    vals, cnt = np.unique(offs, return_counts=True)
    print("  full: " + " ".join(f"{a}:{c}" for a, c in zip(vals, cnt)), flush=True)


h = sl.build_store(spec, 10, seeds, dev)
healthy(h)
del h
torch.cuda.empty_cache()
st = sl.build_store(spec, 50, seeds, dev)
captured(st)
dream_offsets(st)
