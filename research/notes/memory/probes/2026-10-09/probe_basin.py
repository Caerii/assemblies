"""Exploratory (not registered): a LOGIT LENS on replay, and how deep a spurious attractor's basin is.
Random-successor reuse at U = 30, 40, 50, 60 uses per word, (10000, 75, 0.48), tau = 67, rho = 0.05
(Amendment 45's cell, judged; as probe_collapse.py and probe_control.py). Seeds 980-999.

Logit lens. Every replay step's net drive is read as a score for every stored token: the token's
logit is the mean drive over its k neurons, in units of the k-WTA threshold (the k-th largest
drive), so 1.0 means "as driven as the weakest winner". A word's logit is the max over its tokens.
Read per step: whether the correct next word has the top logit, its rank, and its margin over the
best other word.

Depth candidates, from the EVEN-indexed sequences replayed masked (the standard read):
  gap      at each failing sequence's first wrong step, the attractor's logit minus the correct
           word's logit -- how far the false state out-drives the true one
  capture  the share of failed replays whose state 8 steps after the cue overlaps the attractor
           (the brain's k commonest end-state neurons) by >= 0.3
  hold     the attractor's overlap with itself after 5 masked steps started from it
  margin   the correct word's mean margin over successful steps (the healthy transitions' slack)
Gains, from the ODD-indexed sequences: session adaptation (charge c x raw drive, decay exp(-1/T),
T = 50, kept across sequences) at c = 0.03, 0.1, 0.2, paired against masked. Which candidate
predicts a failing brain's gain beyond its masked reliability?

A second cell: python probe_basin.py n k p tau U... (log probe_basin_<n>.log)."""
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

seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
USES = (30, 40, 50, 60)
if len(sys.argv) > 1:   # n k p tau U... (a second cell)
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
B = len(seeds)
dev = "cuda"
ar = torch.arange(B, device=dev)
CHARGES = (0.03, 0.1, 0.2)
rows = []


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / rg.LENGTH)); L = M * rg.LENGTH; V = round(L / U)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX,
                         norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=dev,
                         bias_decay=math.exp(-1 / tau))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(rg.LENGTH)])[0] for q in range(M)]
    allst = torch.cat(seqs, 0).long()                                   # [L, B, k]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()   # [L, B]
    area = mem.area
    saved = area.bias.clone()

    def lens(drive):
        """token logits [L, B] and word logits [B, V], in units of the k-WTA threshold."""
        thr = torch.topk(drive, k, dim=1).values[:, -1].clamp_min(1e-9)
        tok = torch.gather(drive.unsqueeze(0).expand(L, B, n), 2, allst).mean(2) / thr
        wl = torch.full((B, V), -1e9, device=dev).scatter_reduce(1, wordof.T, tok.T, "amax")
        return tok, wl, thr

    def replay(q, bias, charge=0.0, decay=1.0, attractor=None, log=None):
        st = seqs[q]
        area.bias = bias
        area.winners = torch.stack([md.cue(st[0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        end, first_gap = None, torch.full((B,), float("nan"), device=dev)
        for j in range(1, rg.LENGTH):
            x, drive = area.project(1, [mem.fiber], freeze=True, mask_bias=False, return_drive=True)
            raw = drive + bias
            if charge:
                bias.mul_(decay)
                bias.scatter_add_(1, x, torch.gather(raw, 1, x) * charge)
            tok, wl, thr = lens(drive)
            want = wordof[q * rg.LENGTH + j]
            wc = wl[ar, want]
            other = wl.scatter(1, want.view(-1, 1), -1e9).max(1).values
            hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
            ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
            ok = wordof[ov.argmax(0), ar] == want
            if attractor is not None:
                al = torch.gather(drive, 1, attractor).mean(1) / thr
                newly = alive & ~ok
                first_gap = torch.where(newly, al - wc, first_gap)
                if log is not None:
                    log.setdefault(j, []).append(torch.stack([al, wc, other, alive.float()], 1))
            if log is not None and attractor is None:
                log.setdefault("steps", []).append(torch.stack([(wc > other).float(), wc - other,
                                                                (wl > wc.view(-1, 1)).sum(1).float()], 1))
            alive &= ok
            if j == 8:
                end = x.clone()
        return alive, end, first_gap

    zero = torch.zeros(B, n, device=dev)
    even, odd = list(range(0, M, 2)), list(range(1, M, 2))
    # pass 1 (even, masked): the attractor, and the logit lens on every step
    counts = torch.zeros(B, n, device=dev)
    ends, lens_log = [], {}
    rel_even = torch.zeros(B, device=dev)
    for q in even:
        ok, end, _ = replay(q, zero.clone(), log=lens_log)
        rel_even += ok.float()
        counts.scatter_add_(1, end, (~ok).float().view(-1, 1).expand(-1, k).contiguous())
        ends.append((ok, end))
    rel_even /= len(even)
    A = torch.topk(counts, k, dim=1).indices
    steps = torch.stack(lens_log["steps"])                              # [S, B, 3]
    top1, margin_all, rank = steps[..., 0].mean(0), steps[..., 1], steps[..., 2].mean(0)
    # pass 2 (even, masked): the gap at the first wrong step
    gaps = [[] for _ in range(B)]
    curve = {}
    for q in even:
        _, _, g = replay(q, zero.clone(), attractor=A, log=curve)
        for i in range(B):
            if not math.isnan(float(g[i])):
                gaps[i].append(float(g[i]))
    gap = torch.tensor([np.mean(v) if v else float("nan") for v in gaps])
    capA = torch.zeros(B, n, device=dev); capA.scatter_(1, A, 1.0)
    fails = torch.zeros(B, device=dev); caught = torch.zeros(B, device=dev)
    for ok, end in ends:
        o = torch.gather(capA, 1, end).sum(1) / k
        fails += (~ok).float(); caught += ((~ok) & (o >= 0.3)).float()
    capture = (caught / fails.clamp_min(1)).cpu()
    # hold: start from the attractor itself
    area.bias = zero.clone()
    area.winners = A.clone()
    for _ in range(5):
        x = area.project(1, [mem.fiber], freeze=True, mask_bias=False)
    hold = (torch.gather(capA, 1, x.long()).sum(1) / k).cpu()
    # healthy slack: the correct word's margin on successful steps of the even half
    marg = margin_all.clone(); marg[steps[..., 0] < 0.5] = float("nan")
    slack = torch.nanmean(marg, 0).cpu()
    # pass 3 (odd): masked and adapted, paired
    res = {}
    for c in (0.0,) + CHARGES:
        b = zero.clone(); acc = torch.zeros(B, device=dev)
        for q in odd:
            acc += replay(q, b, charge=c, decay=math.exp(-1 / 50))[0].float()
        res[c] = (acc / len(odd)).cpu()
    area.bias = saved
    masked = res[0.0]
    print(f"\nU = {U} (M {M}, V {V}): masked odd {float(masked.mean()):.3f}; "
          + "  ".join(f"c {c}: {float(res[c].mean()):.3f}" for c in CHARGES), flush=True)
    print(f"  lens: correct word top-1 per step {float(top1.mean()):.3f}; mean rank of correct word "
          f"{float(rank.mean()):.2f} of {V}", flush=True)
    for i in range(B):
        r = {"U": U, "seed": seeds[i], "masked": float(masked[i]), "even": float(rel_even[i]),
             "gap": float(gap[i]), "capture": float(capture[i]), "hold": float(hold[i]),
             "slack": float(slack[i]), "top1": float(top1[i]), "rank": float(rank[i])}
        for c in CHARGES:
            r[f"g{c}"] = float(res[c][i] - masked[i])
        rows.append(r)
        print(f"  brain {seeds[i]}: masked {r['masked']:.2f} (even {r['even']:.2f})  gain c.03 {r['g0.03']:+.2f} "
              f"c.1 {r['g0.1']:+.2f} c.2 {r['g0.2']:+.2f} | gap {r['gap']:+.3f} capture {r['capture']:.2f} "
              f"hold {r['hold']:.2f} slack {r['slack']:.3f} top1 {r['top1']:.2f} rank {r['rank']:.1f}", flush=True)
    # the capture curve: attractor vs correct-word logit by step, brains collapsed vs not
    col = (rel_even < 0.2)
    for name, sel in (("collapsed", col), ("other", ~col)):
        if int(sel.sum()) == 0:
            continue
        line = []
        for j in (1, 2, 4, 8, 12, 15):
            t = torch.cat(curve[j], 0).view(-1, B, 4)[:, sel]
            line.append(f"j{j} A {float(t[..., 0].mean()):.2f}/w {float(t[..., 1].mean()):.2f}")
        print(f"  logits by step ({name}, {int(sel.sum())} brains): " + "  ".join(line), flush=True)
    del mem, seqs, allst
    torch.cuda.empty_cache()

# ---- which depth candidate predicts a failing brain's gain beyond its masked reliability?
fail = [r for r in rows if r["masked"] < 0.5 and not math.isnan(r["gap"])]
print(f"\nfailing brain-arms (masked < 0.5): {len(fail)}", flush=True)
m = np.array([r["masked"] for r in fail])
for c in CHARGES if len(fail) >= 3 else ():
    g = np.array([r[f"g{c}"] for r in fail])
    coef = np.polyfit(m, g, 1)
    res_g = g - np.polyval(coef, m)
    out = [f"masked rho {spearman(m, g):+.2f}"]
    for d in ("gap", "capture", "hold", "slack", "top1", "U"):
        x = np.array([r[d] for r in fail], dtype=float)
        out.append(f"{d} rho {spearman(x, g):+.2f} (partial {spearman(x, res_g):+.2f})")
    print(f"gain at c {c}: " + "; ".join(out), flush=True)
