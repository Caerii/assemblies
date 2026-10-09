"""Exploratory (not registered): can a PLAUSIBLE circuit do the separation Amendment 49 registers?
Amendment 49's check compares each previewed write with every stored token -- an oracle no brain
has. Two hippocampal mechanisms that use only signals local to the circuit are tested here:

  ach g        CHOLINERGIC ENCODING MODE (Hasselmo): while writing, recurrent synapses are
               scaled by g < 1 (acetylcholine presynaptically suppresses CA3 recurrents during
               encoding), so stored assemblies pull less on the state being written. Global: no
               detector.
  mismatch th  A CA1-STYLE COMPARATOR (Lisman & Grace; Hasselmo): before each write two
               projections from the current state are compared -- recall alone (recurrence only,
               no input: what memory PREDICTS) and the full write (input + recurrence). If they
               agree on >= th of their winners, memory is capturing the write; the predicted
               neurons are then inhibited for this write only. Needs no stored-token lookup and
               no word labels: one recall pass and one overlap.
  mismatch-ach th  the same detector, but a detection switches recurrence OFF for that write
               (a phasic cholinergic response to the familiarity signal) instead.
  oracle       Amendment 49's label-free check (0.3), for reference.

For every rule: masked replay reliability, captured share, interventions, and the detector's
agreement with the oracle on the writes it judges (hit rate: oracle-flagged writes the rule
flags; false alarms: unflagged writes it flags). (10000, 75, 0.48), tau = 67 (judged cell),
seeds 980-999, U = 10 and 50."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_load_law as ml
from research.experiments import memory_load_drift as md
from research.experiments import memory_write_separation as ws
from research.experiments.seq_capacity_scaling import to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._hashed import StimulusFiber

seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
USES = (10, 50)
RULES = (("standard", None), ("oracle", 0.3), ("ach", 0.5), ("ach", 0.25), ("ach", 0.0),
         ("mismatch", 0.3), ("mismatch", 0.5), ("mismatch-ach", 0.3))
if len(sys.argv) > 1:
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
dev = "cuda"
LEN = rg.LENGTH
B = len(seeds)
ar = torch.arange(B, device=dev)


class Scaled:
    """A fiber whose drive is scaled by g (writes and everything else pass through)."""

    def __init__(self, f, g):
        self.f, self.g = f, g

    def contribute(self, drive, rows):
        tmp = torch.zeros_like(drive)
        self.f.contribute(tmp, rows)
        drive += self.g * tmp

    def __getattr__(self, name):
        return getattr(self.f, name)


def hot_of(x):
    h = torch.zeros(B, n, device=dev)
    h.scatter_(1, x.long(), 1.0)
    return h


def store(mem, elem_seeds, stored, rule, par, stats):
    area, fib = mem.area, mem.fiber
    area.inhibit()
    states = []
    for e, sds in enumerate(elem_seeds):
        stim = StimulusFiber(sds, k, n, p, beta=mem.beta, w_max=mem.w_max, norm_init=mem.norm_init,
                             max_rounds=1, device=dev)
        rec = Scaled(fib, par) if rule == "ach" else fib
        added, flag = None, torch.zeros(B, dtype=torch.bool, device=dev)
        if e > 0 and stored:
            keep_b, keep_w = area.bias.clone(), area.winners.clone()
            area.bias.mul_(area.bias_decay)
            P = area.project(1, [rec, stim], freeze=True, mask_bias=False)
            area.winners = keep_w.clone()
            R = area.project(1, [fib], freeze=True, mask_bias=False) if rule.startswith("mismatch") else None
            area.bias, area.winners = keep_b, keep_w
            hP = hot_of(P)
            T = torch.stack(stored)
            ov = torch.gather(hP.unsqueeze(0).expand(T.shape[0], B, n), 2, T).sum(2) / k
            oracle = (ov >= 0.3).any(0)
            if rule == "oracle":
                flag = oracle
                if bool(flag.any()):
                    added = torch.zeros(B, n, device=dev)
                    for t, b in (ov >= 0.3).nonzero().tolist():
                        added[b, T[t, b]] = 1e4
            elif rule.startswith("mismatch"):
                fam = torch.gather(hP, 1, R.long()).sum(1) / k
                flag = fam >= par
                stats["fam"].append(fam.cpu())
                if rule == "mismatch" and bool(flag.any()):
                    added = torch.zeros(B, n, device=dev)
                    added.scatter_(1, R.long(), 1e4 * flag.float().view(-1, 1).expand(-1, k).contiguous())
            stats["oracle"] += int(oracle.sum()); stats["flag"] += int(flag.sum())
            stats["hit"] += int((flag & oracle).sum()); stats["judged"] += B
        fibers = [rec, stim]
        if rule == "mismatch-ach" and bool(flag.any()):
            # recurrence off for the flagged brains only: zero their rows' recurrent drive
            fibers = [Scaled(fib, (~flag).float().view(-1, 1)), stim]
        if added is not None:
            area.bias += added
        win = area.project(1, fibers, defer_overflow=True)
        if added is not None:
            area.bias -= added * area.bias_decay
        states.append(win.clone()); stored.append(win.clone())
    mem.items += 1
    return torch.stack(states)


for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    print(f"\nU = {U} (V {V}, {L * B} writes)", flush=True)
    for rule, par in RULES:
        mem = ws.build(n, k, p, tau, seeds, dev)
        stored, seqs = [], []
        stats = {"oracle": 0, "flag": 0, "hit": 0, "judged": 0, "fam": []}
        for q in range(M):
            es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
                  for e in range(LEN)]
            seqs.append(store(mem, es, stored, rule, par, stats))
        mem.area.check_overflow()
        allst = torch.stack(stored).long()
        area = mem.area
        rel = torch.zeros(B, device=dev)
        for q in range(M):
            area.bias = torch.zeros(B, n, device=dev)
            area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
            alive = torch.ones(B, dtype=torch.bool, device=dev)
            for j in range(1, LEN):
                x = area.project(1, [mem.fiber], freeze=True, mask_bias=False)
                h = hot_of(x)
                alive &= wordof[torch.gather(h.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
            rel += alive.float()
        rel = (rel / M).cpu().numpy()
        cap = []
        for i in range(B):
            H = torch.zeros(L, n, device=dev, dtype=torch.float16); H.scatter_(1, allst[:, i], 1.0)
            O = (H @ H.T).float() / k; O.fill_diagonal_(0.0)
            diffw = wordof[:, i].view(-1, 1) != wordof[:, i].view(1, -1)
            cap.append(float(((O >= 0.5) & diffw).any(1).float().mean()))
            del H, O, diffw
        hit = stats["hit"] / max(1, stats["oracle"])
        fa = (stats["flag"] - stats["hit"]) / max(1, stats["judged"] - stats["oracle"])
        fam = torch.cat(stats["fam"]).numpy() if stats["fam"] else None
        name = rule if par is None else f"{rule} {par}"
        print(f"  {name:16s} reliability {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}); captured "
              f"{np.mean(cap):.4f}; flagged {stats['flag'] / max(1, stats['judged']):.4f} of writes; oracle-flagged "
              f"{stats['oracle'] / max(1, stats['judged']):.4f}; hit {hit:.2f}, false alarm {fa:.4f}"
              + (f"; familiarity mean {fam.mean():.3f} p99 {np.quantile(fam, 0.99):.3f}" if fam is not None else ""),
              flush=True)
        del mem, seqs, allst, stored
        torch.cuda.empty_cache()
