"""Exploratory (not registered): a SLOW EXCITABILITY TRACE as an allocation primitive. Biology biases
allocation two ways: after-hyperpolarisation pushes recently active cells away (seconds; this
model's refraction, tau = 67 rounds), and CREB-driven excitability pulls a new memory onto cells
allocated in the last hours, linking memories close in time (Cai et al. 2016; Rashid et al. 2016).
Here a trace E is added: after every write each winner gains a * its raw drive, E decaying by
exp(-1 / tau_link) per write, tau_link >> tau; E RAISES drive at the next writes (bias - E). Net
kernel: recent winners pushed away, older ones pulled toward -- a slowly drifting context.
An ADDITIVE trace (E += a raw) ran away: at a = 0.1, tau_link = 512 starts sharing a first word
overlapped 0.877 and replay fell to 0.000 (probe_excitability_additive.log; the arms after it were
stopped). The trace is therefore BOUNDED, as CREB-driven excitability is: a winner's E is SET to
max(E, a raw), not incremented, so no cell's boost exceeds a fraction a of a winner's drive.

Questions, at (10000, 75, 0.48), tau = 67 (judged cell), seeds 980-999:
  starts    sequences sharing a first word: mean overlap of their first tokens (the stimulus alone
            writes them; with no context they nearly coincide and the cue is ambiguous)
  reliability  masked replay of every sequence, word-level
  linking   replay continued 3 steps past a sequence's end: does it land in the sequence WRITTEN
            NEXT (tokens 0-3 of q+1, overlap >= 0.3) more than in the one written before (q-1)?
Arms at U = 50 with the comparator (Amendment 50) on, and at U = 10 without it."""
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
ARMS = ((50, True, 0.02, 512), (50, True, 0.05, 512), (50, True, 0.1, 512), (50, True, 0.05, 128),
        (10, False, 0.0, 1), (10, False, 0.05, 512), (10, False, 0.1, 512))
dev = "cuda"
LEN = rg.LENGTH
B = len(seeds)
ar = torch.arange(B, device=dev)


def hot_of(x):
    h = torch.zeros(B, n, device=dev)
    h.scatter_(1, x.long(), 1.0)
    return h


for U, compare, a, tlink in ARMS:
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    mem = ws.build(n, k, p, tau, seeds, dev)
    area, fib = mem.area, mem.fiber
    E = torch.zeros(B, n, device=dev)
    dl = math.exp(-1.0 / tlink)
    stored, seqs, flags = [], [], 0
    for q in range(M):
        area.inhibit()
        states = []
        for e in range(LEN):
            sds = [to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
            stim = StimulusFiber(sds, k, n, p, beta=mem.beta, w_max=mem.w_max, norm_init=mem.norm_init,
                                 max_rounds=1, device=dev)
            added = torch.zeros(B, n, device=dev)
            if a:
                added -= E                                                # excitability raises drive
            if compare and e > 0:
                keep_b, keep_w = area.bias.clone(), area.winners.clone()
                area.bias.add_(added).mul_(area.bias_decay)
                P = area.project(1, [fib, stim], freeze=True, mask_bias=False)
                area.winners = keep_w.clone()
                R = area.project(1, [fib], freeze=True, mask_bias=False)
                area.bias, area.winners = keep_b, keep_w
                flag = (torch.gather(hot_of(P), 1, R.long()).sum(1) / k) >= 0.5
                if bool(flag.any()):
                    added.scatter_add_(1, R.long(), 1e4 * flag.float().view(-1, 1).expand(-1, k).contiguous())
                    flags += int(flag.sum())
            area.bias += added
            win, drive = area.project(1, [fib, stim], defer_overflow=True, return_drive=True)
            area.bias -= added * area.bias_decay
            if a:
                raw = drive + area.bias                                   # drive net of the restored bias
                E.mul_(dl)
                E.scatter_reduce_(1, win, torch.gather(raw, 1, win) * a, "amax")   # BOUNDED: set, not add
            states.append(win.clone()); stored.append(win.clone())
        seqs.append(torch.stack(states))
        mem.items += 1
    mem.area.check_overflow()
    allst = torch.stack(stored).long()
    # starts: first tokens of sequences sharing a first word
    first = allst[0::LEN]                                                 # [M, B, k]
    fw = wordof[0::LEN]                                                   # [M, B]
    ovs = []
    for i in range(B):
        H = torch.zeros(M, n, device=dev); H.scatter_(1, first[:, i], 1.0)
        O = (H @ H.T) / k
        same = (fw[:, i].view(-1, 1) == fw[:, i].view(1, -1)) & ~torch.eye(M, dtype=torch.bool, device=dev)
        if same.any():
            ovs.append(float(O[same].mean()))
    # replay, and continuation past the end
    rel = torch.zeros(B, device=dev); fwd = torch.zeros(B, device=dev); bwd = torch.zeros(B, device=dev); cnt = 0
    for q in range(M):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, LEN):
            x = area.project(1, [fib], freeze=True, mask_bias=False)
            h = hot_of(x)
            alive &= wordof[torch.gather(h.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
        rel += alive.float()
        if 0 < q < M - 1:
            hitf = torch.zeros(B, dtype=torch.bool, device=dev); hitb = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(3):
                x = area.project(1, [fib], freeze=True, mask_bias=False)
                h = hot_of(x)
                for t in range(4):
                    hitf |= torch.gather(h, 1, allst[(q + 1) * LEN + t]).sum(1) / k >= 0.3
                    hitb |= torch.gather(h, 1, allst[(q - 1) * LEN + t]).sum(1) / k >= 0.3
            fwd += (hitf & alive).float(); bwd += (hitb & alive).float(); cnt += 1
    rel = (rel / M).cpu().numpy()
    alive_n = max(1, cnt)
    name = f"U={U} {'comparator' if compare else 'standard  '} E a={a} tau_link={tlink}"
    print(f"{name}: reliability {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}); same-first-word start overlap "
          f"{np.mean(ovs):.3f}; continuation into q+1 {float(fwd.mean()) / alive_n:.3f} vs q-1 {float(bwd.mean()) / alive_n:.3f}"
          f"; comparator flags {flags}", flush=True)
    del mem, seqs, allst, stored, E
    torch.cuda.empty_cache()
