"""Does one hidden margin decide which brains replay, and which a read-time
adaptation can rescue? A logit lens on replay, registered.

Registered in PREREG_refraction_memory.md, Amendment 48.

A LOGIT LENS reads every replay step's net drive as a score for every stored
token: the token's logit is the mean drive over its k neurons in units of the
k-WTA threshold (the k-th largest drive); a word's logit is the max over its
tokens. A brain's SLACK is the correct next word's mean margin over the best
other word, on the steps it is top-1, from its EVEN-indexed sequences replayed
masked (the standard read); its RELATIVE slack x divides by the median slack
of a reference arm at the same cell (U = 10, random successors). Its masked and
adapted reliability (session adaptation c = 0.1, T = 50, as Amendments 46 and
47) come from its ODD-indexed sequences. Exploratory probes found replay a
logistic function of x, at the same position at two cells, and adaptation
moving that position down; brains the masked read leaves at 0.00 are rescued
in order of their slack.

    arms: reference U = 10; recurrence U = 30, 40, 50, 60, 70 (random successors)

    python -m research.runner signal_margin --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_reuse_grammar as rg             # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((7500, 55, 0.6, 68), (12500, 85, 0.42, 74))
SMOKE_CELL = (2000, 60, 0.5, 17)
REFERENCE = 10
USES = (30, 40, 50, 60, 70)
CHARGE, DECAY_T = 0.1, 50
SEEDS = tuple(range(742, 762))
COLLAPSED = 0.2
#: the probes' pooled fits (relative slack x; masked and adapted reliability)
MASKED_FIT, HABIT_FIT = (0.763, 0.040), (0.642, 0.0915)
POSITION = (0.72, 0.81)
SHIFT, MIN_SIDE, MIN_COLLAPSED, RANK, SIZE = 0.06, 5, 8, 0.6, 0.1


def plan(cells, *, smoke=False):
    uses = (REFERENCE, USES[1]) if smoke else (REFERENCE,) + USES
    return [{"n": n, "k": k, "p": p, "tau": tau, "uses": u, "rho": rg.RHO,
             "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n)}
            for n, k, p, tau in cells for u in uses]


def measure(spec, seeds, device):
    """Per brain: slack and per-step top-1 (even half, masked; logit lens) and
    masked and adapted reliability (odd half), word-level."""
    import numpy as np
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / rg.LENGTH))
    L = M * rg.LENGTH
    V = max(8, round(L / spec["uses"]))
    salt = L * 1000 + spec["uses"]
    out = {"L": L, "M": M, "V": V, "slack": [], "top1": [], "masked": [], "habit": []}
    even, odd = list(range(0, M, 2)), list(range(1, M, 2))
    for i0 in range(0, len(seeds), spec["batch"]):
        part = list(seeds[i0:i0 + spec["batch"]])
        B = len(part)
        mem = AssemblyMemory(seeds_for(part), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                             norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4,
                             device=device, bias_decay=math.exp(-1.0 / spec["tau"]))
        words = [rg.walks(sd, M, V, None, salt) for sd in part]
        seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                     for i, sd in enumerate(part)] for e in range(rg.LENGTH)])[0]
                for q in range(M)]
        allst = torch.cat(seqs, 0).long()
        wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device).long()
        ar = torch.arange(B, device=device)
        area = mem.area
        saved = area.bias.clone()

        def replay(q, bias, charge, lens):
            area.bias = bias
            area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, device)
                                        for i, sd in enumerate(part)]).long()
            alive = torch.ones(B, dtype=torch.bool, device=device)
            for j in range(1, rg.LENGTH):
                x, drive = area.project(1, [mem.fiber], freeze=True, mask_bias=False, return_drive=True)
                want = wordof[q * rg.LENGTH + j]
                if lens is not None:
                    thr = torch.topk(drive, k, dim=1).values[:, -1].clamp_min(1e-9)
                    tok = torch.gather(drive.unsqueeze(0).expand(L, B, n), 2, allst).mean(2) / thr
                    wl = torch.full((B, V), -1e9, device=device).scatter_reduce(1, wordof.T, tok.T, "amax")
                    wc = wl[ar, want]
                    other = wl.scatter(1, want.view(-1, 1), -1e9).max(1).values
                    lens.append(torch.stack([(wc > other).float(), wc - other], 1))
                if charge:
                    raw = drive + bias
                    bias.mul_(math.exp(-1.0 / DECAY_T))
                    bias.scatter_add_(1, x, torch.gather(raw, 1, x) * charge)
                hot = torch.zeros(B, n, device=device)
                hot.scatter_(1, x.long(), 1.0)
                ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
                alive &= wordof[ov.argmax(0), ar] == want
            return alive

        lens = []
        for q in even:
            replay(q, torch.zeros(B, n, device=device), 0.0, lens)
        st = torch.stack(lens)                                          # [S, B, 2]
        win = st[..., 0] > 0.5
        slack = torch.where(win, st[..., 1], torch.zeros_like(st[..., 1])).sum(0) / win.sum(0).clamp_min(1)
        out["slack"] += slack.tolist()
        out["top1"] += st[..., 0].mean(0).tolist()
        for mode, c in (("masked", 0.0), ("habit", CHARGE)):
            bias = torch.zeros(B, n, device=device)
            done = torch.zeros(B, device=device)
            for q in odd:
                done += replay(q, bias, c, None).float()
            out[mode] += (done / len(odd)).tolist()
        area.bias = saved
        del mem, seqs, allst
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][str(spec["uses"])] = m
        print(f"({n}, {k}, {p}) U={spec['uses']}: slack {sum(m['slack']) / len(m['slack']):.3f}  "
              f"masked {sum(m['masked']) / len(m['masked']):.3f}  habit {sum(m['habit']) / len(m['habit']):.3f}",
              flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def logistic(x, mid, width):
    return 1.0 / (1.0 + math.exp(-(x - mid) / width))


def fit(xs, ys):
    """(midpoint, width) of the least-squares logistic through (x, y): a grid
    (midpoint 0.30-1.10 by 0.01, width 0.005-0.200 by 0.005), then refined
    around its best point (midpoint by 0.001, width by 0.0005)."""
    def err(mid, w):
        return sum((logistic(x, mid, w) - y) ** 2 for x, y in zip(xs, ys))
    _, mid, w = min((err(0.30 + 0.01 * i, 0.005 + 0.005 * j), 0.30 + 0.01 * i, 0.005 + 0.005 * j)
                    for i in range(81) for j in range(40))
    _, mid, w = min((err(mid + 0.001 * i, w + 0.0005 * j), mid + 0.001 * i, w + 0.0005 * j)
                    for i in range(-10, 11) for j in range(-9, 10) if w + 0.0005 * j > 0)
    return round(mid, 4), round(w, 5)


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def evaluate(observations):
    """Amendment 48's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("M1", "M2", "M3", "M4")
    out: dict = {"bars": {}, "cells": {}}
    arms = {str(REFERENCE)} | {str(u) for u in USES}
    if not all(c[:3] in cells and arms <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    for n, k, p, _ in CELLS:
        cell = cells[(n, k, p)]["arms"]
        ref = sorted(cell[str(REFERENCE)]["slack"])
        ref = (ref[(len(ref) - 1) // 2] + ref[len(ref) // 2]) / 2
        x, masked, habit = [], [], []
        for u in USES:
            a = cell[str(u)]
            x += [s / ref for s in a["slack"]]
            masked += a["masked"]
            habit += a["habit"]
        sides = min(sum(m < 0.5 for m in masked), sum(m >= 0.5 for m in masked))
        m0, w0 = fit(x, masked)
        m1, w1 = fit(x, habit)
        col = [i for i, m in enumerate(masked) if m < COLLAPSED]
        rank = spearman([x[i] for i in col], [habit[i] for i in col]) if len(col) >= 3 else None
        obs = sum(habit[i] for i in col) / len(col) if col else None
        pred = sum(logistic(x[i], *HABIT_FIT) for i in col) / len(col) if col else None
        info = {"reference_slack": ref, "masked_mid": m0, "masked_width": w0, "habit_mid": m1,
                "habit_width": w1, "sides": sides, "collapsed": len(col), "rank": rank,
                "collapsed_habit": obs, "collapsed_predicted": pred}
        out["cells"][f"{n}/{k}/{p:g}"] = info
        fitted = sides >= MIN_SIDE
        ok["M1"] &= fitted and POSITION[0] <= m0 <= POSITION[1]
        ok["M2"] &= fitted and m1 <= m0 - SHIFT
        ok["M3"] &= len(col) >= MIN_COLLAPSED and rank is not None and rank >= RANK
        ok["M4"] &= (len(col) >= MIN_COLLAPSED and obs is not None and pred is not None
                     and abs(obs - pred) <= SIZE)
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Signal margin", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 48 is registered on seeds 742..761")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.signal-margin", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": rg.LENGTH, "charge": CHARGE, "decay_t": DECAY_T,
                    "strength": ml.STRENGTH, "match": ml.MATCH, "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
