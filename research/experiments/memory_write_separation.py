"""Does pattern separation at WRITE prevent the recurrence collapse? A registered test of a control
at the source.

Registered in PREREG_refraction_memory.md, Amendment 49.

Exploratory probes after Amendment 48 found the recurrence collapse written into the store: in
collapsed brains tokens of different words are laid onto one assembly, in a cascade that starts at
a brain-specific point of the writing, accelerates, and then fails replay of the whole store.
Writing is itself a recurrent step, so a captured state pulls later writes in. The control tested
here acts before a write: the element's round is PREVIEWED frozen (the refraction bias decayed as
the write would decay it); if the previewed state shares at least CAPTURE of its neurons with any
stored token -- label-free: no word identity is used -- those tokens' neurons are inhibited for
this write only, and the real write runs. The written-out store loop is
AssemblyMemory.store_sequence's, checked bit-equal to it at the start of every run.

    arms: U = 10, 50, 60 uses per word (random successors), each stored twice on the same brains
    -- standard and separated -- and replayed masked (word-level, every sequence).

    python -m research.runner write_separation --registration PATH --tag NAME [--smoke]
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

CELLS = ((9500, 70, 0.5, 68), (11500, 80, 0.45, 72))
SMOKE_CELL = (2000, 60, 0.5, 17)
USES = (10, 50, 60)
CAPTURE, INHIBIT = 0.3, 1e4
SEEDS = tuple(range(762, 782))
#: bars
HEALTHY, RESCUED, LIFT, REACH, COST, CAPTURED, COLLAPSED, MAX_COLLAPSED = 0.97, 0.85, 0.4, 0.7, 0.03, 0.001, 0.2, 2


def plan(cells, *, smoke=False):
    uses = (10, 40) if smoke else USES
    return [{"n": n, "k": k, "p": p, "tau": tau, "uses": u, "rho": rg.RHO,
             "beta": round(tl.theta(n, k, p), 5)}
            for n, k, p, tau in cells for u in uses]


def build(n, k, p, tau, seeds, device):
    from research.experiments.seq_capacity_scaling import seeds_for
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    return AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX,
                          norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=device,
                          bias_decay=math.exp(-1.0 / tau))


def store(mem, elem_seeds, stored, separate, device):
    """One sequence, store_sequence's loop written out: elem_seeds [LEN][B]; `stored` the list of
    tokens [B, k] written so far (appended to). Returns (states [LEN, B, k], interventions)."""
    import torch
    from neural_assemblies.core.torch_engine._hashed import StimulusFiber
    area, fib = mem.area, mem.fiber
    B = mem.B
    area.inhibit()
    states, hits = [], 0
    for sds in elem_seeds:
        stim = StimulusFiber(sds, mem.k, mem.n, mem.p, beta=mem.beta, w_max=mem.w_max,
                             norm_init=mem.norm_init, max_rounds=1, device=device)
        added = None
        if separate and stored:
            keep_b, keep_w = area.bias.clone(), area.winners.clone()
            area.bias.mul_(area.bias_decay)
            x = area.project(1, [fib, stim], freeze=True, mask_bias=False)
            area.bias, area.winners = keep_b, keep_w
            hot = torch.zeros(B, mem.n, device=device)
            hot.scatter_(1, x.long(), 1.0)
            T = torch.stack(stored)
            ov = torch.gather(hot.unsqueeze(0).expand(T.shape[0], B, mem.n), 2, T).sum(2) / mem.k
            bad = ov >= CAPTURE
            if bool(bad.any()):
                added = torch.zeros(B, mem.n, device=device)
                for t, b in bad.nonzero().tolist():
                    added[b, T[t, b]] = INHIBIT
                hits += int(bad.any(0).sum())
                area.bias += added
        win = area.project(1, [fib, stim], defer_overflow=True)
        if added is not None:
            area.bias -= added * area.bias_decay
        states.append(win.clone())
        stored.append(win.clone())
    mem.items += 1
    return torch.stack(states), hits


def equivalent(device):
    """The written loop, separation off, equals store_sequence (3 sequences, 4 brains, n = 2000)."""
    import torch
    from research.experiments.seq_capacity_scaling import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    sd = list(range(980, 984))
    a, b = build(2000, 60, 0.5, 17, sd, device), build(2000, 60, 0.5, 17, sd, device)
    for q in range(3):
        es = [[to_i32(_seeding.fnv1a_pair_seed(s, f"w{q}-{e}", "A")) for s in sd] for e in range(rg.LENGTH)]
        sa = a.store_sequence(es)[0]
        sb, _ = store(b, es, [], False, device)
        if not torch.equal(torch.sort(sa, -1).values, torch.sort(sb, -1).values):
            return False
    return True


def measure(spec, seeds, device):
    """Standard and separated stores of the same memories on the same brains: per brain, masked
    word-level reliability over every sequence, captured share (tokens sharing >= 0.5 with a
    token of another word), largest cluster; per store, interventions."""
    import numpy as np
    import torch
    from research.experiments.seq_capacity_scaling import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    n, k, p, tau = spec["n"], spec["k"], spec["p"], spec["tau"]
    LEN = rg.LENGTH
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / LEN))
    L = M * LEN
    V = max(8, round(L / spec["uses"]))
    B = len(seeds)
    words = [rg.walks(sd, M, V, None, L * 1000 + spec["uses"]) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device).long()
    ar = torch.arange(B, device=device)
    out: dict = {"L": L, "M": M, "V": V, "writes": L * B}
    for arm in ("standard", "separated"):
        mem = build(n, k, p, tau, seeds, device)
        stored, seqs, hits = [], [], 0
        for q in range(M):
            es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                   for i, sd in enumerate(seeds)] for e in range(LEN)]
            st, h = store(mem, es, stored, arm == "separated", device)
            seqs.append(st)
            hits += h
        mem.area.check_overflow()
        allst = torch.stack(stored).long()
        area = mem.area
        rel = torch.zeros(B, device=device)
        for q in range(M):
            area.bias = torch.zeros(B, n, device=device)
            area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, device)
                                        for i, sd in enumerate(seeds)]).long()
            alive = torch.ones(B, dtype=torch.bool, device=device)
            for j in range(1, LEN):
                x = area.project(1, [mem.fiber], freeze=True, mask_bias=False)
                hot = torch.zeros(B, n, device=device)
                hot.scatter_(1, x.long(), 1.0)
                ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
                alive &= wordof[ov.argmax(0), ar] == wordof[q * LEN + j]
            rel += alive.float()
        captured, cluster = [], []
        for i in range(B):
            H = torch.zeros(L, n, device=device, dtype=torch.float16)
            H.scatter_(1, allst[:, i], 1.0)
            O = (H @ H.T).float() / k
            O.fill_diagonal_(0.0)
            diffw = wordof[:, i].view(-1, 1) != wordof[:, i].view(1, -1)
            captured.append(float(((O >= 0.5) & diffw).any(1).float().mean()))
            cluster.append(int((O >= 0.5).sum(1).max()))
            del H, O, diffw
        out[arm] = {"reliability": (rel / M).tolist(), "captured": captured, "cluster": cluster,
                    "interventions": hits}
        del mem, seqs, allst, stored
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    if not equivalent(device):
        raise RuntimeError("the written store loop does not reproduce store_sequence")
    print("equivalence: the written store loop reproduces store_sequence", flush=True)
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][str(spec["uses"])] = m
        print(f"({n}, {k}, {p}) U={spec['uses']}: " + "  ".join(
            f"{a} {sum(m[a]['reliability']) / len(m[a]['reliability']):.3f} (captured "
            f"{sum(m[a]['captured']) / len(m[a]['captured']):.4f}, interventions {m[a]['interventions']})"
            for a in ("standard", "separated")), flush=True)
    return {"cells": out, "equivalent": True, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 49's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("W1", "W2", "W3", "W4", "W5", "W6")
    out: dict = {"bars": {}, "cells": {}}
    arms = {str(u) for u in USES}
    if not observations.get("equivalent") or not all(
            c[:3] in cells and arms <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    mean = lambda v: sum(v) / len(v)                                    # noqa: E731
    for n, k, p, _ in CELLS:
        a = cells[(n, k, p)]["arms"]
        info: dict = {u: {arm: {"reliability": mean(a[u][arm]["reliability"]), "captured": mean(a[u][arm]["captured"]),
                          "cost": a[u][arm]["interventions"] / a[u]["writes"]}
                    for arm in ("standard", "separated")} for u in a}
        collapsed = sum(r < COLLAPSED for u in ("50", "60") for r in a[u]["separated"]["reliability"])
        info["collapsed_separated"] = collapsed
        out["cells"][f"{n}/{k}/{p:g}"] = info
        ok["W1"] &= info["10"]["separated"]["reliability"] >= HEALTHY
        ok["W2"] &= (info["50"]["separated"]["reliability"] >= RESCUED and
                     info["50"]["separated"]["reliability"] - info["50"]["standard"]["reliability"] >= LIFT)
        ok["W3"] &= info["60"]["separated"]["reliability"] >= REACH
        ok["W4"] &= collapsed <= MAX_COLLAPSED
        ok["W5"] &= all(info[u]["separated"]["cost"] <= COST for u in a)
        ok["W6"] &= all(info[u]["separated"]["captured"] <= CAPTURED for u in a)
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Write separation", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 49 is registered on seeds 762..781")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.write-separation", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": rg.LENGTH, "capture": CAPTURE, "inhibit": INHIBIT,
                    "strength": ml.STRENGTH, "match": ml.MATCH, "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
