"""Does a LOCAL comparator separate as the oracle does? A registered test of a circuit for pattern
separation at write.

Registered in PREREG_refraction_memory.md, Amendment 50.

Amendment 49 prevented the recurrence collapse with a check that compares each previewed write
with every stored token -- an oracle. The comparator needs neither the store nor word labels.
Before each write (not a sequence's first element, which has no prior state) two projections
from the current state are compared, as CA1 is held to compare CA3's recall with the cortical
input: RECALL (recurrence alone, no input: what memory predicts next) and the WRITE (input and
recurrence, the refraction bias decayed as the write would decay it). If they share at least
THRESHOLD of their winners, memory is capturing the write, and the predicted neurons are
inhibited for that write only. The oracle's flag (the write overlaps a stored token by >= 0.3)
is computed alongside, never acted on, to score the comparator's detections.

    arms: U = 10, 50, 60 uses per word (random successors), each stored twice on the same brains
    -- standard and comparator -- and replayed masked (word-level, every sequence).

    python -m research.runner comparator --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_reuse_grammar as rg             # noqa: E402
from research.experiments import memory_write_separation as ws          # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((8500, 65, 0.5, 65), (12000, 85, 0.43, 71))
SMOKE_CELL = (2000, 60, 0.5, 17)
USES = (10, 50, 60)
THRESHOLD, INHIBIT, ORACLE = 0.5, 1e4, 0.3
SEEDS = tuple(range(782, 802))
#: bars
HEALTHY, RESCUED, LIFT, MAX_COLLAPSED, COLLAPSED = 0.97, 0.8, 0.4, 2, 0.2
COST_REUSE, COST_HEALTHY, HIT, CAPTURED = 0.05, 0.01, 0.6, 0.001


def plan(cells, *, smoke=False):
    uses = (10, 40) if smoke else USES
    return [{"n": n, "k": k, "p": p, "tau": tau, "uses": u, "rho": lib.RHO,
             "beta": round(lib.theta(n, k, p), 5)}
            for n, k, p, tau in cells for u in uses]


def store(mem, elem_seeds, stored, compare, device, stats):
    """One sequence, store_sequence's loop written out (as memory_write_separation.store), with the
    comparator acting when `compare`. `stats` accumulates judged writes, comparator flags, oracle
    flags and their coincidences."""
    import torch
    from neural_assemblies.core.torch_engine._hashed import StimulusFiber
    area, fib = mem.area, mem.fiber
    B, n, k = mem.B, mem.n, mem.k
    area.inhibit()
    states = []
    for e, sds in enumerate(elem_seeds):
        stim = StimulusFiber(sds, k, n, mem.p, beta=mem.beta, w_max=mem.w_max,
                             norm_init=mem.norm_init, max_rounds=1, device=device)
        added = None
        if e > 0:
            keep_b, keep_w = area.bias.clone(), area.winners.clone()
            area.bias.mul_(area.bias_decay)
            P = area.project(1, [fib, stim], freeze=True, mask_bias=False)
            area.winners = keep_w.clone()
            R = area.project(1, [fib], freeze=True, mask_bias=False)
            area.bias, area.winners = keep_b, keep_w
            hP = torch.zeros(B, n, device=device)
            hP.scatter_(1, P.long(), 1.0)
            flag = (torch.gather(hP, 1, R.long()).sum(1) / k) >= THRESHOLD
            T = torch.stack(stored)
            ov = torch.gather(hP.unsqueeze(0).expand(T.shape[0], B, n), 2, T).sum(2) / k
            oracle = (ov >= ORACLE).any(0)
            stats["judged"] += B
            stats["flag"] += int(flag.sum())
            stats["oracle"] += int(oracle.sum())
            stats["hit"] += int((flag & oracle).sum())
            if compare and bool(flag.any()):
                added = torch.zeros(B, n, device=device)
                added.scatter_(1, R.long(), INHIBIT * flag.float().view(-1, 1).expand(-1, k).contiguous())
                area.bias += added
        win = area.project(1, [fib, stim], defer_overflow=True)
        if added is not None:
            area.bias -= added * area.bias_decay
        states.append(win.clone())
        stored.append(win.clone())
    mem.items += 1
    return torch.stack(states)


def measure(spec, seeds, device):
    """Standard and comparator stores of the same memories on the same brains: per brain, masked
    word-level reliability over every sequence and captured share; per store, the comparator's
    flags and their agreement with the oracle."""
    import numpy as np
    import torch
    from research.experiments.memory_lib.seeding import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    n, k, p, tau = spec["n"], spec["k"], spec["p"], spec["tau"]
    LEN = lib.LENGTH
    M = max(1, round(spec["rho"] * lib.unit(n, k, p) / LEN))
    L = M * LEN
    V = max(8, round(L / spec["uses"]))
    B = len(seeds)
    words = [rg.walks(sd, M, V, None, L * 1000 + spec["uses"]) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device).long()
    ar = torch.arange(B, device=device)
    out: dict = {"L": L, "M": M, "V": V}
    for arm in ("standard", "comparator"):
        mem = ws.build(n, k, p, tau, seeds, device)
        stored, seqs = [], []
        stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
        for q in range(M):
            es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                   for i, sd in enumerate(seeds)] for e in range(LEN)]
            seqs.append(store(mem, es, stored, arm == "comparator", device, stats))
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
        captured = []
        for i in range(B):
            H = torch.zeros(L, n, device=device, dtype=torch.float16)
            H.scatter_(1, allst[:, i], 1.0)
            O = (H @ H.T).float() / k
            O.fill_diagonal_(0.0)
            diffw = wordof[:, i].view(-1, 1) != wordof[:, i].view(1, -1)
            captured.append(float(((O >= 0.5) & diffw).any(1).float().mean()))
            del H, O, diffw
        out[arm] = {"reliability": (rel / M).tolist(), "captured": captured, **stats}
        del mem, seqs, allst, stored
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    if not ws.equivalent(device):
        raise RuntimeError("the written store loop does not reproduce store_sequence")
    print("equivalence: the written store loop reproduces store_sequence", flush=True)
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][str(spec["uses"])] = m
        c = m["comparator"]
        print(f"({n}, {k}, {p}) U={spec['uses']}: standard {sum(m['standard']['reliability']) / len(seeds):.3f}  "
              f"comparator {sum(c['reliability']) / len(seeds):.3f} (flags {c['flag'] / max(1, c['judged']):.4f}, "
              f"hit {c['hit'] / max(1, c['oracle']):.2f}, captured {sum(c['captured']) / len(seeds):.4f})", flush=True)
    return {"cells": out, "equivalent": True, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 50's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("C1", "C2", "C3", "C4", "C5", "C6")
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
        h, r = a["10"], a["50"]
        info = {
            "healthy": mean(h["comparator"]["reliability"]),
            "standard50": mean(r["standard"]["reliability"]),
            "comparator50": mean(r["comparator"]["reliability"]),
            "collapsed50": sum(x < COLLAPSED for x in r["comparator"]["reliability"]),
            "flags10": h["comparator"]["flag"] / max(1, h["comparator"]["judged"]),
            "flags50": r["comparator"]["flag"] / max(1, r["comparator"]["judged"]),
            "hit50": r["comparator"]["hit"] / r["comparator"]["oracle"] if r["comparator"]["oracle"] else None,
            "captured": max(mean(a[u]["comparator"]["captured"]) for u in a),
            "comparator60": mean(a["60"]["comparator"]["reliability"]),
        }
        out["cells"][f"{n}/{k}/{p:g}"] = info
        ok["C1"] &= info["healthy"] >= HEALTHY
        ok["C2"] &= info["comparator50"] >= RESCUED and info["comparator50"] - info["standard50"] >= LIFT
        ok["C3"] &= info["collapsed50"] <= MAX_COLLAPSED
        ok["C4"] &= info["flags50"] <= COST_REUSE and info["flags10"] <= COST_HEALTHY
        ok["C5"] &= info["hit50"] is not None and info["hit50"] >= HIT
        ok["C6"] &= info["captured"] <= CAPTURED
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Comparator", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 50 is registered on seeds 782..801")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.comparator", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "threshold": THRESHOLD, "inhibit": INHIBIT,
                    "oracle": ORACLE, "strength": lib.STRENGTH, "match": lib.MATCH, "w_max": lib.W_MAX,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
