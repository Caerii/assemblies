"""Does a self-limiting SLEEP repair a captured store and spare a healthy one? A registered test of
offline unlearning gated by settling contrast.

Registered in PREREG_refraction_memory.md, Amendment 51.

Unlearning (Hopfield, Feinstein & Palmer 1983; the function Crick & Mitchison 1983 proposed for
REM sleep): free dynamics from noise fall into the states a store is drawn to, and each
transition taken loses one potentiation count on every synapse that has one (never below zero;
the connectome's baseline is untouched). Ungated it has a window and then erases any store.
GATED by settling contrast -- the dream winners' mean drive over the area's mean drive -- it
acts only while the dream sits in a captured cluster: a transition is unlearned only if its
contrast exceeds the largest contrast a healthy REFERENCE store's dreams reach (300 dreams of a
U = 10 store on separate reference brains at the same cell), times 1.02.

    stores: U = 50 (collapsing) and U = 10 (healthy) on the subject brains, standard
    store_sequence; sleep episodes of 8 free steps from k random neurons, unlearning from step 3;
    doses cumulative 100, 300, 1000, 3000; masked replay of every sequence after each dose.

    python -m research.runner sleep --registration PATH --tag NAME [--smoke]
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

CELLS = ((9000, 65, 0.5, 69), (11000, 85, 0.42, 65))
SMOKE_CELL = (2000, 60, 0.5, 17)
USES = (50, 10)
DOSES = (0, 100, 300, 1000, 3000)
SMOKE_DOSES = (0, 30)
STEPS, SETTLE, MARGIN, CALIBRATION = 8, 2, 1.02, 300
SEEDS = tuple(range(802, 822))
REFERENCE_SEEDS = tuple(range(822, 842))
NOISE_SEED, CAL_SEED = 4242, 777
#: bars
REPAIRED, LIFT, PLATEAU, HEALTHY, HEALTHY_COST, CLOSED, FRUGAL, MAX_COLLAPSED, COLLAPSED = (
    0.6, 0.4, 0.05, 0.97, 0.001, 0.02, 0.10, 2, 0.2)


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tau": tau, "rho": lib.RHO, "beta": round(lib.theta(n, k, p), 5),
             "doses": list(SMOKE_DOSES if smoke else DOSES)} for n, k, p, tau in cells]


def build_store(spec, U, seeds, device):
    import numpy as np
    import torch
    from research.experiments.memory_lib.seeding import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    n, k, p, tau = spec["n"], spec["k"], spec["p"], spec["tau"]
    LEN = lib.LENGTH
    M = max(1, round(spec["rho"] * lib.unit(n, k, p) / LEN))
    L = M * LEN
    V = max(8, round(L / U))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device).long()
    mem = ws.build(n, k, p, tau, seeds, device)
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(LEN)])[0] for q in range(M)]
    mem.area.check_overflow()
    if not hasattr(mem.fiber, "C"):
        raise RuntimeError("sleep edits the count matrix of the organ fiber")
    return {"mem": mem, "seqs": seqs, "allst": torch.cat(seqs, 0).long(), "wordof": wordof,
            "M": M, "L": L, "seeds": list(seeds)}


def dream(mem, gen, threshold, device, contrasts=None):
    """One sleep episode for every brain: (counts removed, steps gated, steps judged)."""
    import torch
    area, fib = mem.area, mem.fiber
    C = fib.C
    B, n, k = mem.B, mem.n, mem.k
    ar = torch.arange(B, device=device)
    area.bias = torch.zeros(B, n, device=device)
    area.winners = torch.argsort(torch.rand(B, n, device=device, generator=gen), dim=1)[:, :k]
    removed = gated = judged = 0
    for s in range(STEPS):
        prev = area.winners.clone()
        x, drive = area.project(1, [fib], freeze=True, mask_bias=False, return_drive=True)
        x = x.long()
        if s < SETTLE:
            continue
        con = torch.gather(drive, 1, x).mean(1) / drive.mean(1).clamp_min(1e-9)
        if contrasts is not None:
            contrasts.append(con.cpu())
        judged += B
        if threshold is None:
            continue
        g = con >= threshold
        if bool(g.any()):
            bi, ri, ci = ar.view(B, 1, 1), prev.view(B, k, 1), x.view(B, 1, k)
            old = C[bi, ri, ci]
            dec = ((old > 0) & g.view(B, 1, 1)).to(old.dtype)
            C[bi, ri, ci] = old - dec
            removed += int(dec.sum())
            gated += int(g.sum())
    return removed, gated, judged


def reliability(store, device):
    import torch
    mem, seqs, allst, wordof, M, L, seeds = (store[x] for x in ("mem", "seqs", "allst", "wordof", "M", "L", "seeds"))
    area, fib = mem.area, mem.fiber
    B, n, k = mem.B, mem.n, mem.k
    ar = torch.arange(B, device=device)
    rel = torch.zeros(B, device=device)
    for q in range(M):
        area.bias = torch.zeros(B, n, device=device)
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, device)
                                    for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=device)
        for j in range(1, lib.LENGTH):
            x = area.project(1, [fib], freeze=True, mask_bias=False)
            hot = torch.zeros(B, n, device=device)
            hot.scatter_(1, x.long(), 1.0)
            ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
            alive &= wordof[ov.argmax(0), ar] == wordof[q * lib.LENGTH + j]
        rel += alive.float()
    return (rel / M).tolist()


def measure(spec, seeds, device):
    import torch
    out: dict = {}
    ref = build_store(spec, 10, REFERENCE_SEEDS, device)
    cal: list = []
    g0 = torch.Generator(device=device).manual_seed(CAL_SEED)
    for _ in range(CALIBRATION):
        dream(ref["mem"], g0, None, device, cal)
    cal_t = torch.cat(cal)
    threshold = float(cal_t.max()) * MARGIN
    out["calibration"] = {"mean": float(cal_t.mean()), "max": float(cal_t.max()), "threshold": threshold}
    del ref
    torch.cuda.empty_cache()
    for U in USES:
        st = build_store(spec, U, seeds, device)
        held = int(st["mem"].fiber.C.sum())
        before: list = []
        dream(st["mem"], torch.Generator(device=device).manual_seed(99), None, device, before)
        bt = torch.cat(before)
        gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
        arm = {"held": held, "contrast_before": float(bt.mean()), "above_before": float((bt >= threshold).float().mean()),
               "doses": []}
        done = removed = gated = judged = 0
        for dose in spec["doses"]:
            g_int = j_int = 0
            while done < dose:
                r, g, j = dream(st["mem"], gen, threshold, device)
                removed += r; gated += g; judged += j; g_int += g; j_int += j
                done += 1
            arm["doses"].append({"episodes": dose, "reliability": reliability(st, device),
                                 "removed": removed / held, "gated_interval": g_int / j_int if j_int else 0.0})
        out[str(U)] = arm
        del st
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
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": spec["tau"], **m}
        print(f"({n}, {k}, {p}) threshold {m['calibration']['threshold']:.3f} (reference mean "
              f"{m['calibration']['mean']:.3f}, max {m['calibration']['max']:.3f})", flush=True)
        for U in USES:
            a = m[str(U)]
            print(f"  U={U}: contrast before {a['contrast_before']:.3f}; " + "  ".join(
                f"{d['episodes']}: {sum(d['reliability']) / len(d['reliability']):.3f} "
                f"(removed {d['removed']:.4f}, gated {d['gated_interval']:.3f})" for d in a["doses"]), flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 51's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("S1", "S2", "S3", "S4", "S5", "S6")
    out: dict = {"bars": {}, "cells": {}}
    if not all(c[:3] in cells and all(str(u) in cells[c[:3]] for u in USES) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    mean = lambda v: sum(v) / len(v)                                    # noqa: E731
    for n, k, p, _ in CELLS:
        c = cells[(n, k, p)]
        sick = {d["episodes"]: d for d in c["50"]["doses"]}
        well = {d["episodes"]: d for d in c["10"]["doses"]}
        if not ({0, 300, 1000, 3000} <= set(sick) and {0, 3000} <= set(well)):
            ok = {b: False for b in names}
            break
        info = {"before": mean(sick[0]["reliability"]), "at300": mean(sick[300]["reliability"]),
                "at3000": mean(sick[3000]["reliability"]), "healthy3000": mean(well[3000]["reliability"]),
                "healthy_removed": well[3000]["removed"], "gated_last": sick[3000]["gated_interval"],
                "removed": sick[3000]["removed"],
                "collapsed300": sum(r < COLLAPSED for r in sick[300]["reliability"])}
        out["cells"][f"{n}/{k}/{p:g}"] = info
        ok["S1"] &= info["at300"] >= REPAIRED and info["at300"] - info["before"] >= LIFT
        ok["S2"] &= info["at3000"] >= info["at300"] - PLATEAU
        ok["S3"] &= info["healthy3000"] >= HEALTHY and info["healthy_removed"] <= HEALTHY_COST
        ok["S4"] &= info["gated_last"] <= CLOSED
        ok["S5"] &= info["removed"] <= FRUGAL
        ok["S6"] &= info["collapsed300"] <= MAX_COLLAPSED
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Sleep", engines=("hashed_assembly_memory",), default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 51 is registered on seeds 802..821 (reference brains 822..841)")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.sleep", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "steps": STEPS, "settle": SETTLE, "margin": MARGIN,
                    "calibration": CALIBRATION, "reference_seeds": list(REFERENCE_SEEDS), "uses": list(USES),
                    "strength": lib.STRENGTH, "match": lib.MATCH, "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
