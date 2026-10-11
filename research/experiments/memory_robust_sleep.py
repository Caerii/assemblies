"""A sleep threshold one reference brain cannot move. A registered test.

Registered in PREREG_refraction_memory.md, Amendment 54.

Amendments 51 to 53 set the contrast gate of sleep at 1.02 x the MAXIMUM contrast reached by 300
dreams of a healthy U = 10 store on reference brains. Amendment 52's L3 failed where one reference
brain's dreams reached far past the others' and raised that threshold; on the same brains robust
statistics recovered the repair (probes/2026-10-09/probe_threshold). The rule here is the MEDIAN over
reference brains of each brain's maximum contrast: fewer than half the reference brains cannot move
it, however often they settle (a pooled quantile is moved by one brain that settles often). At new
cells, both rules act on the same stores:

    per cell: calibration on reference brains -> thresholds MAX and MEDIAN (each x 1.02);
    a healthy U = 10 store slept 300 episodes under each rule (from the same written state);
    a comparator U = 100 store, replayed, then slept 300 episodes under each rule (same state).

    python -m research.runner robust_sleep --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_comparator as mc                # noqa: E402
from research.experiments import memory_lifecycle as lc                 # noqa: E402
from research.experiments import memory_sleep as sl                     # noqa: E402
from research.experiments import memory_write_separation as ws          # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((9000, 68, 0.5, 66), (10500, 80, 0.45, 66))
SMOKE_CELL = (2000, 60, 0.5, 17)
HEALTHY_USES, USES, SMOKE_USES = 10, 100, 40
EPISODES, SMOKE_EPISODES = 300, 30
RULES = ("max", "median")
SEEDS = tuple(range(1040, 1060))
REFERENCE_SEEDS = tuple(range(1060, 1080))
#: bars
SAFE, SAFE_COST, REACH, NOT_WORSE, SPARING, MAX_COLLAPSED, COLLAPSED = 0.99, 0.001, 0.45, 0.02, 0.02, 2, 0.2


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tau": tau, "rho": lib.RHO, "beta": round(lib.theta(n, k, p), 5),
             "uses": SMOKE_USES if smoke else USES, "episodes": SMOKE_EPISODES if smoke else EPISODES}
            for n, k, p, tau in cells]


def thresholds(contrasts, brains):
    """the two rules, from the reference contrasts as dream() records them (a 1-D float tensor,
    brain-minor: element i belongs to brain i % brains)"""
    per_brain = contrasts.view(-1, brains).max(0).values
    return {"max": float(contrasts.max()) * sl.MARGIN, "median": float(per_brain.median()) * sl.MARGIN}


def under_each_rule(st, thr, episodes, device):
    """replay after sleeping the same written state under each rule: {rule: (reliability, removed)}"""
    backup = st["mem"].fiber.C.cpu()
    out = {}
    for rule in RULES:
        st["mem"].fiber.C.copy_(backup.to(device))
        removed = lc.slept(st, episodes, thr[rule], device)
        out[rule] = (sl.reliability(st, device), removed)
    del backup
    return out


def measure(spec, seeds, device):
    import torch
    ref = sl.build_store(spec, HEALTHY_USES, REFERENCE_SEEDS, device)
    cal: list = []
    g0 = torch.Generator(device=device).manual_seed(sl.CAL_SEED)
    for _ in range(sl.CALIBRATION):
        sl.dream(ref["mem"], g0, None, device, cal)
    del ref
    torch.cuda.empty_cache()
    cal_t = torch.cat(cal).float()
    B = len(REFERENCE_SEEDS)
    thr = thresholds(cal_t, B)
    out: dict = {"calibration": {"mean": float(cal_t.mean()), "max": float(cal_t.max()),
                                 "brain_max": cal_t.view(-1, B).max(0).values.tolist(), "thresholds": thr}}
    st = sl.build_store(spec, HEALTHY_USES, seeds, device)
    healthy = {"before": sl.reliability(st, device)}
    for rule, (rel, removed) in under_each_rule(st, thr, spec["episodes"], device).items():
        healthy[rule], healthy[f"{rule}_removed"] = rel, removed
    out["healthy"] = healthy
    del st
    torch.cuda.empty_cache()
    st, stats = lc.comparator_store(spec, spec["uses"], seeds, device)
    reuse = {"comparator": sl.reliability(st, device), "flags": stats["flag"] / max(1, stats["judged"])}
    for rule, (rel, removed) in under_each_rule(st, thr, spec["episodes"], device).items():
        reuse[rule], reuse[f"{rule}_removed"] = rel, removed
    out["reuse"] = reuse
    del st
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
    mean = lambda v: sum(v) / len(v)                                    # noqa: E731
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": spec["tau"], **m}
        c, h, r = m["calibration"], m["healthy"], m["reuse"]
        print(f"({n}, {k}, {p}) reference contrasts mean {c['mean']:.3f}, max {c['max']:.3f}; "
              f"brain maxima {min(c['brain_max']):.3f}-{max(c['brain_max']):.3f}", flush=True)
        for rule in RULES:
            print(f"  {rule:5s} threshold {c['thresholds'][rule]:.3f}: healthy {mean(h['before']):.3f} -> "
                  f"{mean(h[rule]):.3f} (removed {h[rule + '_removed']:.4f}); U={spec['uses']} comparator "
                  f"{mean(r['comparator']):.3f} -> {mean(r[rule]):.3f} (removed {r[rule + '_removed']:.4f}, "
                  f"lowest {min(r[rule]):.2f})", flush=True)
    return {"cells": out, "equivalent": True, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 54's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("R1", "R2", "R3", "R4", "R5")
    out: dict = {"bars": {}, "cells": {}}
    if not observations.get("equivalent") or not all(c[:3] in cells for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    mean = lambda v: sum(v) / len(v)                                    # noqa: E731
    for n, k, p, _ in CELLS:
        c = cells[(n, k, p)]
        h, r = c["healthy"], c["reuse"]
        info = {"healthy": {x: mean(h[x]) for x in ("before",) + RULES},
                "healthy_removed": {x: h[f"{x}_removed"] for x in RULES},
                "reuse": {x: mean(r[x]) for x in ("comparator",) + RULES},
                "reuse_removed": {x: r[f"{x}_removed"] for x in RULES},
                "collapsed": {x: sum(v < COLLAPSED for v in r[x]) for x in RULES}}
        out["cells"][f"{n}/{k}/{p:g}"] = info
        ok["R1"] &= info["healthy"]["median"] >= SAFE and info["healthy_removed"]["median"] <= SAFE_COST
        ok["R2"] &= info["reuse"]["median"] >= REACH
        ok["R3"] &= info["reuse"]["median"] >= info["reuse"]["max"] - NOT_WORSE
        ok["R4"] &= info["reuse_removed"]["median"] <= SPARING
        ok["R5"] &= info["collapsed"]["median"] <= MAX_COLLAPSED
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Robust sleep", engines=("hashed_assembly_memory",), default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 54 is registered on seeds 1040..1059 (reference brains 1060..1079)")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.robust_sleep", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "comparator_threshold": mc.THRESHOLD,
                    "sleep_steps": sl.STEPS, "sleep_margin": sl.MARGIN, "rule": "median of per-brain maxima",
                    "reference_seeds": list(REFERENCE_SEEDS), "strength": lib.STRENGTH, "match": lib.MATCH,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
