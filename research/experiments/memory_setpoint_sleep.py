"""A sleep gate fixed at birth, with no reference brains. A registered test.

Registered in PREREG_refraction_memory.md, Amendment 55.

Amendments 51 to 54 calibrate sleep's contrast gate on twenty separate healthy reference brains.
Here each brain fixes its own gate once, before it learns anything: 1.02 x the largest contrast
its own EMPTY network's dreams reach (300 dreams). Healthy dreams do not settle, so a healthy
store's dreams should stay below that set point while a captured store's exceed it
(probes/2026-10-10/probe_setpoint). Against Amendment 54's median-of-reference-brains rule, on the
same stores:

    per cell: set points from the subjects' empty networks; the MEDIAN rule from reference brains;
    a healthy U = 10 store (its dream contrasts against each brain's set point; slept 300 episodes
    under each gate from the same written state); a standard U = 50 store and a comparator U = 100
    store, each replayed, then slept 300 episodes under each gate from the same written state.

    python -m research.runner setpoint_sleep --registration PATH --tag NAME [--smoke]
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

CELLS = ((8400, 70, 0.52, 60), (11600, 80, 0.44, 72))
SMOKE_CELL = (2000, 60, 0.5, 17)
HEALTHY_USES, STANDARD_USES, COMPARATOR_USES = 10, 50, 100
SMOKE_STANDARD_USES, SMOKE_COMPARATOR_USES = 30, 40
EPISODES, SMOKE_EPISODES = 300, 30
GATES = ("median", "setpoint")
SEEDS = tuple(range(1080, 1100))
REFERENCE_SEEDS = tuple(range(1100, 1120))
#: bars
SAFE, SAFE_COST, REPAIRED, REACH, MATCH, MAX_COLLAPSED, COLLAPSED, FRUGAL50, FRUGAL100 = (
    0.99, 0.001, 0.6, 0.45, 0.05, 2, 0.2, 0.10, 0.05)


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tau": tau, "rho": lib.RHO, "beta": round(lib.theta(n, k, p), 5),
             "standard_uses": SMOKE_STANDARD_USES if smoke else STANDARD_USES,
             "comparator_uses": SMOKE_COMPARATOR_USES if smoke else COMPARATOR_USES,
             "episodes": SMOKE_EPISODES if smoke else EPISODES}
            for n, k, p, tau in cells]


def brain_maxima(mem, brains, device):
    """each brain's largest contrast over 300 calibration dreams (memory_sleep's calibration)"""
    import torch
    cal: list = []
    g0 = torch.Generator(device=device).manual_seed(sl.CAL_SEED)
    for _ in range(sl.CALIBRATION):
        sl.dream(mem, g0, None, device, cal)
    return torch.cat(cal).float().view(-1, brains).max(0).values


def under_each_gate(st, gates, episodes, device):
    """replay after sleeping the same written state under each gate: {gate: (reliability, removed)}"""
    backup = st["mem"].fiber.C.cpu()
    out = {}
    for g in GATES:
        st["mem"].fiber.C.copy_(backup.to(device))
        removed = lc.slept(st, episodes, gates[g], device)
        out[g] = (sl.reliability(st, device), removed)
    del backup
    return out


def measure(spec, seeds, device):
    import torch
    n, k, p, tau = spec["n"], spec["k"], spec["p"], spec["tau"]
    B = len(seeds)
    empty = ws.build(n, k, p, tau, seeds, device)
    empty_max = brain_maxima(empty, B, device)
    del empty
    ref = sl.build_store(spec, HEALTHY_USES, REFERENCE_SEEDS, device)
    ref_max = brain_maxima(ref["mem"], len(REFERENCE_SEEDS), device)
    del ref
    torch.cuda.empty_cache()
    setpoint = (empty_max * sl.MARGIN).to(device)
    gates = {"median": float(ref_max.median()) * sl.MARGIN,           # Amendment 54's rule
             "setpoint": setpoint}
    out: dict = {"empty_max": empty_max.tolist(), "reference_max": ref_max.tolist(),
                 "setpoint": setpoint.tolist(), "median_threshold": gates["median"]}
    st = sl.build_store(spec, HEALTHY_USES, seeds, device)
    own = brain_maxima(st["mem"], B, device)
    healthy = {"before": sl.reliability(st, device), "max_contrast": own.tolist()}
    for g, (rel, removed) in under_each_gate(st, gates, spec["episodes"], device).items():
        healthy[g], healthy[f"{g}_removed"] = rel, removed
    out["healthy"] = healthy
    del st
    torch.cuda.empty_cache()
    for name, build in (("standard", lambda: sl.build_store(spec, spec["standard_uses"], seeds, device)),
                        ("comparator", lambda: lc.comparator_store(spec, spec["comparator_uses"], seeds, device)[0])):
        st = build()
        arm = {"before": sl.reliability(st, device)}
        for g, (rel, removed) in under_each_gate(st, gates, spec["episodes"], device).items():
            arm[g], arm[f"{g}_removed"] = rel, removed
        out[name] = arm
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
        h = m["healthy"]
        ratio = [a / b for a, b in zip(h["max_contrast"], m["setpoint"])]
        print(f"({n}, {k}, {p}) empty maxima {min(m['empty_max']):.3f}-{max(m['empty_max']):.3f}; healthy maxima "
              f"{min(h['max_contrast']):.3f}-{max(h['max_contrast']):.3f}; healthy max / own set point "
              f"{min(ratio):.3f}-{max(ratio):.3f}; reference median threshold {m['median_threshold']:.3f}", flush=True)
        for name in ("healthy", "standard", "comparator"):
            a = m[name]
            print(f"  {name:10s} before {mean(a['before']):.3f}; " + "; ".join(
                f"{g} {mean(a[g]):.3f} (removed {a[g + '_removed']:.4f}, lowest {min(a[g]):.2f})" for g in GATES),
                flush=True)
    return {"cells": out, "equivalent": True, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 55's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
    out: dict = {"bars": {}, "cells": {}}
    if not observations.get("equivalent") or not all(c[:3] in cells for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    mean = lambda v: sum(v) / len(v)                                    # noqa: E731
    for n, k, p, _ in CELLS:
        c = cells[(n, k, p)]
        h = c["healthy"]
        info = {a: {g: mean(c[a][g]) for g in ("before",) + GATES} for a in ("healthy", "standard", "comparator")}
        removed = {a: {g: c[a][f"{g}_removed"] for g in GATES} for a in ("healthy", "standard", "comparator")}
        collapsed = {a: sum(v < COLLAPSED for v in c[a]["setpoint"]) for a in ("standard", "comparator")}
        below = sum(a < b for a, b in zip(h["max_contrast"], c["setpoint"]))
        out["cells"][f"{n}/{k}/{p:g}"] = {"replay": info, "removed": removed, "collapsed": collapsed,
                                         "brains_below_setpoint": below}
        ok["G1"] &= info["healthy"]["setpoint"] >= SAFE and removed["healthy"]["setpoint"] <= SAFE_COST
        ok["G2"] &= info["standard"]["setpoint"] >= REPAIRED
        ok["G3"] &= info["comparator"]["setpoint"] >= REACH
        ok["G4"] &= all(info[a]["setpoint"] >= info[a]["median"] - MATCH for a in ("standard", "comparator"))
        ok["G5"] &= all(v <= MAX_COLLAPSED for v in collapsed.values())
        ok["G6"] &= (removed["standard"]["setpoint"] <= FRUGAL50 and removed["comparator"]["setpoint"] <= FRUGAL100)
        ok["G7"] &= below == len(h["max_contrast"])
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Set-point sleep", engines=("hashed_assembly_memory",), default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 55 is registered on seeds 1080..1099 (reference brains 1100..1119)")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.setpoint_sleep", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "comparator_threshold": mc.THRESHOLD,
                    "sleep_steps": sl.STEPS, "sleep_margin": sl.MARGIN, "reference_seeds": list(REFERENCE_SEEDS),
                    "strength": lib.STRENGTH, "match": lib.MATCH, "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
