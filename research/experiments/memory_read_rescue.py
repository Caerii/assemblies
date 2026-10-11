"""Does adaptation at read time rescue collapsed brains? Amendment 46's control,
registered paired and conditional on collapse.

Registered in PREREG_refraction_memory.md, Amendment 47.

Amendment 46 registered the rescue's size as a cell mean, which depends on how
many brains collapse, and missed at one cell; paired by brain after the run,
every brain that changed improved (36 of 36) and collapsed brains rose from 0.01
to 0.39. Here the bars are on the brains themselves: which brains collapse under
the standard (masked) read, and what the same memory gives them under a session
adaptation during replay (c = 0.1, T = 50; memory_read_adaptation.measure).

    arms (U, b): recurrence at U = 40 and U = 50 (random successors) -- two levels
    so that brains collapse; repetition U = 10, b = 3; healthy U = 10, random.

    python -m research.runner read_rescue --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_read_adaptation as ra            # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((9000, 70, 0.5, 64), (13000, 90, 0.4, 72))
SMOKE_CELL = (2000, 60, 0.5, 17)
ARMS = (("recurrence40", 40, None), ("recurrence50", 50, None), ("repetition", 10, 3), ("healthy", 10, None))
SEEDS = tuple(range(722, 742))
FAILING, COLLAPSED = 0.5, 0.2
MIN_FAILING, MIN_COLLAPSED = 5, 4
UP, EPS, GAIN, HEALTHY, MARGIN = 0.9, 0.005, 0.2, 0.97, 0.2


def plan(cells, *, smoke=False):
    arms = ARMS[:1] if smoke else ARMS
    return [{"n": n, "k": k, "p": p, "tau": tau, "arm": a, "uses": u, "b": b, "rho": lib.RHO,
             "beta": round(lib.theta(n, k, p), 5), "batch": md.batch_size(n)}
            for n, k, p, tau in cells for a, u, b in arms]


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = ra.measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][spec["arm"]] = m
        print(f"({n}, {k}, {p}) {spec['arm']} U={spec['uses']} b={spec['b'] or 'V'}: "
              + "  ".join(f"{mode} {sum(v) / len(v):.3f}" for mode, v in m["modes"].items()), flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def paired(arms, names):
    """(masked, habit) per brain, pooled over the named arms."""
    out = []
    for a in names:
        modes = arms[a]["modes"]
        out += list(zip(modes["masked"], modes["habit"]))
    return out


def evaluate(observations):
    """Amendment 47's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("Q1", "Q2", "Q3", "Q4")
    out: dict = {"bars": {}, "cells": {}}
    if not all(c[:3] in cells and {a for a, _, _ in ARMS} <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    for n, k, p, _ in CELLS:
        arms = cells[(n, k, p)]["arms"]
        rec = paired(arms, ("recurrence40", "recurrence50"))
        failing = [(m, h) for m, h in rec if m < FAILING]
        collapsed = [(m, h) for m, h in rec if m < COLLAPSED]
        up = sum(h - m > EPS for m, h in failing)
        down = sum(h - m < -EPS for m, h in failing)
        gain_col = sum(h - m for m, h in collapsed) / len(collapsed) if collapsed else None
        rep = paired(arms, ("repetition",))
        gain_rep = sum(h - m for m, h in rep) / len(rep)
        hea = arms["healthy"]["modes"]["habit"]
        info = {"failing": len(failing), "up": up, "down": down, "collapsed": len(collapsed),
                "gain_collapsed": gain_col, "gain_repetition": gain_rep, "healthy_habit": sum(hea) / len(hea)}
        out["cells"][f"{n}/{k}/{p:g}"] = info
        ok["Q1"] &= len(failing) >= MIN_FAILING and up >= UP * len(failing) and down <= 1
        ok["Q2"] &= len(collapsed) >= MIN_COLLAPSED and gain_col >= GAIN
        ok["Q3"] &= info["healthy_habit"] >= HEALTHY
        ok["Q4"] &= gain_col is not None and gain_col - gain_rep >= MARGIN
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Read rescue", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 47 is registered on seeds 722..741")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.read-rescue", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "modes": [list(m) for m in ra.MODES],
                    "strength": lib.STRENGTH, "match": lib.MATCH, "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
