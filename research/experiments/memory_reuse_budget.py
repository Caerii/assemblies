"""Is the reuse budget separable? Two edges -- repeated transitions and recurring
elements -- and a product law between them: a registered test at new cells.

Registered in PREREG_refraction_memory.md, Amendment 45.

Amendment 44 found that about five repetitions per transition destroy word-level
replay. Exploratory probes after it found two edges at (12000, 80, 0.45): a
REPETITION edge (word-level reliability 1.00 -> 0.00 as repeats per transition
R rise from 1 to ~4, at 10 uses per word) and a RECURRENCE edge (1.00 -> 0.00 as
uses per word U rise from 20 to 60, with random successors, whole brains
collapsing), and the interior points matched the product of the two marginal
curves, P(U, R) ~ f(R) g(U). Tested here at two new cells: each cell's
marginals are measured, and the product predicts its interior points.

    arms (U, b): recurrence marginal, random successors: U = 10, 20, 30, 40, 60;
    repetition marginal at U = 10: b = 10, 5, 4, 3 (R ~ 1, 2, 2.5, 3.3);
    interior: (20, 8), (12, 4), (24, 16), (40, 32), (30, 10).
    Reliability as in Amendment 44 (memory_reuse_grammar.measure): word-level,
    each brain its own grammar, total load rho = 0.05.

    python -m research.runner reuse_budget --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_reuse_grammar as rg             # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells and their recovery times (Amendment 41's rule)
CELLS = ((10000, 75, 0.48, 67), (7000, 60, 0.6, 58))
SMOKE_CELL = (2000, 60, 0.5, 17)
RECURRENCE = ((10, None), (20, None), (30, None), (40, None), (60, None))
REPETITION = ((10, 10), (10, 5), (10, 4), (10, 3))
INTERIOR = ((20, 8), (12, 4), (24, 16), (40, 32), (30, 10))
ARMS = RECURRENCE + REPETITION + INTERIOR
SEEDS = tuple(range(682, 702))
HIGH, LOW, TOL, INFORMATIVE = 0.9, 0.2, 0.15, (0.15, 0.85)


def name(u, b):
    return f"U{u}/b{b if b else 'V'}"


def plan(cells, *, smoke=False):
    arms = ((10, None), (10, 3)) if smoke else ARMS
    return [{"n": n, "k": k, "p": p, "tau": tau, "uses": u, "b": b, "rho": lib.RHO,
             "beta": round(lib.theta(n, k, p), 5), "batch": md.batch_size(n)}
            for n, k, p, tau in cells for u, b in arms]


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = rg.measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][name(spec["uses"], spec["b"])] = m
        mean = lambda v: sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v))   # noqa: E731
        print(f"({n}, {k}, {p}) U={spec['uses']} b={spec['b'] or 'V'} V={m['V']} R={mean(m['repeats']):.2f}: "
              f"word {mean(m['word']):.3f} (brains {min(m['word']):.2f}-{max(m['word']):.2f})", flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def interpolate(points, x):
    """Piecewise-linear in x through sorted (x, y) points, constant beyond the ends."""
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return pts[-1][1]


def evaluate(observations):
    """Amendment 45's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out: dict = {"bars": {}, "word": {}, "predicted": {}, "informative": {}}
    if not all(c[:3] in cells and {name(u, b) for u, b in ARMS} <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {"S1": False, "S2": False}
        return out
    s1, s2 = True, True
    for n, k, p, _ in CELLS:
        key = f"{n}/{k}/{p:g}"
        arms = cells[(n, k, p)]["arms"]
        word = {a: _mean(m["word"]) for a, m in arms.items()}
        R = {a: _mean(m["repeats"]) for a, m in arms.items()}
        out["word"][key] = word
        g = [(u, word[name(u, b)]) for u, b in RECURRENCE]
        f = [(R[name(u, b)], word[name(u, b)]) for u, b in ((10, None),) + REPETITION]
        pred = {name(u, b): interpolate(f, R[name(u, b)]) * interpolate(g, u) for u, b in INTERIOR}
        out["predicted"][key] = pred
        informative = [a for a, v in pred.items() if INFORMATIVE[0] < v < INFORMATIVE[1]]
        out["informative"][key] = informative
        s1 &= (word[name(10, None)] >= HIGH and word[name(60, None)] <= LOW and word[name(10, 3)] <= LOW)
        s2 &= len(informative) >= 2 and all(abs(word[a] - v) <= TOL for a, v in pred.items())
    out["bars"] = {"S1": s1, "S2": s2}
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Reuse budget", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 45 is registered on seeds 682..701")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.reuse-budget", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "strength": lib.STRENGTH, "match": lib.MATCH,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
