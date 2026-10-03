"""Recognition and recall, each at its own best write.

Registered in PREREG_refraction_memory.md, Amendment 22.

Amendment 20 compared the refracted memory's recognition (rank-1: the recall
is nearer the cued item than any other) with its recall (distinct
completion) at the RECALL-optimal write, and found the gap grows as fan-in
falls. Read after its bars, its record showed recognition's own best write is
far weaker -- at the bottom of the grid, 0.026 to 0.045, where the int8
counts of the time could not go lower -- and there recognition held at least
32768 items (the cap) at k p <= 10 against ~1300 for recall. With int16
counts (a38c1977) the write can be as weak as needed. This measures both
capacities each at its own optimum:

    the k sweep:  n = 4000, p = 0.5, k = 10, 20, 40, 80, 160 (k p 5 to 80;
                  the in-degree d = 2000 throughout, so recall stays put by
                  Amendment 21 while recognition is free to move)
    the n sweep:  k = 40, p = 0.5, n = 2000, 8000 (d = 1000, 4000)

each swept over 0.0015 x 2^(j/2) up to 0.30 theta, theta = sqrt((1 - p) ln n
/ (p k)). A store runs until both rank-1 and distinct completion have
fallen, to a cap of 131072 items (32768 at k >= 80).

    python -m research.runner recognition --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_regimes as rg                   # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

K_SWEEP = tuple((4000, k, 0.5) for k in (10, 20, 40, 80, 160))
N_SWEEP = ((2000, 40, 0.5), (8000, 40, 0.5))
CELLS = K_SWEEP + N_SWEEP
B_LO, F_TOP = 0.0015, 0.30
GIVE_UP = 8192
STOP_FROM = 32
SEEDS = tuple(range(242, 262))
#: Amendment 19's recall capacity at (4000, 40, 0.5) (QV)
A19 = {(4000, 40, 0.5): 1473.0}
RATIO_MIN, RHO_MAX, SPREAD_MIN, WEAKER = 2.0, -0.9, 5.0, 0.5


def cap_for(n, k, p):
    return 32768 if k >= 80 else 131072


def betas(n, k, p):
    top = F_TOP * tl.theta(n, k, p)
    out, j = [], 0
    while B_LO * 2 ** (j / 2) <= top + 1e-12:
        out.append(round(B_LO * 2 ** (j / 2), 5))
        j += 1
    return sorted(set(out))


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        grid = betas(n, k, p)
        out.append({"n": n, "k": k, "p": p, "theta": tl.theta(n, k, p),
                    "above_floor": tl.above_floor(n, k, p),
                    "betas": [grid[len(grid) // 2], grid[-2]] if smoke else grid,
                    "cap": 32 if smoke else cap_for(n, k, p),
                    "give_up": 16 if smoke else GIVE_UP})
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        grid = spec["betas"]
        results = lr.run_betas(n, k, grid, seeds, spec["cap"], device,
                               {b: profiles[lr.profile_name(b)] for b in grid},
                               stop_on=("rank1", "complete_distinct"), p=p, grid_start=2,
                               give_up=spec["give_up"], stop_from=STOP_FROM)
        sweep = {}
        for beta, (c, cache) in zip(grid, results):
            windows = {m: ws.edges([(M, ensemble_from_values(r[m]).mean)
                                    for M, r in cache.items()])
                       for m in ("rank1", "complete", "complete_distinct")}
            sweep[f"{beta:g}"] = {
                "beta": beta, "c_first_item": c, "windows": windows,
                "ensembles": {M: {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                  for nm, v in r.items()} for M, r in sorted(cache.items())},
            }
            print(f"({n}, {k}, {p}) beta={beta:g} ({beta / spec['theta']:.3f} theta): "
                  f"rank-1 {lr.capacity(windows['rank1']):.0f}"
                  f"{'*' if windows['rank1']['upper_censored'] else ''}, distinct "
                  f"{lr.capacity(windows['complete_distinct']):.0f}", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "theta": spec["theta"],
                                 "above_floor": spec["above_floor"], "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def cell_reading(cell):
    items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
    bs = [s["beta"] for s in items]
    rank1 = [lr.capacity(s["windows"]["rank1"]) for s in items]
    distinct = [lr.capacity(s["windows"]["complete_distinct"]) for s in items]
    r = max(range(len(bs)), key=lambda i: rank1[i])
    d = max(range(len(bs)), key=lambda i: distinct[i])
    return {"kp": cell["k"] * cell["p"], "rank1": rank1[r], "beta_rank1": bs[r],
            "rank1_censored": items[r]["windows"]["rank1"]["upper_censored"],
            "rank1_at_grid_start": r == 0,
            "distinct": distinct[d], "beta_distinct": bs[d],
            "ratio": rank1[r] / distinct[d] if distinct[d] > 0 else None}


def evaluate(observations, *, a19=None):
    """Amendment 22's bars."""
    a19 = A19 if a19 is None else a19
    rows = {(c["n"], c["k"], c["p"]): cell_reading(c) for c in observations["cells"].values()}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    out["bars"]["QV"] = all(abs(rows[c]["distinct"] / v - 1) <= 0.15
                            for c, v in a19.items() if c in rows)
    out["bars"]["Q1"] = bool(rows) and all(
        r["ratio"] is not None and r["ratio"] >= RATIO_MIN for r in rows.values())
    sweep = [rows[c] for c in K_SWEEP if c in rows and rows[c]["ratio"] is not None]
    rho = (rg.spearman([r["kp"] for r in sweep], [r["ratio"] for r in sweep])
           if len(sweep) > 2 else 0.0)
    out["rho_kp_ratio"] = rho
    lo, hi = rows.get(K_SWEEP[0]), rows.get(K_SWEEP[-1])
    spread = (lo["ratio"] / hi["ratio"]
              if lo and hi and lo["ratio"] and hi["ratio"] else None)
    out["spread"] = spread
    out["bars"]["Q2"] = (len(sweep) == len(K_SWEEP) and rho <= RHO_MAX
                         and spread is not None and spread >= SPREAD_MIN)
    out["bars"]["Q3"] = bool(rows) and all(
        r["beta_rank1"] <= WEAKER * r["beta_distinct"] for r in rows.values())
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Recognition", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all seven)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 22 is registered on seeds 242..261")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.recognition", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "b_lo": B_LO, "f_top": F_TOP, "give_up": GIVE_UP,
                    "stop_from": STOP_FROM, "strength": lr.STRENGTH, "w_max": pe.W_MAX,
                    "rounds": pe.T, "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
