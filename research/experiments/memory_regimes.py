"""Two capacity regimes either side of the floor, and recognition against
recall as fan-in falls.

Registered in PREREG_refraction_memory.md, Amendments 19 and 20 (one run).

Amendment 17 found, after the data, that below the floor k p >= 3 ln n the
refracted memory's best distinct-completion capacity is flat in k (1356,
1382, 1452 at n = 4000, p = 0.5, k = 10, 20, 40) where (n/k)^2 falls 16-fold,
and that identification outlasts completion: rank-1 capacity ran past the
8192-item cap where completion was zero, and the PNAS 2020 cell identified
up to 112 items and completed none.

Amendment 19 asks whether capacity changes LAW at the floor: flat in k and
linear in n below it, falling with k above it. Amendment 20 asks whether
recognition and recall DIVERGE as fan-in falls (the dual-process reading).
Fifteen cells:

    the k sweep:     n = 4000, p = 0.5, k = 10, 14, 20, 28, 40 | 56, 80, 112, 160
                     (k p = 5 to 80; the floor is k p = 24.9)
    the n sweep:     k = 20, p = 0.5, n = 2000, 4000, 8000 (below the floor)
    the PNAS family: n = 10000, p = 0.01, k = 100, 200, 400, 800 (k p = 1 to 8)

each swept over a grid that holds both metrics' optima,

    theta x 0.04 x 2^(j/4) below 0.14 theta (identification's weak writes),
    theta x 0.14 x 2^(j/6) up to 0.40 theta (completion's onset and peak),

at or above 0.025, with theta = sqrt((1 - p) ln n / (p k)). A store runs until
BOTH rank-1 and distinct completion have fallen (or to 32768 items), so
identification's capacity is read, not censored at completion's stop.

    python -m research.runner regimes --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

K_SWEEP = tuple((4000, k, 0.5) for k in (10, 14, 20, 28, 40, 56, 80, 112, 160))
N_SWEEP = ((2000, 20, 0.5), (4000, 20, 0.5), (8000, 20, 0.5))
PNAS = tuple((10000, k, 0.01) for k in (100, 200, 400, 800))
CELLS = tuple(dict.fromkeys(K_SWEEP + N_SWEEP + PNAS))
F_RANK1, F_SPLIT, F_TOP = 0.04, 0.14, 0.40
CAP = 32768
GIVE_UP = 8192
MIN_BETA = tl.MIN_BETA
STOP_FROM = 32
SEEDS = tuple(range(182, 202))
#: Amendment 17's best distinct capacities (XV)
A17 = {(4000, 40, 0.5): 1451.57, (4000, 80, 0.5): 1312.0}
FLAT, BELOW_RATIO, ABOVE_FALL = 0.15, (1.6, 2.5), 0.8
RHO_MAX, DIVERGENCE, COMPLETES = -0.7, 3.0, 32


def betas(n, k, p):
    theta = lib.theta(n, k, p)
    grid = set()
    j = 0
    while F_RANK1 * 2 ** (j / 4) < F_SPLIT - 1e-12:
        grid.add(round(theta * F_RANK1 * 2 ** (j / 4), 4))
        j += 1
    j = 0
    while F_SPLIT * 2 ** (j / 6) <= F_TOP + 1e-12:
        grid.add(round(theta * F_SPLIT * 2 ** (j / 6), 4))
        j += 1
    return sorted(b for b in grid if b >= MIN_BETA)


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        grid = betas(n, k, p)
        out.append({"n": n, "k": k, "p": p, "above_floor": lib.above_floor(n, k, p),
                    "theta": lib.theta(n, k, p),
                    "betas": [grid[0], grid[len(grid) // 2]] if smoke else grid,
                    "cap": 32 if smoke else CAP,
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
                               {b: profiles[lib.profile_name(b)] for b in grid},
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
                  f"distinct {lr.capacity(windows['complete_distinct']):.0f}, rank-1 "
                  f"{lr.capacity(windows['rank1']):.0f}", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p,
                                 "above_floor": spec["above_floor"],
                                 "theta": spec["theta"], "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def cell_reading(cell):
    items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
    bs = [s["beta"] for s in items]
    distinct = [lr.capacity(s["windows"]["complete_distinct"]) for s in items]
    rank1 = [lr.capacity(s["windows"]["rank1"]) for s in items]
    censored = [s["windows"]["rank1"]["upper_censored"] for s in items]
    d_best = max(range(len(bs)), key=lambda i: distinct[i])
    r_best = max(range(len(bs)), key=lambda i: rank1[i])
    out = {"kp": cell["k"] * cell["p"], "above_floor": cell["above_floor"],
           "betas": bs, "distinct": distinct, "rank1": rank1,
           "best_distinct": distinct[d_best], "beta_distinct": bs[d_best],
           "best_rank1": rank1[r_best], "beta_rank1": bs[r_best],
           "rank1_censored_at_best": censored[r_best],
           "R": None, "R_censored": None}
    if distinct[d_best] > 0:
        # recognition against recall at the recall-optimal write
        out["R"] = rank1[d_best] / distinct[d_best]
        out["R_censored"] = censored[d_best]
    return out


def evaluate(observations, *, a17=None):
    """Amendments 19 (X) and 20 (R)."""
    a17 = A17 if a17 is None else a17
    rows = {(c["n"], c["k"], c["p"]): cell_reading(c) for c in observations["cells"].values()}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    bars = out["bars"]
    bars["XV"] = all(abs(rows[c]["best_distinct"] / v - 1) <= 0.15
                     for c, v in a17.items() if c in rows)
    below = [rows[c]["best_distinct"] for c in K_SWEEP if c in rows and not rows[c]["above_floor"]]
    mean = sum(below) / len(below) if below else 0.0
    bars["X1"] = bool(below) and mean > 0 and all(abs(v / mean - 1) <= FLAT for v in below)
    ns = [rows[c]["best_distinct"] for c in N_SWEEP if c in rows]
    ratios = [b / a for a, b in zip(ns, ns[1:]) if a > 0]
    out["n_ratios"] = ratios
    bars["X2"] = (len(ratios) == len(N_SWEEP) - 1
                  and all(BELOW_RATIO[0] <= r <= BELOW_RATIO[1] for r in ratios))
    above = [rows[c]["best_distinct"] for c in K_SWEEP if c in rows and rows[c]["above_floor"]]
    bars["X3"] = (len(above) >= 2 and all(a > b for a, b in zip(above, above[1:]))
                  and above[-1] <= ABOVE_FALL * above[0])
    sweep = [rows[c] for c in K_SWEEP if c in rows and rows[c]["R"] is not None]
    rho = spearman([r["kp"] for r in sweep], [r["R"] for r in sweep]) if len(sweep) > 2 else 0.0
    out["rho_kp_R"] = rho
    bars["R1"] = len(sweep) == len(K_SWEEP) and rho <= RHO_MAX
    fam = [rows[c] for c in PNAS if c in rows]
    bars["R2"] = (len(fam) == len(PNAS)
                  and all(r["best_rank1"] >= COMPLETES for r in fam)
                  and fam[0]["best_distinct"] == 0 and fam[-1]["best_distinct"] >= COMPLETES)
    lo, hi = rows.get(K_SWEEP[0]), rows.get(K_SWEEP[-1])
    bars["R3"] = bool(lo and hi and lo["R"] is not None and hi["R"] is not None
                      and lo["R"] >= DIVERGENCE * hi["R"])
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Regimes", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all fifteen cells)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendments 19-20 are registered on seeds 182..201")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(b): lib.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.regimes", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "f_rank1": F_RANK1, "f_split": F_SPLIT, "f_top": F_TOP,
                    "min_beta": MIN_BETA, "stop_from": STOP_FROM, "strength": lib.STRENGTH,
                    "w_max": lib.W_MAX, "rounds": lib.ROUNDS, "half_bar": lib.HALF_BAR,
                    "complete": lib.COMPLETE, "recall_sample": lib.RECALL_SAMPLE,
                    "measurement_seed": lib.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
