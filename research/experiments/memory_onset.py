"""Where does the refracted memory begin to complete, and is its best write
the weakest one that completes?

Registered in PREREG_refraction_memory.md, Amendment 18.

Amendment 17 found, after the data, that at p = 0.5 the best capacity sits at
the FIRST rate of its grid that completes at all (4 of 6 cells) or one step
above it, and that the weakest completing rate opens a load WINDOW: at
(4000, 60) completion holds only between 869 and 1589 stored items, one grid
step up between 31 and 1388. At p = 0.05 the first completing rate completes
from the first checkpoint, and capacity keeps rising for three steps above it.
On Amendment 17's grid (0.2 theta x 2^(j/4)) the first completing rate was
0.168 theta at eight of ten cells and 0.200 theta at two.

This measures the onset on a grid twelve steps to the octave,

    beta_j = theta x 0.12 x 2^(j/12),   up to 0.30 theta (0.42 theta below the floor),

with theta = sqrt((1 - p) ln n / (p k)), at ten cells:

    above the floor:  (2000, 60, 0.5), (4000, 60, 0.5), (4000, 80, 0.5),
                      (4000, 160, 0.5), (8000, 120, 0.5), (8000, 240, 0.25)
    below the floor:  (4000, 10, 0.5), (4000, 20, 0.5),
                      (8000, 100, 0.05), (8000, 200, 0.05)

The ONSET beta_on of a cell is its weakest grid rate whose distinct-completion
capacity is at least 32 items; the BEST is its rate of largest capacity. A
rate that never completes stores to 8192 items before giving up, so a window
that opens late is still seen.

    python -m research.runner onset --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_recall_law as rl                # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((2000, 60, 0.5), (4000, 60, 0.5), (4000, 80, 0.5), (4000, 160, 0.5),
         (8000, 120, 0.5), (8000, 240, 0.25),
         (4000, 10, 0.5), (4000, 20, 0.5), (8000, 100, 0.05), (8000, 200, 0.05))
F_LO, F_HI_ABOVE, F_HI_BELOW, STEPS_PER_OCTAVE = 0.12, 0.305, 0.42, 12
ONSET_ITEMS = 32
GIVE_UP = 8192
MIN_BETA = tl.MIN_BETA
STOP_FROM = 32
SEEDS = tuple(range(162, 182))
#: Amendment 17's best capacity at (4000, 60, 0.5) (OV)
A17 = {(4000, 60, 0.5): 1589.12}
#: the bars' bands
FRACTION_BAND = (0.13, 0.21)
FRACTION_CV = 0.10
LOAD_ASSISTED = 64
DENSE_BEST_OVER_ONSET = 2 ** 0.25
SPARSE_BEST_OVER_ONSET = 1.4


def betas(n, k, p):
    theta = tl.theta(n, k, p)
    top = F_HI_ABOVE if tl.above_floor(n, k, p) else F_HI_BELOW
    out, j = [], 0
    while F_LO * 2 ** (j / STEPS_PER_OCTAVE) <= top + 1e-12:
        b = round(theta * F_LO * 2 ** (j / STEPS_PER_OCTAVE), 4)
        if b >= MIN_BETA:
            out.append(b)
        j += 1
    return sorted(set(out))


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = max(rl.predicted(n, k, p), 64.0)
        grid = betas(n, k, p)
        out.append({"n": n, "k": k, "p": p, "above_floor": tl.above_floor(n, k, p),
                    "theta": tl.theta(n, k, p),
                    "betas": grid[len(grid) // 2: len(grid) // 2 + 2] if smoke else grid,
                    "cap": 32 if smoke else int(min(8 * guess, 40000)),
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
                               stop_on=("complete_distinct",), p=p, grid_start=2,
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
            w = windows["complete_distinct"]
            print(f"({n}, {k}, {p}) beta={beta:g} ({beta / spec['theta']:.3f} theta): "
                  f"distinct [{w.get('lower')}, {w.get('upper')}]", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p,
                                 "above_floor": spec["above_floor"],
                                 "theta": spec["theta"], "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def cell_reading(cell):
    """Onset, best, and the onset's load window, from one cell's sweep."""
    items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
    bs = [s["beta"] for s in items]
    caps = [lr.capacity(s["windows"]["complete_distinct"]) for s in items]
    on = next((i for i, c in enumerate(caps) if c >= ONSET_ITEMS), None)
    best = max(range(len(caps)), key=lambda i: caps[i])
    theta = cell["theta"]
    reading = {"betas": bs, "capacity": caps, "best": caps[best], "beta_best": bs[best],
               "onset": None, "fraction": None, "onset_lower": None,
               "best_over_onset": None, "onset_at_grid_start": on == 0}
    if on is not None:
        lower = items[on]["windows"]["complete_distinct"]["lower"]
        reading.update(onset=bs[on], fraction=bs[on] / theta, onset_lower=lower,
                       best_over_onset=bs[best] / bs[on])
    return reading


def evaluate(observations, *, a17=None):
    """Amendment 18's bars."""
    a17 = A17 if a17 is None else a17
    rows = {}
    for cell in observations["cells"].values():
        rows[(cell["n"], cell["k"], cell["p"])] = cell_reading(cell)
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    out["bars"]["OV"] = all(abs(rows[c]["best"] / v - 1) <= 0.15
                            for c, v in a17.items() if c in rows)
    fractions = [r["fraction"] for r in rows.values()]
    resolved = [f for f in fractions if f is not None]
    mean = sum(resolved) / len(resolved) if resolved else None
    cv = (math.sqrt(sum((f - mean) ** 2 for f in resolved) / len(resolved)) / mean
          if resolved else None)
    out["fraction_mean"], out["fraction_cv"] = mean, cv
    # an onset at the grid's first rate is not an onset that was found
    out["bars"]["O1"] = (len(resolved) == len(rows)
                         and not any(r["onset_at_grid_start"] for r in rows.values())
                         and all(FRACTION_BAND[0] <= f <= FRACTION_BAND[1] for f in resolved)
                         and cv is not None and cv <= FRACTION_CV)
    dense = [r for (n, k, p), r in rows.items() if p == 0.5]
    sparse = [r for (n, k, p), r in rows.items() if p == 0.05]
    out["bars"]["O2"] = bool(dense) and all(
        r["onset"] is not None and r["onset_lower"] is not None
        and r["onset_lower"] >= LOAD_ASSISTED for r in dense)
    out["bars"]["O3"] = bool(sparse) and all(
        r["onset"] is not None and r["onset_lower"] is None for r in sparse)
    out["bars"]["O4"] = bool(dense) and all(
        r["best_over_onset"] is not None
        and r["best_over_onset"] <= DENSE_BEST_OVER_ONSET * (1 + 1e-9) for r in dense)
    out["bars"]["O5"] = bool(sparse) and all(
        r["best_over_onset"] is not None
        and r["best_over_onset"] >= SPARSE_BEST_OVER_ONSET for r in sparse)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Onset", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: the ten cells)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 18 is registered on seeds 162..181")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.onset", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "f_lo": F_LO, "f_hi_above": F_HI_ABOVE,
                    "f_hi_below": F_HI_BELOW, "steps_per_octave": STEPS_PER_OCTAVE,
                    "onset_items": ONSET_ITEMS, "min_beta": MIN_BETA,
                    "stop_from": STOP_FROM, "strength": lr.STRENGTH, "w_max": pe.W_MAX,
                    "rounds": pe.T, "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
