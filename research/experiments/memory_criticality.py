"""Is the recall onset a phase transition? Finite-size scaling of the
pseudo-critical point, and critical slowing down of the read-out.

Registered in PREREG_refraction_memory.md, Amendment 24.

Amendment 18 put recall's onset -- the weakest write under which the refracted
memory completes its items -- at 0.16 to 0.18 theta at every cell, and found
capacity jumping from nothing to its maximum within one grid step. A jump is
not yet a transition. Two signatures separate a phase transition from a
threshold that merely looks sharp:

    the pseudo-critical point sharpens with size: each brain has its own
    onset (its weakest completing rate), and at a transition their spread
    across brains shrinks as n grows while their mean converges;

    critical slowing down: the read-out's relaxation time (frozen rounds until
    the winner set repeats with period one or two -- synchronous read-outs end
    in 2-cycles as often as at fixed points) grows on approach to the onset.

Four cells at fixed k p = 30, p = 0.5: n = 2000, 4000, 8000, 16000 (k = 60),
each on a grid of 24 rates to the octave, theta x 0.14 x 2^(j/24) up to 0.22
theta, with a separate 32-round read of every sampled cue at every checkpoint
for its settling round.

    python -m research.runner criticality --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_degree_law as dl                # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = tuple((n, 60, 0.5) for n in (2000, 4000, 8000, 16000))
F_LO, F_HI, STEPS_PER_OCTAVE = 0.14, 0.22, 24
SETTLE_ROUNDS = 32
ONSET_FROM = 32
STOP_FROM = 32
SEEDS = tuple(range(282, 302))
#: Amendment 18's best capacity at (4000, 60, 0.5) (CV)
A18 = {(4000, 60, 0.5): 1597.0}
SHARPEN, SETTLE_RATIO, BAND, DRIFT = 0.6, 1.5, (0.13, 0.21), 0.10


def betas(n, k, p):
    theta = tl.theta(n, k, p)
    out, j = [], 0
    while F_LO * 2 ** (j / STEPS_PER_OCTAVE) <= F_HI + 1e-12:
        out.append(round(theta * F_LO * 2 ** (j / STEPS_PER_OCTAVE), 5))
        j += 1
    return sorted(set(out))


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = dl.predicted(n, k, p)
        grid = betas(n, k, p)
        out.append({"n": n, "k": k, "p": p, "theta": tl.theta(n, k, p),
                    "betas": grid[6:8] if smoke else grid,
                    "cap": 32 if smoke else int(3 * guess + 256),
                    "give_up": 16 if smoke else int(max(8192, 2 * guess))})
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
                               give_up=spec["give_up"], stop_from=STOP_FROM,
                               settle_rounds=parameters["settle_rounds"])
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
            print(f"({n}, {k}, {p}) beta={beta:g} ({beta / spec['theta']:.4f} theta): "
                  f"distinct [{w.get('lower')}, {w.get('upper')}]", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "theta": spec["theta"],
                                 "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _per_seed(ensemble):
    return dict(zip(ensemble["keys"], ensemble["values"]))


def seed_onsets(cell):
    """Each brain's onset: its weakest rate whose own distinct completion
    exceeds one half at some checkpoint with at least ONSET_FROM items."""
    items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
    onsets = {}
    for s in items:
        for M, metrics in s["ensembles"].items():
            if int(M) < ONSET_FROM:
                continue
            for seed, v in _per_seed(metrics["complete_distinct"]).items():
                if v > pe.HALF_BAR and seed not in onsets:
                    onsets[seed] = s["beta"] / cell["theta"]
    return onsets


def settle_at(s):
    """Mean settling round over the checkpoints inside the rate's distinct
    window (where the ensemble's distinct completion exceeds one half)."""
    values = [statistics.fmean(m["settle"]["values"]) for M, m in s["ensembles"].items()
              if "settle" in m and statistics.fmean(m["complete_distinct"]["values"]) > pe.HALF_BAR]
    return statistics.fmean(values) if values else None


def cell_reading(cell):
    items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
    caps = [lr.capacity(s["windows"]["complete_distinct"]) for s in items]
    onsets = seed_onsets(cell)
    values = list(onsets.values())
    first = next((i for i, c in enumerate(caps) if c >= ONSET_FROM), None)
    half_up = None
    if first is not None:
        target = items[first]["beta"] * 2 ** 0.5
        half_up = min(range(len(items)), key=lambda i: abs(items[i]["beta"] - target))
    unsettled = {f"{s['beta'] / cell['theta']:.4f}": statistics.fmean(
                     statistics.fmean(m["unsettled"]["values"]) for m in s["ensembles"].values()
                     if "unsettled" in m) for s in items
                 if any("unsettled" in m for m in s["ensembles"].values())}
    return {"best": max(caps), "n_onsets": len(values), "unsettled_by_rate": unsettled,
            "onset_mean": statistics.fmean(values) if values else None,
            "onset_sd": statistics.pstdev(values) if len(values) > 1 else None,
            "settle_onset": settle_at(items[first]) if first is not None else None,
            "settle_half_octave_up": settle_at(items[half_up]) if half_up is not None else None}


def evaluate(observations, *, a18=None):
    """Amendment 24's bars."""
    a18 = A18 if a18 is None else a18
    rows = {(c["n"], c["k"], c["p"]): cell_reading(c) for c in observations["cells"].values()}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    out["bars"]["CV"] = all(abs(rows[c]["best"] / v - 1) <= 0.15 for c, v in a18.items() if c in rows)
    seq = [rows[c] for c in CELLS if c in rows]
    sds = [r["onset_sd"] for r in seq]
    # non-increasing within 10% (the grid resolves onsets to ~0.005 theta, so
    # spreads at that floor tie), and sharper by 0.6 from n = 2000 to 16000
    out["bars"]["F1"] = (len(seq) == len(CELLS) and None not in sds
                         and all(b <= 1.1 * a for a, b in zip(sds, sds[1:]))
                         and sds[-1] <= SHARPEN * sds[0])
    means = [r["onset_mean"] for r in seq]
    out["bars"]["F2"] = (len(seq) == len(CELLS) and None not in means
                         and all(BAND[0] <= m <= BAND[1] for m in means)
                         and abs(means[-1] / means[-2] - 1) <= DRIFT)
    out["bars"]["F3"] = (len(seq) == len(CELLS) and all(
        r["settle_onset"] is not None and r["settle_half_octave_up"]
        and r["settle_onset"] >= SETTLE_RATIO * r["settle_half_octave_up"] for r in seq))
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Criticality", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all four)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 24 is registered on seeds 282..301")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.criticality", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "f_lo": F_LO, "f_hi": F_HI,
                    "steps_per_octave": STEPS_PER_OCTAVE, "settle_rounds": SETTLE_ROUNDS,
                    "onset_from": ONSET_FROM, "stop_from": STOP_FROM,
                    "strength": lr.STRENGTH, "w_max": pe.W_MAX, "rounds": pe.T,
                    "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
