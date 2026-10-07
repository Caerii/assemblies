"""The tiling deadline: a refracted area's sequence tiles it, and replay
breaks where the tiling wraps.

Registered in PREREG_refraction_memory.md, Amendment 28.

Amendment 27 found the refracted area's sequence replay breaking at step
n/k on nearly every brain at the sparse cells. Step n/k is when a sequence of
k-neuron states has used n neurons. Refraction charges every winner a bias
that never decays, so while unused neurons remain each new state is drawn
from them: the sequence TILES the area, and at n/k the fresh pool is empty.
This study measures the tiling (each element's share of never-fired
winners), whether replay breaks at multiples of n/k, and whether a bias that
recovers -- zeroed every n/(2k) elements -- removes the deadline.

    python -m research.runner tiling --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 30, 0.5))
STRENGTH, MATCH, TILINGS = 0.5, 0.3, 4
SEEDS = tuple(range(362, 382))
FRESH_IN, FRESH_OUT, NEAR, NEAR_MIN, ON_DEADLINE, MIN_BREAKS = 0.95, 0.10, 0.05, 2, 0.8, 10
RESET_FLOOR, RESET_GAIN = 0.9, 0.3


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        tile = n / k
        out.append({"n": n, "k": k, "p": p, "tile": tile, "beta": round(tl.theta(n, k, p), 5),
                    "length": 24 if smoke else int(round(TILINGS * tile)),
                    "reset": max(1, int(tile // 2))})
    return out


def run_arm(spec, reset, seeds, device):
    """One sequence per brain; per-element fresh share [L, B] and each brain's
    first-break step (L - 1 if it replays the whole sequence)."""
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p, L = spec["n"], spec["k"], spec["p"], spec["length"]
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                         norm_init=True, rounds=1, strength=STRENGTH, max_items=4, device=device)
    elements = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"q0e{e}", "A")) for sd in seeds]
                for e in range(L)]
    states, _, fresh = mem.store_sequence(elements, rounds_per_element=1, bias_reset=reset,
                                          return_fresh=True)
    x = states[0][:, :k // 2]
    alive = torch.ones(len(seeds), dtype=torch.bool, device=states.device)
    steps = torch.zeros(len(seeds), device=states.device)
    for j in range(1, L):
        x = mem.recall(x, rounds=1)
        alive &= sq._overlap(x, states[j]) >= MATCH
        steps += alive.float()
    return fresh.mean(dim=1).tolist(), steps.tolist()


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    out = {}
    for spec in parameters["cells"]:
        cell = dict(spec, arms={})
        for name, reset in (("refracted", None), ("reset", spec["reset"])):
            fresh, breaks = run_arm(spec, reset, seeds, parameters["device"])
            cell["arms"][name] = {"reset": reset, "fresh_mean": fresh,
                                  "break": asdict(ensemble_from_values(breaks, keys=seeds,
                                                                       label="break"))}
            print(f"({spec['n']}, {spec['k']}, {spec['p']}) {name}: breaks "
                  f"{sorted(int(b) for b in breaks)} of {spec['length'] - 1} "
                  f"(n/k {spec['tile']:.1f})", flush=True)
        out[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def tiled(cell):
    """Mean fresh share >= FRESH_IN at every element <= 0.9 n/k, and <=
    FRESH_OUT at every element in [1.1, 1.5] n/k."""
    fresh, tile = cell["arms"]["refracted"]["fresh_mean"], cell["tile"]
    early = [v for e, v in enumerate(fresh) if e <= 0.9 * tile]
    late = [v for e, v in enumerate(fresh) if 1.1 * tile <= e <= 1.5 * tile]
    return bool(early and late and min(early) >= FRESH_IN and max(late) <= FRESH_OUT)


def on_deadline(step, tile):
    """Within max(NEAR x n/k, NEAR_MIN) steps of a positive multiple of n/k."""
    j = max(1, round(step / tile))
    return abs(step - j * tile) <= max(NEAR * tile, NEAR_MIN)


def evaluate(observations):
    """Amendment 28's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    full = all(c in cells for c in CELLS)
    rows, breaks = {}, []
    for key, c in cells.items():
        b = c["arms"]["refracted"]["break"]["values"]
        broke = [s for s in b if s < c["length"] - 1]
        breaks += [on_deadline(s, c["tile"]) for s in broke]
        rows[f"{key[0]}/{key[1]}/{key[2]:g}"] = {
            "tiled": tiled(c), "broke": len(broke),
            "on_deadline": sum(on_deadline(s, c["tile"]) for s in broke),
            "refracted_fraction": statistics.fmean(s / (c["length"] - 1) for s in b),
            "reset_fraction": statistics.fmean(s / (c["length"] - 1)
                                               for s in c["arms"]["reset"]["break"]["values"])}
    out = {"cells": rows, "bars": {}}
    if not full:
        out["bars"] = {"D1": False, "D2": False, "D3": False}
        return out
    out["bars"]["D1"] = all(r["tiled"] for r in rows.values())
    out["bars"]["D2"] = (len(breaks) >= MIN_BREAKS
                         and sum(breaks) >= ON_DEADLINE * len(breaks))
    sparse = [f"{n}/{k}/{p:g}" for n, k, p in CELLS if n / k > 100]
    out["bars"]["D3"] = (all(r["reset_fraction"] >= RESET_FLOOR for r in rows.values())
                         and all(rows[c]["reset_fraction"] - rows[c]["refracted_fraction"]
                                 >= RESET_GAIN for c in sparse))
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Tiling", engines=("hashed_assembly_memory",),
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
        ap.error("Amendment 28 is registered on seeds 362..381")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.tiling", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": STRENGTH, "match": MATCH,
                    "tilings": TILINGS, "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
