"""How long a sequence can one area recall by itself?

Registered in PREREG_refraction_memory.md, Amendment 27.

A chosen sequence of L elements -- each its own stimulus, written in ONE
stimulus-and-recurrence round (`AssemblyMemory.store_sequence`), the area
inhibited once before the first -- at a write equal to the convergence
threshold theta, the construction under which the calculus's own
`ordered_recall` advances (ORDERED-RECALL-BY-TRANSITIONS). Replay: half of
element 0's winners, then frozen masked rounds each fed the previous round's
winners; a brain's REPLAY FRACTION is the number of elements matched in order
(own overlap >= 0.3, the strict xfail's measure) before the first miss, over
L - 1. Two arms: the Hebbian write (no refraction) and the memory's write
(refraction 0.5 beta). The length limit of an arm at a cell is the L at
which the mean replay fraction falls through one half, log-interpolated on a
ladder of L = 16 x 2^(j/4).

    python -m research.runner sequence_length --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: two equal-n/k pairs (33: 2000/60 and 4000/120; 133: 8000/60 and 4000/30)
#: and a change of p at fixed n/k (4000/60 at 0.5 and 0.25)
CELLS = ((2000, 60, 0.5), (4000, 120, 0.5), (4000, 60, 0.5), (4000, 60, 0.25),
         (8000, 60, 0.5), (4000, 30, 0.5))
ARMS = (("hebbian", 0.0), ("refracted", 0.5))
LADDER = tuple(int(round(16 * 2 ** (j / 4))) for j in range(37))       # 16 .. 8192
MATCH, STOP = 0.3, 0.2
SEEDS = tuple(range(342, 362))
#: the probe's constant for L_H / (p (n/k)^2), and the bars' tolerances
LAW_BAND, PAIR_TOL, SHARP_IN, SHARP_OUT, SHARP_STEP, EXTENDS = (0.06, 0.12), 0.30, 0.9, 0.2, 2 ** 0.5, 2.0


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "theta": lib.theta(n, k, p),
             "beta": round(lib.theta(n, k, p), 5),
             "ladder": list(LADDER[:3] if smoke else LADDER)} for n, k, p in cells]


def replay_fraction(n, k, p, beta, s, L, seeds, device):
    """One chosen sequence of L elements per brain; each brain's replay
    fraction [B]."""
    import torch
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=beta, w_max=lib.W_MAX,
                         norm_init=True, rounds=1, strength=s, max_items=4, device=device)
    elements = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"q0e{e}", "A")) for sd in seeds]
                for e in range(L)]
    states, _ = mem.store_sequence(elements, rounds_per_element=1)
    x = states[0][:, :k // 2]
    alive = torch.ones(len(seeds), dtype=torch.bool, device=states.device)
    steps = torch.zeros(len(seeds), device=states.device)
    for j in range(1, L):
        x = mem.recall(x, rounds=1)
        alive &= lib.overlap(x, states[j]) >= MATCH
        steps += alive.float()
    return (steps / (L - 1)).tolist()


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        cell = {"n": n, "k": k, "p": p, "theta": spec["theta"], "beta": spec["beta"], "arms": {}}
        for name, s in ARMS:
            curve, risen, low = {}, False, 0
            for L in spec["ladder"]:
                values = replay_fraction(n, k, p, spec["beta"], s, L, seeds, device)
                curve[str(L)] = asdict(ensemble_from_values(values, keys=seeds, label="replay"))
                mean = statistics.fmean(values)
                risen = risen or mean > 0.5
                low = low + 1 if mean < STOP else 0
                if low >= 2 or (not risen and low >= 1):
                    break
            limit = sq.capacity({L: e["mean"] for L, e in curve.items()})
            print(f"({n}, {k}, {p}) {name}: limit {limit:.0f} "
                  f"({limit / (p * (n / k) ** 2):.4f} p (n/k)^2)", flush=True)
            cell["arms"][name] = {"s": s, "curve": curve, "limit": limit}
        out[f"{n}/{k}/{p:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def law_constant(cell):
    return cell["arms"]["hebbian"]["limit"] / (cell["p"] * (cell["n"] / cell["k"]) ** 2)


def sharp(arm):
    """The Hebbian failure is a cliff: mean replay fraction >= SHARP_IN at the
    largest ladder point at or below limit / sqrt 2, and <= SHARP_OUT at the
    smallest at or above limit x sqrt 2."""
    limit = arm["limit"]
    points = sorted((int(L), e["mean"]) for L, e in arm["curve"].items())
    below = [v for L, v in points if L <= limit / SHARP_STEP]
    above = [v for L, v in points if L >= limit * SHARP_STEP]
    return bool(below and above and below[-1] >= SHARP_IN and above[0] <= SHARP_OUT)


def evaluate(observations):
    """Amendment 27's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    full = all(c in cells for c in CELLS)
    out = {"cells": {f"{n}/{k}/{p:g}": {"hebbian": c["arms"]["hebbian"]["limit"],
                                         "refracted": c["arms"]["refracted"]["limit"],
                                         "constant": law_constant(c),
                                         "sharp": sharp(c["arms"]["hebbian"])}
                     for (n, k, p), c in cells.items()}, "bars": {}}
    if not full:
        out["bars"] = {b: False for b in ("L1", "L2", "L3", "L4")}
        return out
    h = {c: cells[c]["arms"]["hebbian"]["limit"] for c in CELLS}
    out["bars"]["L1"] = all(LAW_BAND[0] <= law_constant(cells[c]) <= LAW_BAND[1] for c in CELLS)
    pairs = (((2000, 60, 0.5), (4000, 120, 0.5)), ((8000, 60, 0.5), (4000, 30, 0.5)))
    out["bars"]["L2"] = all(abs(h[a] / h[b] - 1) <= PAIR_TOL for a, b in pairs)
    out["bars"]["L3"] = all(sharp(cells[c]["arms"]["hebbian"]) for c in CELLS)
    out["bars"]["L4"] = all(cells[c]["arms"]["refracted"]["limit"] >= EXTENDS * h[c] for c in CELLS)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Sequence length", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all six)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 27 is registered on seeds 342..361")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.sequence-length", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "arms": dict(ARMS), "match": MATCH, "stop": STOP,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
