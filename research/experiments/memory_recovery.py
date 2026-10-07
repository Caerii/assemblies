"""A refraction that recovers: how the recovery time sets how long a sequence
one area can recall.

Registered in PREREG_refraction_memory.md, Amendment 29.

Amendment 27 found the Hebbian area's single-sequence limit at about
0.09 p (n/k)^2 (a merging cliff), and Amendment 28 the refracted area's at a
deadline of n/k elements (the never-decaying bias tiles the area). Both are
ends of one knob: a bias that decays by exp(-1/tau) every writing round
(`AssemblyMemory(bias_decay=...)`) is the Hebbian area at tau = 0 (decay 0,
tested bit for bit) and the cumulative bias at tau = infinity. This study
sweeps tau and measures each one's length limit -- the L at which the mean
replay fraction of one chosen sequence per brain falls through one half.

    python -m research.runner recovery --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
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
#: recovery times in writing rounds (= elements); 0 is the Hebbian area,
#: None the cumulative (never-decaying) bias
TAUS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, None)
LADDER = tuple(int(round(16 * 2 ** (j / 2))) for j in range(21))        # 16 .. 16384
STRENGTH, MATCH, STOP = 0.5, 0.3, 0.2
SEEDS = tuple(range(382, 402))
GAIN, EDGE_BAND, LAW_SPREAD = 2.0, (0.5, 4.0), 1.5


def tau_name(tau):
    return "inf" if tau is None else str(tau)


def decay_of(tau):
    if tau is None:
        return None
    return 0.0 if tau == 0 else math.exp(-1.0 / tau)


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tile": n / k, "beta": round(tl.theta(n, k, p), 5),
             "taus": [tau_name(t) for t in ((0, 16, None) if smoke else TAUS)],
             "ladder": list(LADDER[:3] if smoke else LADDER)} for n, k, p in cells]


def replay_fraction(spec, tau, L, seeds, device):
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                         norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                         device=device, bias_decay=decay_of(tau))
    elements = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"q0e{e}", "A")) for sd in seeds]
                for e in range(L)]
    states, _ = mem.store_sequence(elements, rounds_per_element=1)
    x = states[0][:, :k // 2]
    alive = torch.ones(len(seeds), dtype=torch.bool, device=states.device)
    steps = torch.zeros(len(seeds), device=states.device)
    for j in range(1, L):
        x = mem.recall(x, rounds=1)
        alive &= sq._overlap(x, states[j]) >= MATCH
        steps += alive.float()
    return (steps / (L - 1)).tolist()


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    out = {}
    for spec in parameters["cells"]:
        cell = dict(spec, taus={})
        for name in spec["taus"]:
            tau = None if name == "inf" else int(name)
            curve, risen, low = {}, False, 0
            for L in spec["ladder"]:
                values = replay_fraction(spec, tau, L, seeds, parameters["device"])
                curve[str(L)] = asdict(ensemble_from_values(values, keys=seeds, label="replay"))
                mean = statistics.fmean(values)
                risen = risen or mean > 0.5
                low = low + 1 if mean < STOP else 0
                if low >= 2 or (not risen and low >= 1):
                    break
            limit = sq.capacity({L: e["mean"] for L, e in curve.items()})
            cell["taus"][name] = {"curve": curve, "limit": limit}
            print(f"({spec['n']}, {spec['k']}, {spec['p']}) tau={name}: limit {limit:.0f}",
                  flush=True)
        out[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def cell_reading(cell):
    limits = {name: t["limit"] for name, t in cell["taus"].items()}
    finite = {int(t): v for t, v in limits.items() if t != "inf"}
    best_tau = max(finite, key=lambda t: finite[t])
    best = finite[best_tau]
    # the upper edge: the largest finite tau whose limit is at least half the best
    edge = max(t for t, v in finite.items() if v >= 0.5 * best)
    return {"limits": limits, "best_tau": best_tau, "best": best,
            "hebbian": limits.get("0"), "cumulative": limits.get("inf"),
            "edge": edge, "edge_over_tile": edge / cell["tile"],
            "law": best / cell["tile"] ** 2}


def evaluate(observations):
    """Amendment 29's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    rows = {key: cell_reading(c) for key, c in cells.items()}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    if not all(c in rows for c in CELLS):
        out["bars"] = {b: False for b in ("B1", "B2", "B3", "B4")}
        return out
    out["bars"]["B1"] = all(r["best"] >= GAIN * max(r["hebbian"], r["cumulative"])
                            for r in rows.values())
    largest = max(t for t in TAUS if t is not None)
    out["bars"]["B2"] = all(0 < r["best_tau"] < largest for r in rows.values())
    out["bars"]["B3"] = all(EDGE_BAND[0] <= r["edge_over_tile"] <= EDGE_BAND[1]
                            for r in rows.values())
    laws = [r["law"] for r in rows.values()]
    out["bars"]["B4"] = max(laws) <= LAW_SPREAD * min(laws)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Recovery", engines=("hashed_assembly_memory",),
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
        ap.error("Amendment 29 is registered on seeds 382..401")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.recovery", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": STRENGTH, "match": MATCH, "stop": STOP,
                    "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
