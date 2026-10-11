"""When must the write happen? Burst-timing-dependent plasticity, and the two
controls that take it apart.

Registered in PREREG_refraction_memory.md, Amendment 25.

The memory's write (the ROUND write) is spike-timing plasticity at the scale
of one round: every round, a synapse whose pre fired the round before and
whose post fires now gains a count, and the counts feed straight back into
the item's next round. Burst-timing-dependent plasticity (Butts, Kanold &
Shatz 2007, at developing retinogeniculate synapses) differs from it twice:
it potentiates pairs whose BURSTS coincide on a long window, ORDER within it
ignored, and it acts on the item's activity as a whole, not round by round.
Four rules separate the two departures:

    round          online, every firing, causal (the memory's write)
    online_burst   online, causal, only between neurons that have already
                   fired `burst_min` = 2 times in the item (burst gating alone)
    deferred       the round write's own counts, written AFTER the item's
                   rounds (deferral alone: nothing written feeds back)
    burst          deferred, symmetric, one count between every two neurons
                   that fired at least twice in the item (the BTDP rule)

Each rule is swept on its own grid of write rates, and its capacity read at
its own best rate. A second reading asks what a deferred write stores: one
frozen round from half of the item's round-t winners, scored against round
t + 1's.

    python -m research.runner write_rules --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
#: registered names, owned by research.experiments.memory_lib since 2026-10-10 (re-exported)
from research.experiments.memory_lib.readout import overlap as _overlap # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

RULES = ("round", "online_burst", "deferred", "burst")
BURST_MIN = 2
CELLS = ((2000, 60, 0.5), (4000, 20, 0.5), (4000, 160, 0.5))
#: theta x 0.1 x 2^(j/4), j = 0..20: 0.1 to 3.2 theta, four rates to the octave
F_LO, STEPS_PER_OCTAVE, N_RATES = 0.1, 4, 21
#: a rule that has not completed by this load stops (capacity 0 is then read
#: from M = 2 to here); the round write runs to its capacity
NULL_LOAD = 256
STOP_FROM = 32
SEEDS = tuple(range(302, 322))
#: the trajectory reading: items stored, the rate (in theta); the items read
#: are the last and those a quarter, half and three quarters back
TRAJ_ITEMS, TRAJ_RATE = 200, 1.0
#: Amendment 18's best capacities at these cells (CV)
A18 = {(2000, 60, 0.5): 506.7, (4000, 20, 0.5): 1553.6, (4000, 160, 0.5): 1083.2}
CV_BAND = (0.75, 1.10)
GATING_RATIO, NEXT_FLOOR, NEXT_OVER_SAME, OWN_CEIL, ATTRACTOR_FLOOR = 0.5, 0.4, 5.0, 0.1, 0.5


def betas(n, k, p):
    theta = lib.theta(n, k, p)
    return sorted({round(theta * F_LO * 2 ** (j / STEPS_PER_OCTAVE), 5) for j in range(N_RATES)})


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = dl.predicted(n, k, p)
        grid = betas(n, k, p)
        out.append({"n": n, "k": k, "p": p, "theta": lib.theta(n, k, p),
                    "betas": grid[7:9] if smoke else grid,
                    "cap": 32 if smoke else int(3 * guess + 256),
                    "give_up": 16 if smoke else NULL_LOAD,
                    "traj_items": 24 if smoke else TRAJ_ITEMS})
    return out


def reads(items):
    """The items the trajectory reading reads: 199, 150, 100, 50 of 200."""
    return (items - 1, 3 * items // 4, items // 2, items // 4)


def trajectory(n, k, p, beta, seeds, rule, items, device):
    """Store `items` items under `rule`; for the items `reads(items)`, one
    frozen round from half of round t's winners scored against round t + 1's
    ("next") and round t's ("same"), and the rounds' own overlap of round t
    with t + 1 ("own"), per brain, averaged over t and items."""
    import torch
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=beta, w_max=lib.W_MAX,
                         norm_init=True, rounds=lib.ROUNDS, strength=lib.STRENGTH,
                         max_items=items + 1, device=device, write_rule=rule,
                         burst_min=BURST_MIN)
    project, trace, burst = mem.area.project, [], []

    def recorded(*args, **kwargs):
        kwargs.setdefault("record", [])
        out = project(*args, **kwargs)
        trace.append(list(kwargs["record"]))
        return out
    mem.area.project = recorded
    for a in range(items):
        mem.store([to_i32(_seeding.fnv1a_pair_seed(s, f"s{a}", "A")) for s in seeds],
                  stim_size=k)
        if rule == "burst":
            burst.append(mem.last_burst.sum(dim=1).float() / k)
    mem.area.project = project
    nxt, same, own = [], [], []
    for i in reads(items):
        r = trace[i]
        for t in range(len(r) - 1):
            got = mem.recall(r[t][:, :k // 2], rounds=1)
            nxt.append(_overlap(got, r[t + 1]))
            same.append(_overlap(got, r[t]))
            own.append(_overlap(r[t], r[t + 1]))
    out = {name: torch.stack(v).mean(dim=0).tolist()
           for name, v in (("next", nxt), ("same", same), ("own", own))}
    if burst:
        out["burst_over_k"] = torch.stack(burst).mean(dim=0).tolist()
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
        cell = {"n": n, "k": k, "p": p, "theta": spec["theta"], "rules": {}}
        for rule in parameters["rules"]:
            results = lr.run_betas(n, k, grid, seeds, spec["cap"], device,
                                   {b: profiles[lib.profile_name(b)] for b in grid},
                                   stop_on=("complete_distinct",), p=p, grid_start=2,
                                   give_up=spec["give_up"], stop_from=STOP_FROM,
                                   write_rule=rule, burst_min=BURST_MIN)
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
            best = max(lr.capacity(s["windows"]["complete_distinct"]) for s in sweep.values())
            print(f"({n}, {k}, {p}) {rule}: best capacity {best:.1f}", flush=True)
            beta_t = round(spec["theta"] * parameters["traj_rate"], 5)
            traj = trajectory(n, k, p, beta_t, seeds, rule, spec["traj_items"], device)
            print(f"({n}, {k}, {p}) {rule} at {parameters['traj_rate']} theta: "
                  f"next {statistics.fmean(traj['next']):.3f} same "
                  f"{statistics.fmean(traj['same']):.3f} own {statistics.fmean(traj['own']):.3f}",
                  flush=True)
            cell["rules"][rule] = {"sweep": sweep, "trajectory": {
                "beta": beta_t, **{nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                   for nm, v in traj.items()}}}
        out[f"{n}/{k}/{p:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def rule_reading(rule):
    caps = {s["beta"]: lr.capacity(s["windows"]["complete_distinct"])
            for s in rule["sweep"].values()}
    best_beta = max(caps, key=lambda b: caps[b])
    t = rule["trajectory"]
    return {"best": caps[best_beta], "best_beta": best_beta,
            "next": t["next"]["mean"], "same": t["same"]["mean"], "own": t["own"]["mean"],
            **({"burst_over_k": t["burst_over_k"]["mean"]} if "burst_over_k" in t else {})}


def evaluate(observations, *, a18=None):
    """Amendment 25's bars."""
    a18 = A18 if a18 is None else a18
    rows = {(c["n"], c["k"], c["p"]): {r: rule_reading(v) for r, v in c["rules"].items()}
            for c in observations["cells"].values()}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    full = len(rows) == len(CELLS) and all(set(RULES) <= set(r) for r in rows.values())
    # the grid is a quarter octave coarse, so its best rate can sit up to an
    # eighth of an octave off Amendment 18's optimum: the band is wider below
    out["bars"]["CV"] = full and all(CV_BAND[0] <= rows[c]["round"]["best"] / v <= CV_BAND[1]
                                     for c, v in a18.items())
    out["bars"]["W1"] = full and all(r["deferred"]["best"] == 0 for r in rows.values())
    out["bars"]["W2"] = full and all(r["burst"]["best"] == 0 for r in rows.values())
    out["bars"]["W3"] = full and all(r["online_burst"]["best"] <= GATING_RATIO * r["round"]["best"]
                                     for r in rows.values())
    out["bars"]["W4"] = full and all(
        r["deferred"]["next"] >= NEXT_FLOOR
        and r["deferred"]["next"] >= NEXT_OVER_SAME * r["deferred"]["same"]
        and r["deferred"]["own"] <= OWN_CEIL and r["round"]["own"] >= ATTRACTOR_FLOOR
        for r in rows.values())
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Write rules", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all three)")
    ap.add_argument("--rules", help="comma-separated subset of " + ",".join(RULES))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    rules = args.rules.split(",") if args.rules else list(RULES)
    if any(r not in RULES for r in rules):
        ap.error(f"rules must be members of {RULES}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 25 is registered on seeds 302..321")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(b): lib.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.write_rules", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "rules": rules, "burst_min": BURST_MIN, "f_lo": F_LO,
                    "steps_per_octave": STEPS_PER_OCTAVE, "n_rates": N_RATES,
                    "null_load": NULL_LOAD, "stop_from": STOP_FROM,
                    "traj_rate": TRAJ_RATE,
                    "traj_reads": [list(reads(s["traj_items"])) for s in specs],
                    "strength": lib.STRENGTH, "w_max": lib.W_MAX, "rounds": lib.ROUNDS,
                    "half_bar": lib.HALF_BAR, "complete": lib.COMPLETE,
                    "recall_sample": lib.RECALL_SAMPLE,
                    "measurement_seed": lib.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
