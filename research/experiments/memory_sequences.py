"""Sequences from adaptation: one recurrent area, one causal Hebbian rule,
and the refraction that moves its activity, store either attractors or
sequences.

Registered in PREREG_refraction_memory.md, Amendment 26.

Amendment 25 found that the round write's counts, written AFTER the item's
rounds instead of during them, store no attractor but the item's TRAJECTORY:
refraction relocates the unwritten activity every round, and the write
records each round's transition to the next. This study asks whether that
is a sequence memory -- whether the area REPLAYS a stored trajectory by
itself, from a partial cue of its first state -- how many sequences it
holds, and whether the ratio of refraction to plasticity (s / beta) is the
switch between the two memories when the write stays online:

    deferred, s = 0.5 beta    the round write's counts, written after the item
    online,   s = 1.5 beta    the round write, with refraction stronger than
                              plasticity (activity moves every round as it is
                              written)
    online,   s = 0.5 beta    the memory's own write (the attractor control)

REPLAY: the item's round-0 winners, first half, then frozen masked rounds,
each fed the previous round's winners; the replay LENGTH is the number of
steps before the overlap with the item's own round j first falls below one
half, as a fraction of T - 1. Sequence capacity is the load at which the mean
replay length over the stored items falls through one half.

    python -m research.runner sequences --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_lib as lib                      # noqa: E402
#: registered names, owned by research.experiments.memory_lib since 2026-10-10 (re-exported)
from research.experiments.memory_lib.readout import overlap as _overlap # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: (n, k, p): n/k = 33, 67, 33 and in-degree d = n p = 1000, 2000, 2000 --
#: the first and third share n/k, the second and third share d
CELLS = ((2000, 60, 0.5), (4000, 60, 0.5), (4000, 120, 0.5))
#: (write rule, s / beta, rates in theta)
ARMS = (("deferred", 0.5, (0.5, 0.7, 1.0, 1.4, 2.0, 2.8)),
        ("round", 1.5, (0.5, 0.7, 1.0, 1.4, 2.0, 2.8)),
        ("round", 0.5, (0.2,)))
T = 8
#: checkpoints: 32 x 2^(j/4) up to 16384 items
CHECKPOINTS = tuple(sorted({int(round(32 * 2 ** (j / 4))) for j in range(37)}))
READ_ITEMS = 16
#: a rate whose replay never exceeds one half stops past this load; one that
#: has, stops after two consecutive checkpoints at or below STOP_LEN
NULL_LOAD, STOP_LEN = 1024, 0.25
SEEDS = tuple(range(322, 342))
#: the switch reading: during-write consecutive overlap over items 16..31
#: (every rate stores at least 32)
OWN_FROM, OWN_AT = 16, 32
FULL, ONLINE_FLOOR, OWN_HIGH, OWN_LOW, SAME, APART = 0.9, 0.7, 0.5, 0.1, 0.25, 2.0


def arm_name(rule, s):
    return f"{rule}-s{s:g}"


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        theta = lib.theta(n, k, p)
        arms = []
        for rule, s, rates in ARMS:
            grid = (rates[2:3] or rates[:1]) if smoke else rates
            arms.append({"name": arm_name(rule, s), "rule": rule, "s": s,
                         "betas": [round(theta * f, 5) for f in grid]})
        out.append({"n": n, "k": k, "p": p, "theta": theta, "arms": arms,
                    "checkpoints": [m for m in CHECKPOINTS if m <= (64 if smoke else 16384)]})
    return out


def replay(mem, trace, items, k):
    """Mean over `items` of each brain's replay length (fraction of T - 1)
    and of its last step's overlap: [B], [B]."""
    import torch
    lengths, last = [], []
    for i in items:
        r = trace[i].long()
        x = r[0][:, :k // 2]
        alive = torch.ones(r.shape[1], dtype=torch.bool, device=r.device)
        length = torch.zeros(r.shape[1], device=r.device)
        o = None
        for j in range(1, r.shape[0]):
            x = mem.recall(x, rounds=1)
            o = _overlap(x, r[j])
            alive &= o >= 0.5
            length += alive.float()
        lengths.append(length / (r.shape[0] - 1))
        last.append(o)
    return torch.stack(lengths).mean(dim=0), torch.stack(last).mean(dim=0)


def run_rate(n, k, p, beta, rule, s, seeds, checkpoints, device):
    """Store items under (rule, s) at `beta`, reading replay (or, for the
    attractor control, 8-round half-cue completion of the item's last round)
    at each checkpoint; returns {M: {metric: [B]}} and the during-write
    consecutive overlap at OWN_AT items."""
    import torch
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=beta, w_max=lib.W_MAX,
                         norm_init=True, rounds=T, strength=s,
                         max_items=checkpoints[-1] + 1, device=device, write_rule=rule)
    project, trace = mem.area.project, []

    def recorded(*args, **kwargs):
        kwargs.setdefault("record", [])
        out = project(*args, **kwargs)
        # int16 holds any neuron index here (n <= 4000) at a quarter the memory
        trace.append(torch.stack(kwargs["record"]).to(torch.int16))    # [T, B, k]
        return out
    mem.area.project = recorded
    cache, own, a, risen, low = {}, None, 0, False, 0
    attractor = rule == "round" and s < 1
    for M in checkpoints:
        while a < M:
            mem.store([to_i32(_seeding.fnv1a_pair_seed(sd, f"s{a}", "A")) for sd in seeds],
                      stim_size=k)
            a += 1
            if a == OWN_AT:
                own = torch.stack([_overlap(trace[i][t].long(), trace[i][t + 1].long())
                                   for i in range(OWN_FROM, OWN_AT)
                                   for t in range(T - 1)]).mean(dim=0).tolist()
        mem.area.project = project
        items = [round(i * (M - 1) / (READ_ITEMS - 1)) for i in range(READ_ITEMS)]
        if attractor:
            done = torch.stack([_overlap(mem.recall(trace[i][-1][:, :k // 2].long()),
                                         trace[i][-1].long())
                                for i in items]).mean(dim=0)
            cache[M] = {"complete": done.tolist()}
            value = float(done.mean())
        else:
            length, last = replay(mem, trace, items, k)
            cache[M] = {"replay": length.tolist(), "last": last.tolist()}
            value = float(length.mean())
        mem.area.project = recorded
        risen = risen or value > 0.5
        low = low + 1 if (risen and value <= STOP_LEN) else 0
        if low >= 2 or (not risen and M >= NULL_LOAD):
            break
    return cache, own


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        cell = {"n": n, "k": k, "p": p, "theta": spec["theta"], "arms": {}}
        for arm in spec["arms"]:
            rates = {}
            for beta in arm["betas"]:
                cache, own = run_rate(n, k, p, beta, arm["rule"], arm["s"], seeds,
                                      spec["checkpoints"], device)
                metric = "complete" if "complete" in next(iter(cache.values())) else "replay"
                cap = capacity({M: statistics.fmean(v[metric]) for M, v in cache.items()})
                print(f"({n}, {k}, {p}) {arm['name']} beta={beta:g} "
                      f"({beta / spec['theta']:.2f} theta): capacity {cap:.0f}"
                      + (f", own {statistics.fmean(own):.3f}" if own else ""), flush=True)
                rates[f"{beta:g}"] = {
                    "beta": beta, "metric": metric, "capacity": cap,
                    "own": asdict(ensemble_from_values(own, keys=seeds, label="own")) if own else None,
                    "ensembles": {str(M): {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                           for nm, v in r.items()} for M, r in cache.items()},
                }
            cell["arms"][arm["name"]] = {"rule": arm["rule"], "s": arm["s"], "rates": rates}
        out[f"{n}/{k}/{p:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def capacity(curve):
    """The load at which the mean read falls through one half, log-
    interpolated between the last checkpoint above it and the next; 0 if it
    never rises above one half; the last checkpoint if it never falls."""
    points = sorted((int(M), v) for M, v in curve.items())
    above = [i for i, (_, v) in enumerate(points) if v > 0.5]
    if not above:
        return 0.0
    i = above[-1]
    if i == len(points) - 1:
        return float(points[i][0])
    (m0, v0), (m1, v1) = points[i], points[i + 1]
    t = (v0 - 0.5) / (v0 - v1)
    return math.exp(math.log(m0) + t * (math.log(m1) - math.log(m0)))


def _mean(ensemble):
    return ensemble["mean"]


def arm_reading(arm):
    rates = sorted(arm["rates"].values(), key=lambda r: r["beta"])
    best = max(rates, key=lambda r: r["capacity"])
    # the peak mean read over loads and rates: the mechanism, whatever the
    # cell's capacity
    peak = max(_mean(m[r["metric"]]) for r in rates for m in r["ensembles"].values())
    own = [_mean(r["own"]) for r in rates if r["own"]]
    return {"best": best["capacity"], "best_beta": best["beta"], "peak_read": peak,
            "own_max": max(own) if own else None, "own_min": min(own) if own else None}


def evaluate(observations):
    """Amendment 26's bars."""
    rows = {(c["n"], c["k"], c["p"]): {name: arm_reading(a) for name, a in c["arms"].items()}
            for c in observations["cells"].values()}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    full = len(rows) == len(CELLS)
    d, o, a = arm_name("deferred", 0.5), arm_name("round", 1.5), arm_name("round", 0.5)
    # every rate of an arm must move (own_max) or hold (own_min)
    out["bars"]["S1"] = full and all(r[d]["peak_read"] >= FULL and r[d]["own_max"] <= OWN_LOW
                                     for r in rows.values())
    out["bars"]["S2"] = full and all(r[a]["own_min"] >= OWN_HIGH and r[o]["own_max"] <= OWN_LOW
                                     and r[o]["peak_read"] >= ONLINE_FLOOR
                                     for r in rows.values())
    if full:
        c = {cell: rows[cell][d]["best"] for cell in CELLS}
        same_nk = c[CELLS[0]] / c[CELLS[2]] if c[CELLS[2]] else float("inf")
        same_d = c[CELLS[1]] / c[CELLS[2]] if c[CELLS[2]] else float("inf")
        out["law"] = {"same_n_over_k": same_nk, "same_degree": same_d}
        out["bars"]["S3"] = abs(same_nk - 1) <= SAME and same_d >= APART
        out["bars"]["S3d"] = abs(same_d - 1) <= SAME and same_nk <= 1 / APART
    else:
        out["bars"]["S3"] = out["bars"]["S3d"] = False
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Sequences", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all three)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 26 is registered on seeds 322..341")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(b): lib.profile(b)
                for s in specs for arm in s["arms"] for b in arm["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.sequences", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "rounds": T, "read_items": READ_ITEMS,
                    "null_load": NULL_LOAD, "stop_len": STOP_LEN, "own_items": [OWN_FROM, OWN_AT],
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
