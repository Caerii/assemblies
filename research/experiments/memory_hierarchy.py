"""Sequences of sequences: a chunk area that orders whole sequences stored
once in a sequence area.

Registered in PREREG_refraction_memory.md, Amendment 33.

Within one area the refraction that keeps contexts apart also gives a shared
element a new code in every sequence, so two stored sequences cannot be glued
at a shared junction (exploratory probe, Amendment 31's notes: replay never
crosses the junction). This study splits the two jobs across two areas:

    S, the SEQUENCE area: each chunk -- a chosen sequence of Lc elements --
       is written ONCE (`store_sequence`), and is reused by every plan.
    C, the CHUNK area: each plan -- an order of chunks -- is written as a
       chosen sequence of chunk symbols, one round per chunk; refraction
       there codes each plan's occurrence of a chunk on its own neurons.
    C -> S, a fiber (`DenseOrganFiber`, n_C -> n_S): every plan position's
       C state is linked onto its chunk's first S state, written `link`
       times after the plans.

Recall runs on two clocks. From a plan's first C state, for each of its
chunks: the C state drives S through C -> S (one frozen masked round), S
replays the chunk by frozen masked rounds, and C advances one frozen round.
A plan's whole-replay fraction is the elements -- chunk starts and contents,
in plan order -- matched (own overlap >= 0.3) before the first miss.

    python -m research.runner hierarchy --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: (n_S, n_C), both k = 60, p = 0.5
CELLS = ((4000, 2000), (8000, 4000))
K, P = 60, 0.5
CHUNKS, CHUNK_LEN, PLAN_LEN, PLANS = 8, 12, 5, 6
LOAD_PLAN_LEN = 8
TAU, STRENGTH, MATCH = 33, 0.5, 0.3
LINKS = (0, 2)
#: (chunks, plans) of PLAN_LEN chunks each; the first is the mechanism load,
#: the others the load at which the probe saw the links, then the chunk
#: area, give way
LOADS = ((8, 6), (32, 32), (64, 96))
#: plans recalled per load (evenly spaced over the stored ones)
READ_PLANS = 16
PLAN_SEED = 20261007
SEEDS = tuple(range(462, 482))
FULL, NONE = 0.9, 0.1


def plans(chunks=None, plan_len=None, count=None, seed=PLAN_SEED):
    """`count` orders of `plan_len` distinct chunks; the last two share their
    second and third chunks (a run of two) and differ elsewhere -- the plans
    whose chunk contexts the chunk area must keep apart."""
    chunks = CHUNKS if chunks is None else chunks
    plan_len = PLAN_LEN if plan_len is None else plan_len
    count = PLANS if count is None else count
    rng = random.Random(seed)
    out = [rng.sample(range(chunks), plan_len) for _ in range(count - 2)]
    a = rng.sample(range(chunks), plan_len)
    rest = [c for c in range(chunks) if c not in a[1:3]]
    b = [rng.choice([c for c in rest if c != a[0]])] + a[1:3]
    b += rng.sample([c for c in rest if c not in b and c not in a[3:]] or rest, plan_len - 3)
    return out + [a, b]


def plan(cells, *, smoke=False):
    return [{"n_s": ns, "n_c": nc, "k": K, "p": P,
             "beta_s": round(lib.theta(ns, K, P), 5), "beta_c": round(lib.theta(nc, K, P), 5),
             "links": list(LINKS), "loads": [[8, 6]] if smoke else [list(l) for l in LOADS],
             "chunk_len": 4 if smoke else CHUNK_LEN} for ns, nc in cells]


def load_shape(load):
    """(chunks, plans, plan length) of a load; the mechanism load keeps the
    registered five-chunk plans with their shared run."""
    chunks, count = load
    return chunks, count, (PLAN_LEN if (chunks, count) == LOADS[0] else LOAD_PLAN_LEN)


def build(spec, link, seeds, device, load=None):
    """Write the chunks into S, the plans into C, and the links C -> S.
    Returns (S, C, CS, chunk_states [chunks][L, B, k], plan_states [plans][Lp, B, k])."""
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._hashed import DenseOrganFiber
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    k, p = spec["k"], spec["p"]
    decay = math.exp(-1.0 / TAU)

    def area(n, beta):
        return AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=beta, w_max=lib.W_MAX,
                              norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                              device=device, bias_decay=decay)

    def els(prefix, names):
        return [[to_i32(_seeding.fnv1a_pair_seed(sd, f"{prefix}{nm}", "A")) for sd in seeds]
                for nm in names]
    S, C = area(spec["n_s"], spec["beta_s"]), area(spec["n_c"], spec["beta_c"])
    n_chunks, n_plans, plan_len = load_shape(load or LOADS[0])
    chunks = [S.store_sequence(els(f"x{c}e", range(spec["chunk_len"])))[0] for c in range(n_chunks)]
    orders = plans(n_chunks, plan_len, n_plans)
    plan_states = [C.store_sequence(els("chunk", order))[0] for order in orders]
    CS = DenseOrganFiber([to_i32(_seeding.fnv1a_pair_seed(sd, "C->S", "A")) for sd in seeds],
                         spec["n_c"], spec["n_s"], p, beta=spec["beta_s"], w_max=lib.W_MAX,
                         norm_init=True, device=device)
    for _ in range(link):
        for order, states in zip(orders, plan_states):
            for i, c in enumerate(order):
                CS.observe(states[i], chunks[c][0])
    return S, C, CS, chunks, orders, plan_states


def recall_plan(S, C, CS, chunks, order, states, k):
    """Each brain's whole-replay fraction of one plan [B], the chunk-start
    hits [B] (chunks whose start the C state evoked), and the C chain's
    in-order fraction [B]."""
    import torch
    B = states.shape[1]
    total = len(order) * chunks[0].shape[0]
    alive = torch.ones(B, dtype=torch.bool, device=states.device)
    steps = torch.zeros(B, device=states.device)
    starts = torch.zeros(B, device=states.device)
    c_alive = torch.ones(B, dtype=torch.bool, device=states.device)
    c_steps = torch.zeros(B, device=states.device)
    c = states[0]
    for i, chunk in enumerate(order):
        s = S.area.project(1, [CS], rows_for={id(CS): c}, freeze=True, mask_bias=True,
                           manage_episodes=False)
        hit = lib.overlap(s, chunks[chunk][0]) >= MATCH
        starts += hit.float()
        alive &= hit
        steps += alive.float()
        for j in range(1, chunks[chunk].shape[0]):
            s = S.recall(s, rounds=1)
            alive &= lib.overlap(s, chunks[chunk][j]) >= MATCH
            steps += alive.float()
        if i + 1 < len(order):
            c = C.recall(c, rounds=1)
            c_alive &= lib.overlap(c, states[i + 1]) >= MATCH
            c_steps += c_alive.float()
    return ((steps / total).tolist(), (starts / len(order)).tolist(),
            (c_steps / (len(order) - 1)).tolist())


def run_load(spec, load, link, seeds, device):
    """Build one (load, link) and recall its read plans: {plan index: readings}."""
    S, C, CS, chunks, orders, plan_states = build(spec, link, seeds, device, load)
    read = {round(i * (len(orders) - 1) / (READ_PLANS - 1)) for i in range(READ_PLANS)}
    if tuple(load) == LOADS[0]:
        read |= {len(orders) - 2, len(orders) - 1}      # the plans that share a run
    rows = {}
    for idx in sorted(read):
        whole, starts, chain = recall_plan(S, C, CS, chunks, orders[idx], plan_states[idx], spec["k"])
        rows[str(idx)] = {"order": orders[idx],
                          "whole": asdict(ensemble_from_values(whole, keys=seeds, label="whole")),
                          "starts": asdict(ensemble_from_values(starts, keys=seeds, label="starts")),
                          "chain": asdict(ensemble_from_values(chain, keys=seeds, label="chain"))}
    return rows


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    out = {}
    for spec in parameters["cells"]:
        cell = {"n_s": spec["n_s"], "n_c": spec["n_c"], "links": {}}
        for load in spec["loads"]:
            for link in spec["links"]:
                rows = run_load(spec, load, link, seeds, parameters["device"])
                cell["links"][f"{load[0]}/{load[1]}/{link}"] = rows
                mean = lambda read: sum(r[read]["mean"] for r in rows.values()) / len(rows)
                print(f"(S {spec['n_s']}, C {spec['n_c']}) load {load} link {link}: whole "
                      f"{mean('whole'):.2f}, starts {mean('starts'):.2f}, chain {mean('chain'):.2f}",
                      flush=True)
        out[f"{spec['n_s']}/{spec['n_c']}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 33's bars."""
    import statistics
    cells = {(c["n_s"], c["n_c"]): c for c in observations["cells"].values()}
    keys = [f"{a}/{b}/{l}" for a, b in LOADS for l in LINKS]
    out = {"bars": {}}
    if not all(c in cells for c in CELLS) or not all(k in c["links"] for c in cells.values() for k in keys):
        out["bars"] = {b: False for b in ("H1", "H2", "H3", "H4")}
        return out
    cs = [cells[c] for c in CELLS]
    mech = f"{LOADS[0][0]}/{LOADS[0][1]}"

    def rows(c, load, link):
        return list(c["links"][f"{load}/{link}"].values())

    def mean(c, load, link, read):
        return statistics.fmean(r[read]["mean"] for r in rows(c, load, link))
    out["bars"]["H1"] = all(r["whole"]["mean"] >= FULL for c in cs for r in rows(c, mech, 2))
    out["bars"]["H2"] = all(r["starts"]["mean"] <= NONE for c in cs for a, b in LOADS
                            for r in rows(c, f"{a}/{b}", 0))
    shared = [r for c in cs for r in rows(c, mech, 2)[-2:]]
    out["bars"]["H3"] = all(r["chain"]["mean"] >= FULL and r["whole"]["mean"] >= FULL for r in shared)
    heavy = f"{LOADS[2][0]}/{LOADS[2][1]}"
    small, big = cs
    out["bars"]["H4"] = (mean(small, heavy, 2, "starts") < mean(small, heavy, 2, "chain") - 0.1
                         and mean(big, heavy, 2, "starts") >= mean(small, heavy, 2, "starts") + 0.1)
    out["summary"] = {f"{ns}/{nc}": {key: {read: round(statistics.fmean(r[read]["mean"] for r in rr.values()), 3)
                                           for read in ("whole", "starts", "chain")}
                                     for key, rr in c["links"].items()}
                      for (ns, nc), c in cells.items()}
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Hierarchy", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n_S:n_C pairs (default: both)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(int(v) for v in t.split(":")) for t in args.cells.split(",")]
             if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 33 is registered on seeds 462..481")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(b): lib.profile(b) for s in specs for b in (s["beta_s"], s["beta_c"])}
    path = run_experiment(
        script=__file__, protocol="memory.hierarchy", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "plan_len": PLAN_LEN, "load_plan_len": LOAD_PLAN_LEN,
                    "read_plans": READ_PLANS,
                    "tau": TAU, "strength": STRENGTH, "match": MATCH, "plan_seed": PLAN_SEED,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
