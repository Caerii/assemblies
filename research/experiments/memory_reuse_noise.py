"""Words that recur, and noise during replay, inside the load law's budget: a
registered test.

Registered in PREREG_refraction_memory.md, Amendment 42.

The load law (Amendments 37 to 41) is for sequences of distinct elements
replayed without noise. A program's sequences reuse their elements -- the same
word in many sentences -- and replay is never noiseless. Here, many sequences of
16 elements per brain at a total load rho = L k ln n / (n^2 p):

    REUSE   each brain's elements are WORDS drawn i.i.d. from a vocabulary of
            V = L / U (U uses per word on average), the same word the same
            stimulus wherever it occurs; each brain draws its own sequences.
            The fraction of a brain's sequences replayed whole (noiselessly,
            from a uniformly random half of the first element), and the overlap
            of two occurrences of the same word in different sequences (CODE
            OVERLAP) against two different words' (BASELINE).
    NOISE   distinct elements; at every replay step a fraction nu of the
            winners replaced by uniformly random neurons (Amendment 36's
            uniform noise); the fraction replayed whole.

    python -m research.runner reuse_noise --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_noise as mn                     # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells and their recovery times (Amendment 41's rule)
CELLS = ((4000, 100, 0.35, 40), (10000, 120, 0.3, 42))
SMOKE_CELL = (2000, 60, 0.5, 17)
LENGTH = 16
#: reuse arms: uses per word (None: every element distinct), at two loads
USES, REUSE_RHO = (None, 2, 5, 20), (0.05, 0.08)
#: noise arms, at three loads
NUS, NOISE_RHO = (0.0, 0.03, 0.05, 0.1), (0.05, 0.08, 0.11)
SEEDS = tuple(range(622, 642))
#: W1's ceiling: a type code would overlap >= 0.5, chance k/n is 0.01 to 0.03
TOKEN, FREE, MILD, COMPOUND = 0.3, 0.98, 0.98, 0.2
#: occurrence pairs sampled per brain for the code overlaps
PAIRS = 200


def plan(cells, *, smoke=False):
    out = []
    for n, k, p, tau in cells:
        base = {"n": n, "k": k, "p": p, "tau": tau, "beta": round(tl.theta(n, k, p), 5),
                "batch": md.batch_size(n)}
        for rho in (REUSE_RHO[:1] if smoke else REUSE_RHO):
            for u in ((None, 5) if smoke else USES):
                out.append({**base, "kind": "reuse", "rho": rho, "uses": u, "nu": 0.0})
        for rho in (NOISE_RHO[:1] if smoke else NOISE_RHO):
            for nu in ((0.0, 0.05) if smoke else NUS):
                out.append({**base, "kind": "noise", "rho": rho, "uses": None, "nu": nu})
    return out


def _words(seed, M, V, salt):
    import numpy as np
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(salt)])
    return rng.integers(0, V, size=(M, LENGTH))


def measure(spec, seeds, device):
    """Per brain: the fraction of its sequences replayed whole, and (reuse arms)
    the mean same-word and different-word code overlaps."""
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / LENGTH))
    L = M * LENGTH
    V = None if spec["uses"] is None else max(2, round(L / spec["uses"]))
    salt = L * 1000 + (V or 0)
    out = {"L": L, "M": M, "V": V, "whole": [], "same": [], "different": []}
    for i in range(0, len(seeds), spec["batch"]):
        part = list(seeds[i:i + spec["batch"]])
        mem = AssemblyMemory(seeds_for(part), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                             norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4,
                             device=device, bias_decay=math.exp(-1.0 / spec["tau"]))
        if V is None:
            words = [None] * len(part)
            name = lambda b, q, e: f"L{L}q{q}e{e}"                      # noqa: E731
        else:
            words = [_words(sd, M, V, salt) for sd in part]
            name = lambda b, q, e: f"w{int(words[b][q, e])}"            # noqa: E731
        seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, name(b, q, e), "A"))
                                     for b, sd in enumerate(part)] for e in range(LENGTH)])[0]
                for q in range(M)]
        gen = torch.Generator(device=device).manual_seed(salt)
        done = torch.zeros(len(part), device=device)
        for q, st in enumerate(seqs):
            x = torch.stack([md.cue(st[0, b], sd, L * 100_000 + q, k, device) for b, sd in enumerate(part)])
            alive = torch.ones(len(part), dtype=torch.bool, device=device)
            for j in range(1, LENGTH):
                x = mem.recall(x, rounds=1)
                if spec["nu"]:
                    x = mn._corrupt(x, spec["nu"], n, gen, uniform=True)
                alive &= sq._overlap(x, st[j]) >= ml.MATCH
            done += alive.float()
        out["whole"] += (done / M).tolist()
        if V is not None:
            for b in range(len(part)):
                where = {}
                for q in range(M):
                    for e in range(LENGTH):
                        where.setdefault(int(words[b][q, e]), []).append((q, e))
                same, diff, prev = [], [], None
                for occ in where.values():
                    cross = [(a, c) for a in occ for c in occ if a[0] < c[0]]
                    if cross and len(same) < PAIRS:
                        (q1, e1), (q2, e2) = cross[0]
                        same.append(float(sq._overlap(seqs[q1][e1, b:b + 1], seqs[q2][e2, b:b + 1])))
                    if prev is not None and len(diff) < PAIRS:
                        diff.append(float(sq._overlap(seqs[prev[0]][prev[1], b:b + 1],
                                                      seqs[occ[0][0]][occ[0][1], b:b + 1])))
                    prev = occ[0]
                out["same"].append(sum(same) / len(same) if same else None)
                out["different"].append(sum(diff) / len(diff) if diff else None)
        del mem, seqs
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"],
                                                 "reuse": {}, "noise": {}})
        if spec["kind"] == "reuse":
            cell["reuse"].setdefault(str(spec["rho"]), {})[str(spec["uses"])] = m
        else:
            cell["noise"].setdefault(str(spec["rho"]), {})[str(spec["nu"])] = m
        mean = sum(m["whole"]) / len(m["whole"])
        same = [s for s in m["same"] if s is not None]
        print(f"({n}, {k}, {p}) {spec['kind']} rho={spec['rho']} uses={spec['uses']} nu={spec['nu']} "
              f"L={m['L']} V={m['V']}: whole {mean:.3f}"
              + (f", same-word overlap {sum(same) / len(same):.3f}" if same else ""), flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def evaluate(observations):
    """Amendment 42's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out: dict = {"bars": {}, "whole": {}, "same": {}, "different": {}}
    names = ("W1", "W2", "N1", "N2")
    try:
        judged = [cells[(n, k, p)] for n, k, p, _ in CELLS]
        for c in judged:
            for rho in REUSE_RHO:
                for u in USES:
                    c["reuse"][str(rho)][str(u)]
            for rho in NOISE_RHO:
                for nu in NUS:
                    c["noise"][str(rho)][str(nu)]
    except KeyError:
        out["bars"] = {b: False for b in names}
        return out
    for (n, k, p, _), c in zip(CELLS, judged):
        key = f"{n}/{k}/{p:g}"
        out["whole"][key] = {"reuse": {r: {u: _mean(m["whole"]) for u, m in a.items()} for r, a in c["reuse"].items()},
                             "noise": {r: {nu: _mean(m["whole"]) for nu, m in a.items()} for r, a in c["noise"].items()}}
        out["same"][key] = {r: {u: _mean(m["same"]) for u, m in a.items() if u != "None"} for r, a in c["reuse"].items()}
        out["different"][key] = {r: {u: _mean(m["different"]) for u, m in a.items() if u != "None"}
                                 for r, a in c["reuse"].items()}
    out["bars"]["W1"] = all(v is not None and v <= TOKEN for s in out["same"].values()
                            for a in s.values() for v in a.values())
    out["bars"]["W2"] = all(w["reuse"][str(r)][str(u)] >= FREE for w in out["whole"].values()
                            for r in REUSE_RHO for u in (None, 2, 5))
    out["bars"]["N1"] = all(w["noise"][str(r)][str(nu)] >= MILD for w in out["whole"].values()
                            for r in (0.05, 0.08) for nu in (0.0, 0.03, 0.05))
    out["bars"]["N2"] = all(w["noise"]["0.05"]["0.1"] - w["noise"]["0.11"]["0.1"] >= COMPOUND
                            for w in out["whole"].values())
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Reuse and noise", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 42 is registered on seeds 622..641")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.reuse-noise", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": LENGTH, "strength": ml.STRENGTH, "match": ml.MATCH,
                    "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
