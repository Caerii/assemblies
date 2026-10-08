"""Is the critical load additive over sequences? Many short sequences against one
long one at the same total load: a registered test.

Registered in PREREG_refraction_memory.md, Amendment 40.

The load law (Amendments 37 to 39) is for one sequence per area: replay fails at
a critical rho = L k ln n / (n^2 p), and with the refraction recovering over
tau = n/k / 2 it is reliable while rho <= 0.09. A compiler spends that budget
on sequences of whatever length a program needs. An exploratory probe found
the failure of many short sequences GRADED -- each brain loses a share of its
sequences, as a per-step hazard compounded over l - 1 steps predicts -- where
one long sequence fails all or none; short sequences outlast the single
sequence's cliff. Tested here: the single-sequence safe rule holds for any
split, short sequences degrade later, and their failure is per sequence.

    RELIABILITY  for each total L on a ladder of rho, per brain either one
                 sequence of L elements or M = L / l sequences of l elements,
                 each written by its own store_sequence (the area inhibited
                 before its first element, so sequences are not linked; the
                 refraction carries over). Every sequence is replayed
                 noiselessly from a uniformly random half of its first element
                 (drawn from the brain's seed and the sequence's index); the
                 mean over brains of the fraction of a brain's sequences
                 replayed whole.

    python -m research.runner load_many --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_load_tau as mt                  # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells: n/k = 100 and 171, both in regime, tau by the rule
CELLS = ((8000, 80, 0.5), (12000, 70, 0.5))
SMOKE_CELL = (4000, 60, 0.5)
#: sequence lengths: one sequence of the whole load, and two short ones
ARMS = ("single", 64, 16)
LADDER = md.LADDER                                                      # rho 0.071 .. 0.26
SEEDS = tuple(range(582, 602))
GAIN, SAFE, MIXED = 1.1, ml.SAFE, 0.5
#: a ladder point counts as empty below this mean whole fraction (stop rule)
EMPTY = 0.02


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        for arm in (("single", 16) if smoke else ARMS):
            ladder = []
            for r in (LADDER[:2] if smoke else LADDER):
                L = max(8, int(round(r * ml.unit(n, k, p))))
                if arm != "single":
                    L = max(1, round(L / arm)) * arm
                ladder.append(L)
            out.append({"n": n, "k": k, "p": p, "tau": mt.rule(n, k), "arm": arm,
                        "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n),
                        "ladder": ladder})
    return out


def whole(spec, L, seeds, device):
    """Each brain's fraction of its sequences replayed whole, for total load L."""
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    length = L if spec["arm"] == "single" else int(spec["arm"])
    M = L // length
    out = []
    for i in range(0, len(seeds), spec["batch"]):
        part = list(seeds[i:i + spec["batch"]])
        mem = AssemblyMemory(seeds_for(part), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                             norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4,
                             device=device, bias_decay=math.exp(-1.0 / spec["tau"]))
        seqs = []
        for q in range(M):
            els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"L{L}l{length}q{q}e{e}", "A")) for sd in part]
                   for e in range(length)]
            seqs.append(mem.store_sequence(els)[0])
        done = torch.zeros(len(part), device=device)
        for q, states in enumerate(seqs):
            x = torch.stack([md.cue(states[0, b], sd, L * 100_000 + q, k, device)
                             for b, sd in enumerate(part)])
            alive = torch.ones(len(part), dtype=torch.bool, device=device)
            for j in range(1, length):
                x = mem.recall(x, rounds=1)
                alive &= sq._overlap(x, states[j]) >= ml.MATCH
                if not bool(alive.any()):
                    break
            done += alive.float()
        out += (done / M).tolist()
        del mem, seqs
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p, arm = spec["n"], spec["k"], spec["p"], spec["arm"]
        rows, zeros = {}, 0
        for L in spec["ladder"]:
            fr = whole(spec, L, seeds, device)
            full = sum(fr) / len(fr)
            rows[str(L)] = {"rho": L / ml.unit(n, k, p), "whole": fr, "full": full}
            print(f"({n}, {k}, {p}) tau={spec['tau']} arm={arm} L={L} rho={L / ml.unit(n, k, p):.3f}: "
                  f"whole {full:.3f}", flush=True)
            zeros = zeros + 1 if full < EMPTY else 0
            if zeros >= 2:
                break
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][str(arm)] = {"ladder": rows}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def mixed(rows):
    """Per ladder point, the share of brains with some sequences whole and some
    not (whole fraction strictly between 0.1 and 0.9)."""
    return {L: sum(0.1 < f < 0.9 for f in r["whole"]) / len(r["whole"]) for L, r in rows.items()}


def evaluate(observations):
    """Amendment 40's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out: dict = {"bars": {}, "rho": {}, "ratio": {}, "mixed": {}}
    names = ("A1", "A2", "A3")
    graded = []
    if not all(c in cells and {str(a) for a in ARMS} <= set(cells[c]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    for n, k, p in CELLS:
        key = f"{n}/{k}/{p:g}"
        arms = cells[(n, k, p)]["arms"]
        r = {str(a): {q: ml.crossing(arms[str(a)]["ladder"], q) for q in (0.9, 0.5, 0.1)} for a in ARMS}
        out["rho"][key] = r
        single = r["single"][0.5]
        out["ratio"][key] = {str(a): (r[str(a)][0.5] / single if single and r[str(a)][0.5] else None)
                             for a in ARMS[1:]}
        out["mixed"][key] = {str(a): mixed(arms[str(a)]["ladder"]) for a in ARMS[1:]}
        for a in ARMS[1:]:
            rows = arms[str(a)]["ladder"]
            inside = [L for L, row in rows.items() if 0.1 < row["full"] < 0.9]
            graded.append(bool(inside) and all(out["mixed"][key][str(a)][L] >= MIXED for L in inside))
    out["bars"]["A1"] = all(q[0.9] is not None and q[0.9] >= SAFE
                            for r in out["rho"].values() for q in r.values())
    out["bars"]["A2"] = all(v[a] is not None and v[a] >= GAIN
                            for v in out["ratio"].values() for a in v)
    out["bars"]["A3"] = all(graded)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Load many", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 40 is registered on seeds 582..601")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.load-many", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": ml.STRENGTH, "match": ml.MATCH,
                    "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
