"""Is the load law's constant a constant? A registered test between two forms at
cells where they disagree.

Registered in PREREG_refraction_memory.md, Amendment 38.

Amendment 37 confirmed that single-sequence replay fails at a critical load
rho = L k ln n / (n^2 p), rho_50 ~ 0.142, at two unseen cells -- but one of them
came in 19% low, and over the nine in-regime cells measured, rho_50 is lower at
n/k >= 133 than at n/k <= 67. A post hoc fit,

    rho_50 = exp(-1.5224) (n/k)^-0.1683 (k p / ln n)^0.1718,

halves the residual spread. At n/k = 300 the two forms part by a factor 1.35.

    RELIABILITY  as in Amendment 37 (memory_load_law): for each L on a ladder of
                 rho, a fresh sequence of L elements per brain, replayed
                 noiselessly from a uniformly random half of element 0; the
                 fraction of brains replaying all L - 1 steps. Brains run in
                 batches (the dense counts of 20 brains at n = 24000 do not fit
                 the device), so each brain's cue is drawn from its own seed:
                 a brain's record does not depend on the batch it ran in.

    python -m research.runner load_drift --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells: n/k = 300 at two (n, k, p), both in regime (k p >= 3 ln n)
CELLS = ((18000, 60, 0.6), (24000, 80, 0.5))
#: the smoke exercises the code on a survey cell, never on a judged one
SMOKE_CELL = (4000, 60, 0.5)
TAU, STRENGTH, MATCH = ml.TAU, ml.STRENGTH, ml.MATCH
LADDER = ml.LADDER[2:]                                                  # rho 0.071 .. 0.26
SEEDS = tuple(range(542, 562))
#: the post hoc fit over the nine in-regime cells (Amendments 37 and its survey)
FIT = (-1.52243353, -0.16834888, 0.1718313)
#: the drift's band: two residual sd (7.6%, three parameters spent on nine cells),
#: widened for the extrapolation from n/k <= 150 to 300
CONSTANT, CONSTANT_BAND, DRIFT_BAND = ml.RHO_50, ml.BAND, 1.2
SAFE, SHARP = ml.SAFE, ml.SHARP
#: dense int8 counts per batch, bytes
BUDGET = 3e9


def drift(n, k, p):
    """The fitted form's rho_50."""
    a, b, c = FIT
    return math.exp(a + b * math.log(n / k) + c * math.log(k * p / math.log(n)))


def batch_size(n):
    return max(1, int(BUDGET // (n * n)))


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "beta": round(tl.theta(n, k, p), 5), "batch": batch_size(n),
             "ladder": [max(8, int(round(r * ml.unit(n, k, p)))) for r in (LADDER[:2] if smoke else LADDER)]}
            for n, k, p in cells]


def cue(state, seed, L, k, device):
    """A uniformly random half of one brain's first state, drawn from its seed."""
    import torch
    gen = torch.Generator(device=device).manual_seed(int(seed) * 1_000_003 + int(L))
    perm = torch.argsort(torch.rand(k, device=device, generator=gen))
    return state[perm[:k // 2]]


def reliability(spec, L, seeds, device):
    """Each brain's replay length, brains in batches of spec["batch"]."""
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    out = []
    for i in range(0, len(seeds), spec["batch"]):
        part = list(seeds[i:i + spec["batch"]])
        mem = AssemblyMemory(seeds_for(part), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                             norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                             device=device, bias_decay=math.exp(-1.0 / TAU))
        els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"L{L}e{e}", "A")) for sd in part] for e in range(L)]
        states = mem.store_sequence(els)[0]
        x = torch.stack([cue(states[0, b], sd, L, k, device) for b, sd in enumerate(part)])
        alive = torch.ones(len(part), dtype=torch.bool, device=device)
        steps = torch.zeros(len(part), device=device)
        for j in range(1, L):
            x = mem.recall(x, rounds=1)
            alive &= sq._overlap(x, states[j]) >= MATCH
            steps += alive.float()
            if not bool(alive.any()):
                break
        out += steps.tolist()
        del mem, states, x
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        rows, zeros = {}, 0
        for L in spec["ladder"]:
            steps = reliability(spec, L, seeds, device)
            full = sum(s >= L - 1 for s in steps) / len(steps)
            rows[str(L)] = {"rho": L / ml.unit(n, k, p), "steps": steps, "full": full}
            print(f"({n}, {k}, {p}) L={L} rho={L / ml.unit(n, k, p):.3f}: full {full:.2f}", flush=True)
            zeros = zeros + 1 if full == 0 else 0
            if zeros >= 2:
                break
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "ladder": rows}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 38's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"bars": {}, "rho": {}, "nearer": {}}
    if not all(c in cells for c in CELLS):
        out["bars"] = {b: False for b in ("K1", "D1", "S1", "S2")}
        return out
    judged = []
    for n, k, p in CELLS:
        r = {q: ml.crossing(cells[(n, k, p)]["ladder"], q) for q in (0.9, 0.5, 0.1)}
        key = f"{n}/{k}/{p:g}"
        out["rho"][key] = r
        if r[0.5] is not None:
            gap_c = abs(math.log(r[0.5] / CONSTANT))
            gap_d = abs(math.log(r[0.5] / drift(n, k, p)))
            out["nearer"][key] = "constant" if gap_c < gap_d else "drift"
        judged.append((r, drift(n, k, p)))
    out["bars"]["K1"] = all(r[0.5] is not None and CONSTANT / CONSTANT_BAND <= r[0.5] <= CONSTANT * CONSTANT_BAND
                            for r, _ in judged)
    out["bars"]["D1"] = all(r[0.5] is not None and d / DRIFT_BAND <= r[0.5] <= d * DRIFT_BAND for r, d in judged)
    out["bars"]["S1"] = all(r[0.9] is not None and r[0.9] >= SAFE for r, _ in judged)
    out["bars"]["S2"] = all(r[0.9] and r[0.1] and r[0.1] / r[0.9] <= SHARP for r, _ in judged)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Load drift", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 38 is registered on seeds 542..561")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.load-drift", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "tau": TAU, "strength": STRENGTH, "match": MATCH,
                    "w_max": pe.W_MAX, "fit": list(FIT), "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
