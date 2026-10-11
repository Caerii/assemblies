"""The reliability of single-sequence replay against interference load, at cells
never probed: a registered prediction.

Registered in PREREG_refraction_memory.md, Amendment 37.

At the convergence-threshold write (beta = theta) the learned signal of a stored
transition, against the base drive's fluctuation, is sqrt(ln n) per unit overlap
at every cell, and the interference of L stored transitions adds a variance of
about L (k/n)^2 beta^2 k p, which relative to the base variance k p (1 - p) is

    rho = L k ln n / (n^2 p).

An exploratory survey (8 cells, tau = 64) found replay fails at rho_50 = 0.142
(0.119 to 0.171) at every cell with k p >= 3 ln n. Here, at two cells never run:

    RELIABILITY  for each L on a ladder of rho = 0.06 x 2^(j/8), a fresh sequence
                 of L elements per brain (refraction 0.5 beta recovering over 64
                 rounds), replayed noiselessly from a uniformly random half of
                 element 0; the fraction of brains replaying all L - 1 steps.
                 rho_q is where that fraction falls through q (log-interpolated).

    python -m research.runner load_law --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_lib as lib                      # noqa: E402
#: registered names, owned by research.experiments.memory_lib since 2026-10-10 (re-exported)
from research.experiments.memory_lib.laws import unit                   # noqa: E402
from research.experiments.memory_lib.model import STRENGTH              # noqa: E402
from research.experiments.memory_lib.readout import MATCH               # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the held-out cells (judged) and an out-of-regime cell (reported)
CELLS = ((6000, 90, 0.4), (12000, 80, 0.5))
REPORTED = ((6000, 40, 0.5),)
TAU = 64
LADDER = tuple(0.06 * 2 ** (j / 8) for j in range(18))                  # rho 0.06 .. 0.26
SEEDS = tuple(range(522, 542))
RHO_50, BAND, SAFE, SHARP = 0.142, 1.3, 0.09, 1.6


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "beta": round(lib.theta(n, k, p), 5),
             "ladder": [max(8, int(round(r * unit(n, k, p)))) for r in (LADDER[6:8] if smoke else LADDER)]}
            for n, k, p in cells]


def reliability(spec, L, seeds, device):
    """Each brain's replay length [B] for one fresh sequence of L elements."""
    import torch
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=spec["beta"], w_max=lib.W_MAX,
                         norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                         device=device, bias_decay=math.exp(-1.0 / TAU))
    els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"L{L}e{e}", "A")) for sd in seeds] for e in range(L)]
    states = mem.store_sequence(els)[0]
    gen = torch.Generator(device=device).manual_seed(L)
    B = states.shape[1]
    perm = torch.argsort(torch.rand(B, k, device=device, generator=gen), dim=1)
    x = torch.gather(states[0], 1, perm)[:, :k // 2]
    alive = torch.ones(B, dtype=torch.bool, device=device)
    steps = torch.zeros(B, device=device)
    for j in range(1, L):
        x = mem.recall(x, rounds=1)
        alive &= lib.overlap(x, states[j]) >= MATCH
        steps += alive.float()
        if not bool(alive.any()):
            break
    out = steps.tolist()
    del mem, states
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
            rows[str(L)] = {"rho": L / unit(n, k, p), "steps": steps, "full": full}
            print(f"({n}, {k}, {p}) L={L} rho={L / unit(n, k, p):.3f}: full {full:.2f}", flush=True)
            zeros = zeros + 1 if full == 0 else 0
            if zeros >= 2:
                break
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "ladder": rows}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def crossing(rows, q):
    """rho at which the full-replay fraction falls through q, log-interpolated;
    None if it never does."""
    pts = sorted((r["rho"], r["full"]) for r in rows.values())
    for (r0, v0), (r1, v1) in zip(pts, pts[1:]):
        if v0 >= q > v1:
            t = (v0 - q) / (v0 - v1)
            return math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
    return None


def evaluate(observations):
    """Amendment 37's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"bars": {}, "rho": {}}
    if not all(c in cells for c in CELLS):
        out["bars"] = {b: False for b in ("R1", "R2", "R3")}
        return out
    for c, cell in cells.items():
        out["rho"][f"{c[0]}/{c[1]}/{c[2]:g}"] = {q: crossing(cell["ladder"], q) for q in (0.9, 0.5, 0.1)}
    judged = [out["rho"][f"{n}/{k}/{p:g}"] for n, k, p in CELLS]
    out["bars"]["R1"] = all(r[0.5] is not None and RHO_50 / BAND <= r[0.5] <= RHO_50 * BAND for r in judged)
    out["bars"]["R2"] = all(r[0.9] is not None and r[0.9] >= SAFE for r in judged)
    out["bars"]["R3"] = all(r[0.9] and r[0.1] and r[0.1] / r[0.9] <= SHARP for r in judged)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Load law", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 37 is registered on seeds 522..541")
    cells = CELLS + REPORTED
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.load-law", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "tau": TAU, "strength": STRENGTH, "match": MATCH,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
