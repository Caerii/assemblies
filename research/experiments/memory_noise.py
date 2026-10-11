"""Replay under UNIFORMLY random activity noise, and from a random half cue.

Registered in PREREG_refraction_memory.md, Amendment 36.

Amendment 34's noise replaced the FIRST nu k slots of each replay step's
winners, and the k-WTA returns its winners ordered by drive, strongest first:
the slots it replaced were the most driven winners, nearly all of them members
of the true next element. Its "half cue" was likewise the strongest half of
the first element. Here, on the same cells and recovery time:

    NOISE  one sequence of L = 400; at every replay step nu k of the winners
           are replaced by random neurons, the replaced slots chosen either as
           Amendment 34 chose them (TOP) or uniformly at random (UNIFORM);
           each brain's first-miss step (own overlap < 0.3), mean over four
           noise draws (L - 1 if none).
    CUE    the strongest half of the first element (Amendment 34's cue) or a
           uniformly random half, with a fraction eta of it replaced at random;
           the replay fraction.

    python -m research.runner noise --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_robustness as mr                # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = mr.CELLS
#: Amendment 34's recorded medians at nu = 0.1 (TOP slots)
A34_MEDIAN = {(4000, 60, 0.5): 39.0, (8000, 60, 0.5): 232.0}
NOISE_LEN, NUS, DRAWS = 400, (0.05, 0.07, 0.1, 0.2), 4
SLOTS = ("top", "uniform")
CUES, ETAS = ("strong", "random"), (0.0, 0.25)
NOISE_SEED = 13
SEEDS = tuple(range(502, 522))
MILDER, FULL_BRAINS, A34_TOL, CUE_FLOOR = 2.0, 18, 0.35, 0.9


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "beta": round(lib.theta(n, k, p), 5),
             "noise_len": 40 if smoke else NOISE_LEN} for n, k, p in cells]


def _shuffle(x, gen):
    import torch
    perm = torch.argsort(torch.rand(x.shape, device=x.device, generator=gen), dim=1)
    return torch.gather(x, 1, perm)


def _corrupt(x, frac, n, gen, uniform):
    """Replace round(frac k) slots of x [B, k] with random neurons: the first
    slots (TOP, the strongest winners) or uniformly chosen ones."""
    return mr._corrupt(_shuffle(x, gen) if uniform else x, frac, n, gen)


def replay(mem, states, *, nu=0.0, uniform=False, cue="strong", eta=0.0, gen=None):
    """Each brain's first-miss step [B] (L - 1 if none)."""
    import torch
    n, k, L = mem.n, mem.k, states.shape[0]
    first = states[0] if cue == "strong" else _shuffle(states[0], gen)
    x = _corrupt(first[:, :k // 2], eta, n, gen, uniform=True)
    alive = torch.ones(states.shape[1], dtype=torch.bool, device=states.device)
    steps = torch.zeros(states.shape[1], device=states.device)
    for j in range(1, L):
        x = _corrupt(mem.recall(x, rounds=1), nu, n, gen, uniform)
        alive &= lib.overlap(x, states[j]) >= mr.MATCH
        steps += alive.float()
    return steps


def experiment(record):
    import torch
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        cell = {"n": spec["n"], "k": spec["k"], "p": spec["p"], "noise": {}, "cue": {}}
        mem = mr._memory(spec, seeds, device)
        L = spec["noise_len"]
        states = mem.store_sequence(mr._elements(seeds, "noise", L))[0]
        gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
        for slots in SLOTS:
            cell["noise"][slots] = {}
            for nu in NUS:
                first = torch.stack([replay(mem, states, nu=nu, uniform=slots == "uniform", gen=gen)
                                     for _ in range(DRAWS)]).mean(dim=0)
                cell["noise"][slots][f"{nu:g}"] = asdict(ensemble_from_values(first.tolist(), keys=seeds, label="first"))
                print(f"({spec['n']}, {spec['k']}) {slots} nu={nu:g}: first miss median "
                      f"{statistics.median(first.tolist()):.1f} of {L - 1}", flush=True)
        for cue in CUES:
            cell["cue"][cue] = {}
            for eta in ETAS:
                frac = replay(mem, states, cue=cue, eta=eta, gen=gen) / (L - 1)
                cell["cue"][cue][f"{eta:g}"] = asdict(ensemble_from_values(frac.tolist(), keys=seeds, label="cue"))
                print(f"({spec['n']}, {spec['k']}) cue {cue} eta={eta:g}: {float(frac.mean()):.2f}", flush=True)
        out[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 36's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"bars": {}}
    if not all(c in cells for c in CELLS):
        out["bars"] = {b: False for b in ("U1", "U2", "U3", "U4")}
        return out
    med = {c: {s: statistics.median(cells[c]["noise"][s]["0.1"]["values"]) for s in SLOTS} for c in CELLS}
    out["medians_nu_0.1"] = {f"{n}/{k}/{p:g}": v for (n, k, p), v in med.items()}
    small, big = CELLS
    out["bars"]["U1"] = med[small]["uniform"] >= MILDER * med[small]["top"]
    out["bars"]["U2"] = sum(v >= NOISE_LEN - 1 for v in cells[big]["noise"]["uniform"]["0.1"]["values"]) >= FULL_BRAINS
    out["bars"]["U3"] = all(abs(med[c]["top"] / A34_MEDIAN[c] - 1) <= A34_TOL for c in CELLS)
    out["bars"]["U4"] = all(cells[c]["cue"]["random"]["0"]["mean"] >= CUE_FLOOR for c in CELLS)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Noise", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 36 is registered on seeds 502..521")
    specs = plan(CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.noise", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "tau": mr.TAU, "strength": mr.STRENGTH, "nus": list(NUS),
                    "draws": DRAWS, "slots": list(SLOTS), "cues": list(CUES), "etas": list(ETAS),
                    "noise_seed": NOISE_SEED, "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
