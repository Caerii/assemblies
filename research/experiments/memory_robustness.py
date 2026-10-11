"""Robustness of the refracted sequence memory: many sequences at once,
activity noise during replay, and corrupted cues.

Registered in PREREG_refraction_memory.md, Amendment 34.

Every earlier sequence study stored one sequence per brain and replayed it
noiselessly. Here, at the recovery time that maximised length (tau = 33,
Amendment 29):

    MANY   M chosen sequences of 32 elements per brain, written one after
           another; the mean replay fraction over up to 16 of them, on a
           ladder of M. The TOTAL at which it falls through one half (M x 32,
           log-interpolated) against the single-sequence limit of Amendment 29.
    NOISE  one sequence of L = 400; at every replay step a fraction nu of the
           winners is replaced by random neurons; each brain's first-miss step
           (own overlap < 0.3), mean over four noise draws (L - 1 if none).
    CUE    the half cue with a fraction eta of its neurons replaced; the
           replay fraction.

    python -m research.runner robustness --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((4000, 60, 0.5), (8000, 60, 0.5))
#: Amendment 29's best single-sequence limits at these cells
A29 = {(4000, 60, 0.5): 2308.9, (8000, 60, 0.5): 6890.3}
TAU, STRENGTH, MATCH = 33, 0.5, 0.3
SEQ_LEN = 32
M_LADDER = tuple(int(round(8 * 2 ** (j / 2))) for j in range(15))     # 8 .. 1024
NOISE_LEN, NUS, DRAWS = 400, (0.05, 0.07, 0.1, 0.2), 4
ETAS = (0.25, 0.5)
NOISE_SEED = 11
SEEDS = tuple(range(482, 502))
BUDGET_BAND, HARMLESS, HORIZON_GAIN, CUE_FLOOR = (0.6, 1.6), 18, 2.0, 0.9


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "beta": round(lib.theta(n, k, p), 5),
             "m_ladder": list(M_LADDER[:2] if smoke else M_LADDER),
             "noise_len": 40 if smoke else NOISE_LEN} for n, k, p in cells]


def _memory(spec, seeds, device):
    from research.experiments.memory_lib.seeding import seeds_for
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    return AssemblyMemory(seeds_for(list(seeds)), spec["n"], spec["k"], spec["p"], beta=spec["beta"],
                          w_max=lib.W_MAX, norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                          device=device, bias_decay=math.exp(-1.0 / TAU))


def _elements(seeds, tag, length):
    from research.experiments.memory_lib.seeding import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    return [[to_i32(_seeding.fnv1a_pair_seed(sd, f"{tag}e{e}", "A")) for sd in seeds]
            for e in range(length)]


def _corrupt(x, frac, n, gen):
    import torch
    m = int(round(frac * x.shape[1]))
    if m <= 0:
        return x
    x = x.clone()
    x[:, :m] = torch.randint(0, n, (x.shape[0], m), device=x.device, generator=gen)
    return x


def replay(mem, states, *, nu=0.0, eta=0.0, gen=None):
    """Each brain's first-miss step [B] (L - 1 if none)."""
    import torch
    n, k, L = mem.n, mem.k, states.shape[0]
    x = _corrupt(states[0][:, :k // 2], eta, n, gen)
    alive = torch.ones(states.shape[1], dtype=torch.bool, device=states.device)
    steps = torch.zeros(states.shape[1], device=states.device)
    for j in range(1, L):
        x = _corrupt(mem.recall(x, rounds=1), nu, n, gen)
        alive &= lib.overlap(x, states[j]) >= MATCH
        steps += alive.float()
    return steps


def experiment(record):
    import torch
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        cell = {"n": spec["n"], "k": spec["k"], "p": spec["p"], "many": {}, "noise": {}, "cue": {}}
        # MANY
        risen, low = False, 0
        for M in spec["m_ladder"]:
            mem = _memory(spec, seeds, device)
            seqs = [mem.store_sequence(_elements(seeds, f"q{q}", SEQ_LEN))[0] for q in range(M)]
            read = sorted({round(i * (M - 1) / 15) for i in range(16)})
            frac = torch.stack([replay(mem, seqs[i]) / (SEQ_LEN - 1) for i in read]).mean(dim=0)
            cell["many"][str(M)] = asdict(ensemble_from_values(frac.tolist(), keys=seeds, label="replay"))
            mean = float(frac.mean())
            print(f"({spec['n']}, {spec['k']}) many M={M} ({M * SEQ_LEN} elements): {mean:.2f}", flush=True)
            risen = risen or mean > 0.5
            low = low + 1 if mean < 0.2 else 0
            if low >= 2:
                break
        # NOISE and CUE, on one sequence
        mem = _memory(spec, seeds, device)
        L = spec["noise_len"]
        states = mem.store_sequence(_elements(seeds, "noise", L))[0]
        gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
        for nu in (0.0,) + NUS:
            first = torch.stack([replay(mem, states, nu=nu, gen=gen) for _ in range(DRAWS if nu else 1)]).mean(dim=0)
            cell["noise"][f"{nu:g}"] = asdict(ensemble_from_values(first.tolist(), keys=seeds, label="first"))
            print(f"({spec['n']}, {spec['k']}) noise nu={nu:g}: first miss median "
                  f"{statistics.median(first.tolist()):.0f} of {L - 1}", flush=True)
        for eta in ETAS:
            frac = replay(mem, states, eta=eta, gen=gen) / (L - 1)
            cell["cue"][f"{eta:g}"] = asdict(ensemble_from_values(frac.tolist(), keys=seeds, label="cue"))
            print(f"({spec['n']}, {spec['k']}) cue eta={eta:g}: {float(frac.mean()):.2f}", flush=True)
        out[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def budget(cell):
    return SEQ_LEN * sq.capacity({M: e["mean"] for M, e in cell["many"].items()})


def evaluate(observations):
    """Amendment 34's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"bars": {}}
    if not all(c in cells for c in CELLS):
        out["bars"] = {b: False for b in ("N1", "N2", "N3", "N4")}
        return out
    out["budget"] = {f"{n}/{k}/{p:g}": budget(c) / A29[(n, k, p)] for (n, k, p), c in cells.items()}
    out["bars"]["N1"] = all(BUDGET_BAND[0] <= v <= BUDGET_BAND[1] for v in out["budget"].values())
    out["bars"]["N2"] = all(sum(v >= NOISE_LEN - 1 for v in c["noise"]["0.05"]["values"]) >= HARMLESS
                            for c in cells.values())
    small, big = cells[CELLS[0]], cells[CELLS[1]]
    out["bars"]["N3"] = (statistics.median(big["noise"]["0.1"]["values"])
                         >= HORIZON_GAIN * statistics.median(small["noise"]["0.1"]["values"]))
    out["bars"]["N4"] = all(c["cue"]["0.25"]["mean"] >= CUE_FLOOR for c in cells.values())
    out["medians"] = {f"{n}/{k}/{p:g}": {nu: statistics.median(e["values"]) for nu, e in c["noise"].items()}
                      for (n, k, p), c in cells.items()}
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Robustness", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: both)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 34 is registered on seeds 482..501")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.robustness", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "tau": TAU, "strength": STRENGTH, "seq_len": SEQ_LEN,
                    "nus": list(NUS), "draws": DRAWS, "etas": list(ETAS), "noise_seed": NOISE_SEED,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
