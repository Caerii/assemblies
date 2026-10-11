"""Context: does one area tell the same element apart in two sequences?

Registered in PREREG_refraction_memory.md, Amendment 30.

Two chosen sequences per brain share a run of m identical elements:
S1 = A0 A1 A2 C0..C(m-1) D0 D1 D2 and S2 = B0 B1 B2 C0..C(m-1) E0 E1 E2,
written one after the other (`store_sequence`, one round per element), with
G unrelated elements written between them. Replay from half of A0 (B0) must
end in D (E): possible only if the shared elements' states differ between the
two sequences. Three refractions: none (tau = 0), recovering over 33 writing
rounds (the length optimum of Amendment 29), and the cumulative bias that
never recovers.

    python -m research.runner context --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((4000, 60, 0.5), (8000, 60, 0.5))
TAUS = ("0", "33", "inf")
SHARED = (1, 4, 16)
GAPS = (0, 600)
PREFIX = SUFFIX = 3
STRENGTH = 0.5
SEEDS = tuple(range(402, 422))
SEP_MAX, SEP_HEBB, SEP_RECENCY, NEED, HEBB_MAX = 0.05, 0.5, 0.15, 18, 10


def decay_of(name):
    return None if name == "inf" else (0.0 if name == "0" else math.exp(-1.0 / int(name)))


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "beta": round(lib.theta(n, k, p), 5),
             "taus": list(TAUS), "shared": [4] if smoke else list(SHARED),
             "gaps": [0, 16] if smoke else list(GAPS)} for n, k, p in cells]


def run_case(spec, tau, m, gap, seeds, device):
    """Per brain: the shared states' mean overlap across the two sequences,
    and whether each sequence's replay ends on its OWN continuation (own
    overlap >= 0.5 and above the other sequence's) at the branch."""
    import torch
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=spec["beta"], w_max=lib.W_MAX,
                         norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                         device=device, bias_decay=decay_of(tau))

    def els(names):
        return [[to_i32(_seeding.fnv1a_pair_seed(sd, nm, "A")) for sd in seeds] for nm in names]
    shared = [f"C{i}" for i in range(m)]
    x1, _ = mem.store_sequence(els(["A0", "A1", "A2"] + shared + ["D0", "D1", "D2"]))
    if gap:
        mem.store_sequence(els([f"F{i}" for i in range(gap)]))
    x2, _ = mem.store_sequence(els(["B0", "B1", "B2"] + shared + ["E0", "E1", "E2"]))
    sep = torch.stack([lib.overlap(x1[PREFIX + i], x2[PREFIX + i]) for i in range(m)]).mean(dim=0)
    branch = PREFIX + m
    correct = []
    for own, other in ((x1, x2), (x2, x1)):
        x = own[0][:, :k // 2]
        for _ in range(branch):
            x = mem.recall(x, rounds=1)
        o, t = lib.overlap(x, own[branch]), lib.overlap(x, other[branch])
        correct.append(((o >= 0.5) & (o > t)).float())
    return sep.tolist(), (correct[0] * correct[1]).tolist()


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    out = {}
    for spec in parameters["cells"]:
        cell = {"n": spec["n"], "k": spec["k"], "p": spec["p"], "cases": {}}
        for tau in spec["taus"]:
            for m in spec["shared"]:
                for gap in spec["gaps"]:
                    sep, both = run_case(spec, tau, m, gap, seeds, parameters["device"])
                    cell["cases"][f"{tau}/{m}/{gap}"] = {
                        "tau": tau, "m": m, "gap": gap,
                        "separation": asdict(ensemble_from_values(sep, keys=seeds, label="sep")),
                        "both_correct": asdict(ensemble_from_values(both, keys=seeds, label="both"))}
                    print(f"({spec['n']}, {spec['k']}, {spec['p']}) tau={tau} m={m} gap={gap}: "
                          f"shared overlap {sum(sep) / len(sep):.3f}, both branches right on "
                          f"{int(sum(both))} of {len(both)}", flush=True)
        out[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _case(cell, tau, m, gap):
    return cell["cases"].get(f"{tau}/{m}/{gap}")


def evaluate(observations):
    """Amendment 30's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"bars": {}}
    if not all(c in cells for c in CELLS):
        out["bars"] = {b: False for b in ("C1", "C2", "C3")}
        return out

    def right(case):
        return sum(case["both_correct"]["values"])
    c1 = []
    for c in cells.values():
        for tau in ("33", "inf"):
            for m in SHARED:
                case = _case(c, tau, m, 0)
                c1.append(case["separation"]["mean"] <= SEP_MAX and right(case) >= NEED)
    out["bars"]["C1"] = all(c1)
    out["bars"]["C2"] = all(_case(c, "0", 16, 0)["separation"]["mean"] >= SEP_HEBB
                            and right(_case(c, "0", 16, 0)) <= HEBB_MAX for c in cells.values())
    out["bars"]["C3"] = all(_case(c, "33", m, 600)["separation"]["mean"] >= SEP_RECENCY
                            for c in cells.values() for m in SHARED)
    out["summary"] = {f"{n}/{k}/{p:g}": {key: (round(v["separation"]["mean"], 3), right(v))
                                         for key, v in c["cases"].items()}
                      for (n, k, p), c in cells.items()}
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Context", engines=("hashed_assembly_memory",),
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
        ap.error("Amendment 30 is registered on seeds 402..421")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.context", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": STRENGTH, "prefix": PREFIX, "suffix": SUFFIX,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
