"""Bidirectional recall: one area replays a stored sequence forward or
backward, from any point, with long-range inhibition choosing the direction.

Registered in PREREG_refraction_memory.md, Amendment 31 (reverse links
alone) and Amendment 32 (BALANCED: extra forward links as well, protocol
version 2, `--design balanced`).

The round write records only forward transitions, so replay runs forward
only. `store_sequence(reverse_counts=r)` adds every element's reverse
transition r times after the sequence (a stand-in for a rule that
potentiates post-before-pre pairs from a trace of the previous state). A
chain linked both ways points each state at both neighbours; long-range
inhibition (LRI) at recall vetoes the state the replay just left, so the
direction of travel is set by where it came from. The cue is entered into
the LRI history (`prime_lri`), and a replay from the middle is primed with the
neighbour it came from.

    python -m research.runner bidirectional --registration PATH --tag NAME [--smoke]
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
REVERSE = (0, 1, 2, 3)
#: Amendment 32: (extra forward counts, reverse counts); (0, 2) is A31's
#: unbalanced arm, the reference
BALANCED = ((0, 2), (1, 2), (2, 2), (2, 3))
SEEDS_BALANCED = tuple(range(442, 462))
LENGTH, TAU, STRENGTH = 200, 33, 0.5
LRI_PERIOD, LRI_STRENGTH = 4, 100.0
SEEDS = tuple(range(422, 442))
READS = ("forward_masked", "forward_lri", "backward_lri", "middle_backward", "middle_forward")
FULL, NONE, WEAK, LOST = 0.9, 0.1, 0.5, 0.5


def plan(cells, *, smoke=False, design="reverse"):
    arms = ([[0, 0], [0, 2]] if smoke else [[0, r] for r in REVERSE]) if design == "reverse"         else ([[0, 2], [1, 2]] if smoke else [list(a) for a in BALANCED])
    return [{"n": n, "k": k, "p": p, "beta": round(lib.theta(n, k, p), 5),
             "length": 24 if smoke else LENGTH, "arms": arms} for n, k, p in cells]


def arm_name(forward, reverse):
    """A31's arms are named by their reverse count alone; A32's by both."""
    return str(reverse) if forward == 0 else f"{forward}+{reverse}"


def _chain(mem, states, order, k):
    import torch
    x = states[order[0]][:, :k // 2]
    alive = torch.ones(states.shape[1], dtype=torch.bool, device=states.device)
    steps = torch.zeros(states.shape[1], device=states.device)
    for idx in order[1:]:
        x = mem.recall(x, rounds=1)
        alive &= lib.overlap(x, states[idx]) >= 0.3
        steps += alive.float()
    return (steps / (len(order) - 1)).tolist()


def _lri_chain(mem, states, order, k, came_from=None):
    mem.area.set_lri(LRI_PERIOD, LRI_STRENGTH)
    if came_from is not None:
        mem.area.prime_lri(states[came_from])
    mem.area.prime_lri(states[order[0]])
    out = _chain(mem, states, order, k)
    mem.area.set_lri(0, 0.0)
    return out


def run_case(spec, reverse, seeds, device, forward=0):
    from research.experiments.memory_lib.seeding import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p, L = spec["n"], spec["k"], spec["p"], spec["length"]
    mem = AssemblyMemory(seeds_for(list(seeds)), n, k, p, beta=spec["beta"], w_max=lib.W_MAX,
                         norm_init=True, rounds=1, strength=STRENGTH, max_items=4,
                         device=device, bias_decay=math.exp(-1.0 / TAU))
    elements = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"q0e{e}", "A")) for sd in seeds]
                for e in range(L)]
    states, _ = mem.store_sequence(elements, reverse_counts=reverse, forward_counts=forward)
    mid = L // 2
    fwd, bwd = list(range(L)), list(range(L - 1, -1, -1))
    return {"forward_masked": _chain(mem, states, fwd, k),
            "forward_lri": _lri_chain(mem, states, fwd, k),
            "backward_lri": _lri_chain(mem, states, bwd, k),
            "middle_backward": _lri_chain(mem, states, list(range(mid, -1, -1)), k, mid + 1),
            "middle_forward": _lri_chain(mem, states, list(range(mid, L)), k, mid - 1)}


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    out = {}
    for spec in parameters["cells"]:
        cell = {"n": spec["n"], "k": spec["k"], "p": spec["p"], "reverse": {}}
        for f, r in spec["arms"]:
            reads = run_case(spec, r, seeds, parameters["device"], forward=f)
            cell["reverse"][arm_name(f, r)] = {name: asdict(ensemble_from_values(v, keys=seeds, label=name))
                                               for name, v in reads.items()}
            print(f"({spec['n']}, {spec['k']}, {spec['p']}) forward +{f} reverse {r}: " + ", ".join(
                f"{name} {sum(v) / len(v):.2f}" for name, v in reads.items()), flush=True)
        out[f"{spec['n']}/{spec['k']}/{spec['p']:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 31's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"bars": {}}
    if not all(c in cells for c in CELLS) or not all(
            str(r) in c["reverse"] for c in cells.values() for r in REVERSE):
        out["bars"] = {b: False for b in ("R1", "R2", "R3", "R4")}
        return out

    def m(c, r, read):
        return c["reverse"][str(r)][read]["mean"]
    cs = list(cells.values())
    out["bars"]["R1"] = all(m(c, 0, "forward_masked") >= FULL and m(c, 0, "backward_lri") <= NONE
                            for c in cs)
    out["bars"]["R2"] = all(m(c, 1, "backward_lri") <= WEAK for c in cs)
    out["bars"]["R3"] = all(m(c, r, read) >= FULL for c in cs for r in (2, 3)
                            for read in ("forward_lri", "backward_lri", "middle_backward",
                                         "middle_forward"))
    out["bars"]["R4"] = all(m(c, 2, "forward_masked") <= LOST for c in cs)
    out["summary"] = {f"{n}/{k}/{p:g}": {r: {read: round(v["mean"], 3) for read, v in reads.items()}
                                         for r, reads in c["reverse"].items()}
                      for (n, k, p), c in cells.items()}
    return out


def evaluate_balanced(observations):
    """Amendment 32's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = [arm_name(f, r) for f, r in BALANCED]
    out = {"bars": {}}
    if not all(c in cells for c in CELLS) or not all(
            a in c["reverse"] for c in cells.values() for a in names):
        out["bars"] = {b: False for b in ("Q1", "Q2", "Q3")}
        return out
    cs = list(cells.values())
    four = ("forward_lri", "backward_lri", "middle_backward", "middle_forward")

    def m(c, arm, read):
        return c["reverse"][arm][read]["mean"]
    # Q1: the balanced arm (forward ~3.35 counts against reverse 3) both ways
    out["bars"]["Q1"] = all(m(c, "2+3", read) >= FULL for c in cs for read in four)
    # Q2: the stronger direction wins -- forward-heavy (2+2) goes forward and
    # not back; reverse-heavy (2, A31's arm) goes back and not forward
    out["bars"]["Q2"] = all(m(c, "2+2", "forward_lri") >= FULL and m(c, "2+2", "backward_lri") <= LOST
                            and m(c, "2", "backward_lri") >= FULL and m(c, "2", "forward_lri") <= LOST
                            for c in cs)
    # Q3: without LRI the balanced chain has no direction
    out["bars"]["Q3"] = all(m(c, "2+3", "forward_masked") <= LOST for c in cs)
    out["summary"] = {f"{n}/{k}/{p:g}": {a: {read: round(v["mean"], 3) for read, v in reads.items()}
                                         for a, reads in c["reverse"].items()}
                      for (n, k, p), c in cells.items()}
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Bidirectional", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: both)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--design", choices=("reverse", "balanced"), default="reverse")
    args = ap.parse_args(argv)
    balanced = args.design == "balanced"
    registered = SEEDS_BALANCED if balanced else SEEDS
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(registered):
        ap.error("Amendment 32 is registered on seeds 442..461" if balanced
                 else "Amendment 31 is registered on seeds 422..441")
    specs = plan(cells, smoke=args.smoke, design=args.design)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.bidirectional", protocol_version="2" if balanced else "1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "design": args.design, "tau": TAU, "strength": STRENGTH,
                    "lri_period": LRI_PERIOD, "lri_strength": LRI_STRENGTH,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
