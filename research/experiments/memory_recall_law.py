"""What does the refracted memory RECALL, against the Hebbian control, and in
which variables does it scale?

Registered in PREREG_refraction_memory.md, Amendment 14.

Amendment 13 fixed the refracted memory's completion-optimal learning rate,
ln(1 + beta*) = 0.29 / sqrt(k p / 2), and saw (unregistered) that at that
optimum the distinct-completion capacity follows M* = C n^2 p / (k ln(n/k)),
the sparse associative memory's synapse-count form, rather than (n/k)^2.
P1's published multiplier (23 to 38x the Hebbian ceiling) is a rank-1
IDENTIFICATION figure at beta = 0.1, which Amendments 11 to 13 showed is an
over-strong write that flatters identification. This study measures both
memories on DISTINCT completion (>= 0.8 of the item recovered AND the recall
nearest it), each at its own best learning rate, at ten (n, k, p) cells that
separate the candidate laws:

    A  k p = 30, p = 0.5, n/k = 16.7 / 33 / 67 / 133   the ln(n/k) factor
    B  k p = 60, p = 0.5, n/k = 16.7 / 33 / 67          the same at k p = 60
    C  sparse p at n/k = 33: (4000, 120, 0.25) and (8000, 240, 0.125) at
       k p = 30; (8000, 240, 0.25) at k p = 60          p and k only via k p?

The refracted memory (0.5 beta, T = 8, arm B) is swept over five learning
rates beta_pred x 2^(j/4), j = -2..2, around the law's prediction; the
Hebbian control (strength 0) over beta = 0.0125 to 0.4 in factors of 2. A
memory's capacity at a cell is its best distinct-completion upper edge over
its sweep. Checkpoints are the geometric grid from 2.

    python -m research.runner recall-law --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies import describe_assembly_memory                  # noqa: E402
from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: (n, k, p) -> block
CELLS = {(1000, 60, 0.5): "A", (2000, 60, 0.5): "A", (4000, 60, 0.5): "A",
         (8000, 60, 0.5): "A",
         (2000, 120, 0.5): "B", (4000, 120, 0.5): "B", (8000, 120, 0.5): "B",
         (4000, 120, 0.25): "C", (8000, 240, 0.125): "C", (8000, 240, 0.25): "C"}
#: equal (n/k, k p), different p: W3
EQUAL_GROUPS = (((2000, 60, 0.5), (4000, 120, 0.25), (8000, 240, 0.125)),
                ((4000, 120, 0.5), (8000, 240, 0.25)))
#: Amendment 13's best distinct completion (WV)
A13 = {(2000, 60, 0.5): 457.0, (4000, 60, 0.5): 1505.0, (4000, 120, 0.5): 1245.0,
       (8000, 120, 0.5): 4157.0}
CONTROL_BETAS = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)
SEEDS = tuple(range(82, 102))
C_GUESS = 0.06


def beta_pred(k, p):
    return math.expm1(lr.GAMMA / math.sqrt(k * p / 2))


def refracted_betas(k, p):
    return [round(beta_pred(k, p) * 2 ** (j / 4), 4) for j in range(-2, 3)]


def synapse_scale(n, k, p):
    """n^2 p / (k ln(n/k)): the synapse-count form."""
    return n * n * p / (k * math.log(n / k))


def predicted(n, k, p):
    return C_GUESS * synapse_scale(n, k, p)


def name(memory, beta):
    return f"{memory}-beta-{beta:g}"


def profile(memory, beta):
    return describe_assembly_memory(w_max=lib.W_MAX, beta=beta,
                                    strength=lib.STRENGTH if memory == "refracted" else 0.0,
                                    gate=False, norm_init=True, synaptic_scaling=False)


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = predicted(n, k, p)
        out.append({
            "n": n, "k": k, "p": p, "block": CELLS[(n, k, p)],
            "refracted": {"betas": [beta_pred(k, p)] if smoke else refracted_betas(k, p),
                          "cap": 32 if smoke else int(min(8 * guess, 40000)),
                          "give_up": 16 if smoke else int(4 * guess) + 64},
            "control": {"betas": [0.1] if smoke else list(CONTROL_BETAS),
                        "cap": 32 if smoke else int(0.2 * (n / k) ** 2) + 256,
                        "give_up": 16 if smoke else int(0.1 * (n / k) ** 2) + 64},
        })
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        cell = {"n": n, "k": k, "p": p, "block": spec["block"]}
        for memory in ("refracted", "control"):
            plan_m = spec[memory]
            sweep = {}
            betas = plan_m["betas"]
            results = lr.run_betas(
                n, k, betas, seeds, plan_m["cap"], device,
                {b: profiles[name(memory, b)] for b in betas},
                stop_on=("rank1", "complete_distinct") if memory == "control"
                else ("complete_distinct",),
                p=p, strength=lib.STRENGTH if memory == "refracted" else 0.0,
                grid_start=2, give_up=plan_m["give_up"])
            for beta, (c, cache) in zip(betas, results):
                windows = {m: ws.edges([(M, ensemble_from_values(r[m]).mean)
                                        for M, r in cache.items()])
                           for m in ("rank1", "complete", "complete_distinct")}
                sweep[f"{beta:g}"] = {
                    "beta": beta, "c_first_item": c, "windows": windows,
                    "ensembles": {M: {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                      for nm, v in r.items()} for M, r in sorted(cache.items())},
                    "readings": {M: r for M, r in sorted(cache.items())},
                }
                w = windows["complete_distinct"]
                print(f"({n}, {k}, {p}) {memory} beta={beta:g}: distinct "
                      f"{w.get('lower')} .. {w.get('upper')}, rank-1 "
                      f"{windows['rank1'].get('upper')}", flush=True)
            cell[memory] = sweep
        out[f"{n}/{k}/{p:g}"] = cell
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def best(sweep, metric="complete_distinct"):
    """(capacity, beta, index) of the best learning rate in a sweep; a
    metric that never exceeds 0.5 has capacity 0."""
    items = sorted(sweep.values(), key=lambda s: s["beta"])
    caps = [lr.capacity(s["windows"][metric]) for s in items]
    i = max(range(len(caps)), key=lambda j: caps[j])
    return caps[i], items[i]["beta"], i, len(caps)


def _cv(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values)) / m


def evaluate(observations, *, a13=None):
    """Amendment 14's bars."""
    a13 = A13 if a13 is None else a13
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out = {"cells": {}, "bars": {}}
    for nkp, cell in cells.items():
        n, k, p = nkp
        ref, ref_beta, ref_i, ref_n = best(cell["refracted"])
        ctl, ctl_beta, _, _ = best(cell["control"])
        ref_r1, _, _, _ = best(cell["refracted"], "rank1")
        ctl_r1, _, _, _ = best(cell["control"], "rank1")
        out["cells"][f"{n}/{k}/{p:g}"] = {
            "block": cell["block"], "kp": k * p, "n_over_k": n / k,
            "refracted": ref, "refracted_beta": ref_beta, "refracted_interior": 0 < ref_i < ref_n - 1,
            "refracted_index": ref_i,
            "control": ctl, "control_beta": ctl_beta,
            "multiplier": ref / max(ctl, 2.0),
            "refracted_rank1": ref_r1, "control_rank1": ctl_r1,
            "per_nk2": ref / (n / k) ** 2,
            "C": ref / synapse_scale(n, k, p),
        }
    rows = out["cells"]

    def row(nkp):
        return rows[f"{nkp[0]}/{nkp[1]}/{nkp[2]:g}"]
    out["bars"]["WV"] = all(abs(row(nkp)["refracted"] / v - 1) <= 0.15
                            for nkp, v in a13.items() if nkp in cells)
    out["bars"]["W1"] = all(row(nkp)["multiplier"] >= 10 for nkp in cells)
    w2 = True
    for block in ("A", "B"):
        members = [nkp for nkp in cells if cells[nkp]["block"] == block]
        cs = [row(nkp)["C"] for nkp in members]
        pn = [row(nkp)["per_nk2"] for nkp in members]
        mean = sum(cs) / len(cs)
        tight = all(abs(c / mean - 1) <= 0.15 for c in cs)
        out[f"block_{block}"] = {"C": cs, "per_nk2": pn, "cv_C": _cv(cs), "cv_per_nk2": _cv(pn)}
        w2 &= tight and _cv(cs) < _cv(pn)
    out["bars"]["W2"] = w2
    w3 = True
    for group in EQUAL_GROUPS:
        caps = [row(nkp)["refracted"] for nkp in group]
        idx = [row(nkp)["refracted_index"] for nkp in group]
        mean = sum(caps) / len(caps)
        w3 &= all(abs(c / mean - 1) <= 0.15 for c in caps) and max(idx) - min(idx) <= 1
    out["bars"]["W3"] = w3
    out["bars"]["W4"] = all(row(nkp)["refracted_interior"] for nkp in cells if nkp[2] < 0.5)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Recall law", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: the ten cells)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {sorted(CELLS)}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 14 is registered on seeds 82..101")
    specs = plan(cells, smoke=args.smoke)
    profiles = {}
    for spec in specs:
        for memory in ("refracted", "control"):
            for beta in spec[memory]["betas"]:
                profiles[name(memory, beta)] = profile(memory, beta)
    path = run_experiment(
        script=__file__, protocol="memory.recall-law", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "gamma": lr.GAMMA, "c_guess": C_GUESS,
                    "strength": lib.STRENGTH, "w_max": lib.W_MAX, "rounds": lib.ROUNDS,
                    "half_bar": lib.HALF_BAR, "complete": lib.COMPLETE,
                    "recall_sample": lib.RECALL_SAMPLE, "measurement_seed": lib.MEASUREMENT_SEED,
                    "grid_start": 2, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
