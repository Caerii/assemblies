"""Does a recovery time scaled with n/k restore the critical load? A registered
test of a design rule at cells never run.

Registered in PREREG_refraction_memory.md, Amendment 39.

Amendment 38 found the load law's cliff at rho = 0.081 at n/k = 300, below its
safe rule, with the refraction recovering over 64 rounds -- a fifth of the
n/k steps between a neuron's uses. Exploratory probes after it found the cliff
moves with the recovery time: at (15000, 50, 0.7), rho_50 ~ 0.088 at tau = 64
and ~ 0.115 at tau = 128 and 256, while tau = 512 brings back the tiling
deadline (brains fail near step n/k). The rule tested here:

    tau = n/k / 2

against tau = 64 at the same cells, brains and sequences.

    RELIABILITY  as in Amendments 37 and 38 (memory_load_drift.reliability):
                 for each L on a ladder of rho, a fresh sequence per brain,
                 replayed noiselessly from a uniformly random half of element 0;
                 the fraction of brains replaying all L - 1 steps.

    python -m research.runner load_tau --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells: n/k = 300 and 200, both in regime (k p >= 3 ln n)
CELLS = ((21000, 70, 0.5), (16000, 80, 0.45))
#: the smoke exercises the code on a survey cell, never on a judged one
SMOKE_CELL = (4000, 60, 0.5)
CONTROL = 64
LADDER = md.LADDER                                                      # rho 0.071 .. 0.26
SEEDS = tuple(range(562, 582))
GAIN, SAFE, LOW, HIGH, SHARP = 1.2, ml.SAFE, 0.100, 0.135, ml.SHARP


def rule(n, k):
    """The recovery time the rule sets: half the steps between a neuron's uses."""
    return int(round(n / k / 2))


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        taus = (CONTROL, 32) if smoke else (CONTROL, rule(n, k))
        for tau in taus:
            out.append({"n": n, "k": k, "p": p, "tau": tau, "beta": round(tl.theta(n, k, p), 5),
                        "batch": md.batch_size(n),
                        "ladder": [max(8, int(round(r * ml.unit(n, k, p))))
                                   for r in (LADDER[:2] if smoke else LADDER)]})
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p, tau = spec["n"], spec["k"], spec["p"], spec["tau"]
        rows, zeros = {}, 0
        for L in spec["ladder"]:
            steps = md.reliability(spec, L, seeds, device)
            full = sum(s >= L - 1 for s in steps) / len(steps)
            rows[str(L)] = {"rho": L / ml.unit(n, k, p), "steps": steps, "full": full}
            print(f"({n}, {k}, {p}) tau={tau} L={L} rho={L / ml.unit(n, k, p):.3f}: full {full:.2f}",
                  flush=True)
            zeros = zeros + 1 if full == 0 else 0
            if zeros >= 2:
                break
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": {}})
        cell["tau"][str(tau)] = {"ladder": rows}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 39's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out: dict = {"bars": {}, "rho": {}, "gain": []}
    names = ("T1", "T2", "T3", "T4")
    if not all(c in cells and {str(CONTROL), str(rule(c[0], c[1]))} <= set(cells[c]["tau"])
               for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ruled, gains = [], []
    for n, k, p in CELLS:
        key = f"{n}/{k}/{p:g}"
        r = {tau: {q: ml.crossing(cells[(n, k, p)]["tau"][str(tau)]["ladder"], q) for q in (0.9, 0.5, 0.1)}
             for tau in (CONTROL, rule(n, k))}
        out["rho"][key] = r
        ruled.append(r[rule(n, k)])
        a, b = r[rule(n, k)][0.5], r[CONTROL][0.5]
        gains.append(a / b if a and b else None)
    out["gain"] = gains
    out["bars"]["T1"] = gains[0] is not None and gains[0] >= GAIN and gains[1] is not None and gains[1] > 1
    out["bars"]["T2"] = all(r[0.9] is not None and r[0.9] >= SAFE for r in ruled)
    out["bars"]["T3"] = all(r[0.5] is not None and LOW <= r[0.5] <= HIGH for r in ruled)
    out["bars"]["T4"] = all(r[0.9] and r[0.1] and r[0.1] / r[0.9] <= SHARP for r in ruled)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Load tau", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 39 is registered on seeds 562..581")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.load-tau", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": ml.STRENGTH, "match": ml.MATCH,
                    "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
