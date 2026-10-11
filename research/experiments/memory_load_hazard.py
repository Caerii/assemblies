"""Is the load law's cliff a hazard compounded over length? Short sequences
predict long ones: a registered test.

Registered in PREREG_refraction_memory.md, Amendment 43.

Amendment 40 found many short sequences fail one by one where one long sequence
fails all or none. A model of a sequence of l elements -- the cue captures the
first step with probability c, every later step fails with hazard h, both set by
the total load rho -- gives

    P(whole | l, rho) = c(rho) (1 - h(rho))^(l - 1),

and, fitted to Amendment 40's 16- and 64-element arms after the fact, located
its single sequences' cliffs (5,000 to 13,000 elements) to one ladder step. The
cliff is then where l h(rho) ~ 1, and rho_50 falls with ln l. Tested here at
cells never run: c and h from 16- and 64-element sequences predict, with no
further parameter, 256- and 1024-element sequences and one sequence of the
whole load.

    RELIABILITY  memory_load_many.whole: per brain, M = L / l sequences of l
                 elements (or one of L), each by its own store_sequence, every
                 one replayed noiselessly from a uniformly random half of its
                 first element; the mean over brains of the fraction whole.
                 Every arm at a ladder point holds the same total L.

    python -m research.runner load_hazard --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_load_many as mm                 # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells and their recovery times (Amendment 41's rule: n/k / 2)
CELLS = ((10000, 70, 0.5, 71), (8000, 50, 0.7, 80))
SMOKE_CELL = (2000, 60, 0.5, 17)
#: the fitting arms, the predicted arms, and one sequence of the whole load
FIT, PREDICT = (16, 64), (256, 1024)
ARMS = FIT + PREDICT + ("single",)
QUANTUM = 1024
LADDER = tuple(0.08 * 2 ** (j / 12) for j in range(17))                 # rho 0.08 .. 0.20
SEEDS = tuple(range(642, 662))
CLIFF = 0.05
EMPTY = 0.02


def plan(cells, *, smoke=False):
    out = []
    for n, k, p, tau in cells:
        q = 16 if smoke else QUANTUM
        ladder = [max(1, round(r * lib.unit(n, k, p) / q)) * q for r in (LADDER[:2] if smoke else LADDER)]
        for arm in ((16, "single") if smoke else ARMS):
            out.append({"n": n, "k": k, "p": p, "tau": tau, "arm": arm,
                        "beta": round(lib.theta(n, k, p), 5), "batch": md.batch_size(n), "ladder": ladder})
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
            fr = mm.whole(spec, L, seeds, device)
            full = sum(fr) / len(fr)
            rows[str(L)] = {"rho": L / lib.unit(n, k, p), "whole": fr, "full": full}
            print(f"({n}, {k}, {p}) tau={spec['tau']} arm={arm} L={L} rho={L / lib.unit(n, k, p):.3f}: "
                  f"whole {full:.3f}", flush=True)
            zeros = zeros + 1 if full < EMPTY else 0
            if zeros >= 2:
                break
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][str(arm)] = {"ladder": rows}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def fit(p16, p64):
    """(c, h) from the 16- and 64-element arms at one load: h from their ratio
    (0 if the longer is not worse), c from the shorter."""
    if p16 <= 0:
        return 0.0, 1.0
    h = 0.0 if p64 >= p16 else 1 - (max(p64, 1e-9) / p16) ** (1 / (FIT[1] - FIT[0]))
    return min(1.0, p16 / (1 - h) ** (FIT[0] - 1)), h


def predict(c, h, length):
    return c * (1 - h) ** (length - 1)


def evaluate(observations):
    """Amendment 43's bars: every predicted arm's rho_50 -- the load at which the
    model, fitted at each ladder point to the 16- and 64-element arms, puts its
    whole fraction through one half -- against the measured one."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out: dict = {"bars": {}, "rho50": {}, "fits": {}}
    names = ("K1", "K2", "K3", "K4")
    if not all(c[:3] in cells and {str(a) for a in ARMS} <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    for n, k, p, _ in CELLS:
        key = f"{n}/{k}/{p:g}"
        rows = {a: cells[(n, k, p)]["arms"][str(a)]["ladder"] for a in ARMS}
        common = [L for L in rows[16] if L in rows[64]]
        fits, predicted = {}, {str(a): {} for a in PREDICT + ("single",)}
        for L in common:
            c, h = fit(rows[16][L]["full"], rows[64][L]["full"])
            rho = rows[16][L]["rho"]
            fits[L] = {"rho": rho, "c": c, "h": h}
            for a in PREDICT + ("single",):
                length = int(L) if a == "single" else a
                predicted[str(a)][L] = {"rho": rho, "full": predict(c, h, length)}
        out["fits"][key] = fits
        measured = {str(a): ml.crossing(rows[a], 0.5) for a in ARMS}
        pred = {a: ml.crossing(r, 0.5) for a, r in predicted.items()}
        out["rho50"][key] = {"measured": measured, "predicted": pred}
        for bar, a in (("K1", "256"), ("K2", "1024"), ("K3", "single")):
            ok[bar] &= (pred[a] is not None and measured[a] is not None
                        and abs(pred[a] / measured[a] - 1) <= CLIFF)
        seq = [measured[str(a)] for a in ARMS]
        ok["K4"] &= all(x is not None for x in seq) and all(a > b for a, b in zip(seq, seq[1:]))
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Load hazard", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 43 is registered on seeds 642..661")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.load-hazard", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": lib.STRENGTH, "match": lib.MATCH,
                    "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
