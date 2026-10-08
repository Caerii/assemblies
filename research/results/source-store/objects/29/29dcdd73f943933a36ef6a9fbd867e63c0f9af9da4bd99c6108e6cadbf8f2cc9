"""Where does a recovery time of n/k beat one of n/k / 2? The gain of the tau peak
across area size: a registered test.

Registered in PREREG_refraction_memory.md, Amendment 41.

Amendment 39 set tau = n/k / 2 and restored the load law's safe rule at large
areas. Exploratory probes after Amendment 40 found the critical load peaks near
tau = n/k -- at n/k = 20 rho_50 rises from ~0.12 at n/k / 2 to ~0.20 at
1.1 to 1.25 n/k -- but not at n/k = 300, where tau = n/k is slightly worse
(0.110 against 0.115 to 0.117). Tested here: the gain of tau = n/k over
tau = n/k / 2, at four new cells that differ in n/k alone (k p ~ 35,
k p / ln n ~ 4).

    RELIABILITY  as in Amendments 37 to 39 (memory_load_drift.reliability): one
                 sequence of L elements per brain and ladder point, replayed
                 noiselessly from a uniformly random half of element 0; the
                 fraction of brains replaying all L - 1 steps.

    python -m research.runner load_peak --registration PATH --tag NAME [--smoke]
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

#: the judged cells, n/k = 20, 50, 100, 200 at k p ~ 35
CELLS = ((6000, 300, 0.12), (5000, 100, 0.35), (10000, 100, 0.35), (14000, 70, 0.5))
SMOKE_CELL = (2000, 60, 0.5)
#: tau as a multiple of n/k: the Amendment 39 rule and the peak
ARMS = (0.5, 1.0)
LADDER = tuple(0.05 * 2 ** (j / 12) for j in range(34))                 # rho 0.05 .. 0.35
SEEDS = tuple(range(602, 622))
PEAK_GAIN, FLAT_GAIN, SAFE, MONOTONE = 1.3, 1.1, ml.SAFE, -0.8


def tau(n, k, f):
    return max(1, int(round(f * n / k)))


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "f": f, "tau": tau(n, k, f), "beta": round(tl.theta(n, k, p), 5),
             "batch": md.batch_size(n),
             "ladder": [max(8, int(round(r * ml.unit(n, k, p)))) for r in (LADDER[:2] if smoke else LADDER)]}
            for n, k, p in cells for f in ARMS]


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        rows, zeros = {}, 0
        for L in spec["ladder"]:
            steps = md.reliability(spec, L, seeds, device)
            full = sum(s >= L - 1 for s in steps) / len(steps)
            rows[str(L)] = {"rho": L / ml.unit(n, k, p), "steps": steps, "full": full}
            print(f"({n}, {k}, {p}) tau={spec['tau']} ({spec['f']} n/k) L={L} "
                  f"rho={L / ml.unit(n, k, p):.3f}: full {full:.2f}", flush=True)
            zeros = zeros + 1 if full == 0 else 0
            if zeros >= 2:
                break
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "arms": {}})
        cell["arms"][str(spec["f"])] = {"tau": spec["tau"], "ladder": rows}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = ranks(x), ranks(y)
    m = len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (m * (m * m - 1))


def evaluate(observations):
    """Amendment 41's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    out: dict = {"bars": {}, "rho": {}, "gain": {}}
    names = ("P1", "P2", "P3", "P4")
    if not all(c in cells and {str(f) for f in ARMS} <= set(cells[c]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    gains = []
    for n, k, p in CELLS:
        key = f"{n}/{k}/{p:g}"
        r = {str(f): {q: ml.crossing(cells[(n, k, p)]["arms"][str(f)]["ladder"], q) for q in (0.9, 0.5, 0.1)}
             for f in ARMS}
        out["rho"][key] = r
        a, b = r["1.0"][0.5], r["0.5"][0.5]
        g = a / b if a and b else None
        out["gain"][key] = g
        gains.append(g)
    if any(g is None for g in gains):
        out["bars"] = {b: False for b in names}
        return out
    out["spearman"] = _spearman([n / k for n, k, _ in CELLS], gains)
    out["bars"]["P1"] = gains[0] >= PEAK_GAIN
    out["bars"]["P2"] = out["spearman"] <= MONOTONE
    out["bars"]["P3"] = all(r["1.0"][0.9] is not None and r["1.0"][0.9] >= SAFE for r in out["rho"].values())
    out["bars"]["P4"] = gains[-1] <= FLAT_GAIN
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Load peak", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 41 is registered on seeds 602..621")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.load-peak", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "strength": ml.STRENGTH, "match": ml.MATCH,
                    "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
