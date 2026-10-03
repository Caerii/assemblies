"""Does the learning-rate law need the connectivity-noise factor (1 - p)?

Registered in PREREG_refraction_memory.md, Amendment 16.

Amendment 13 fixed ln(1 + beta*) = 0.29 / sqrt(k p / 2) at p = 0.5.
Amendment 14 found that at equal (n/k, k p) a sparser area stores as much but
wants a HIGHER learning rate. The number of cue synapses a neuron receives is
Binomial(k/2, p): its spread is sqrt(k p (1 - p) / 2), not sqrt(k p / 2). If
the write must clear THAT spread, the law is

    ln(1 + beta*) = 0.29 sqrt(2 (1 - p)) / sqrt(k p / 2),

identical at p = 0.5, about 1.22x and 1.32x higher at p = 0.25 and 0.125, and
0.71x LOWER at p = 0.75 -- where the plain fan-in law predicts no change. The
dense side is the discriminating test.

Six cells at n/k = 33: k p = 30 at p = 0.75, 0.5, 0.25, 0.125 and k p = 60 at
p = 0.5, 0.25. The refracted memory (0.5 beta, T = 8, w_max 20, arm B,
ungated) is swept over beta_pred x 2^(j/4), j = -4..4, beta_pred the
(1 - p) law's value, kept >= 0.025 (exact int8 saturation); distinct
completion as Amendments 13-14, checkpoints from M = 2.

    python -m research.runner sparse-law --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_recall_law as rl                # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((1333, 40, 0.75), (2000, 60, 0.5), (4000, 120, 0.25), (8000, 240, 0.125),
         (4000, 120, 0.5), (8000, 240, 0.25))
KP30 = ((1333, 40, 0.75), (2000, 60, 0.5), (4000, 120, 0.25), (8000, 240, 0.125))
MIN_BETA = 0.025
#: no stop decision before this many items (Amendment 15's early-stop artifact)
STOP_FROM = 32
SEEDS = tuple(range(122, 142))
#: Amendment 14's capacities at the shared cells (SV)
A14 = {(2000, 60, 0.5): 492.0, (4000, 120, 0.25): 442.0, (8000, 240, 0.125): 502.0,
       (4000, 120, 0.5): 1208.0, (8000, 240, 0.25): 1109.0}


def beta_pred(k, p):
    """The (1 - p) law: 0.29 sqrt(2 (1 - p)) / sqrt(k p / 2)."""
    return math.expm1(lr.GAMMA * math.sqrt(2 * (1 - p)) / math.sqrt(k * p / 2))


def betas(k, p):
    grid = {round(beta_pred(k, p) * 2 ** (j / 4), 4) for j in range(-4, 5)}
    return sorted(b for b in grid if b >= MIN_BETA)


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = rl.predicted(n, k, p)
        out.append({"n": n, "k": k, "p": p,
                    "betas": [round(beta_pred(k, p), 4)] if smoke else betas(k, p),
                    "beta_pred": round(beta_pred(k, p), 4),
                    "cap": 32 if smoke else int(min(8 * guess, 40000)),
                    "give_up": 16 if smoke else int(4 * guess) + 64})
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        sweep = {}
        betas = spec["betas"]
        results = lr.run_betas(n, k, betas, seeds, spec["cap"], device,
                               {b: profiles[lr.profile_name(b)] for b in betas},
                               stop_on=("complete_distinct",), p=p, grid_start=2,
                               give_up=spec["give_up"], stop_from=STOP_FROM)
        for beta, (c, cache) in zip(betas, results):
            windows = {m: ws.edges([(M, ensemble_from_values(r[m]).mean)
                                    for M, r in cache.items()])
                       for m in ("rank1", "complete", "complete_distinct")}
            sweep[f"{beta:g}"] = {
                "beta": beta, "c_first_item": c, "windows": windows,
                "ensembles": {M: {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                  for nm, v in r.items()} for M, r in sorted(cache.items())},
            }
            print(f"({n}, {k}, {p}) beta={beta:g} c={c}: distinct "
                  f"{windows['complete_distinct'].get('upper')}", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "beta_pred": spec["beta_pred"],
                                 "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _cv(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values)) / m


def evaluate(observations, *, a14=None):
    """Amendment 16's bars."""
    a14 = A14 if a14 is None else a14
    rows = {}
    for cell in observations["cells"].values():
        items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
        bs = [s["beta"] for s in items]
        caps = [lr.capacity(s["windows"]["complete_distinct"]) for s in items]
        star = lr.optimum(bs, caps)
        n, k, p = cell["n"], cell["k"], cell["p"]
        kp = k * p
        rows[(n, k, p)] = {
            "betas": bs, "capacity": caps, "best": max(caps), "beta_star": star,
            "gamma_plain": None if star is None else math.log1p(star) * math.sqrt(kp / 2),
            "gamma_noise": (None if star is None else
                            math.log1p(star) * math.sqrt(kp / 2) / math.sqrt(2 * (1 - p))),
        }
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    out["bars"]["SV"] = all(abs(rows[c]["best"] / v - 1) <= 0.15
                            for c, v in a14.items() if c in rows)
    out["bars"]["S1"] = all(r["beta_star"] is not None for r in rows.values())
    if out["bars"]["S1"]:
        noise = [r["gamma_noise"] for r in rows.values()]
        plain = [r["gamma_plain"] for r in rows.values()]
        mean = sum(noise) / len(noise)
        out["cv"] = {"noise": _cv(noise), "plain": _cv(plain)}
        out["bars"]["S2"] = (all(abs(g / mean - 1) <= 0.15 for g in noise)
                             and _cv(noise) < _cv(plain))
        # the (1 - p) law predicts 0.71x; the plain fan-in law 1.0x
        out["dense_ratio"] = (rows[(1333, 40, 0.75)]["beta_star"]
                              / rows[(2000, 60, 0.5)]["beta_star"])
        out["bars"]["S3"] = out["dense_ratio"] <= 0.85
    else:
        out["bars"]["S2"] = out["bars"]["S3"] = False
    caps = [rows[c]["best"] for c in KP30 if c in rows]
    mean = sum(caps) / len(caps)
    out["bars"]["S4"] = len(caps) == len(KP30) and all(abs(c / mean - 1) <= 0.15 for c in caps)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Sparse law", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: the six cells)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 16 is registered on seeds 122..141")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.sparse-law", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "gamma": lr.GAMMA, "min_beta": MIN_BETA,
                    "strength": lr.STRENGTH, "w_max": pe.W_MAX, "rounds": pe.T,
                    "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "stop_from": STOP_FROM, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
