"""Is the number of write rounds a DEPTH whose per-round learning rate should
scale as 1/T?

Registered in PREREG_refraction_memory.md, Amendment 15.

An item's write is T rounds of stimulus plus recurrence, each potentiating
the pairs that fire together by (1 + beta), so the log-weight an item puts on
its own assembly grows with T at a fixed beta. Amendment 13 fixed the
completion-optimal per-item write (ln(1 + beta*) = 0.29 / sqrt(k p / 2) at
T = 8). The depth-muP reading is that rounds are depth and the per-round rate
should shrink with them so the per-item write stays fixed:
beta_T = expm1(ln(1 + beta*_8) x 8 / T). Old results read the other way --
T = 16 stored far less than T = 8 at beta = 0.1 -- and this asks whether that
was over-writing.

The refracted memory (0.5 beta, w_max 20, arm B, ungated) at (2000, 60) and
(4000, 60) (k p = 30), T in {6, 8, 12, 16}, recall held at 8 rounds; at each
T, beta over beta_T x 2^(j/4), j = -2..3, plus beta = 0.1; distinct
completion as Amendments 13-14, checkpoints from M = 2. beta stays >= 0.025,
where the int8 count matrix saturates exactly at the clip.

    python -m research.runner time-depth --registration PATH --tag NAME [--smoke]
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

CELLS = ((2000, 60), (4000, 60))
DEPTHS = (6, 8, 12, 16)
RECALL_ROUNDS = 8
MIN_BETA = 0.025
SEEDS = tuple(range(102, 122))
#: Amendment 14's T = 8 capacities (DV)
A14 = {(2000, 60): 492.0, (4000, 60): 1616.0}


def beta_for_depth(T, k, p=pe.P):
    """The per-round rate that keeps the per-item write of the T = 8 law."""
    return math.expm1(math.log1p(rl.beta_pred(k, p)) * 8 / T)


def betas(T, k):
    grid = [round(beta_for_depth(T, k) * 2 ** (j / 4), 4) for j in range(-2, 4)]
    return sorted({b for b in grid if b >= MIN_BETA} | {0.1})


def plan(cells, *, smoke=False):
    out = []
    for n, k in cells:
        guess = rl.predicted(n, k, pe.P)
        for T in ((8,) if smoke else DEPTHS):
            out.append({"n": n, "k": k, "T": T,
                        "betas": [0.1] if smoke else betas(T, k),
                        "law_beta": round(beta_for_depth(T, k), 4),
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
        n, k, T = spec["n"], spec["k"], spec["T"]
        sweep = {}
        for beta in spec["betas"]:
            c, cache = lr.run_beta(n, k, beta, seeds, spec["cap"], device,
                                   profiles[lr.profile_name(beta)],
                                   stop_on=("complete_distinct",), grid_start=2,
                                   give_up=spec["give_up"], rounds=T,
                                   recall_rounds=parameters["recall_rounds"])
            windows = {m: ws.edges([(M, ensemble_from_values(r[m]).mean)
                                    for M, r in cache.items()])
                       for m in ("rank1", "complete", "complete_distinct")}
            sweep[f"{beta:g}"] = {
                "beta": beta, "c_first_item": c, "windows": windows,
                "ensembles": {M: {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                  for nm, v in r.items()} for M, r in sorted(cache.items())},
            }
            print(f"({n}, {k}) T={T} beta={beta:g} c={c}: distinct "
                  f"{windows['complete_distinct'].get('upper')}", flush=True)
        out[f"{n}/{k}/T{T}"] = {"n": n, "k": k, "T": T, "law_beta": spec["law_beta"],
                                "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _cv(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values)) / m


def evaluate(observations, *, a14=None):
    """Amendment 15's bars."""
    a14 = A14 if a14 is None else a14
    rows = {}
    for cell in observations["cells"].values():
        items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
        every = {s["beta"]: lr.capacity(s["windows"]["complete_distinct"]) for s in items}
        # the regular grid; the extra beta = 0.1 point serves D1 only
        bs = [b for b in sorted(every) if b != 0.1]
        caps = [every[b] for b in bs]
        law = min(bs, key=lambda b: abs(math.log(b / cell["law_beta"])))
        rows[(cell["n"], cell["k"], cell["T"])] = {
            "betas": bs, "capacity": caps, "best": max(caps),
            "beta_star": lr.optimum(bs, caps),
            "at_law": every[law], "at_0.1": every[0.1],
        }
    out = {"cells": {f"{n}/{k}/T{T}": r for (n, k, T), r in rows.items()}, "bars": {}}
    cells = sorted({(n, k) for n, k, _ in rows})
    depths = sorted({T for _, _, T in rows})
    out["bars"]["DV"] = all(abs(rows[(n, k, 8)]["best"] / a14[(n, k)] - 1) <= 0.15
                            for n, k in cells if (n, k) in a14)
    out["bars"]["D1"] = all(rows[(n, k, 16)]["at_0.1"] < rows[(n, k, 8)]["at_0.1"]
                            for n, k in cells)
    d2 = True
    for n, k in cells:
        ref = rows[(n, k, 8)]["at_law"]
        d2 &= ref > 0 and all(abs(rows[(n, k, T)]["at_law"] / ref - 1) <= 0.15
                              for T in depths)
    out["bars"]["D2"] = d2
    d3, d4 = True, True
    out["collapse"] = {}
    for n, k in cells:
        stars = [rows[(n, k, T)]["beta_star"] for T in depths]
        if None in stars:
            d3 = False
        else:
            scaled = [math.log1p(s) * T for s, T in zip(stars, depths)]
            raw = [math.log1p(s) for s in stars]
            mean = sum(scaled) / len(scaled)
            out["collapse"][f"{n}/{k}"] = {"cv_scaled": _cv(scaled), "cv_raw": _cv(raw)}
            d3 &= all(abs(v / mean - 1) <= 0.20 for v in scaled) and _cv(scaled) < _cv(raw)
        bests = [rows[(n, k, T)]["best"] for T in depths]
        mean = sum(bests) / len(bests)
        d4 &= all(abs(b / mean - 1) <= 0.15 for b in bests)
    out["bars"]["D3"] = d3
    out["bars"]["D4"] = d4
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Time depth", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--nk", help="n:k pairs (default: the two cells)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(int(v) for v in pair.split(":")) for pair in args.nk.split(",")]
             if args.nk else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 15 is registered on seeds 102..121")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.time-depth", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "recall_rounds": RECALL_ROUNDS, "min_beta": MIN_BETA,
                    "strength": lr.STRENGTH, "w_max": pe.W_MAX, "p": pe.P,
                    "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
