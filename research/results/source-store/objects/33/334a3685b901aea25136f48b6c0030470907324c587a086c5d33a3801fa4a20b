"""Does the refracted memory's best learning rate transfer across scale when
it is scaled with fan-in?

Registered in PREREG_refraction_memory.md, Amendment 12.

Amendment 11 found that the write a memory needs scales down with the number
of cue synapses each neuron sums (X6), the assemblies counterpart of muP's
per-layer learning rate. This study sweeps the plasticity rate beta of the
REAL refracted AssemblyMemory (both fibers: recurrent and stimulus, whose
fan-ins are equal here, k p) at five cells spanning k p = 30 to 120 at two
n/k levels, with refraction at the adopted 0.5 beta -- strength is a multiple
of the learning rate, because the churn transition sits at a fixed MULTIPLE
of beta (Amendment 6), so holding the charge fixed instead would push low
beta past it -- T = 8, w_max = 20, arm B. At every (cell, beta) it stores items
with the capacity study's stimuli and reads, at every checkpoint of the
geometric grid from 16, the module's own masked half-cue recall:

    rank-1      the recall is nearer the cued item than any other stored item
    completion  the fraction of sampled items whose recall recovers at least
                0.8 of the item (Amendment 11's criterion)

Storing stops once both metrics have been at or below 0.5 at two consecutive
checkpoints after either was above, or at the cell's cap. Each metric's load
window is `memory_write_strength.edges`; capacity is its upper edge.

The optimum beta* per cell is the vertex of a parabola in log beta through
the best grid beta and its two neighbours (censored at a grid end). Two
parameterisations are then compared: STANDARD (beta itself transfers) and
FAN-IN SCALED (gamma = ln(1 + beta) sqrt(k p / 2) transfers).

    python -m research.runner learning-rate --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies import describe_assembly_memory                  # noqa: E402
from neural_assemblies.core.numpy_engine import _seeding                # noqa: E402
from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.experiments.seq_capacity_scaling import seeds_for, to_i32  # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

STRENGTH = 0.5
BETAS = (0.025, 0.0354, 0.05, 0.0707, 0.1, 0.1414, 0.2, 0.2828)
#: (n, k) -> the refracted rank-1 ceiling at beta = 0.1 where measured
#: (Amendments 7 and 8); (8000, 240) is new and has none
CELLS = {(2000, 60): 431.31709832368625, (4000, 120): 383.15474360005896,
         (8000, 240): None, (4000, 60): 1961.400398770613,
         (8000, 120): 2229.7301845665493}
LEVELS = (((2000, 60), (4000, 120), (8000, 240)), ((4000, 60), (8000, 120)))
CAP_FACTOR = 16

#: Amendment 13: completion must be DISTINCT (recovered >= 0.8 AND rank-1),
#: fresh brains, a finer grid, and two cells Amendment 12 never saw
#: ((3000, 90), (6000, 180)) where gamma* = 0.29 PREDICTS beta*.
DISTINCT_BETAS = tuple(round(0.025 * 2 ** (i / 4), 4) for i in range(11))   # 0.025 .. 0.1414
DISTINCT_CELLS = {(2000, 60): 431.31709832368625, (3000, 90): None,
                  (4000, 120): 383.15474360005896, (6000, 180): None,
                  (8000, 240): None, (4000, 60): 1961.400398770613,
                  (8000, 120): 2229.7301845665493}
DISTINCT_LEVELS = (((2000, 60), (3000, 90), (4000, 120), (6000, 180), (8000, 240)),
                   ((4000, 60), (8000, 120)))
DISTINCT_SEEDS = tuple(range(62, 82))
GAMMA = 0.29


def profile_name(beta):
    return f"beta-{beta:g}"


def profile(beta):
    return describe_assembly_memory(w_max=pe.W_MAX, beta=beta, strength=STRENGTH,
                                    gate=False, norm_init=True, synaptic_scaling=False)


def readings(mem, St, M, n, k, recall_rounds=None):
    """Per-brain rank-1, own overlap and completed fraction from the module's
    own masked recall on `sample_for(M)` (`recall_rounds` frozen rounds;
    default the write's)."""
    torch = pe._torch()
    B = St.shape[1]
    flat = St.permute(1, 0, 2)                                       # [B, M, k]
    hits = torch.zeros(B, device=St.device)
    own = torch.zeros(B, device=St.device)
    done = torch.zeros(B, device=St.device)
    distinct = torch.zeros(B, device=St.device)
    sample = pe.sample_for(M)
    for i in sample:
        rec = mem.recall(St[int(i)][:, : k // 2], rounds=recall_rounds)
        mask = torch.zeros(B, n, dtype=torch.bool, device=St.device)
        mask.scatter_(1, rec, True)
        ov = torch.stack([mask[b][flat[b]].sum(1) for b in range(B)])  # [B, M]
        hit = ov.argmax(1) == int(i)
        hits += hit.float()
        o = ov[:, int(i)].float() / k
        own += o
        done += (o >= ws.COMPLETE).float()
        distinct += ((o >= ws.COMPLETE) & hit).float()
    S = len(sample)
    return {"rank1": (hits / S).tolist(), "own": (own / S).tolist(),
            "complete": (done / S).tolist(),
            "complete_distinct": (distinct / S).tolist()}


def run_beta(n, k, beta, seeds, cap, device, organ_semantics,
             stop_on=("rank1", "complete"), *, p=None, strength=STRENGTH,
             grid_start=16, give_up=None, rounds=None, recall_rounds=None):
    """Store with the capacity study's stimuli and read every checkpoint.

    Stops once a `stop_on` metric has been above 0.5 and all have then been
    at or below it at two consecutive checkpoints, at `cap`, or -- when
    `give_up` is set -- at the first checkpoint at or past it if none has
    yet risen. `p`, `strength` and `grid_start` default to Amendments 12-13."""
    torch = pe._torch()
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    mem = AssemblyMemory(seeds_for(seeds), n, k, pe.P if p is None else p, beta=beta,
                         w_max=pe.W_MAX, norm_init=True, synaptic_scaling=False,
                         rounds=pe.T if rounds is None else rounds,
                         strength=strength, gate=False, max_items=cap,
                         device=device, organ_semantics=organ_semantics)
    grid = [m for m in pe.geometric_grid(grid_start, cap)]
    stored, cache = [], {}
    seen, below, c = False, 0, None
    for a in range(cap):
        ss = [to_i32(_seeding.fnv1a_pair_seed(seed, f"s{a}", "A")) for seed in seeds]
        stored.append(mem.store(ss, stim_size=k).clone())
        M = a + 1
        if M == 1:
            c = pe.first_item_count(mem.fiber.C, mem.fiber.pres, stored[0], n)
        if M not in grid:
            continue
        cache[M] = readings(mem, torch.stack(stored), M, n, k, recall_rounds)
        cache[M]["fill"] = mem.fill.cpu().numpy().tolist()
        means = [ensemble_from_values(cache[M][m]).mean for m in stop_on]
        if any(v > pe.HALF_BAR for v in means):
            seen, below = True, 0
        elif seen:
            below += 1
        if seen and below >= 2:
            break
        if give_up is not None and not seen and M >= give_up:
            break
    del mem, stored
    torch.cuda.empty_cache()
    return c, cache


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    out = {}
    distinct = parameters.get("completion") == "distinct"
    stop_on = ("rank1", "complete_distinct") if distinct else ("rank1", "complete")
    for spec in parameters["cells"]:
        n, k, cap = spec["n"], spec["k"], spec["cap"]
        sweep = {}
        for beta in parameters["betas"]:
            c, cache = run_beta(n, k, beta, seeds, cap, device, profiles[profile_name(beta)],
                                stop_on=stop_on)
            windows = ws.windows(cache)
            if distinct:
                windows["complete_distinct"] = ws.edges(
                    [(M, ensemble_from_values(r["complete_distinct"]).mean)
                     for M, r in cache.items()])
            sweep[f"{beta:g}"] = {
                "beta": beta, "c_first_item": c,
                "windows": windows,
                "ensembles": {M: {name: asdict(ensemble_from_values(v, keys=seeds, label=name))
                                  for name, v in r.items()} for M, r in sorted(cache.items())},
                "readings": {M: r for M, r in sorted(cache.items())},
            }
            w = sweep[f"{beta:g}"]["windows"]
            print(f"({n}, {k}) beta={beta:g} c={c}: rank-1 {w['rank1']['lower']} .. "
                  f"{w['rank1']['upper']}, completion {w['complete']['lower']} .. "
                  f"{w['complete']['upper']}"
                  + (f", distinct {w['complete_distinct']['lower']} .. "
                     f"{w['complete_distinct']['upper']}" if distinct else ""), flush=True)
        out[f"{n}/{k}"] = {"n": n, "k": k, "cap": cap, "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def capacity(window):
    """Upper edge; a censored window counts at its last point above (a lower
    bound); a metric that never exceeds 0.5 has capacity 0."""
    if window["never"]:
        return 0.0
    return window["last_above"] if window["upper_censored"] else window["upper"]


def optimum(betas, values):
    """beta* by a parabola in log beta through the best grid point and its
    neighbours; None when the best is at a grid end or every value is 0."""
    if max(values) <= 0:
        return None
    i = int(np.argmax(values))
    if i == 0 or i == len(values) - 1:
        return None
    x = np.log(np.asarray(betas[i - 1:i + 2]))
    y = np.asarray(values[i - 1:i + 2], dtype=float)
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0:
        return float(betas[i])
    return float(np.exp(min(max(-b / (2 * a), x[0]), x[2])))


def evaluate(observations, *, anchors=None):
    """Amendment 12's bars."""
    anchors = CELLS if anchors is None else anchors
    cells = {(c["n"], c["k"]): c for c in observations["cells"].values()}
    out = {"cells": {}, "bars": {}}
    for nk, cell in cells.items():
        betas = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
        bs = [s["beta"] for s in betas]
        comp = [capacity(s["windows"]["complete"]) for s in betas]
        rank = [capacity(s["windows"]["rank1"]) for s in betas]
        b_star = optimum(bs, comp)
        kp = nk[1] * pe.P
        out["cells"][f"{nk[0]}/{nk[1]}"] = {
            "betas": bs, "completion": comp, "rank1": rank,
            "c_first_item": [s["c_first_item"] for s in betas],
            "beta_star": b_star,
            "gamma_star": None if b_star is None else math.log1p(b_star) * math.sqrt(kp / 2),
            "best_completion_per_nk2": max(comp) / (nk[0] / nk[1]) ** 2,
            "rank1_at_0.1": rank[bs.index(0.1)] if 0.1 in bs else None,
        }
    rows = out["cells"]

    def row(nk):
        return rows[f"{nk[0]}/{nk[1]}"]
    known = [nk for nk in cells if anchors.get(nk)]
    out["bars"]["TV"] = bool(known) and all(
        row(nk)["rank1_at_0.1"] is not None
        and abs(row(nk)["rank1_at_0.1"] / anchors[nk] - 1) <= 0.10 for nk in known)
    out["bars"]["T1"] = all(row(nk)["beta_star"] is not None for nk in cells)

    def decreasing(level, key):
        values = [row(nk)[key] for nk in level if nk in cells]
        return (len(values) == len(level) and None not in values
                and all(a > b for a, b in zip(values, values[1:])))
    out["bars"]["T2"] = all(decreasing(level, "beta_star") for level in LEVELS)
    gammas = [row(nk)["gamma_star"] for nk in cells]
    if None in gammas:
        out["bars"]["T3"] = False
    else:
        mean = sum(gammas) / len(gammas)
        out["bars"]["T3"] = all(abs(g / mean - 1) <= 0.25 for g in gammas)
    def increasing(level, key):
        values = [row(nk)[key] for nk in level if nk in cells]
        return len(values) == len(level) and all(a < b for a, b in zip(values, values[1:]))
    out["bars"]["T4"] = all(increasing(level, "best_completion_per_nk2") for level in LEVELS)
    # descriptive: how well each exponent collapses beta* (coefficient of variation)
    stars = {nk: row(nk)["beta_star"] for nk in cells}
    if None not in stars.values():
        cv = {}
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            g = [math.log1p(stars[nk]) * (nk[1] * pe.P / 2) ** a for nk in cells]
            m = sum(g) / len(g)
            cv[a] = math.sqrt(sum((x - m) ** 2 for x in g) / len(g)) / m
        out["collapse_cv_by_exponent"] = cv
    return out


def collapse_cv(stars, exponent):
    g = [math.log1p(b) * (nk[1] * pe.P / 2) ** exponent for nk, b in stars.items()]
    m = sum(g) / len(g)
    return math.sqrt(sum((x - m) ** 2 for x in g) / len(g)) / m


def evaluate_distinct(observations, *, anchors=None):
    """Amendment 13's bars, on DISTINCT completion."""
    anchors = DISTINCT_CELLS if anchors is None else anchors
    cells = {(c["n"], c["k"]): c for c in observations["cells"].values()}
    out = {"cells": {}, "bars": {}}
    for nk, cell in cells.items():
        betas = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
        bs = [s["beta"] for s in betas]
        comp = [capacity(s["windows"]["complete_distinct"]) for s in betas]
        rank = [capacity(s["windows"]["rank1"]) for s in betas]
        b_star = optimum(bs, comp)
        kp = nk[1] * pe.P
        out["cells"][f"{nk[0]}/{nk[1]}"] = {
            "betas": bs, "completion": comp, "rank1": rank,
            "beta_star": b_star,
            "predicted_beta_star": math.expm1(GAMMA / math.sqrt(kp / 2)),
            "gamma_star": None if b_star is None else math.log1p(b_star) * math.sqrt(kp / 2),
            "best_completion_per_nk2": max(comp) / (nk[0] / nk[1]) ** 2,
            "rank1_at_0.1": rank[bs.index(0.1)] if 0.1 in bs else None,
        }
    rows = out["cells"]

    def row(nk):
        return rows[f"{nk[0]}/{nk[1]}"]
    known = [nk for nk in cells if anchors.get(nk)]
    out["bars"]["UV"] = bool(known) and all(
        row(nk)["rank1_at_0.1"] is not None
        and abs(row(nk)["rank1_at_0.1"] / anchors[nk] - 1) <= 0.10 for nk in known)
    out["bars"]["U1"] = all(row(nk)["beta_star"] is not None for nk in cells)
    out["bars"]["U2"] = out["bars"]["U1"] and all(
        abs(row(nk)["gamma_star"] / GAMMA - 1) <= 0.15 for nk in cells)
    stars = {nk: row(nk)["beta_star"] for nk in cells}
    if None not in stars.values():
        cv = {a: collapse_cv(stars, a) for a in (0.0, 0.25, 0.5, 0.75, 1.0)}
        out["collapse_cv_by_exponent"] = cv
        out["bars"]["U3"] = min(cv, key=lambda a: cv[a]) == 0.5 and cv[0.5] < 0.10
    else:
        out["bars"]["U3"] = False

    def increasing(level):
        values = [row(nk)["best_completion_per_nk2"] for nk in level if nk in cells]
        return len(values) == len(level) and all(a < b for a, b in zip(values, values[1:]))
    out["bars"]["U4"] = all(increasing(level) for level in DISTINCT_LEVELS)
    return out


def plan(cells, *, smoke=False, table=None):
    table = CELLS if table is None else table
    out = []
    for n, k in cells:
        anchor = table[(n, k)] or (n / k) ** 2 * 0.4
        out.append({"n": n, "k": k,
                    "cap": 64 if smoke else int(min(CAP_FACTOR * anchor, 40000))})
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Learning rate", engines=("hashed_assembly_memory",),
                           default_seeds=tuple(range(42, 62)))
    ap.add_argument("--registration", required=True)
    ap.add_argument("--nk", help="n:k pairs (default: the five cells)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--distinct", action="store_true",
                    help="Amendment 13: distinct completion, its cells, grid and seeds")
    args = ap.parse_args(argv)
    table = DISTINCT_CELLS if args.distinct else CELLS
    cells = ([tuple(int(v) for v in pair.split(":")) for pair in args.nk.split(",")]
             if args.nk else list(table))
    if any(nk not in table for nk in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {sorted(table)}")
    if args.distinct and not args.smoke and list(args.seeds) != list(DISTINCT_SEEDS):
        ap.error("Amendment 13 is registered on seeds 62..81 (pass --seeds 62 ... 81)")
    grid = DISTINCT_BETAS if args.distinct else BETAS
    betas = [0.05, 0.1, 0.1414] if args.smoke else list(grid)
    path = run_experiment(
        script=__file__, protocol="memory.learning-rate",
        protocol_version="2" if args.distinct else "1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment,
        organ_semantics={profile_name(b): profile(b) for b in betas},
        parameters={"cells": plan(cells, smoke=args.smoke, table=table), "betas": betas,
                    "completion": "distinct" if args.distinct else "any",
                    "strength": STRENGTH, "p": pe.P, "w_max": pe.W_MAX, "rounds": pe.T,
                    "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
