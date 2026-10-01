"""What does the circuit store as a function of how hard each item is written?

Registered in PREREG_refraction_memory.md, Amendment 10.

Amendment 9 found the random-pattern ceiling of the refracted memory's circuit
moves by +66 to +144% when each item is written two counts weaker (PE-S). The
potentiation is multiplicative, so a synapse shared by s items written c times
each weighs min((1 + beta)^(c s), w_max): the rule's SHAPE changes with c --
nearly linear in s at c = 1, convex in the middle, and BINARY (1 or w_max) from
c = 32 on, where one count of sharing already reaches the clip. This study
traces the ceiling across that whole range on the same circuit as
`memory_pattern_efficiency` (presence, norm_init, chain table, half-cue
masked k-WTA recall), with the same independent random patterns at every c
of a cell (paired), and reads two ceilings per write strength:

    rank-1      the recall is nearer the cued item than any other (the
                capacity study's criterion: IDENTIFICATION)
    completion  the recall recovers at least 0.8 of the cued item
                (COMPLETION), per item; the ceiling is where the
                fraction of items completed crosses 0.5

The model's own write strength c is measured as in Amendment 9 (the rounded
mean count on the first stored item's internal pairs).

Checkpoints are found per curve by doubling from a power of two until the
ensemble mean crosses 0.5, then adding the 1.5x point inside the bracket
(`adaptive_ceiling`); every evaluated point is recorded.

    python -m research.runner write-strength --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies import describe_assembly_memory                  # noqa: E402
from neural_assemblies.core.numpy_engine import _seeding                # noqa: E402
from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments._substrate import ceiling_from_curve          # noqa: E402
from research.experiments.seq_capacity_scaling import seeds_for, to_i32  # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

COUNTS = (1, 2, 3, 4, 5, 6, 8, 11, 16, 23, 32)
COMPLETE = 0.8
CAP = 1 << 17
CELLS = tuple(pe.ANCHORS)
#: Amendment 9's random-pattern rank-1 ceilings at the model's own c (WS-V)
RANDOM_ANCHORS = {(2000, 60): 818.0, (4000, 120): 762.0, (4000, 60): 3583.0,
                  (8000, 120): 4088.0, (8000, 60): 15288.0, (2000, 30): 5043.0,
                  (4000, 30): 20741.0}


def readings(W, patterns, sample, k):
    """rank-1 hit, mean own overlap and completed fraction for one brain."""
    torch = pe._torch()
    idx = torch.as_tensor(sample, device=patterns.device, dtype=torch.int64)
    rec = pe.recall(W, patterns[idx][:, : k // 2], k)
    mask = torch.zeros(len(sample), W.shape[0], dtype=torch.bool, device=W.device)
    mask.scatter_(1, rec, True)
    ov = torch.stack([mask[i][patterns].sum(1) for i in range(len(sample))])
    own = ov[torch.arange(len(sample), device=W.device), idx].float() / k
    return ((ov.argmax(1) == idx).float().mean().item(), own.mean().item(),
            (own >= COMPLETE).float().mean().item())


def evaluate_at(M, brains, c, k, n, pres_bits, invdj, tab):
    """Per-brain readings at M with c counts per internal pair; `brains`
    maps brain index -> its patterns [>= M, k]."""
    out = {"rank1": [], "own": [], "complete": [], "potentiated": [], "pair_sharing": []}
    for b, pats in brains.items():
        present = pe.presence_of(pres_bits, b, n)
        N = pe.pattern_counts(pats[:M], n)
        pot, share = pe.pattern_stats(N, present, M, k)
        W = pe.weights((N * c).clamp_(max=tab.numel() - 1), present, invdj[b], tab)
        hit, own, done = readings(W, pats[:M], pe.sample_for(M), k)
        for name, value in (("rank1", hit), ("own", own), ("complete", done),
                            ("potentiated", pot), ("pair_sharing", share)):
            out[name].append(value)
        del N, W, present
    return out


def adaptive_ceiling(measure_at, metric, start, cap, cache):
    """Double from `start` until the ensemble mean of `metric` is <= 0.5
    (halving first if it already is), then add the 1.5x point inside the
    bracket. `cache` maps M -> readings and is shared between metrics."""
    def mean_at(M):
        if M not in cache:
            cache[M] = measure_at(M)
        return ensemble_from_values(cache[M][metric]).mean

    M = start
    while mean_at(M) <= pe.HALF_BAR and M > 16:
        M //= 2
    if mean_at(M) <= pe.HALF_BAR:
        return
    while M < cap and mean_at(M) > pe.HALF_BAR:
        M *= 2
    if M <= cap and mean_at(M) <= pe.HALF_BAR:
        mean_at(3 * M // 4)


def scan(measure_at, cap, give_up, cache):
    """Amendment 11's instrument: every power of two from 16, read for BOTH
    metrics, so a curve that fails at low load and works later is seen.

    Doubling stops once some metric has been above 0.5 and both have then
    been at or below it at two consecutive points, at `cap`, or at `give_up`
    if neither metric has yet been above 0.5. Then the 1.5x point is added
    inside every doubling across which either metric changes side of 0.5."""
    metrics = ("rank1", "complete")

    def means(M):
        if M not in cache:
            cache[M] = measure_at(M)
        return {m: ensemble_from_values(cache[M][m]).mean for m in metrics}

    M, seen, below = 16, False, 0
    while M <= cap:
        above = any(v > pe.HALF_BAR for v in means(M).values())
        if above:
            seen, below = True, 0
        elif seen:
            below += 1
        if (seen and below >= 2) or (not seen and M >= give_up):
            break
        M *= 2
    points = sorted(cache)
    for a, b in zip(points, points[1:]):
        if b == 2 * a:
            ma, mb = means(a), means(b)
            if any((ma[m] > pe.HALF_BAR) != (mb[m] > pe.HALF_BAR) for m in metrics):
                means(3 * a // 2)


def edges(points):
    """Lower and upper edges of the load window where a metric exceeds 0.5,
    interpolated in log2 M. `lower` is None when the curve already exceeds
    0.5 at its first point; `upper` is None when it never does and
    `upper_censored` when it still does at its last."""
    import math
    pts = sorted(points)
    above = [i for i, (_, v) in enumerate(pts) if v > pe.HALF_BAR]

    def cross(i, j):
        (m0, v0), (m1, v1) = pts[i], pts[j]
        t = 0.0 if v0 == v1 else (v0 - pe.HALF_BAR) / (v0 - v1)
        t = min(max(t, 0.0), 1.0)
        return 2.0 ** (math.log2(m0) + t * (math.log2(m1) - math.log2(m0)))

    if not above:
        return {"lower": None, "upper": None, "upper_censored": False, "never": True}
    first, last = above[0], above[-1]
    return {"lower": None if first == 0 else cross(first - 1, first),
            "upper": None if last == len(pts) - 1 else cross(last, last + 1),
            "upper_censored": last == len(pts) - 1, "never": False,
            "last_above": pts[last][0]}


def windows(cache):
    out = {}
    for metric in ("rank1", "complete"):
        out[metric] = edges([(M, ensemble_from_values(r[metric]).mean)
                             for M, r in cache.items()])
    return out


def ceilings(cache, seeds):
    out = {}
    for metric in ("rank1", "complete"):
        curve = [(M, ensemble_from_values(r[metric]).mean) for M, r in sorted(cache.items())]
        out[metric] = pe._ceiling(curve)
    return out


def summarise(cache, seeds):
    return {"ceilings": ceilings(cache, seeds), "windows": windows(cache),
            "ensembles": {M: {name: asdict(ensemble_from_values(values, keys=seeds, label=name))
                              for name, values in r.items()}
                          for M, r in sorted(cache.items())},
            "readings": {M: r for M, r in sorted(cache.items())}}


def random_brains(n, k, seeds, cap, device, salt="ws"):
    """Each brain's own independent random k-subsets, generated in chunks
    from a per-(seed, cell, chunk) generator so the set is a function of the
    brain alone."""
    torch = pe._torch()
    brains = {}
    for b, seed in enumerate(seeds):
        parts = []
        for s in range(0, cap, pe.CHUNK):
            g = torch.Generator(device=device)
            g.manual_seed(to_i32(_seeding.fnv1a_pair_seed(seed, f"{salt}/{n}/{k}/{s}", "A")) & 0x7FFFFFFF)
            m = min(pe.CHUNK, cap - s)
            parts.append(torch.rand(m, n, generator=g, device=device).topk(k, dim=1).indices)
        brains[b] = torch.cat(parts)
    return brains


def circuit(n, k, seeds, device, organ_semantics):
    """The cell's presence, in-degree and the model's own write strength c
    (one stored item, the capacity study's first stimulus)."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    mem = AssemblyMemory(seeds_for(seeds), n, k, pe.P, beta=pe.BETA, w_max=pe.W_MAX,
                         norm_init=True, synaptic_scaling=False, rounds=pe.T,
                         strength=pe.STRENGTH, gate=False, max_items=1, device=device,
                         organ_semantics=organ_semantics)
    ss = [to_i32(_seeding.fnv1a_pair_seed(seed, "s0", "A")) for seed in seeds]
    first = mem.store(ss, stim_size=k)
    mem.check()
    c = pe.first_item_count(mem.fiber.C, mem.fiber.pres, first, n)
    return c, mem.fiber.pres.clone(), mem.fiber.invdj.clone()


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    torch = pe._torch()
    tab = pe.chain(device)
    cap = parameters["cap"]
    out = {}
    for spec in parameters["cells"]:
        n, k = spec["n"], spec["k"]
        c_model, pres_bits, invdj = circuit(n, k, seeds, device, profiles["refracted"])
        start = 1 << max(4, int(np.floor(np.log2(0.05 * (n / k) ** 2))))
        cap_k = cap if k <= 60 else cap // 2      # pattern memory at k = 120
        search = parameters.get("search", "adaptive")
        brains = random_brains(n, k, seeds, cap_k, device,
                               salt="ws" if search == "adaptive" else "ws-scan")
        give_up = 1 << int(np.ceil(np.log2(8 * (n / k) ** 2)))
        sweep = {}
        for c in parameters["counts"]:
            cache = {}
            def at(M, c=c):
                return evaluate_at(M, brains, c, k, n, pres_bits, invdj, tab)
            if search == "adaptive":
                adaptive_ceiling(at, "rank1", start, cap_k, cache)
                adaptive_ceiling(at, "complete", start, cap_k, cache)
            else:
                scan(at, cap_k, give_up, cache)
            sweep[str(c)] = summarise(cache, seeds)
            w = sweep[str(c)]["windows"]
            print(f"({n}, {k}) c={c}: rank-1 window {w['rank1']['lower']} .. {w['rank1']['upper']}, "
                  f"completion window {w['complete']['lower']} .. {w['complete']['upper']}", flush=True)
        del brains
        torch.cuda.empty_cache()
        cell = {"n": n, "k": k, "c_model": c_model, "cap": cap_k, "sweep": sweep}
        out[f"{n}/{k}"] = cell
        del pres_bits, invdj
        torch.cuda.empty_cache()
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations, *, random_anchors=None):
    """Amendment 10's bars. `random_anchors` maps (n, k) -> Amendment 9's
    random-pattern rank-1 ceiling at the model's own c."""
    random_anchors = RANDOM_ANCHORS if random_anchors is None else random_anchors
    cells = {(c["n"], c["k"]): c for c in observations["cells"].values()}
    judged = [nk for nk in pe.IN_REGIME if nk in cells]
    out = {"cells": {}, "bars": {}}

    def ceil(cell, c, metric="rank1"):
        return cell["sweep"][str(c)]["ceilings"][metric]

    def m(cell, c, metric="rank1"):
        return ceil(cell, c, metric)["m_star"]

    def resolved(cell, c, metric="rank1"):
        x = ceil(cell, c, metric)
        return not x["censored"] and not x["below_at_first"]

    for nk, cell in cells.items():
        n, k = nk
        counts = sorted(int(c) for c in cell["sweep"])
        r = {c: m(cell, c) for c in counts}
        comp = {c: (0.0 if ceil(cell, c, "complete")["below_at_first"]
                    else m(cell, c, "complete")) for c in counts}
        out["cells"][f"{n}/{k}"] = {
            "c_model": cell["c_model"],
            "rank1_per_nk2": {c: r[c] / (n / k) ** 2 for c in counts},
            "complete_per_nk2": {c: comp[c] / (n / k) ** 2 for c in counts},
            "rank1_argmax": max(counts, key=lambda c: r[c]),
            "complete_argmax": max(counts, key=lambda c: comp[c]),
            "span": max(r.values()) / min(r.values()),
            "unresolved": [c for c in counts if not resolved(cell, c)],
        }
    rows = out["cells"]

    def row(nk):
        return rows[f"{nk[0]}/{nk[1]}"]

    out["bars"]["WS-V"] = all(
        abs(m(cells[nk], cells[nk]["c_model"]) / random_anchors[nk] - 1) <= 0.10
        for nk in cells if nk in random_anchors)
    out["bars"]["WS-1"] = all(
        resolved(cells[nk], 32) and m(cells[nk], 32) >= 1.2 * m(cells[nk], cells[nk]["c_model"])
        for nk in judged)

    def interior_min(nk):
        cell = cells[nk]
        middle = min(m(cell, c) for c in (4, 5, 6, 8, 11, 16))
        return middle * 1.2 <= m(cell, 2) and middle * 1.2 <= m(cell, 32)
    out["bars"]["WS-2"] = all(interior_min(nk) for nk in judged)
    out["bars"]["WS-3"] = all(row(nk)["complete_argmax"] > row(nk)["rank1_argmax"] for nk in judged)

    def nk_alone(c):
        pairs = (((2000, 60), (4000, 120)), ((4000, 60), (8000, 120)))
        return all(0.8 <= m(cells[a], c) / m(cells[b], c) <= 1.25 for a, b in pairs)
    holds = {c: nk_alone(c) for c in COUNTS if all(str(c) in cells[nk]["sweep"] for nk in judged)}
    out["n_over_k_alone_holds_at"] = [c for c, ok in holds.items() if ok]
    c_models = {cells[nk]["c_model"] for nk in judged}
    out["bars"]["WS-4"] = (holds.get(32, False) and all(holds.get(c, False) for c in c_models
                                                       if c in holds)
                           and not holds.get(1, True) and not holds.get(2, True))
    out["bars"]["WS-5"] = (row((4000, 120))["span"] < row((2000, 60))["span"]
                           and row((8000, 120))["span"] < row((4000, 60))["span"]
                           and (((2000, 30) not in cells)
                                or row((2000, 30))["span"] > row((4000, 60))["span"]))
    return out


def evaluate_scan(observations, *, random_anchors=None):
    """Amendment 11's bars, on window edges from the full scan."""
    random_anchors = RANDOM_ANCHORS if random_anchors is None else random_anchors
    cells = {(c["n"], c["k"]): c for c in observations["cells"].values()}
    judged = [nk for nk in pe.IN_REGIME if nk in cells]
    out = {"cells": {}, "bars": {}}

    def win(nk, c, metric="rank1"):
        return cells[nk]["sweep"][str(c)]["windows"][metric]

    def upper(nk, c, metric="rank1"):
        """Upper edge; 0 when the metric never exceeds 0.5; a censored edge
        is its last point above (a lower bound)."""
        w = win(nk, c, metric)
        if w["never"]:
            return 0.0
        return w["last_above"] if w["upper_censored"] else w["upper"]

    def sq(nk):
        return (nk[0] / nk[1]) ** 2

    for nk, cell in cells.items():
        counts = sorted(int(c) for c in cell["sweep"])
        out["cells"][f"{nk[0]}/{nk[1]}"] = {
            "c_model": cell["c_model"],
            "rank1": {c: {"lower": win(nk, c)["lower"], "upper": upper(nk, c),
                          "censored": win(nk, c)["upper_censored"]} for c in counts},
            "complete": {c: {"lower": win(nk, c, "complete")["lower"],
                             "upper": upper(nk, c, "complete"),
                             "censored": win(nk, c, "complete")["upper_censored"]}
                         for c in counts},
        }
    own = {nk: cells[nk]["c_model"] for nk in cells}
    out["bars"]["XV"] = all(
        not win(nk, own[nk])["never"] and not win(nk, own[nk])["upper_censored"]
        and abs(upper(nk, own[nk]) / random_anchors[nk] - 1) <= 0.10
        for nk in cells if nk in random_anchors)
    out["bars"]["X1"] = all(upper(nk, 32, "complete") >= 1.5 * upper(nk, own[nk], "complete")
                            and upper(nk, 32, "complete") > 0 for nk in judged)
    out["bars"]["X2"] = all(upper(nk, 32) > 0 and upper(nk, 32, "complete") / upper(nk, 32) >= 0.85
                            for nk in judged)
    k60 = [upper(nk, 32, "complete") / sq(nk) for nk in judged if nk[1] == 60]
    k120 = [upper(nk, 32, "complete") / sq(nk) for nk in judged if nk[1] == 120]

    def tight(values):
        mean = sum(values) / len(values)
        return all(abs(v / mean - 1) <= 0.15 for v in values), mean
    ok60, m60 = tight(k60)
    ok120, m120 = tight(k120)
    out["binary_completion_per_nk2"] = {"k60": k60, "k120": k120}
    out["bars"]["X3"] = ok60 and ok120 and m120 >= 1.25 * m60
    pairs = (((2000, 60), (4000, 120)), ((4000, 60), (8000, 120)))
    own_ratio = [upper(a, own[a]) / upper(b, own[b]) for a, b in pairs]
    bin_ratio = [upper(a, 32) / upper(b, 32) for a, b in pairs]
    out["pair_ratios"] = {"own": own_ratio, "binary": bin_ratio}
    out["bars"]["X4"] = (all(0.8 <= r <= 1.25 for r in own_ratio)
                         and all(r <= 0.8 for r in bin_ratio))
    out["bars"]["X5"] = all(win(nk, 3)["lower"] is not None and not win(nk, 3)["never"]
                            and upper(nk, 3) > upper(nk, own[nk])
                            for nk in judged if nk[1] == 60)

    def c_min(nk):
        for c in sorted(int(x) for x in cells[nk]["sweep"]):
            if upper(nk, c, "complete") > 0.05 * sq(nk):
                return c
        return None
    levels = (((2000, 60), (4000, 120)), ((2000, 30), (4000, 60), (8000, 120)),
              ((4000, 30), (8000, 60)))
    mins = {f"{nk[0]}/{nk[1]}": c_min(nk) for nk in cells}
    out["completion_c_min"] = mins

    def falls(level):
        values = [c_min(nk) for nk in level if nk in cells]
        return (None not in values and len(values) == len(level)
                and all(a > b for a, b in zip(values, values[1:])))
    out["bars"]["X6"] = all(falls(level) for level in levels)
    return out


def plan(cells):
    return [{"n": n, "k": k} for n, k in cells]


def main(argv=None):
    ap = experiment_parser(__doc__ or "Write strength", engines=("hashed_assembly_memory",),
                           default_seeds=tuple(range(42, 62)))
    ap.add_argument("--registration", required=True)
    ap.add_argument("--nk", help="n:k pairs (default: the seven cells of the law)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scan", action="store_true",
                    help="Amendment 11: full scan from 16 with both window edges")
    args = ap.parse_args(argv)
    cells = ([tuple(int(v) for v in pair.split(":")) for pair in args.nk.split(",")]
             if args.nk else list(CELLS))
    if any(nk not in pe.ANCHORS for nk in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {sorted(pe.ANCHORS)}")
    profiles = {
        "refracted": describe_assembly_memory(w_max=pe.W_MAX, beta=pe.BETA, strength=pe.STRENGTH,
                                              gate=False, norm_init=True, synaptic_scaling=False),
    }
    path = run_experiment(
        script=__file__, protocol="memory.write-strength", protocol_version="2" if args.scan else "1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": plan(cells),
                    "counts": [1, 5, 32] if args.smoke else list(COUNTS),
                    "cap": 1024 if args.smoke else CAP, "complete": COMPLETE,
                    "p": pe.P, "beta": pe.BETA, "w_max": pe.W_MAX, "rounds": pe.T,
                    "strength": pe.STRENGTH, "half_bar": pe.HALF_BAR,
                    "recall_sample": pe.RECALL_SAMPLE, "measurement_seed": pe.MEASUREMENT_SEED,
                    "device": args.device, "search": "scan" if args.scan else "adaptive"},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
