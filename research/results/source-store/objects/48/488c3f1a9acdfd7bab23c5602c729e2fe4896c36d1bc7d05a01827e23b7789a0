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


#: items whose stimulus seeds go to the device in one copy
STIMULUS_BLOCK = 1024
#: elements of the [brains, items x k] membership gather a reading holds at once
GATHER_ELEMENTS = 1 << 25


def readings(mem, St, M, n, k, recall_rounds=None, settle_rounds=None):
    """Per-brain rank-1, own overlap and completed fraction from the module's
    own masked recall on `sample_for(M)` (`recall_rounds` frozen rounds;
    default the write's). `St` is [M, B, k]. The sampled cues are read in
    one batched recall (`AssemblyMemory.recall_many`, equal per cue to
    `recall`), and each recall's overlap with every stored item is one
    gather over brains, in chunks that bound its memory."""
    torch = pe._torch()
    B = St.shape[1]
    hits = torch.zeros(B, device=St.device)
    own = torch.zeros(B, device=St.device)
    done = torch.zeros(B, device=St.device)
    distinct = torch.zeros(B, device=St.device)
    sample = pe.sample_for(M)
    index = torch.as_tensor(np.asarray(sample), dtype=torch.int64, device=St.device)
    cues = St.index_select(0, index)[:, :, : k // 2].permute(1, 0, 2)  # [B, S, k/2]
    recalled = mem.recall_many(cues.to(torch.int64), rounds=recall_rounds)  # [B, S, k]
    per = max(1, GATHER_ELEMENTS // (M * k))                         # brains per gather
    for b0 in range(0, B, per):
        rows = slice(b0, b0 + per)
        items = St[:, rows].permute(1, 0, 2).reshape(-1, M * k).to(torch.int64)
        for s, i in enumerate(sample):
            mask = torch.zeros(items.shape[0], n, dtype=torch.bool, device=St.device)
            mask.scatter_(1, recalled[rows, s], True)
            ov = mask.gather(1, items).view(-1, M, k).sum(2)             # [brains, M]
            hit = ov.argmax(1) == int(i)
            hits[rows] += hit.float()
            o = ov[:, int(i)].float() / k
            own[rows] += o
            done[rows] += (o >= ws.COMPLETE).float()
            distinct[rows] += ((o >= ws.COMPLETE) & hit).float()
    S = len(sample)
    out = {"rank1": (hits / S).tolist(), "own": (own / S).tolist(),
           "complete": (done / S).tolist(),
           "complete_distinct": (distinct / S).tolist()}
    if settle_rounds:
        # the read-out's relaxation time, from a separate longer read so the
        # registered `recall_rounds` read above is untouched
        _, settled = mem.recall_many(cues.to(torch.int64), rounds=settle_rounds, settle=True)
        out["settle"] = settled.float().mean(dim=1).tolist()
        # the share of read-outs that reach no fixed point or 2-cycle at all
        out["unsettled"] = (settled > settle_rounds).float().mean(dim=1).tolist()
    return out


def run_beta(n, k, beta, seeds, cap, device, organ_semantics,
             stop_on=("rank1", "complete"), **options):
    """One learning rate: `run_betas` with a single rate."""
    return run_betas(n, k, [beta], seeds, cap, device, {beta: organ_semantics},
                     stop_on, **options)[0]


#: device bytes a launch holds beside its brains: the batched recall's two
#: passes of RECALL_BYTES and its bias copy, the readings' gather chunk, the
#: CUDA graph pools and allocator slack
LAUNCH_FIXED_BYTES = 3 << 29
#: device memory left free (the desktop holds ~1-2 GB of the card)
RESERVE_BYTES = 1 << 30


def brain_bytes(n, k, cap, count_bytes=1):
    """One brain's device bytes in a launch: counts (int8, or int16 for a
    weak write whose clip binds past count 127), connectome bits, per-neuron
    state (bias, ever-fired, in-degree, drive rows) and its stored items
    (int32, up to the cap)."""
    return n * n * count_bytes + n * ((n + 31) // 32) * 4 + 32 * n + cap * k * 4


def launch_rates(n, B, *, k=0, cap=0, count_bytes=1):
    """How many learning rates of B brains each one launch holds.

    Sized against the card's REAL free memory, not a fixed byte limit: a
    launch of 340 brains at n = 4000 fit the organ fiber's 6 GiB limit, but
    with its stored items, readings and the desktop's share the card ran
    out, and on Windows the driver then pages device memory to system RAM
    instead of failing -- the run slowed several-fold (Amendment 18's first
    attempt, 2.08 GB paged)."""
    import torch
    from neural_assemblies.core.torch_engine._hashed import DenseOrganFiber
    matrices = n * (n * count_bytes + ((n + 31) // 32) * 4)
    by_limit = DenseOrganFiber.MAX_BYTES // (matrices * B)
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    room = free - RESERVE_BYTES - LAUNCH_FIXED_BYTES
    by_memory = room // (brain_bytes(n, k, cap, count_bytes) * B) if room > 0 else 0
    return max(1, min(by_limit, by_memory))


def run_betas(n, k, betas, seeds, cap, device, organ_semantics,
              stop_on=("rank1", "complete"), *, p=None, strength=STRENGTH,
              grid_start=16, give_up=None, rounds=None, recall_rounds=None,
              stop_from=None, seen_from=None, settle_rounds=None, write_rule="round",
              burst_min=2):
    """Store with the capacity study's stimuli and read every checkpoint, for
    every rate in `betas`; returns [(c, cache)] in their order.

    Each rate's store stops once a `stop_on` metric has been above 0.5 and
    all have then been at or below it at two consecutive checkpoints, at
    `cap`, or -- when `give_up` is set -- at the first checkpoint at or past
    it if none has yet risen. `p`, `strength` and `grid_start` default to
    Amendments 12-13. `organ_semantics` maps each rate to its profile.

    THE RATES ARE BRAINS OF ONE LAUNCH (DESIGN_memory_throughput.md): rate g's
    brains are the seeds' brains with that rate's chain table and charge,
    and a rate whose store has stopped is dropped from the launch. A brain's
    arithmetic is the arithmetic of its solo run, so every rate's readings
    equal its own run's (tested against the one-rate launch)."""
    import torch
    from neural_assemblies.core.torch_engine._hashed import clip_count
    betas = [float(b) for b in betas]
    # rates whose clip binds past count 127 need int16 counts: they launch
    # apart from the int8 rates, which would otherwise pay double memory
    wide = [b for b in betas if (clip_count(b, pe.W_MAX) or 0) > 127]
    groups = [(g, 2 if g is wide else 1) for g in (wide, [b for b in betas if b not in wide]) if g]
    # an overcommit must raise, not page: cap the allocator at what is free
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, (torch.cuda.memory_reserved() + free - RESERVE_BYTES // 2) / total))
    results = {}
    try:
        for group, count_bytes in groups:
            per = launch_rates(n, len(seeds), k=k, cap=cap, count_bytes=count_bytes)
            # equal launches: 17 rates at 15 a launch run as 9 + 8, not 15 + 2
            per = -(-len(group) // -(-len(group) // per))
            for g0 in range(0, len(group), per):
                chunk = group[g0:g0 + per]
                done = _run_launch(n, k, chunk, seeds, cap, device,
                                   organ_semantics, stop_on, p=p, strength=strength,
                                   grid_start=grid_start, give_up=give_up, rounds=rounds,
                                   recall_rounds=recall_rounds, stop_from=stop_from,
                                   seen_from=seen_from, settle_rounds=settle_rounds,
                                   write_rule=write_rule, burst_min=burst_min)
                results.update(zip(chunk, done))
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    return [results[b] for b in betas]


def _run_launch(n, k, betas, seeds, cap, device, organ_semantics, stop_on, *,
                p, strength, grid_start, give_up, rounds, recall_rounds, stop_from,
                seen_from=None, settle_rounds=None, write_rule="round", burst_min=2):
    torch = pe._torch()
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    S, G = len(seeds), len(betas)
    brains = seeds_for(seeds)
    mem = AssemblyMemory(brains * G, n, k, pe.P if p is None else p,
                         beta=(betas[0] if G == 1 else [b for b in betas for _ in brains]),
                         w_max=pe.W_MAX, norm_init=True, synaptic_scaling=False,
                         rounds=pe.T if rounds is None else rounds,
                         strength=strength, gate=False, max_items=cap, device=device,
                         graphs=write_rule == "round", write_rule=write_rule,
                         burst_min=burst_min,
                         organ_semantics=(organ_semantics[betas[0]] if G == 1
                                          else {b: organ_semantics[b] for b in betas}))
    grid = set(pe.geometric_grid(grid_start, cap))
    active = list(range(G))                       # the rates still storing, in launch order
    caches = [{} for _ in range(G)]
    c = [None] * G
    seen, below = [False] * G, [0] * G
    # the stored items, [cap, brains, k] int32, allocated once at the cap (the
    # launch's budget counts it) and filled in place: a cat per checkpoint and
    # an index_select per drop each held a second copy beside the first
    buffer = torch.empty(cap, len(brains) * G, k, dtype=torch.int32, device=device)
    filled = 0
    pending = []
    for a in range(cap):
        if a % STIMULUS_BLOCK == 0:
            # the stimuli's seeds go to the device a block at a time: a copy
            # per item made every item wait for the GPU to drain
            stim = torch.tensor([[to_i32(_seeding.fnv1a_pair_seed(seed, f"s{b}", "A"))
                                  for seed in seeds]
                                 for b in range(a, min(a + STIMULUS_BLOCK, cap))],
                                dtype=torch.int32, device=device)
        pending.append(mem.store(stim[a % STIMULUS_BLOCK].repeat(len(active)),
                                 stim_size=k).to(torch.int32))
        M = a + 1
        if M == 1:
            first = pending[0]
            for j, g in enumerate(active):
                part = slice(j * S, (j + 1) * S)
                c[g] = pe.first_item_count(mem.fiber.C[part], mem.fiber.pres[part],
                                           first[part].long(), n)
        if M not in grid:
            continue
        mem.check()
        buffer[filled:M] = torch.stack(pending)
        filled = M
        pending = []
        r = readings(mem, buffer[:M], M, n, k, recall_rounds, settle_rounds)
        fill = mem.fill.cpu().numpy().tolist()
        finished = []
        for j, g in enumerate(active):
            part = slice(j * S, (j + 1) * S)
            caches[g][M] = {name: values[part] for name, values in r.items()}
            caches[g][M]["fill"] = fill[part]
            means = [ensemble_from_values(caches[g][M][m]).mean for m in stop_on]
            # a reading below `seen_from` items does not arm the stop: rank-1
            # is right with probability 1/M by chance, the 0.5 bar at M = 2
            # (Amendment 22's defect: a chance 0.53 at M = 2 stopped stores at
            # M = 32 before their recognition windows opened)
            if any(v > pe.HALF_BAR for v in means) and (seen_from is None or M >= seen_from):
                seen[g], below[g] = True, 0
            elif seen[g]:
                below[g] += 1
            # with a handful of items a recall sample is noise: from M = 2, two
            # readings of 0.48 stopped a store whose capacity was ~1500
            # (Amendment 15); `stop_from` holds the stop decision until then
            if seen[g] and below[g] >= 2 and (stop_from is None or M >= stop_from):
                finished.append(j)
            elif give_up is not None and not seen[g] and M >= give_up:
                finished.append(j)
        if len(finished) == len(active):
            break
        if finished:
            keep_rates = [j for j in range(len(active)) if j not in finished]
            keep = [j * S + b for j in keep_rates for b in range(S)]
            mem.select(keep)
            # compact the kept brains forward in place (keep ascends)
            for j, src in enumerate(keep):
                if src != j:
                    buffer[:filled, j] = buffer[:filled, src]
            buffer = buffer[:, :len(keep)]
            active = [active[j] for j in keep_rates]
    del mem, buffer, pending
    torch.cuda.empty_cache()
    return list(zip(c, caches))


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
        betas = parameters["betas"]
        results = run_betas(n, k, betas, seeds, cap, device,
                            {b: profiles[profile_name(b)] for b in betas}, stop_on=stop_on)
        for beta, (c, cache) in zip(betas, results):
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
