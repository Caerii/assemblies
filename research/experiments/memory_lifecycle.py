"""Do write-time separation and sleep COMPOSE? A registered test of the write-sleep lifecycle.

Registered in PREREG_refraction_memory.md, Amendment 52.

Amendment 50 registered a local comparator that keeps writes off existing assemblies; Amendment 51
a contrast-gated sleep that dissolves captured clusters offline. Each has a reach: in exploratory
probes sleep alone failed past ~60 uses per word and the comparator alone past ~80. Here both act
on the same memories -- the comparator at every write, then 300 episodes of sleep -- against
each alone, at 80 and 100 uses per word.

    per U: a STANDARD store (replayed, then slept 300 episodes and replayed) and a COMPARATOR store
    (replayed, then slept 300 episodes and replayed), same memories and brains; the sleep threshold
    from a U = 10 store on separate reference brains (memory_sleep's calibration).

    python -m research.runner lifecycle --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_comparator as mc                # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_reuse_grammar as rg             # noqa: E402
from research.experiments import memory_sleep as sl                     # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.experiments import memory_write_separation as ws          # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((9500, 75, 0.46, 63), (12500, 90, 0.4, 69))
SMOKE_CELL = (2000, 60, 0.5, 17)
USES = (80, 100)
SMOKE_USES = (40,)
EPISODES, SMOKE_EPISODES = 300, 30
SEEDS = tuple(range(842, 862))
REFERENCE_SEEDS = tuple(range(862, 882))
#: bars
SYNERGY_OVER_COMPARATOR, SYNERGY_OVER_SLEEP, REACH80, REACH100, MAX_COLLAPSED, COLLAPSED, SPARING = (
    0.15, 0.3, 0.65, 0.45, 2, 0.2, 0.02)


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5),
             "uses": list(SMOKE_USES if smoke else USES), "episodes": SMOKE_EPISODES if smoke else EPISODES}
            for n, k, p, tau in cells]


def comparator_store(spec, U, seeds, device):
    """memory_sleep.build_store's store, written through the comparator."""
    import numpy as np
    import torch
    from research.experiments.seq_capacity_scaling import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    n, k, p, tau = spec["n"], spec["k"], spec["p"], spec["tau"]
    LEN = rg.LENGTH
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / LEN))
    L = M * LEN
    V = max(8, round(L / U))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device).long()
    mem = ws.build(n, k, p, tau, seeds, device)
    stored, seqs = [], []
    stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
    for q in range(M):
        es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
              for e in range(LEN)]
        seqs.append(mc.store(mem, es, stored, True, device, stats))
    mem.area.check_overflow()
    return {"mem": mem, "seqs": seqs, "allst": torch.stack(stored).long(), "wordof": wordof, "M": M, "L": L,
            "seeds": list(seeds)}, stats


def slept(store, episodes, threshold, device):
    import torch
    held = int(store["mem"].fiber.C.sum())
    gen = torch.Generator(device=device).manual_seed(sl.NOISE_SEED)
    removed = 0
    for _ in range(episodes):
        removed += sl.dream(store["mem"], gen, threshold, device)[0]
    return removed / held


def measure(spec, seeds, device):
    import torch
    ref = sl.build_store(spec, 10, REFERENCE_SEEDS, device)
    cal: list = []
    g0 = torch.Generator(device=device).manual_seed(sl.CAL_SEED)
    for _ in range(sl.CALIBRATION):
        sl.dream(ref["mem"], g0, None, device, cal)
    cal_t = torch.cat(cal)
    threshold = float(cal_t.max()) * sl.MARGIN
    out: dict = {"calibration": {"mean": float(cal_t.mean()), "max": float(cal_t.max()), "threshold": threshold}}
    del ref
    torch.cuda.empty_cache()
    for U in spec["uses"]:
        arm: dict = {}
        st = sl.build_store(spec, U, seeds, device)
        arm["standard"] = sl.reliability(st, device)
        arm["sleep_removed"] = slept(st, spec["episodes"], threshold, device)
        arm["sleep"] = sl.reliability(st, device)
        del st
        torch.cuda.empty_cache()
        st, stats = comparator_store(spec, U, seeds, device)
        arm["comparator"] = sl.reliability(st, device)
        arm["flags"] = stats["flag"] / max(1, stats["judged"])
        arm["both_removed"] = slept(st, spec["episodes"], threshold, device)
        arm["both"] = sl.reliability(st, device)
        del st
        torch.cuda.empty_cache()
        out[str(U)] = arm
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    if not ws.equivalent(device):
        raise RuntimeError("the written store loop does not reproduce store_sequence")
    print("equivalence: the written store loop reproduces store_sequence", flush=True)
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": spec["tau"], **m}
        print(f"({n}, {k}, {p}) sleep threshold {m['calibration']['threshold']:.3f}", flush=True)
        for U in spec["uses"]:
            a = m[str(U)]
            mean = lambda v: sum(v) / len(v)                            # noqa: E731
            print(f"  U={U}: standard {mean(a['standard']):.3f}  sleep {mean(a['sleep']):.3f} "
                  f"(removed {a['sleep_removed']:.4f})  comparator {mean(a['comparator']):.3f} "
                  f"(flags {a['flags']:.4f})  both {mean(a['both']):.3f} (removed {a['both_removed']:.4f})", flush=True)
    return {"cells": out, "equivalent": True, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 52's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("L1", "L2", "L3", "L4", "L5")
    out: dict = {"bars": {}, "cells": {}}
    if not observations.get("equivalent") or not all(
            c[:3] in cells and all(str(u) in cells[c[:3]] for u in USES) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    mean = lambda v: sum(v) / len(v)                                    # noqa: E731
    for n, k, p, _ in CELLS:
        c = cells[(n, k, p)]
        info = {u: {"standard": mean(c[u]["standard"]), "sleep": mean(c[u]["sleep"]),
                    "comparator": mean(c[u]["comparator"]), "both": mean(c[u]["both"]),
                    "collapsed_both": sum(r < COLLAPSED for r in c[u]["both"]),
                    "both_removed": c[u]["both_removed"]} for u in ("80", "100")}
        out["cells"][f"{n}/{k}/{p:g}"] = info
        h = info["100"]
        ok["L1"] &= (h["both"] - h["comparator"] >= SYNERGY_OVER_COMPARATOR
                     and h["both"] - h["sleep"] >= SYNERGY_OVER_SLEEP)
        ok["L2"] &= info["80"]["both"] >= REACH80
        ok["L3"] &= h["both"] >= REACH100
        ok["L4"] &= all(info[u]["collapsed_both"] <= MAX_COLLAPSED for u in ("80", "100"))
        ok["L5"] &= all(info[u]["both_removed"] <= SPARING for u in ("80", "100"))
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Lifecycle", engines=("hashed_assembly_memory",), default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 52 is registered on seeds 842..861 (reference brains 862..881)")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.lifecycle", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": rg.LENGTH, "comparator_threshold": mc.THRESHOLD,
                    "sleep_steps": sl.STEPS, "sleep_margin": sl.MARGIN, "reference_seeds": list(REFERENCE_SEEDS),
                    "strength": ml.STRENGTH, "match": ml.MATCH, "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
