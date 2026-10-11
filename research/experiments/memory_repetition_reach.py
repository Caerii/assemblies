"""Does the write-sleep lifecycle reach the REPETITION edge? A registered test.

Registered in PREREG_refraction_memory.md, Amendment 53.

Amendment 45 found two reuse edges: recurrence (~40 uses per word with random successors) and
repetition (~2.8 repeats per transition, with each word followed by one of b successors).
Amendment 52 registered that the local comparator at write (Amendment 50) and contrast-gated sleep
(Amendment 51) compose against the first. Here they meet the second: U = 10 uses per word with
b = 3 successors (3.3 repeats per transition, past the edge), b = 5 (2 repeats, near it) and random
successors (a healthy control).

    per b: a STANDARD store (replayed, then slept 300 episodes and replayed) and a COMPARATOR store
    (replayed, then slept 300 episodes and replayed), same memories and brains; the sleep threshold
    from a U = 10 store on separate reference brains (memory_sleep's calibration). On the standard
    store, the overlap of the tokens written at the first two occurrences of each repeated word
    pair (the pair's second word).

    python -m research.runner repetition_reach --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_comparator as mc                # noqa: E402
from research.experiments import memory_lifecycle as lc                 # noqa: E402
from research.experiments import memory_reuse_grammar as rg             # noqa: E402
from research.experiments import memory_sleep as sl                     # noqa: E402
from research.experiments import memory_write_separation as ws          # noqa: E402
from research.experiments import memory_lib as lib                      # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((8500, 64, 0.52, 66), (11000, 82, 0.44, 67))
SMOKE_CELL = (2000, 60, 0.5, 17)
USES = 10
SUCCESSORS = (3, 5, None)
EPISODES, SMOKE_EPISODES = 300, 30
SEEDS = tuple(range(1000, 1020))
REFERENCE_SEEDS = tuple(range(1020, 1040))
#: bars
EDGE, RESCUE, OVER_SLEEP, FLAG_FLOOR, FLAG_RATIO, HEALTHY, NEAR, SPARING, MAX_COLLAPSED, COLLAPSED, MERGE = (
    0.25, 0.75, 0.15, 0.02, 10.0, 0.98, 0.95, 0.02, 2, 0.2, 0.2)


def key(b):
    return "random" if b is None else f"b{b}"


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tau": tau, "rho": lib.RHO, "beta": round(lib.theta(n, k, p), 5),
             "uses": USES, "successors": [key(b) for b in SUCCESSORS],
             "episodes": SMOKE_EPISODES if smoke else EPISODES}
            for n, k, p, tau in cells]


def store(spec, b, seeds, compare, device):
    """M walks of LENGTH words on per-brain grammars with b successors per word (None: random),
    written by the store loop -- store_sequence's when compare is False, the comparator's when
    True."""
    import numpy as np
    import torch
    from research.experiments.memory_lib.seeding import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    n, k, p, tau, U = spec["n"], spec["k"], spec["p"], spec["tau"], spec["uses"]
    LEN = lib.LENGTH
    M = max(1, round(spec["rho"] * lib.unit(n, k, p) / LEN))
    L = M * LEN
    V = max(8, round(L / U))
    words = [rg.walks(sd, M, V, b, L * 1000 + (b or 0) * 7 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device).long()
    mem = ws.build(n, k, p, tau, seeds, device)
    stored, seqs = [], []
    stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
    for q in range(M):
        es = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
              for e in range(LEN)]
        seqs.append(mc.store(mem, es, stored, compare, device, stats))
    mem.area.check_overflow()
    return {"mem": mem, "seqs": seqs, "allst": torch.stack(stored).long(), "wordof": wordof, "M": M, "L": L,
            "seeds": list(seeds)}, stats


def pair_overlap(st, n, k, device):
    """per brain: the mean overlap (/k) of the tokens written at the first two occurrences of each
    repeated word pair (the pair's second word), sequence boundaries excluded; NaN if none."""
    import numpy as np
    import torch
    allst, wordof, L = st["allst"], st["wordof"], st["L"]
    out = []
    for i in range(len(st["seeds"])):
        w = wordof[:, i].cpu().numpy()
        pairs: dict = {}
        for t in range(1, L):
            if t % lib.LENGTH:
                pairs.setdefault((int(w[t - 1]), int(w[t])), []).append(t)
        firsts = [ts[:2] for ts in pairs.values() if len(ts) >= 2]
        if not firsts:
            out.append(float("nan"))
            continue
        a = torch.tensor([f[0] for f in firsts], device=device)
        c = torch.tensor([f[1] for f in firsts], device=device)
        H = torch.zeros(L, n, device=device, dtype=torch.float16)
        H.scatter_(1, allst[:, i], 1.0)
        out.append(float(((H[a] * H[c]).sum(1) / k).mean()))
        del H
    return [None if np.isnan(v) else v for v in out]


def measure(spec, seeds, device):
    import torch
    ref = sl.build_store(spec, 10, REFERENCE_SEEDS, device)
    cal: list = []
    g0 = torch.Generator(device=device).manual_seed(sl.CAL_SEED)
    for _ in range(sl.CALIBRATION):
        sl.dream(ref["mem"], g0, None, device, cal)
    cal_t = torch.cat(cal).float()
    threshold = float(cal_t.max()) * sl.MARGIN
    out: dict = {"calibration": {"mean": float(cal_t.mean()), "max": float(cal_t.max()),
                                 "q99": float(torch.quantile(cal_t, 0.99)), "threshold": threshold}}
    del ref
    torch.cuda.empty_cache()
    for b in SUCCESSORS:
        arm: dict = {}
        st, _ = store(spec, b, seeds, False, device)
        arm["standard"] = sl.reliability(st, device)
        arm["pair_overlap"] = pair_overlap(st, spec["n"], spec["k"], device)
        arm["sleep_removed"] = lc.slept(st, spec["episodes"], threshold, device)
        arm["sleep"] = sl.reliability(st, device)
        del st
        torch.cuda.empty_cache()
        st, stats = store(spec, b, seeds, True, device)
        arm["comparator"] = sl.reliability(st, device)
        arm["flags"] = stats["flag"] / max(1, stats["judged"])
        arm["both_removed"] = lc.slept(st, spec["episodes"], threshold, device)
        arm["both"] = sl.reliability(st, device)
        del st
        torch.cuda.empty_cache()
        out[key(b)] = arm
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    if not ws.equivalent(device):
        raise RuntimeError("the written store loop does not reproduce store_sequence")
    print("equivalence: the written store loop reproduces store_sequence", flush=True)
    out = {}
    mean = lambda v: sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v))  # noqa: E731
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "tau": spec["tau"], **m}
        c = m["calibration"]
        print(f"({n}, {k}, {p}) sleep threshold {c['threshold']:.3f} (mean {c['mean']:.3f}, "
              f"max {c['max']:.3f}, q99 {c['q99']:.3f})", flush=True)
        for b in SUCCESSORS:
            a = m[key(b)]
            print(f"  {key(b)}: standard {mean(a['standard']):.3f} (pair overlap {mean(a['pair_overlap']):.3f})  "
                  f"sleep {mean(a['sleep']):.3f} (removed {a['sleep_removed']:.4f})  "
                  f"comparator {mean(a['comparator']):.3f} (flags {a['flags']:.4f})  "
                  f"both {mean(a['both']):.3f} (removed {a['both_removed']:.4f})", flush=True)
    return {"cells": out, "equivalent": True, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 53's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
    arms = [key(b) for b in SUCCESSORS]
    out: dict = {"bars": {}, "cells": {}}
    if not observations.get("equivalent") or not all(
            c[:3] in cells and all(a in cells[c[:3]] for a in arms) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}

    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")

    for n, k, p, _ in CELLS:
        c = cells[(n, k, p)]
        info = {a: {"standard": mean(c[a]["standard"]), "sleep": mean(c[a]["sleep"]),
                    "comparator": mean(c[a]["comparator"]), "both": mean(c[a]["both"]),
                    "collapsed_both": sum(r < COLLAPSED for r in c[a]["both"]),
                    "flags": c[a]["flags"], "both_removed": c[a]["both_removed"],
                    "pair_overlap": mean(c[a]["pair_overlap"])} for a in arms}
        out["cells"][f"{n}/{k}/{p:g}"] = info
        r, near, ctl = info["b3"], info["b5"], info["random"]
        ok["P1"] &= r["standard"] <= EDGE
        ok["P2"] &= r["both"] >= RESCUE
        ok["P3"] &= r["both"] >= r["comparator"] and r["both"] - r["sleep"] >= OVER_SLEEP
        ok["P4"] &= r["flags"] >= max(FLAG_FLOOR, FLAG_RATIO * ctl["flags"])
        ok["P5"] &= (ctl["both"] >= HEALTHY and near["both"] >= NEAR
                     and all(info[a]["both_removed"] <= SPARING for a in arms))
        ok["P6"] &= all(info[a]["collapsed_both"] <= MAX_COLLAPSED for a in ("b3", "b5"))
        ok["P7"] &= r["pair_overlap"] <= MERGE
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Repetition reach", engines=("hashed_assembly_memory",), default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 53 is registered on seeds 1000..1019 (reference brains 1020..1039)")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lib.profile_name(s["beta"]): lib.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.repetition_reach", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": lib.LENGTH, "comparator_threshold": mc.THRESHOLD,
                    "sleep_steps": sl.STEPS, "sleep_margin": sl.MARGIN, "reference_seeds": list(REFERENCE_SEEDS),
                    "strength": lib.STRENGTH, "match": lib.MATCH, "w_max": lib.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
