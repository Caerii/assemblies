"""Does adaptation at READ time control the recurrence collapse -- and only it? A
registered test of a control.

Registered in PREREG_refraction_memory.md, Amendment 46.

Replay in every amendment so far runs frozen and MASKED: no adaptation acts on
the read-out. Amendment 45 found a recurrence edge (whole brains collapsing at
~40 uses per element) and a repetition edge (~2.8 repeats per transition).
Exploratory probes after it found the collapse is a spurious attractor (failed
replays funnel into a few shared states), causal by a cross-validated lesion,
and that a mild SESSION ADAPTATION during replay -- every replay winner charged
c x its raw drive, the charge decaying by exp(-1/T) per step and kept across a
brain's sequences -- lifts collapsed brains, inside a window of c T ~ 5-10, while
leaving the repetition failure where it was.

    arms (U, b): recurrence U = 40, random successors; repetition U = 10, b = 3;
    healthy U = 10, random successors. Each replayed three ways on the same
    memory: masked (the standard read), habit (c = 0.1, T = 50), strong
    (c = 0.3, T = 50). Word-level reliability as in Amendment 44.

    python -m research.runner read_adaptation --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_load_drift as md                # noqa: E402
from research.experiments import memory_load_law as ml                  # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_reuse_grammar as rg             # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((11000, 80, 0.45, 69), (8000, 60, 0.6, 67))
SMOKE_CELL = (2000, 60, 0.5, 17)
ARMS = (("recurrence", 40, None), ("repetition", 10, 3), ("healthy", 10, None))
#: (name, charge c, decay time T in replay steps); c = 0 is the masked read
MODES = (("masked", 0.0, 1), ("habit", 0.1, 50), ("strong", 0.3, 50))
SEEDS = tuple(range(702, 722))
RESCUE, CEILING, HEALTHY, INERT, HARM = 0.15, 0.85, 0.97, 0.1, 0.1


def plan(cells, *, smoke=False):
    arms = ARMS[:1] if smoke else ARMS
    return [{"n": n, "k": k, "p": p, "tau": tau, "arm": a, "uses": u, "b": b, "rho": rg.RHO,
             "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n)}
            for n, k, p, tau in cells for a, u, b in arms]


def measure(spec, seeds, device):
    """Per mode, each brain's word-level reliability on one memory."""
    import numpy as np
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / rg.LENGTH))
    L = M * rg.LENGTH
    V = max(8, round(L / spec["uses"]))
    salt = L * 1000 + (spec["b"] or 0) * 7 + spec["uses"]
    out = {"L": L, "M": M, "V": V, "modes": {m: [] for m, _, _ in MODES}}
    for i0 in range(0, len(seeds), spec["batch"]):
        part = list(seeds[i0:i0 + spec["batch"]])
        B = len(part)
        mem = AssemblyMemory(seeds_for(part), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                             norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4,
                             device=device, bias_decay=math.exp(-1.0 / spec["tau"]))
        words = [rg.walks(sd, M, V, spec["b"], salt) for sd in part]
        seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                     for i, sd in enumerate(part)] for e in range(rg.LENGTH)])[0]
                for q in range(M)]
        allst = torch.cat(seqs, 0).long()
        wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device)
        ar = torch.arange(B, device=device)
        area = mem.area
        saved = area.bias.clone()
        for mode, c, T in MODES:
            bias = torch.zeros(B, n, device=device)
            done = torch.zeros(B, device=device)
            for q, st in enumerate(seqs):
                area.bias = bias
                area.winners = torch.stack([md.cue(st[0, i], sd, L * 100_000 + q, k, device)
                                            for i, sd in enumerate(part)]).long()
                alive = torch.ones(B, dtype=torch.bool, device=device)
                for j in range(1, rg.LENGTH):
                    x, drive = area.project(1, [mem.fiber], freeze=True, mask_bias=False, return_drive=True)
                    if c:
                        raw = drive + bias
                        bias.mul_(math.exp(-1.0 / T))
                        bias.scatter_add_(1, x, torch.gather(raw, 1, x) * c)
                    hot = torch.zeros(B, n, device=device)
                    hot.scatter_(1, x.long(), 1.0)
                    ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
                    alive &= wordof[ov.argmax(0), ar] == wordof[q * rg.LENGTH + j]
                done += alive.float()
            out["modes"][mode] += (done / M).tolist()
        area.bias = saved
        del mem, seqs, allst
        torch.cuda.empty_cache()
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        m = measure(spec, seeds, device)
        cell = out.setdefault(f"{n}/{k}/{p:g}", {"n": n, "k": k, "p": p, "tau": spec["tau"], "arms": {}})
        cell["arms"][spec["arm"]] = m
        print(f"({n}, {k}, {p}) {spec['arm']} U={spec['uses']} b={spec['b'] or 'V'}: "
              + "  ".join(f"{mode} {sum(v) / len(v):.3f}" for mode, v in m["modes"].items()), flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations):
    """Amendment 46's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("H1", "H2", "H3", "H4")
    out: dict = {"bars": {}, "means": {}}
    if not all(c[:3] in cells and {a for a, _, _ in ARMS} <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    ok = {b: True for b in names}
    for n, k, p, _ in CELLS:
        arms = cells[(n, k, p)]["arms"]
        mean = {a: {m: sum(v) / len(v) for m, v in arms[a]["modes"].items()} for a in arms}
        out["means"][f"{n}/{k}/{p:g}"] = mean
        rec, rep, hea = mean["recurrence"], mean["repetition"], mean["healthy"]
        ok["H1"] &= rec["masked"] <= CEILING and rec["habit"] - rec["masked"] >= RESCUE
        ok["H2"] &= hea["habit"] >= HEALTHY
        ok["H3"] &= rep["habit"] - rep["masked"] <= INERT
        ok["H4"] &= rec["strong"] <= rec["masked"] - HARM
    out["bars"] = ok
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Read adaptation", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 46 is registered on seeds 702..721")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.read-adaptation", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": rg.LENGTH, "modes": [list(m) for m in MODES],
                    "strength": ml.STRENGTH, "match": ml.MATCH, "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
