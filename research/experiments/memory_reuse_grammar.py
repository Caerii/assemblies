"""Does reuse cost through repeated TRANSITIONS rather than recurring elements? A
registered test that separates the two.

Registered in PREREG_refraction_memory.md, Amendment 44.

Amendment 42 found that elements recurring a few times each cost nothing; an
exploratory probe after it found that when each word has only b allowed
successors -- a grammar -- replay fails at the WORD level (0.00 to 0.39 whole)
where random successors hold (0.98), at the same twenty uses per word. What
varies with b at fixed uses is how often each transition (bigram) is stored,
R ~ U / b. Here the two are crossed:

    each brain draws its own grammar: every word of a vocabulary of V = L / U has
    b allowed successors (b = V: i.i.d. words), and its M = L / 16 sequences are
    random walks on it. Arms: U = 20 at R ~ 1 (b = V), 5 (b = 4), 10 (b = 2);
    R = 5 at U = 10 (b = 2), 20 (b = 4), 40 (b = 8).

    WORD-LEVEL RELIABILITY  every sequence replayed noiselessly from a random half
                 of its first state; at every step the read-out is matched to the
                 nearest stored token (any sequence) and scored by that token's
                 word; the mean over brains of the fraction of sequences whose
                 every step names the right word. Also reported: the token-level
                 score (overlap >= 0.3 with this sequence's own state) and the
                 same-word code overlap.

    python -m research.runner reuse_grammar --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_sequences as sq                 # noqa: E402
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

#: the judged cells and their recovery times (Amendment 41's rule)
CELLS = ((9000, 110, 0.33, 41), (12000, 80, 0.45, 75))
SMOKE_CELL = (2000, 60, 0.5, 17)
LENGTH, RHO = 16, 0.05
#: (uses per word U, successors per word b; None = i.i.d.)
ARMS = ((20, None), (20, 4), (20, 2), (10, 2), (40, 8))
SEEDS = tuple(range(662, 682))
GAP, FREE, SPAN, MERGE = 0.3, 0.95, 0.25, 0.03
PAIRS = 200


def plan(cells, *, smoke=False):
    return [{"n": n, "k": k, "p": p, "tau": tau, "uses": u, "b": b, "rho": RHO,
             "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n)}
            for n, k, p, tau in cells for u, b in (((20, None), (20, 4)) if smoke else ARMS)]


def walks(seed, M, V, b, salt):
    """M random walks of LENGTH words on a per-brain grammar (b successors per word)."""
    import numpy as np
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(salt)])
    succ = None if b is None or b >= V else np.stack([rng.choice(V, size=b, replace=False) for _ in range(V)])
    out = np.empty((M, LENGTH), dtype=np.int64)
    for q in range(M):
        w = int(rng.integers(V))
        for e in range(LENGTH):
            out[q, e] = w
            w = int(rng.integers(V)) if succ is None else int(succ[w, rng.integers(b)])
    return out


def measure(spec, seeds, device):
    import numpy as np
    import torch
    from research.experiments.seq_capacity_scaling import seeds_for, to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = spec["n"], spec["k"], spec["p"]
    M = max(1, round(spec["rho"] * ml.unit(n, k, p) / LENGTH))
    L = M * LENGTH
    V = max(8, round(L / spec["uses"]))
    b = spec["b"]
    salt = L * 1000 + (b or 0) * 7 + spec["uses"]
    out = {"L": L, "M": M, "V": V, "word": [], "token": [], "same": [], "repeats": []}
    for i0 in range(0, len(seeds), spec["batch"]):
        part = list(seeds[i0:i0 + spec["batch"]])
        B = len(part)
        mem = AssemblyMemory(seeds_for(part), n, k, p, beta=spec["beta"], w_max=pe.W_MAX,
                             norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4,
                             device=device, bias_decay=math.exp(-1.0 / spec["tau"]))
        words = [walks(sd, M, V, b, salt) for sd in part]
        seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                     for i, sd in enumerate(part)] for e in range(LENGTH)])[0]
                for q in range(M)]
        allst = torch.cat(seqs, 0).long()                              # [L, B, k]
        wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=device)   # [L, B]
        ar = torch.arange(B, device=device)
        wdone = torch.zeros(B, device=device)
        tdone = torch.zeros(B, device=device)
        for q, st in enumerate(seqs):
            x = torch.stack([md.cue(st[0, i], sd, L * 100_000 + q, k, device) for i, sd in enumerate(part)])
            walive = torch.ones(B, dtype=torch.bool, device=device)
            talive = walive.clone()
            for j in range(1, LENGTH):
                x = mem.recall(x, rounds=1)
                talive &= sq._overlap(x, st[j]) >= ml.MATCH
                hot = torch.zeros(B, n, device=device)
                hot.scatter_(1, x.long(), 1.0)
                ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)        # [L, B]
                walive &= wordof[ov.argmax(0), ar] == wordof[q * LENGTH + j]
            wdone += walive.float()
            tdone += talive.float()
        out["word"] += (wdone / M).tolist()
        out["token"] += (tdone / M).tolist()
        for i in range(B):
            w = words[i]
            pairs = {}
            for q in range(M):
                for e in range(LENGTH - 1):
                    pairs[(int(w[q, e]), int(w[q, e + 1]))] = pairs.get((int(w[q, e]), int(w[q, e + 1])), 0) + 1
            out["repeats"].append(sum(pairs.values()) / len(pairs))
            where = {}
            for q in range(M):
                for e in range(LENGTH):
                    where.setdefault(int(w[q, e]), []).append((q, e))
            same = []
            for occ in where.values():
                cross = next(((a, c) for a in occ for c in occ if a[0] < c[0]), None)
                if cross and len(same) < PAIRS:
                    (q1, e1), (q2, e2) = cross
                    same.append(float(sq._overlap(seqs[q1][e1, i:i + 1], seqs[q2][e2, i:i + 1])))
            out["same"].append(sum(same) / len(same) if same else None)
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
        cell["arms"][f"U{spec['uses']}/b{spec['b'] if spec['b'] else 'V'}"] = m
        mean = lambda v: sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v))   # noqa: E731
        print(f"({n}, {k}, {p}) U={spec['uses']} b={spec['b'] or 'V'} L={m['L']} V={m['V']} "
              f"repeats/bigram {mean(m['repeats']):.1f}: word {mean(m['word']):.3f}, token {mean(m['token']):.3f}, "
              f"same-word overlap {mean(m['same']):.3f}", flush=True)
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def evaluate(observations):
    """Amendment 44's bars."""
    cells = {(c["n"], c["k"], c["p"]): c for c in observations["cells"].values()}
    names = ("G1", "G2", "G3", "G4")
    out: dict = {"bars": {}, "word": {}, "same": {}}
    keys = [f"U{u}/b{b if b else 'V'}" for u, b in ARMS]
    if not all(c[:3] in cells and set(keys) <= set(cells[c[:3]]["arms"]) for c in CELLS):
        out["bars"] = {b: False for b in names}
        return out
    for n, k, p, _ in CELLS:
        arms = cells[(n, k, p)]["arms"]
        out["word"][f"{n}/{k}/{p:g}"] = {a: _mean(arms[a]["word"]) for a in keys}
        out["same"][f"{n}/{k}/{p:g}"] = {a: _mean(arms[a]["same"]) for a in keys}
    w, s = out["word"].values(), out["same"].values()
    out["bars"]["G1"] = all(x["U20/bV"] - x["U20/b2"] >= GAP for x in w)
    out["bars"]["G2"] = all(x["U20/bV"] >= FREE for x in w)
    out["bars"]["G3"] = all(max(x[a] for a in ("U10/b2", "U20/b4", "U40/b8"))
                            - min(x[a] for a in ("U10/b2", "U20/b4", "U40/b8")) <= SPAN for x in w)
    out["bars"]["G4"] = all(x["U20/b2"] - x["U20/bV"] >= MERGE for x in s)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Reuse grammar", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 44 is registered on seeds 662..681")
    specs = plan((SMOKE_CELL,) if args.smoke else CELLS, smoke=args.smoke)
    profiles = {lr.profile_name(s["beta"]): lr.profile(s["beta"]) for s in specs}
    path = run_experiment(
        script=__file__, protocol="memory.reuse-grammar", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "length": LENGTH, "strength": ml.STRENGTH, "match": ml.MATCH,
                    "w_max": pe.W_MAX, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
