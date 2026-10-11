"""How much of the substrate's associative capacity does each code realise?

Registered in PREREG_refraction_memory.md, Amendment 9.

The refracted memory's ceiling is ~0.35-0.50 (n/k)^2 assemblies. This study
asks what the SAME circuit stores when the patterns are not the network's own:
the same presence matrix, the same in-degree normalisation, the same
potentiation table and clip, read by the same half-cue, frozen, masked k-WTA
recall and judged by the same rank-1 criterion. Only the stored patterns and
how they were written differ:

    real         the refracted AssemblyMemory itself (the capacity study's
                 store, seeds and stimuli), read from its own count matrix
    gated        the same with the convergence gate (two cells)
    clean        the real assemblies, synapses rebuilt from them alone:
                 c counts on every internal pair, no transient winners
    random       independent uniformly random k-subsets, written cleanly
    random_cm2   the same with c - 2 counts per pair (weight sensitivity)
    random_cp2   the same with c + 2
    balanced     usage-balanced k-subsets (each item takes the k least-used
                 neurons, random tie-break), written cleanly

c is the rounded mean count on the present internal pairs of the condition's
first stored item, across brains -- what one item's write puts on its own
assembly. Every variant is measured per brain at checkpoints on its own grid;
the ceiling M* is where the ensemble-mean rank-1 crosses 0.5
(`ceiling_from_curve`). Two pattern statistics are read from the unclipped
count matrix N = X^T X of each variant's patterns: the POTENTIATED fraction of
present synapses, and PAIR SHARING, sum_{i != j} N_ij (N_ij - 1) / (M k (k-1)),
the mean number of other items that share an internal synapse pair with an
item. For independent random k-subsets its expectation is
(M - 1) k (k - 1) / (n (n - 1)), recorded beside it.

    python -m research.runner pattern-efficiency --registration PATH --tag NAME [--smoke]

Completion leaves every bar UNJUDGED; `evaluate(observations)` applies
Amendment 9's bars to a finished record.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies import describe_assembly_memory                  # noqa: E402
from neural_assemblies.core.numpy_engine import _seeding                # noqa: E402
from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments._substrate import ceiling_from_curve          # noqa: E402
from research.experiments.memory_lib.seeding import seeds_for, to_i32  # noqa: E402
#: registered names, owned by research.experiments.memory_lib since 2026-10-10 (re-exported)
from research.experiments.memory_lib.model import STRENGTH, W_MAX       # noqa: E402
from research.experiments.memory_lib.readout import (                   # noqa: E402
    HALF_BAR, MEASUREMENT_SEED, RECALL_SAMPLE, ROUNDS as T, sample_for)
from research.runner import experiment_parser, run_experiment           # noqa: E402

P = 0.5
BETA = 0.1
CHUNK = 2048

#: Amendment 8's replayed refracted ceilings (A7 for (4000, 60)): they place
#: each variant's checkpoint window, and PE-V checks the real condition
#: against them.
ANCHORS = {(2000, 60): 431.31709832368625, (4000, 120): 383.15474360005896,
           (4000, 60): 1961.400398770613, (8000, 120): 2229.7301845665493,
           (8000, 60): 6995.135435554336, (2000, 30): 1588.5264285670949,
           (4000, 30): 4748.47057822085}
#: Amendments 5 and 6's gated ceilings (legacy records).
GATED_ANCHORS = {(4000, 60): 2645.0, (8000, 60): 8666.0}
IN_REGIME = ((2000, 60), (4000, 120), (4000, 60), (8000, 120), (8000, 60))
#: checkpoint windows in multiples of the anchor
WINDOWS = {"real": (0.25, 3.0), "random": (0.5, 8.0), "balanced": (1.0, 12.0)}
VARIANT_WINDOW = {"real": "real", "gated": "real", "clean": "real",
                  "clean_gated": "real", "random": "random",
                  "random_cm2": "random", "random_cp2": "random",
                  "balanced": "balanced"}
IDEAL = {"random": (False, 0), "random_cm2": (False, -2),
         "random_cp2": (False, 2), "balanced": (True, 0)}


def geometric_grid(lo=16, hi=1 << 17):
    """16, 24, 32, 48, ... : powers of two and 1.5 times them."""
    out, m = [], lo
    while m <= hi:
        out.append(m)
        if m * 3 // 2 <= hi:
            out.append(m * 3 // 2)
        m *= 2
    return sorted(set(out))


def window(anchor, name):
    lo, hi = WINDOWS[name]
    return [m for m in geometric_grid() if lo * anchor <= m <= hi * anchor]


def pair_sharing_expected(n, k, M):
    return (M - 1) * k * (k - 1) / (n * (n - 1))


# ---------------------------------------------------------------- device side
def _torch():
    from neural_assemblies.core._torch_ops import torch_ops
    return torch_ops


def presence_of(pres_bits, b, n):
    """Unpack brain b's presence bitmask [n_pre, words] to bool [n, n]."""
    torch = _torch()
    shifts = torch.arange(32, device=pres_bits.device, dtype=torch.int32)
    bits = ((pres_bits[b].unsqueeze(-1) >> shifts) & 1).reshape(pres_bits.shape[1], -1)
    return bits[:, :n].bool()


def chain(device):
    torch = _torch()
    from neural_assemblies.core._pricing import chain_table
    return torch.from_numpy(chain_table(BETA, W_MAX, 127)).to(device)


def weights(counts, present, invdj_b, tab):
    """The organ fiber's weight: chain-table potentiation, clip, presence,
    divided by the post neuron's in-degree (norm_init)."""
    return present.float() * tab[counts.long().clamp_(max=tab.numel() - 1)] * invdj_b.view(1, -1)


def pattern_counts(patterns, n):
    """Unclipped N = X^T X for patterns [M, k] (int64), as float32."""
    torch = _torch()
    N = torch.zeros(n, n, dtype=torch.float32, device=patterns.device)
    for s in range(0, patterns.shape[0], CHUNK):
        part = patterns[s:s + CHUNK]
        X = torch.zeros(part.shape[0], n, dtype=torch.float32, device=patterns.device)
        X.scatter_(1, part, 1.0)
        N.addmm_(X.t(), X)
    return N


def pattern_stats(N, present, M, k):
    """Potentiated fraction of present synapses and pair sharing."""
    torch = _torch()
    diag = torch.diagonal(N)
    shared = float((N * (N - 1)).sum() - (diag * (diag - 1)).sum())
    pot = float(((N > 0) & present).sum()) / float(present.sum())
    return pot, shared / (M * k * (k - 1))


def recall(W, cues, k, rounds=T):
    """`rounds` frozen k-WTA rounds of recurrence from cues [S, m]."""
    win = cues
    for _ in range(rounds):
        win = W[win].sum(1).topk(k, dim=1).indices
    return win


def rank1(W, patterns, sample, k):
    """Per-brain rank-1 (fraction of sampled items whose half-cue recall
    overlaps them more than any other stored item; ties to the lowest index,
    as `seq_capacity_scaling.measure`) and mean own overlap."""
    torch = _torch()
    idx = torch.as_tensor(sample, device=patterns.device, dtype=torch.int64)
    rec = recall(W, patterns[idx][:, : k // 2], k)
    mask = torch.zeros(len(sample), W.shape[0], dtype=torch.bool, device=W.device)
    mask.scatter_(1, rec, True)
    ov = torch.stack([mask[i][patterns].sum(1) for i in range(len(sample))])  # [S, M]
    hit = (ov.argmax(1) == idx).float().mean().item()
    own = (ov[torch.arange(len(sample), device=W.device), idx].float() / k).mean().item()
    return hit, own


def first_item_count(C, pres_bits, first, n):
    """Rounded mean count on present internal pairs of the first item."""
    values = []
    for b in range(first.shape[0]):
        S = first[b]
        present = presence_of(pres_bits, b, n)[S.view(-1, 1), S.view(1, -1)]
        values.append(C[b][S.view(-1, 1), S.view(1, -1)][present].float().mean().item())
    return int(round(ensemble_from_values(values).mean))


def ideal_patterns(n, k, M, B, balanced, seed, device):
    """[M, B, k] independent (or usage-balanced) k-subsets."""
    torch = _torch()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = torch.empty(M, B, k, dtype=torch.int64, device=device)
    if not balanced:
        for s in range(0, M, CHUNK):
            m = min(CHUNK, M - s)
            out[s:s + m] = torch.rand(m, B, n, generator=g, device=device).topk(k, dim=2).indices
        return out
    use = torch.zeros(B, n, device=device)
    ones = torch.ones(B, k, device=device)
    for a in range(M):
        pick = (use + torch.rand(B, n, generator=g, device=device)).topk(
            k, dim=1, largest=False).indices
        out[a] = pick
        use.scatter_add_(1, pick, ones)
    return out


def _measure_patterns(stored, checkpoints, n, k, c, pres_bits, invdj, tab,
                      real_counts=None):
    """Per-brain readings of one variant at each checkpoint.

    `stored` [M_max, B, k]. With `real_counts` [B, n, n] the variant is read
    through those synapses (the real memory); otherwise through c counts per
    internal pair of its own patterns (a clean write)."""
    torch = _torch()
    B = stored.shape[1]
    out = {m: {"rank1": [], "own": [], "potentiated": [], "pair_sharing": []}
           for m in checkpoints}
    for b in range(B):
        present = presence_of(pres_bits, b, n)
        for M in checkpoints:
            pats = stored[:M, b]
            N = pattern_counts(pats, n)
            pot, share = pattern_stats(N, present, M, k)
            if real_counts is None:
                counts = (N * c).clamp_(max=tab.numel() - 1)
                written = pot
            else:
                counts = real_counts[b]
                written = float(((counts > 0) & present).sum()) / float(present.sum())
            W = weights(counts, present, invdj[b], tab)
            hit, own = rank1(W, pats, sample_for(M), k)
            row = out[M]
            row["rank1"].append(hit)
            row["own"].append(own)
            row["potentiated"].append(written)
            row["pair_sharing"].append(share)
            del N, W, counts
        del present
    torch.cuda.empty_cache()
    return out


def _real_condition(n, k, seeds, checkpoints, gate, device, organ_semantics):
    """Store with the capacity study's protocol; read real and clean."""
    torch = _torch()
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory

    mem = AssemblyMemory(seeds_for(seeds), n, k, P, beta=BETA, w_max=W_MAX,
                         norm_init=True, synaptic_scaling=False, rounds=T,
                         strength=STRENGTH, gate=gate, max_items=max(checkpoints),
                         device=device, organ_semantics=organ_semantics)
    fiber = mem.fiber
    tab = chain(device)
    stored, c = [], None
    real = {}
    clean_at = {}
    module = {}
    for a in range(max(checkpoints)):
        ss = [to_i32(_seeding.fnv1a_pair_seed(seed, f"s{a}", "A")) for seed in seeds]
        stored.append(mem.store(ss, stim_size=k).clone())
        M = a + 1
        if M == 1:
            c = first_item_count(fiber.C, fiber.pres, stored[0], n)
        if M not in checkpoints:
            continue
        St = torch.stack(stored)
        # the module's own recall on the same sampled items: PE-V(a)
        hits = torch.zeros(len(seeds), device=device)
        sample = sample_for(M)
        flat = St.permute(1, 0, 2)                                   # [B, M, k]
        for i in sample:
            rec = mem.recall(St[int(i)][:, : k // 2])
            mask = torch.zeros(len(seeds), n, dtype=torch.bool, device=device)
            mask.scatter_(1, rec, True)
            ov = torch.stack([mask[b][flat[b]].sum(1) for b in range(len(seeds))])
            hits += (ov.argmax(1) == int(i)).float()
        module[M] = (hits / len(sample)).tolist()
        real.update(_measure_patterns(St, [M], n, k, c, fiber.pres, fiber.invdj,
                                      tab, real_counts=fiber.C))
        clean_at.update(_measure_patterns(St, [M], n, k, c, fiber.pres, fiber.invdj, tab))
    for M in real:
        real[M]["rank1_module"] = module[M]
    pres_bits, invdj = fiber.pres.clone(), fiber.invdj.clone()
    del mem, fiber, stored
    torch.cuda.empty_cache()
    return c, real, clean_at, pres_bits, invdj


def _ceiling(points):
    ceiling = ceiling_from_curve(points, threshold=HALF_BAR)
    return {"m_star": ceiling.m_star, "lo": ceiling.lo, "hi": ceiling.hi,
            "censored": bool(ceiling.censored),
            "above_at_first": bool(points and min(points)[1] > HALF_BAR),
            "below_at_first": bool(points and min(points)[1] <= HALF_BAR),
            "interior_points": ceiling.n_interior, "supported": bool(ceiling.supported)}


def summarise(variant, n, k, seeds, checkpoints):
    curve = []
    summary = {}
    for M in sorted(checkpoints):
        reading = checkpoints[M]
        ens = {name: asdict(ensemble_from_values(values, keys=seeds, label=name))
               for name, values in reading.items()}
        summary[M] = ens
        curve.append((M, ens["rank1"]["mean"]))
    return {"variant": variant, "ensembles": summary, "ceiling": _ceiling(curve),
            "pair_sharing_expected": {M: pair_sharing_expected(n, k, M)
                                      for M in sorted(checkpoints)}}


def experiment(record):
    """All variants at every requested cell; completion leaves bars UNJUDGED."""
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    torch = _torch()
    tab = chain(device)
    cells = {}
    for spec in parameters["cells"]:
        n, k = spec["n"], spec["k"]
        grids = {name: [int(m) for m in values] for name, values in spec["grids"].items()}
        variants = {}
        c, real, clean, pres_bits, invdj = _real_condition(
            n, k, seeds, grids["real"], False, device, profiles["refracted"])
        variants["real"] = (real, grids["real"])
        variants["clean"] = (clean, grids["real"])
        constants = {"c": c}
        if "gated" in grids:
            cg, gated, clean_g, _, _ = _real_condition(
                n, k, seeds, grids["gated"], True, device, profiles["gated"])
            variants["gated"] = (gated, grids["gated"])
            variants["clean_gated"] = (clean_g, grids["gated"])
            constants["c_gated"] = cg
        for name, (balanced, dc) in IDEAL.items():
            grid = grids[VARIANT_WINDOW[name]]
            seed = to_i32(_seeding.fnv1a_pair_seed(MEASUREMENT_SEED, f"{name}/{n}/{k}", "A"))
            pats = ideal_patterns(n, k, max(grid), len(seeds), balanced, seed, device)
            reading = _measure_patterns(pats, grid, n, k, max(c + dc, 1), pres_bits, invdj, tab)
            variants[name] = (reading, grid)
            constants[f"c_{name}"] = max(c + dc, 1)
            del pats
            torch.cuda.empty_cache()
        cells[f"{n}/{k}"] = {
            "n": n, "k": k, "constants": constants,
            "variants": {name: summarise(name, n, k, seeds, reading)
                         for name, (reading, _) in variants.items()},
            "readings": {name: {str(M): values for M, values in reading.items()}
                         for name, (reading, _) in variants.items()},
        }
        print(f"cell ({n}, {k}): " + ", ".join(
            f"{name} {cell['ceiling']['m_star']:.0f}"
            for name, cell in cells[f'{n}/{k}']['variants'].items()), flush=True)
    return {"cells": cells, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def _interp(variant, field, m_star):
    """Ensemble mean of `field` at M*, linear in log2 M between checkpoints."""
    pts = sorted((int(M), e[field]["mean"]) for M, e in variant["ensembles"].items())
    if m_star <= pts[0][0]:
        return pts[0][1]
    for (m0, v0), (m1, v1) in zip(pts, pts[1:]):
        if m0 <= m_star <= m1:
            t = (math.log2(m_star) - math.log2(m0)) / (math.log2(m1) - math.log2(m0))
            return v0 + t * (v1 - v0)
    return pts[-1][1]


def evaluate(observations, *, anchors=None, gated_anchors=None):
    """Amendment 9's bars on a finished record's observations."""
    anchors = ANCHORS if anchors is None else anchors
    gated_anchors = GATED_ANCHORS if gated_anchors is None else gated_anchors
    cells = {(c["n"], c["k"]): c for c in observations["cells"].values()}
    judged = [nk for nk in IN_REGIME if nk in cells]
    out = {"cells": {}, "bars": {}}

    def m_star(cell, name):
        return cell["variants"][name]["ceiling"]["m_star"]

    def resolved(cell, name):
        ceil = cell["variants"][name]["ceiling"]
        return not ceil["censored"] and not ceil["below_at_first"]

    for nk, cell in cells.items():
        n, k = nk
        sq = (n / k) ** 2
        row = {name: {"m_star": m_star(cell, name), "per_nk2": m_star(cell, name) / sq,
                      "resolved": resolved(cell, name)}
               for name in cell["variants"]}
        for name in cell["variants"]:
            v = cell["variants"][name]
            ms = m_star(cell, name)
            share = _interp(v, "pair_sharing", ms)
            expected = pair_sharing_expected(n, k, ms)
            row[name].update(potentiated=_interp(v, "potentiated", ms),
                             pair_sharing=share, pair_excess=share / expected)
        row["eta"] = m_star(cell, "real") / m_star(cell, "random")
        if "gated" in cell["variants"]:
            row["eta_gated"] = m_star(cell, "gated") / m_star(cell, "random")
        out["cells"][f"{n}/{k}"] = row

    def every(pred):
        return all(pred(nk, out["cells"][f"{nk[0]}/{nk[1]}"], cells[nk]) for nk in judged)

    # PE-V: the instrument
    va = all(abs(e["rank1"]["mean"] - ensemble_from_values(v["rank1_module"]).mean) <= 0.03
             for c in cells.values()
             for e, v in ((c["variants"]["real"]["ensembles"][M], c["readings"]["real"][M])
                          for M in c["readings"]["real"]))
    vb = all(abs(out["cells"][f"{n}/{k}"]["real"]["m_star"] / anchors[(n, k)] - 1) <= 0.10
             for (n, k) in cells if (n, k) in anchors)
    out["bars"]["PE-V"] = va and vb
    out["bars"]["PE-V(a) module recall"] = va
    out["bars"]["PE-V(b) anchors"] = vb
    # PE-0: the pair-sharing instrument reads 1 on independent patterns
    calib = all(0.9 <= e["pair_sharing"]["mean"] / cell["variants"]["random"]["pair_sharing_expected"][M] <= 1.1
                for cell in cells.values()
                for M, e in cell["variants"]["random"]["ensembles"].items())
    # the substitution: the random memory in the real one's place is eta = 1
    out["bars"]["PE-0"] = calib and not (0.35 <= 1.0 <= 0.70)

    def ok(r, *names):
        return all(r[name]["resolved"] for name in names)

    out["bars"]["PE-1"] = every(lambda nk, r, c: ok(r, "random")
                                and 0.5 <= r["random"]["per_nk2"] <= 1.0
                                and 0.40 <= r["random"]["potentiated"] <= 0.60)
    etas = [out["cells"][f"{n}/{k}"]["eta"] for n, k in judged]
    out["bars"]["PE-2"] = (every(lambda nk, r, c: ok(r, "real", "random")
                                 and 0.35 <= r["eta"] <= 0.70)
                           and max(etas) / min(etas) <= 1.5)
    out["bars"]["PE-3"] = every(lambda nk, r, c: ok(r, "real", "random")
                                and r["real"]["pair_excess"] >= 1.5
                                and 0.75 <= r["real"]["pair_sharing"] / r["random"]["pair_sharing"] <= 1.33)
    out["bars"]["PE-4"] = every(lambda nk, r, c: ok(r, "real", "clean")
                                and r["real"]["m_star"] > r["clean"]["m_star"])
    # a balanced curve still above the bar at its last checkpoint is a LOWER
    # bound (m_star = that checkpoint), which suffices when it clears 1.5x
    out["bars"]["PE-5"] = every(lambda nk, r, c: ok(r, "random")
                                and not c["variants"]["balanced"]["ceiling"]["below_at_first"]
                                and r["balanced"]["m_star"] >= 1.5 * r["random"]["m_star"])
    out["bars"]["PE-S"] = every(lambda nk, r, c: ok(r, "random", "random_cm2", "random_cp2") and all(
        abs(r[v]["m_star"] / r["random"]["m_star"] - 1) <= 0.30 for v in ("random_cm2", "random_cp2")))
    gated = [nk for nk in judged if "gated" in cells[nk]["variants"]]
    if gated:
        ok_r = all(abs(out["cells"][f"{n}/{k}"]["gated"]["m_star"] / gated_anchors[(n, k)] - 1) <= 0.10
                   for n, k in gated)
        ok_6 = True
        for n, k in gated:
            r, c = out["cells"][f"{n}/{k}"], cells[(n, k)]
            common = sorted(set(c["variants"]["real"]["ensembles"]) & set(c["variants"]["gated"]["ensembles"]),
                            key=lambda M: abs(math.log2(int(M)) - math.log2(r["real"]["m_star"])))
            M = common[0]
            share = {name: c["variants"][name]["ensembles"][M]["pair_sharing"]["mean"]
                     for name in ("real", "gated")}
            r["gate_common_M"] = int(M)
            r["gate_pair_sharing"] = share
            ok_6 &= (ok(r, "gated", "real", "random")
                     and r["eta_gated"] > r["eta"] and share["gated"] < share["real"]
                     and 0.75 <= r["gated"]["pair_sharing"] / r["random"]["pair_sharing"] <= 1.33)
        out["bars"]["PE-R"] = ok_r
        out["bars"]["PE-6"] = ok_6
    return out


# ----------------------------------------------------------------- the CLI
def plan(cells, *, smoke=False):
    """Explicit per-cell checkpoint grids (recorded in the run's parameters)."""
    out = []
    for n, k in cells:
        anchor = ANCHORS[(n, k)]
        grids = {"real": window(anchor, "real"), "random": window(anchor, "random"),
                 "balanced": window(anchor, "balanced")}
        if (n, k) in GATED_ANCHORS:
            grids["gated"] = window(GATED_ANCHORS[(n, k)], "real")
        if smoke:
            grids = {name: values[:2] for name, values in grids.items()}
        out.append({"n": n, "k": k, "grids": grids})
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Pattern efficiency", engines=("hashed_assembly_memory",),
                           default_seeds=tuple(range(42, 62)))
    ap.add_argument("--registration", required=True)
    ap.add_argument("--nk", help="n:k pairs (default: the seven cells of the law)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(int(v) for v in pair.split(":")) for pair in args.nk.split(",")]
             if args.nk else list(ANCHORS))
    if any(nk not in ANCHORS for nk in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {sorted(ANCHORS)}")
    profiles = {
        "refracted": describe_assembly_memory(w_max=W_MAX, beta=BETA, strength=STRENGTH,
                                              gate=False, norm_init=True, synaptic_scaling=False),
        "gated": describe_assembly_memory(w_max=W_MAX, beta=BETA, strength=STRENGTH,
                                          gate=True, norm_init=True, synaptic_scaling=False),
    }
    path = run_experiment(
        script=__file__, protocol="memory.pattern-efficiency", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": plan(cells, smoke=args.smoke), "p": P, "beta": BETA,
                    "w_max": W_MAX, "rounds": T, "strength": STRENGTH, "half_bar": HALF_BAR,
                    "recall_sample": RECALL_SAMPLE, "measurement_seed": MEASUREMENT_SEED,
                    "windows": WINDOWS, "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
