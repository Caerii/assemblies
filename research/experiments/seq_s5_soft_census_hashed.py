"""E7 at width: the soft census on the EXPLICIT substrate.

PREREG_s5_cliff_anatomy.md Addendum 3. The numpy census (E7) ran on the
SAMPLED arc; GATE-3 of the sequence port found A1's one short horizon was
the sampler's. This is the same census -- every (state, symbol) pair
stepped once from the cued state, frozen, the output assembly's overlap
with its intended block recorded; a 500-symbol word replayed and its first
deviation compared with the first true-path visit to a bad pair -- on the
hashed organ (`HashedArcFSM`), which equals the materialized engine, over
the same four groups and ten seeds, brains batched per launch.

    python research/experiments/seq_s5_soft_census_hashed.py [--seeds 10] [--groups Z60,S5] [--smoke]
"""
from __future__ import annotations

import inspect
import importlib
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from neural_assemblies import describe_hashed_arc_fsm
from neural_assemblies.programs.word_problems import (
    GROUPS, true_trajectory, word_problem_fsm)
from research.experiments.seq_s5_word_problem import (
    BETA, GROUP_NAMES, K, ORGAN_P, PRESENTATIONS, REFRACTED, SEEDS, sizes)
from research.runner import experiment_parser, run_experiment

LONGEST = 500
LAUNCH_BYTES = 5 << 30


def _bytes_per_brain(n_arc, n_state):
    return int(2 * n_arc * n_state * (2 + 4 / 32) + 4 * n_arc * 64)


def census_at_width(group_name, seeds, longest=LONGEST, presentations=PRESENTATIONS,
                    strength=REFRACTED, norm_init=False):
    torch: Any = importlib.import_module("torch")
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    group = GROUPS[group_name]()
    states, symbols, transitions = word_problem_fsm(group)
    n_arc, n_state = sizes(group, len(symbols))
    table = {(fr, sym): to for fr, sym, to in transitions}
    w_max = inspect.signature(Brain).parameters["w_max"].default
    gain = min((1.0 + BETA) ** presentations, w_max)
    t0 = time.perf_counter()
    fsm = HashedArcFSM(seeds, states, symbols, transitions, n_arc=n_arc,
                       n_state=n_state, k=K, p=ORGAN_P, beta=BETA,
                       refracted_strength=strength, w_max=w_max, norm_init=norm_init,
                       max_potentiations=256, prefix="_wp", zero_or_size=False)
    fsm.train(presentations)
    fsm.check()
    B = len(seeds)
    si = fsm.state_index

    def onblock(labels):
        """[B] fraction of STATE's winners inside the block of `labels` [B]."""
        return ((fsm.state.winners // K) == labels.view(-1, 1)).float().mean(1)

    # -- the word, per brain its own
    words, truths = [], []
    for seed in seeds:
        rng = random.Random(seed + 4242)
        w = [rng.choice(symbols) for _ in range(longest)]
        words.append([fsm.symbol_index[s] for s in w])
        truths.append([si[t] for t in true_trajectory(group, w)])
    W = torch.tensor(words, device="cuda")
    T = torch.tensor(truths, device="cuda")
    start = group.label(group.identity)
    fsm.arc.inhibit()
    fsm.cue_state(start)
    labels = torch.zeros_like(W)
    onb = torch.zeros(B, longest, device="cuda")
    for t in range(longest):
        labels[:, t] = fsm.step(W[:, t])
        onb[:, t] = onblock(labels[:, t])
    bad = (labels != T).cpu().numpy()
    dev = (onb < 1.0).cpu().numpy()

    # -- the census: every (state, symbol) once, frozen, from the cued state
    soft = [[] for _ in seeds]
    hard = [[] for _ in seeds]
    relations = [[] for _ in seeds]
    ov_all = []
    arcs = {}                      # (state, symbol) -> [B, k] test-time arc
    idx_of = {v: k_ for k_, v in si.items()}
    for st in states:
        for sym in symbols:
            fsm.arc.inhibit()
            fsm.cue_state(st)
            label = fsm.step(sym)
            arcs[(st, sym)] = fsm.arc.winners.clone()
            target = si[table[(st, sym)]]
            ov = onblock(torch.full((B,), target, device="cuda", dtype=torch.int64))
            ov_all.append(ov)
            lab, ovc = label.cpu().numpy(), ov.cpu().numpy()
            wins: Any = None
            drive: Any = None
            for b in range(B):
                if int(lab[b]) != target:
                    hard[b].append((st, sym))
                elif ovc[b] < 1.0:
                    soft[b].append(((st, sym), float(ovc[b])))
                    # WHO is the intruder: its block, and that block's
                    # relation to the pair in the Cayley graph
                    if wins is None:
                        wins = (fsm.state.winners // K).cpu().numpy()
                        # the STATE drive this step, recomputed from the arc
                        # assembly (Addendum 5): the intruder's drive is its
                        # present-row count at gain 1; a block member's is
                        # its count x g
                        drive = torch.zeros(B, n_state, device="cuda")
                        fsm.arc_state.contribute(drive, fsm.arc.winners)
                        drive = drive.cpu().numpy()
                    intr = sorted(set(int(x) for x in wins[b] if int(x) != target))
                    win_b = fsm.state.winners[b].cpu().numpy()
                    intr_neurons = [int(x) for x in win_b if int(x) // K != target]
                    c_o = max(float(drive[b, j]) for j in intr_neurons)
                    blk = drive[b, target * K:(target + 1) * K]
                    c_b = float(blk.min()) / gain
                    to = table[(st, sym)]
                    rel = []
                    for blk in intr:
                        name = idx_of[blk]
                        if name == st:
                            rel.append("from")
                        elif any(table[(st, g)] == name for g in symbols if g != sym):
                            rel.append("other_gen")
                        elif any(table[(to, g)] == name for g in symbols):
                            rel.append("next")
                        elif any(table[(name, g)] == to for g in symbols):
                            rel.append("co_parent")
                        else:
                            rel.append("other")
                    relations[b].append({"pair": [st, sym], "to": to,
                                         "intruder_blocks": intr, "relation": rel,
                                         "c_o": c_o, "c_b": c_b})
    ov_all = torch.stack(ov_all).cpu().numpy()                    # [pairs, B]

    # -- the conjunction (A1's P-CONJ): across-SYMBOL overlap (same state,
    # the two generators) and across-STATE overlap (same symbol, another
    # state), per brain; chance is k / n_arc
    def _ov(a, b_):
        return (a.unsqueeze(2) == b_.unsqueeze(1)).any(2).float().mean(1)   # [B]
    xs, xt = [], []
    rng_ = random.Random(7)
    for st in states:
        xs.append(_ov(arcs[(st, symbols[0])], arcs[(st, symbols[1])]))
        for sym in symbols:
            other = rng_.choice([x for x in states if x != st])
            xt.append(_ov(arcs[(st, sym)], arcs[(other, sym)]))
    across_symbol = torch.stack(xs).mean(0).cpu().numpy()
    across_state = torch.stack(xt).mean(0).cpu().numpy()

    rows = []
    for b, seed in enumerate(seeds):
        bad_pairs = set(p for p, _ in soft[b]) | set(hard[b])
        prev, predicted = start, None
        for i, sidx in enumerate(words[b]):
            sym = symbols[sidx]
            if (prev, sym) in bad_pairs:
                predicted = i
                break
            prev = table[(prev, sym)]
        first_bad = int(np.flatnonzero(bad[b])[0]) if bad[b].any() else longest
        first_dev = int(np.flatnonzero(dev[b])[0]) if dev[b].any() else None
        rows.append({
            "group": group_name, "seed": int(seed),
            "first_bad": first_bad, "first_dev": first_dev,
            "predicted_first_dev": predicted,
            "v2_exact": bool(first_dev == predicted),
            "n_soft": len(soft[b]), "n_hard": len(hard[b]),
            "soft_overlaps": sorted(float(o) for _p, o in soft[b]),
            "census_ov_min": float(ov_all[:, b].min()),
            "census_ov_varies": bool(ov_all[:, b].min() < ov_all[:, b].max()),
            "onblock_min": float(onb[b].min()),
            "n_pairs": len(states) * len(symbols),
            "relations": relations[b],
            "across_symbol": float(across_symbol[b]),
            "across_state": float(across_state[b]),
            "strength": strength,
        })
    print(f"    {group_name}: {B} brains, n_arc {n_arc}, n_state {n_state}, "
          f"{len(states) * len(symbols)} pairs, word {longest}  "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    del fsm
    torch.cuda.empty_cache()
    return rows


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    groups = parameters["groups"]
    longest = parameters["longest"]
    pres = parameters["presentations"]
    strength = parameters["strength"]
    norm_init = parameters["norm_init"]
    out = []
    for g in groups:
        G = GROUPS[g]()
        n_arc, n_state = sizes(G, len(G.generators))
        per = max(1, LAUNCH_BYTES // _bytes_per_brain(n_arc, n_state))
        for i in range(0, len(seeds), per):
            out += census_at_width(g, seeds[i:i + per], longest, pres, strength,
                                   norm_init)

    clean, v2, any_bad = 0, True, 0
    for v in out:
        v2 &= v["v2_exact"]
        nb = v["n_soft"] + v["n_hard"]
        any_bad += nb > 0
        clean += (nb == 0 and v["first_bad"] == longest)
    n = len(out)
    xs_ = np.array([v["across_symbol"] for v in out]); xt_ = np.array([v["across_state"] for v in out])
    p_conj = bool(xs_.mean() < 0.15 and xt_.mean() < 0.15)
    cs = [(x["c_o"], x["c_b"]) for v in out for x in v["relations"] if "c_o" in x]
    tail_summary = None
    if cs:
        t3 = sum(1 for co, cb in cs if cb <= 14 and co >= 35)
        tail_summary = {"count": len(cs), "t3": t3,
                        "gain": min((1 + BETA) ** pres, 20.0)}
    pairs = sum(v["n_pairs"] for v in out)
    n_soft = sum(v["n_soft"] for v in out)
    n_hard = sum(v["n_hard"] for v in out)
    w1 = clean >= int(0.9 * n)
    return {"verdict": "VOID" if record["mode"] == "smoke" else (
                "PASS" if w1 and v2 else "FAIL"),
            "w1_clean_organs": clean, "w1_total_organs": n,
            "w2_first_deviation_matches": bool(v2), "p_conj": p_conj,
            "soft_pairs": n_soft, "hard_pairs": n_hard, "pairs": pairs,
            "organs_with_bad_pair": any_bad, "rows": out,
            "tail_summary": tail_summary,
            "scope": "S5 soft census and first-deviation mechanism"}


def main(argv=None):
    parser = experiment_parser(
        __doc__ or "S5 soft census", engines=("hashed_arc_fsm",),
        default_seeds=tuple(SEEDS),
    )
    parser.add_argument("--groups", default=",".join(GROUP_NAMES))
    parser.add_argument("--presentations", type=int, default=PRESENTATIONS)
    parser.add_argument("--norm-init", action="store_true")
    parser.add_argument("--strength", type=float, default=REFRACTED)
    args = parser.parse_args(argv)
    groups = args.groups.split(",")
    unknown = sorted(set(groups) - set(GROUP_NAMES))
    if unknown:
        parser.error(f"unknown groups: {unknown}")
    if args.presentations < 1:
        parser.error("--presentations must be positive")
    parameters = {"groups": groups, "longest": 20 if args.smoke else LONGEST,
                  "presentations": 2 if args.smoke else args.presentations,
                  "strength": args.strength, "norm_init": args.norm_init}
    path = run_experiment(
        script=Path(__file__), protocol="sequence.s5-soft-census",
        protocol_version="2", registration="research/notes/sequence/PREREG_s5_cliff_anatomy.md",
        engine=args.engine, seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        parameters=parameters, organ_semantics=describe_hashed_arc_fsm(
            w_max=20.0, norm_init=args.norm_init,
            refracted_strength=args.strength, zero_or_size=False,
        ), measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
