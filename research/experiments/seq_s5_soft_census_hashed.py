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

import argparse
import inspect
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch                                                              # noqa: E402

from neural_assemblies.core.brain import Brain                            # noqa: E402
from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM  # noqa: E402
from neural_assemblies.programs.word_problems import (                    # noqa: E402
    GROUPS, true_trajectory, word_problem_fsm)
from seq_s5_word_problem import (                                         # noqa: E402
    BETA, GROUP_NAMES, K, ORGAN_P, PRESENTATIONS, REFRACTED, SEEDS, sizes)
from _results import write_result  # noqa: E402

LONGEST = 500
LAUNCH_BYTES = 5 << 30


def _bytes_per_brain(n_arc, n_state):
    return int(2 * n_arc * n_state * (2 + 4 / 32) + 4 * n_arc * 64)


def census_at_width(group_name, seeds, longest=LONGEST, presentations=PRESENTATIONS,
                    strength=REFRACTED, norm_init=False):
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
            wins = None
            drive = None
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--groups", type=str, default=",".join(GROUP_NAMES))
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--presentations", type=int, default=PRESENTATIONS,
                    help="Addendum 5: the potentiated gain (1 + beta)^P")
    ap.add_argument("--norm-init", action="store_true",
                    help="Addendum 7 N1: in-degree normalisation on the organ")
    ap.add_argument("--strength", type=float, default=REFRACTED,
                    help="Addendum 6: the arc's refraction strength (absolute; "
                         "the registered organ is 0.1 = beta)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    # seeds continue the registered ten (42..51) upward
    seeds = list(range(SEEDS[0], SEEDS[0] + (2 if args.smoke else args.seeds)))
    groups = args.groups.split(",")
    longest = 20 if args.smoke else LONGEST
    pres = 2 if args.smoke else args.presentations
    if args.smoke:
        print("*** SMOKE: API only. THESE NUMBERS ARE VOID. ***")
    print("=== E7 at width: the soft census on the explicit substrate ===\n")
    out = []
    for g in groups:
        G = GROUPS[g]()
        n_arc, n_state = sizes(G, len(G.generators))
        per = max(1, LAUNCH_BYTES // _bytes_per_brain(n_arc, n_state))
        for i in range(0, len(seeds), per):
            out += census_at_width(g, seeds[i:i + per], longest, pres, args.strength,
                                   args.norm_init)

    print(f"\n    {'group':7s} {'seed':>4s} {'first_bad':>9s} {'dev':>5s} "
          f"{'pred_dev':>8s} {'V2':>3s} {'soft':>5s} {'hard':>5s} {'min_ov':>7s}")
    clean, v2, any_bad = 0, True, 0
    for v in out:
        dev = "-" if v["first_dev"] is None else str(v["first_dev"])
        pred = "-" if v["predicted_first_dev"] is None else str(v["predicted_first_dev"])
        v2 &= v["v2_exact"]
        nb = v["n_soft"] + v["n_hard"]
        any_bad += nb > 0
        clean += (nb == 0 and v["first_bad"] == longest)
        print(f"    {v['group']:7s} {v['seed']:4d} {v['first_bad']:9d} {dev:>5s} "
              f"{pred:>8s} {str(v['v2_exact'])[0]:>3s} {v['n_soft']:5d} "
              f"{v['n_hard']:5d} {v['census_ov_min']:7.4f}", flush=True)
    n = len(out)
    from collections import Counter
    rel = Counter(r_ for v in out for x in v["relations"] for r_ in x["relation"])
    per_group = {}
    for v in out:
        g_ = per_group.setdefault(v["group"], [0, 0, 0])
        g_[0] += v["n_soft"] + v["n_hard"]; g_[1] += v["n_pairs"]
        g_[2] += v["first_bad"] < longest
    print("\n    per group: bad pairs / pairs (rate)   words derailing")
    for g_, (nb, np_, nd) in per_group.items():
        print(f"      {g_:7s} {nb:4d} / {np_:6d} ({100.0 * nb / np_:.3f}%)   {nd}")
    print(f"    intruder relations: {dict(rel)}")
    xs_ = np.array([v["across_symbol"] for v in out]); xt_ = np.array([v["across_state"] for v in out])
    print(f"    conjunction (strength {args.strength}): across-symbol overlap "
          f"{xs_.mean():.3f} (max {xs_.max():.3f}), across-state {xt_.mean():.3f} "
          f"(max {xt_.max():.3f}); P-CONJ both < 0.15: "
          f"{'PASS' if xs_.mean() < 0.15 and xt_.mean() < 0.15 else 'FAIL'}")
    cs = [(x["c_o"], x["c_b"]) for v in out for x in v["relations"] if "c_o" in x]
    if cs:
        t3 = sum(1 for co, cb in cs if cb <= 14 and co >= 35)
        print(f"    tails (presentations {pres}, gain {min((1 + BETA) ** pres, 20.0):.2f}): "
              f"c_o (intruder) {sorted(round(c, 1) for c, _ in cs)}; "
              f"c_b (weakest block member) {sorted(round(c, 1) for _, c in cs)}; "
              f"T3 (c_b <= 14 and c_o >= 35): {t3}/{len(cs)}")
    pairs = sum(v["n_pairs"] for v in out)
    n_soft = sum(v["n_soft"] for v in out)
    n_hard = sum(v["n_hard"] for v in out)
    print("\n=== BARS (Addendum 3) ===")
    print(f"  soft {n_soft} + hard {n_hard} of {pairs} pairs "
          f"({100.0 * (n_soft + n_hard) / max(pairs, 1):.3f}%); "
          f"organs with any bad pair: {any_bad}/{n}; clean organs (no bad pair, "
          f"word runs {longest}): {clean}/{n}")
    w1 = clean >= int(0.9 * n)
    print(f"  {'PASS' if w1 else 'FAIL'}  W1 the soft pairs were the sampler's "
          f"(>= 90% of organs clean)")
    print(f"  {'PASS' if v2 else 'FAIL'}  W2 first_dev == first true-path visit to "
          f"a bad pair, every organ (vacuous where none)")
    if not args.smoke:
        path = write_result(
            "sequence", f"seq_s5_soft_census_results_hashed{args.tag}.json", out,
        )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
