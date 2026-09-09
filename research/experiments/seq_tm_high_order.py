"""PREREG_temporal_memory.md, cells B: high-order sequence prediction on the
hashed transducer as a temporal memory (state = previous arc, predicted
neurons win), the literature's own benchmark (Bouhadjar et al. 2022).

    set I    A D B E  and  F D B C          (shared middle D B; order 2)
    set II   six five-element sequences with shared elements
    set III  two twelve-element sequences identical in the middle ten
             (order 10)

Each set is presented PRESENTATIONS times to 20 brains; after each
presentation every sequence is replayed frozen and the next word
predicted at every position. TM-4: set I's continuation after D B on
>= 18/20 brains at g = 4. TM-5: set III's order-10 disambiguation on
>= 15/20 at g = 4. TM-6: the presentation at which set I is first
predicted perfectly, reported. The induced state (g = 0, projected) is
the control on every set.

    python research/experiments/seq_tm_high_order.py [--brains 20] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                              # noqa: E402

from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer  # noqa: E402
from _results import results_path                                         # noqa: E402

N, N_ARC, K, P, ORGAN_P, BETA = 4000, 4000, 100, 0.05, 0.2, 0.10
PRESENTATIONS, ROUNDS, GROUND_ROUNDS = 40, 3, 5
SEEDS = list(range(42, 62))

SETS = {
    "I": [list("ADBE"), list("FDBC")],
    "II": [list("ADBEG"), list("FDBCH"), list("IJBEK"), list("LDMCN"), list("OJBCP"), list("QDMEK")],
    "III": [list("A") + list("DBEGHIJKLM") + list("N"),
            list("F") + list("DBEGHIJKLM") + list("O")],
}
VOCAB = sorted({w for seqs in SETS.values() for s in seqs for w in s})


def ambiguous_positions(seqs):
    """(sequence index, position) pairs where the prefix so far is shared
    with another sequence and the next word differs: the order test."""
    out = []
    for i, s in enumerate(seqs):
        for t in range(1, len(s) - 1):
            prefix = tuple(s[1:t + 1])           # after the first, distinguishing word
            for j, o in enumerate(seqs):
                if j != i and len(o) > t + 1 and tuple(o[1:t + 1]) == prefix and o[t + 1] != s[t + 1]:
                    out.append((i, t))
                    break
    return out


def run_set(name, seqs, seeds, gain, mode, presentations):
    t0 = time.perf_counter()
    tr = HashedTransducer(seeds, VOCAB, n=N, n_arc=N_ARC, k=K, p=P, beta=BETA,
                          organ_p=ORGAN_P, w_max=20.0, norm_init=True,
                          max_potentiations=presentations * 4 + 8,
                          state_mode=mode, predict_gain=gain,
                          n_state=(N_ARC if mode == "copy" else None))
    tr.ground(rounds=GROUND_ROUNDS)
    B = len(seeds)
    amb = ambiguous_positions(seqs)
    rng = random.Random(0)
    first_perfect = [None] * B
    per_pres = []
    for pres in range(presentations):
        for s in seqs:
            tr.train_sentence(s, rounds=ROUNDS)
        # replay frozen: rank-1 at every position, and at the ambiguous ones
        hits_all = np.zeros(B); n_all = 0
        hits_amb = np.zeros(B); n_amb = 0
        for i, s in enumerate(seqs):
            tr.reset()
            for t, (a, nxt) in enumerate(zip(s, s[1:])):
                tr.tick(a, rounds=ROUNDS, freeze=True)
                ranked = tr.rank(tr.emit(), rng)
                ok = np.array([r[0] == nxt for r in ranked], dtype=float)
                hits_all += ok; n_all += 1
                if (i, t) in amb:
                    hits_amb += ok; n_amb += 1
        acc_all = hits_all / n_all
        acc_amb = hits_amb / max(n_amb, 1)
        per_pres.append({"presentation": pres + 1, "acc_all": acc_all.tolist(),
                         "acc_ambiguous": acc_amb.tolist()})
        for b in range(B):
            if first_perfect[b] is None and acc_all[b] >= 1.0:
                first_perfect[b] = pres + 1
    last = per_pres[-1]
    print(f"    set {name:3s} mode {mode:7s} g {gain:<3}: rank-1 all {np.mean(last['acc_all']):.3f}, "
          f"ambiguous {np.mean(last['acc_ambiguous']):.3f} on {n_amb} positions; brains perfect "
          f"{sum(a >= 1.0 for a in last['acc_all'])}/{B}; first perfect at "
          f"{sorted(x for x in first_perfect if x)}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    del tr
    torch.cuda.empty_cache()
    return {"set": name, "mode": mode, "gain": gain, "ambiguous_positions": amb,
            "per_presentation": per_pres, "first_perfect": first_perfect,
            "final_acc_ambiguous": last["acc_ambiguous"], "final_acc_all": last["acc_all"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", type=int, default=len(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[: (3 if args.smoke else args.brains)]
    pres = 3 if args.smoke else PRESENTATIONS
    if args.smoke:
        print("*** SMOKE: API only. THESE NUMBERS ARE VOID. ***")
    print(f"=== temporal memory, high-order sequences: {len(seeds)} brains, n {N}, k {K}, "
          f"{pres} presentations ===")
    rows = []
    arms = [("induced", 0.0), ("copy", 0.0), ("copy", 4.0)]
    for name, seqs in SETS.items():
        for mode, g in arms:
            rows.append(run_set(name, seqs, seeds, g, mode, pres))
    print("\n=== BARS ===")
    def brains_ok(name, mode, g, bar):
        r = next(x for x in rows if x["set"] == name and x["mode"] == mode and x["gain"] == g)
        return sum(a >= 1.0 for a in r["final_acc_ambiguous"])
    tm4 = brains_ok("I", "copy", 4.0, 18)
    tm5 = brains_ok("III", "copy", 4.0, 15)
    print(f"  {'PASS' if tm4 >= 18 else 'FAIL'}  TM-4 set I continuation after D B, g=4: {tm4}/{len(seeds)} (>= 18)")
    print(f"  {'PASS' if tm5 >= 15 else 'FAIL'}  TM-5 set III order-10, g=4: {tm5}/{len(seeds)} (>= 15)")
    r1 = next(x for x in rows if x["set"] == "I" and x["mode"] == "copy" and x["gain"] == 4.0)
    print(f"  TM-6 set I first perfect presentation, g=4: {sorted(x for x in r1['first_perfect'] if x)}")
    if not args.smoke:
        path = results_path("sequence", "seq_tm_high_order_results.json")
        with open(path, "w") as fh:
            json.dump({"seeds": seeds, "n": N, "k": K, "presentations": pres, "rows": rows,
                       "tm4": tm4, "tm5": tm5}, fh, indent=1)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
