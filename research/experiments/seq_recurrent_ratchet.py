"""Does per-round mass renormalization lift the RATCHET ceiling?

Implements `research/notes/memory/PREREG_recurrent_ratchet.md`.

Recurrence ON, because self-recurrence is the DEFINITION of an assembly. Three
substrates: NONE (no normalization), B (`norm_init`, INITIAL weights only), C
(`synaptic_scaling`, CURRENT mass, per round). The ratchet -- the competitor
past ~32 items being the already-potentiated assemblies -- is the one collapse
mechanism norm_init provably cannot touch, and substrate C is the only tool
that normalizes learned mass.

Four readouts, three of which exist to stop a silent pass: the documented
failure mode is self-overlap 0.68 (reads as success) alongside rank-1 identity
0.018 against a chance of 0.008.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap, activate_assembly
from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import assembly_overlap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

N, K, BETA, P = 2000, 50, 0.10, 0.05
AREA = "A"
ARMS = ("NONE", "B", "C")
SEEDS = [42, 43, 44]
PART_A_M = [8, 16, 32, 48, 64]          # M-ceiling sweep, T=8
PART_B_T = [5, 8, 12, 20]               # T-window sweep, M=16
CEILING = 0.50                          # rank-1 threshold defining "holds"


def _train(arm, M, T, seed):
    """Store M assemblies in ONE area, with recurrence on."""
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=P, seed=seed, engine="numpy_sparse",
                  recurrent_projection=True,
                  norm_init=(arm == "B"),
                  synaptic_scaling=(arm == "C"))
    brain.add_area(AREA, N, K, BETA)
    stims = []
    for i in range(M):
        s = f"s{i}"
        brain.add_stimulus(s, K)
        stims.append(s)
    stored = []
    for s in stims:
        brain.inhibit_areas([AREA])
        for _ in range(T):
            brain.project({s: [AREA]}, {AREA: [AREA]})
        stored.append(_snap(brain, AREA))
    return brain, stims, stored


def _retrieve(brain, T, stim=None, half_of=None):
    """One retrieval inside a probe. Either stimulus-cued or half-assembly-cued.

    `brain.probe()` because recruitment -- not plasticity -- is the channel by
    which a readout changes what it reads ([[probe-isolation-required]]), and
    the snap is taken INSIDE the block.
    """
    with brain.probe():
        brain.inhibit_areas([AREA])
        if half_of is not None:
            ids = np.asarray(half_of.winners)[: K // 2]
            activate_assembly(brain, Assembly(AREA, NeuronIds(ids)))
            for _ in range(T):
                brain.project({}, {AREA: [AREA]})     # recurrence alone
        else:
            for _ in range(T):
                brain.project({stim: [AREA]}, {AREA: [AREA]})
        return _snap(brain, AREA)


def _score(brain, stims, stored, T):
    """rank-1 identity (full and half cue), self-overlap, distinctness."""
    full_hits = half_hits = 0
    self_ovs = []
    for i, s in enumerate(stims):
        live = _retrieve(brain, T, stim=s)
        ovs = [assembly_overlap(np.asarray(live.winners),
                                np.asarray(st.winners)) for st in stored]
        full_hits += int(np.argmax(ovs) == i)
        self_ovs.append(float(ovs[i]))

        liveh = _retrieve(brain, T, half_of=stored[i])
        ovh = [assembly_overlap(np.asarray(liveh.winners),
                                np.asarray(st.winners)) for st in stored]
        half_hits += int(np.argmax(ovh) == i)

    M = len(stims)
    pair = [assembly_overlap(np.asarray(stored[i].winners),
                             np.asarray(stored[j].winners))
            for i in range(M) for j in range(i + 1, M)]
    return {"rank1_full": full_hits / M, "rank1_half": half_hits / M,
            "self_overlap": float(np.mean(self_ovs)),
            "pairwise": float(np.mean(pair)) if pair else 0.0,
            "chance": K / N}


def worker(arm, M, T, seed):
    brain, stims, stored = _train(arm, M, T, seed)
    r = _score(brain, stims, stored, T)
    r.update(arm=arm, M=M, T=T, seed=seed)
    # The documented silent failure: self-overlap reads as success while
    # rank-1 identity is at chance.
    r["silent_flag"] = bool(r["self_overlap"] > 0.6 and r["rank1_full"] < 0.2)
    return r


def _mean(res, arm, M, T, key):
    return float(np.mean([res[(arm, M, T, s)][key] for s in SEEDS]))


def _ceiling(res, arm, Ts=8):
    """Largest M whose mean rank-1 (full cue) still clears CEILING."""
    hit = [M for M in PART_A_M
           if _mean(res, arm, M, Ts, "rank1_full") >= CEILING]
    return max(hit) if hit else 0


def main():
    from _parallel import run_cells
    cells = [(a, M, 8, s) for a in ARMS for M in PART_A_M for s in SEEDS]
    cells += [(a, 16, T, s) for a in ARMS for T in PART_B_T for s in SEEDS
              if T != 8]
    print("=== recurrent ratchet: does per-round mass renorm lift the ceiling? ===")
    print(f"    n={N} k={K} beta={BETA} p={P}, recurrence ON, "
          f"chance overlap={K/N:.4f}\n")
    res = run_cells(worker, cells, max_workers=min(len(cells), 14))

    print(f"\n--- Part A: M-ceiling at T=8  (rank1 full / half, self-ov, pairwise)")
    print(f"    {'arm':5s} {'M':>4} {'rank1_full':>11} {'rank1_half':>11} "
          f"{'self_ov':>8} {'pairwise':>9}  flags")
    for a in ARMS:
        for M in PART_A_M:
            f = _mean(res, a, M, 8, "rank1_full")
            h = _mean(res, a, M, 8, "rank1_half")
            so = _mean(res, a, M, 8, "self_overlap")
            pw = _mean(res, a, M, 8, "pairwise")
            flags = sum(res[(a, M, 8, s)]["silent_flag"] for s in SEEDS)
            print(f"    {a:5s} {M:4d} {f:11.3f} {h:11.3f} {so:8.3f} "
                  f"{pw:9.4f}  {'SILENT x%d' % flags if flags else ''}")

    print(f"\n--- Part B: T-window at M=16")
    print(f"    {'arm':5s} {'T':>4} {'rank1_full':>11} {'rank1_half':>11}")
    for a in ARMS:
        for T in PART_B_T:
            print(f"    {a:5s} {T:4d} {_mean(res, a, 16, T, 'rank1_full'):11.3f} "
                  f"{_mean(res, a, 16, T, 'rank1_half'):11.3f}")

    ceil = {a: _ceiling(res, a) for a in ARMS}
    print("\n=== BARS ===")
    rc1 = all(_mean(res, a, 8, 8, "rank1_full") > 0.90 for a in ARMS)
    print(f"  {'PASS' if rc1 else 'FAIL'}  RC1 instrument sanity: M=8 rank1 > "
          f"0.90 all arms " +
          str({a: round(_mean(res, a, 8, 8, 'rank1_full'), 3) for a in ARMS}))
    rc2 = _mean(res, "NONE", 48, 8, "rank1_full") < 0.50
    print(f"  {'PASS' if rc2 else 'FAIL'}  RC2 ratchet reproduces: NONE at "
          f"M=48 rank1 {_mean(res, 'NONE', 48, 8, 'rank1_full'):.3f} < 0.50")
    rc3 = ceil["C"] > ceil["B"]
    print(f"  {'PASS' if rc3 else 'FAIL'}  RC3 THE CLAIM: C ceiling "
          f"{ceil['C']} > B ceiling {ceil['B']}  (NONE {ceil['NONE']})")
    mc = ceil["C"]
    rc4 = (mc > 0) and _mean(res, "C", mc, 8, "rank1_half") > 0.50
    print(f"  {'PASS' if rc4 else 'FAIL'}  RC4 pattern completion at C's "
          f"ceiling M={mc}: half-cue rank1 "
          f"{_mean(res, 'C', mc, 8, 'rank1_half') if mc else float('nan'):.3f}"
          f" > 0.50")
    rc5 = _mean(res, "C", 16, 20, "rank1_full") > _mean(res, "B", 16, 20,
                                                        "rank1_full")
    print(f"  {'PASS' if rc5 else 'FAIL'}  RC5 window at M=16,T=20: C "
          f"{_mean(res, 'C', 16, 20, 'rank1_full'):.3f} > B "
          f"{_mean(res, 'B', 16, 20, 'rank1_full'):.3f}")
    if rc3 and not rc4:
        print("        -> C buys CAPACITY without buying ATTRACTORS; report "
              "as capacity, never as robustness.")
    if not rc3:
        print("        -> RC3 failing with C ~ B is the LARGER result: the "
              "ratchet is not a normalization problem and the "
              "capacity/robustness trade is structural.")

    path = os.path.join(_HERE, "seq_recurrent_ratchet_results.json")
    with open(path, "w") as fh:
        json.dump({"n": N, "k": K, "beta": BETA, "p": P, "seeds": SEEDS,
                   "ceilings": ceil,
                   "bars": {"RC1": bool(rc1), "RC2": bool(rc2),
                            "RC3": bool(rc3), "RC4": bool(rc4),
                            "RC5": bool(rc5)},
                   "cells": {f"{a}/{M}/{T}/{s}": res[(a, M, T, s)]
                             for (a, M, T, s) in cells}}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
