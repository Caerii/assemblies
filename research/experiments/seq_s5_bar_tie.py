"""Are the S5 soft defects k-WTA bar ties?  PREREG_bar_tie.md.

Same organs and the same census as the registered study (`soft_hard_census`
is the ONE owner). After training, the state area gets Gaussian input noise
with std 1e-3 -- enough to reorder EXACT ties in integer drive and nothing
else -- and the census is repeated. Ties move; excess mass does not.

    python research/experiments/seq_s5_bar_tie.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seq_s5_word_problem import (  # noqa: E402
    GROUP_NAMES, build, run_tiered, soft_hard_census,
)

SEEDS = [42, 43, 44]
NOISE = 1e-3
REPEATS = 3


def _pairs(soft):
    return {tuple(rec["pair"]) for rec in soft}


def _jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def worker(group_name, seed, _arm="tie"):
    group, fsm, symbols = build(group_name, seed, "trained",
                                norm_init=False, synaptic_scaling=False)
    b = fsm.brain
    soft0, hard0 = soft_hard_census(b, fsm, symbols, group)
    s0 = _pairs(soft0)

    # Noise on the STATE area only, at census time only. `probe()` SAVES AND
    # RESTORES the engine rng state on exit, so a census inside it redraws the
    # same noise every time -- the first run of this test produced three
    # identical "realizations" (Jaccard 1.00 between repeats, trivially). The
    # rng state is therefore reset IN PLACE before each census (same object:
    # the winner selector holds a reference to it), giving REPEATS independent
    # tie resolutions of the same trained organ.
    eng = b._engine_for(b.areas[fsm.state_area])
    st = eng._areas[fsm.state_area]
    st.input_noise_std = NOISE
    b.areas[fsm.state_area].input_noise_std = NOISE
    noisy = []
    for i in range(REPEATS):
        eng._rng.bit_generator.state = np.random.default_rng(
            10_000 * seed + i).bit_generator.state
        soft_i, hard_i = soft_hard_census(b, fsm, symbols, group)
        noisy.append((sorted(_pairs(soft_i)), len(hard_i)))
    st.input_noise_std = 0.0
    b.areas[fsm.state_area].input_noise_std = 0.0

    sets = [set(map(tuple, p)) for p, _h in noisy]
    return {
        "n_soft0": len(s0), "n_hard0": len(hard0),
        "soft0": sorted(s0),
        "noisy_counts": [len(p) for p, _h in noisy],
        "noisy_hard": [h for _p, h in noisy],
        "jaccard_vs_det": [_jaccard(s0, s) for s in sets],
        "jaccard_noisy_pairs": [_jaccard(sets[i], sets[j])
                                for i in range(REPEATS)
                                for j in range(i + 1, REPEATS)],
    }


def main():
    print("=== S5 soft defects: bar ties or excess mass?  (PREREG_bar_tie.md)")
    print(f"    state-area input noise {NOISE} at census only, {REPEATS} "
          f"repeats, seeds {SEEDS}\n")
    r = run_tiered([(g, s, "tie") for g in GROUP_NAMES for s in SEEDS],
                   worker_fn=worker)
    print(f"    {'group':7s} {'seed':>4s} {'soft0':>5s} {'hard0':>5s} "
          f"{'noisy soft':>12s} {'noisy hard':>10s} {'J(det,noisy)':>22s}")
    js, det_total, noisy_total, hard_noisy, organs = [], 0, 0, 0, 0
    for g in GROUP_NAMES:
        for s in SEEDS:
            v = r[(g, s, "tie")]
            det_total += v["n_soft0"]
            noisy_total += sum(v["noisy_counts"])
            hard_noisy += sum(v["noisy_hard"])
            if v["n_soft0"] > 0:
                organs += 1
                js.extend(v["jaccard_vs_det"])
            print(f"    {g:7s} {s:4d} {v['n_soft0']:5d} {v['n_hard0']:5d} "
                  f"{str(v['noisy_counts']):>12s} {str(v['noisy_hard']):>10s} "
                  f"{' '.join(f'{j:.2f}' for j in v['jaccard_vs_det']):>22s}",
                  flush=True)
    # Jaccard is averaged over ORGANS x repeats of one trained brain each --
    # a mean over conditions; the verdict is a set comparison, not a seed
    # statistic with an interval.
    mean_j = float(np.mean(js)) if js else float("nan")
    ratio = noisy_total / max(REPEATS * det_total, 1)
    print(f"\n  organs with soft0>0: {organs}   mean Jaccard(det, noisy) "
          f"{mean_j:.3f}   noisy/det count ratio {ratio:.2f}   hard under "
          f"noise {hard_noisy}")
    t1 = (mean_j <= 0.5) and (0.5 <= ratio <= 2.0) and hard_noisy == 0
    t2 = (mean_j >= 0.9) and abs(ratio - 1.0) < 0.1
    print(f"  {'PASS' if t1 else 'no  '}  T1 TIE: Jaccard <= 0.5, counts "
          f"within [0.5x, 2x], zero hard")
    print(f"  {'PASS' if t2 else 'no  '}  T2 MASS: Jaccard >= 0.9, counts "
          f"unchanged")
    if not (t1 or t2):
        print("  NEITHER -- mixed population; see per-organ rows")
    path = os.path.join(_HERE, "seq_s5_bar_tie_results.json")
    with open(path, "w") as fh:
        json.dump({"seeds": SEEDS, "noise": NOISE,
                   "cells": {f"{g}/{s}": r[(g, s, "tie")]
                             for g in GROUP_NAMES for s in SEEDS},
                   "mean_jaccard": mean_j, "count_ratio": ratio,
                   "T1": t1, "T2": t2}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
