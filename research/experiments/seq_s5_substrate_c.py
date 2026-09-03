"""Substrate C: the census under the theorems' actual homeostasis hypothesis.

Implements `research/notes/PREREG_substrate_c_homeostasis.md` (Amendment 1):
`Brain(norm_init=False, synaptic_scaling=True)` -- per-update write-time
column renormalization, the existing engine mechanism, no new code. Same
organs, seeds, words, census instrument as the registered S5 study and
e73e493. Judged against:

  HC1  zero hard defects on all 40 organs
  HC2  total soft pairs < substrate A's registered 30
  HC3  (conditional on HC2) per-group exact@500 >= registered table
  HC4  full defect anatomy recorded (overlaps, intruders, displaced)

Substrate A control is cited from e73e493's N1 PASS (same engine commit),
not re-run.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.diagnostics import assembly_overlap
from neural_assemblies.programs.word_problems import (
    true_trajectory, word_problem_fsm,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seq_s5_word_problem import (  # noqa: E402
    GROUP_NAMES, SEEDS, build, run_tiered, soft_hard_census,
)

LONGEST = 500

# Registered comparisons (seq_s5_word_problem_results.json, aeb17ce; census
# totals bb5195a / e73e493-N1).
REGISTERED_SOFT_TOTAL = 30
REGISTERED_EXACT500 = {"Z60": 7, "A4xZ5": 9, "A5": 7, "S5": 5}

#: Amendment 2 (AUDIT_refraction_scaling.md): `synaptic_scaling=True` also
#: scaled the REFRACTED arc, and refraction + scaling on one area destroys its
#: assemblies within ~10 presentations. `--scoped` confines scaling to the
#: state area, which is the only area substrate C was ever meant to act on.
SCOPED = "--scoped" in sys.argv
SCALING = frozenset({"_wp_state"}) if SCOPED else True


def _ov(a, b):
    return assembly_overlap(np.asarray(a.winners), np.asarray(b.winners))


def worker(group_name, seed, _arm="homeo"):
    group, fsm, symbols = build(group_name, seed, "trained",
                                norm_init=False, synaptic_scaling=SCALING)
    b = fsm.brain
    rng = random.Random(seed + 4242)
    word = [rng.choice(symbols) for _ in range(LONGEST)]
    start = group.label(group.identity)
    truth = true_trajectory(group, word)

    labels, onblock = [], []
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(start)
        fsm._unfix_state()
        for sym in word:
            labels.append(fsm.step(sym))
            onblock.append(_ov(_snap(b, fsm.state_area),
                               fsm.state_assembly(labels[-1])))
    first_bad = next((i for i, (a, t) in enumerate(zip(labels, truth))
                      if a != t), LONGEST)
    first_dev = next((i for i, o in enumerate(onblock) if o < 1.0), None)

    _states, _syms, transitions = word_problem_fsm(group)
    table = {(fr, sym): to for fr, sym, to in transitions}
    soft, hard = soft_hard_census(b, fsm, symbols, group)

    bad_pairs = {tuple(rec["pair"]) for rec in soft} | set(hard)
    prev, predicted = start, None
    for i, sym in enumerate(word):
        if (prev, sym) in bad_pairs:
            predicted = i
            break
        prev = table[(prev, sym)]

    exact = {str(L): bool(labels[:L] == truth[:L]) for L in (10, 50, 100, 500)}
    return {
        "first_bad": int(first_bad), "first_dev": first_dev,
        "predicted_first_dev": predicted,
        "v2_exact": bool(first_dev == predicted),
        "n_soft": len(soft), "n_hard": len(hard), "soft": soft,
        "exact": exact,
    }


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = SEEDS[:int(argv[0])] if argv else SEEDS
    print("=== substrate C: census under per-round homeostasis ===")
    print(f"    norm_init=False, synaptic_scaling={SCALING!r} (per-update)"
          + ("  [SCOPED to the state area -- Amendment 2]" if SCOPED else "")
          + "\n")
    r = run_tiered([(g, s, "homeo") for g in GROUP_NAMES for s in seeds],
                   worker_fn=worker)

    out, total_soft, total_hard = {}, 0, 0
    exact500 = {}
    print(f"\n    {'group':7s} {'seed':>4s} {'soft':>5s} {'hard':>5s} "
          f"{'first_bad':>9s} {'exact@500':>9s}")
    for g in GROUP_NAMES:
        e500 = 0
        for s in seeds:
            v = r[(g, s, "homeo")]
            out[f"{g}/{s}"] = v
            total_soft += v["n_soft"]
            total_hard += v["n_hard"]
            e500 += v["exact"]["500"]
            print(f"    {g:7s} {s:4d} {v['n_soft']:5d} {v['n_hard']:5d} "
                  f"{v['first_bad']:9d} {str(v['exact']['500']):>9s}",
                  flush=True)
        exact500[g] = e500

    print("\n=== BARS ===")
    hc1 = total_hard == 0
    print(f"  {'PASS' if hc1 else 'FAIL'}  HC1 zero hard defects "
          f"(total {total_hard})")
    hc2 = total_soft < REGISTERED_SOFT_TOTAL
    print(f"  {'PASS' if hc2 else 'FAIL'}  HC2 soft total {total_soft} < "
          f"registered {REGISTERED_SOFT_TOTAL}")
    hc3 = all(exact500[g] >= REGISTERED_EXACT500[g] for g in GROUP_NAMES) \
        if hc2 else None
    print(f"  {('PASS' if hc3 else 'FAIL') if hc3 is not None else 'VOID'}"
          f"  HC3 exact@500 >= registered per group  {exact500} vs "
          f"{REGISTERED_EXACT500}")

    ovs = [rec["overlap"] for v in out.values() for rec in v["soft"]]
    if ovs:
        print(f"  HC4 soft anatomy: n={len(ovs)} overlap "
              f"min {min(ovs):.3f} med {float(np.median(ovs)):.3f}")
    else:
        print("  HC4 soft anatomy: no soft pairs at all")

    payload = {"seeds": seeds, "cells": out, "total_soft": total_soft,
               "total_hard": total_hard, "exact500": exact500,
               "verdicts": {"HC1": hc1, "HC2": hc2, "HC3": hc3}}
    path = os.path.join(_HERE, "seq_s5_substrate_c_results"
                        + ("_scoped" if SCOPED else "") + ".json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
