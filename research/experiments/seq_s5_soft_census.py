"""E7: the soft census, with a LIVE instrument. [[SEQ-EXACT-RECOVERY]]

Implements Addendum 2 of `research/notes/PREREG_s5_cliff_anatomy.md`. The
previous census snapped AFTER `fsm.run`'s internal probe returned; `probe()`
restores winners on exit, so it read the same residue 240 times (margin
1.0000, constant -- the dead-probe signature). Here every snap happens INSIDE
the probe that produced it.

soft defect := transition whose label is CORRECT but whose output assembly is
not exactly the intended block. The zero-parameter prediction V2: per seed,
the first deviation step of the replayed word equals the first step its TRUE
path visits a soft pair.
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
    GROUP_NAMES, SEEDS, build, run_tiered,
)

LONGEST = 500


def _ov(a, b):
    return assembly_overlap(np.asarray(a.winners), np.asarray(b.winners))


def _single_step(fsm, state, sym):
    """One census step, everything read INSIDE the probe that computed it."""
    b = fsm.brain
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(state)
        fsm._unfix_state()
        label = fsm.step(sym)
        live = _snap(b, fsm.state_area)
    return label, live


def worker(group_name, seed, _arm="trained"):
    group, fsm, symbols = build(group_name, seed, "trained")
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
    soft, hard, exact_ov = [], [], []
    for st in fsm.states:
        for sym in symbols:
            label, live = _single_step(fsm, st, sym)
            ov = _ov(live, fsm.state_assembly(table[(st, sym)]))
            exact_ov.append(ov)
            if label != table[(st, sym)]:
                hard.append((st, sym))
            elif ov < 1.0:
                soft.append(((st, sym), float(ov)))

    bad_pairs = set(p for p, _o in soft) | set(hard)
    prev, predicted = start, None
    for i, sym in enumerate(word):
        if (prev, sym) in bad_pairs:
            predicted = i
            break
        prev = table[(prev, sym)]

    ovs = np.asarray(exact_ov)
    return {
        "first_bad": int(first_bad),
        "first_dev": first_dev,
        "predicted_first_dev": predicted,
        "v2_exact": bool(first_dev == predicted),
        "n_soft": len(soft),
        "n_hard": len(hard),
        "soft_overlaps": sorted(float(o) for _p, o in soft),
        "census_ov_min": float(ovs.min()),
        "census_ov_varies": bool(ovs.min() < ovs.max()),
        "onblock_min": float(min(onblock)),
    }


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== E7: soft census, snapped inside the probe ===\n")
    cells = [(g, s, "trained") for g in GROUP_NAMES for s in seeds]
    r = run_tiered(cells, worker_fn=worker)

    print(f"\n    {'group':7s} {'seed':>4s} {'first_bad':>9s} {'dev':>5s} "
          f"{'pred_dev':>8s} {'V2':>3s} {'soft':>5s} {'hard':>5s} "
          f"{'min_ov':>7s}")
    out, v1_alive, v1_soft, v2 = [], True, True, True
    for g in GROUP_NAMES:
        for s in seeds:
            v = r[(g, s, "trained")]
            out.append({"group": g, "seed": s, **v})
            dev = "-" if v["first_dev"] is None else str(v["first_dev"])
            pred = ("-" if v["predicted_first_dev"] is None
                    else str(v["predicted_first_dev"]))
            v2 &= v["v2_exact"]
            if v["first_dev"] is not None:
                v1_soft &= (v["n_soft"] + v["n_hard"]) > 0
            print(f"    {g:7s} {s:4d} {v['first_bad']:9d} {dev:>5s} "
                  f"{pred:>8s} {str(v['v2_exact'])[0]:>3s} {v['n_soft']:5d} "
                  f"{v['n_hard']:5d} {v['census_ov_min']:7.4f}", flush=True)

    print("\n=== BARS ===")
    alive = any(x["census_ov_varies"] for x in out)
    print(f"  {'PASS' if alive else 'FAIL'}  V1a instrument alive "
          f"(census overlaps vary somewhere)")
    print(f"  {'PASS' if v1_soft else 'FAIL'}  V1b deviating seeds have "
          f"soft/hard pairs")
    print(f"  {'PASS' if v2 else 'FAIL'}  V2 first_dev == first true-path "
          f"visit to a bad pair, every seed")

    path = os.path.join(_HERE, "seq_s5_soft_census_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
