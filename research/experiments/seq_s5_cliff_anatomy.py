"""Anatomy of the cliff: defect set or drift? [[SEQ-EXACT-RECOVERY]]

Implements `research/notes/PREREG_s5_cliff_anatomy.md`. The trained organs are
IDENTICAL to the registered S5 study -- same `build`, same seeds, same words --
and only READOUTS are added, all inside `probe()`.

The competing accounts, separable by one zero-parameter prediction:

  H-defect  a small static set of transitions is trained WRONG; first_bad is
            the first step the word exercises one. The census (E3) finds them,
            and E4 predicts first_bad EXACTLY per seed, nothing fitted.
  H-drift   no edge is individually broken; the live state assembly deviates
            from its stored block between visits. The census is clean and the
            on-block trajectory (E2) carries the signal instead.
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
    GROUPS, true_trajectory, word_problem_fsm,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seq_s5_word_problem import (  # noqa: E402
    GROUP_NAMES, LENGTHS, SEEDS, build, run_tiered,
)

LONGEST = max(LENGTHS)


def _traced_run(fsm, word, start):
    """`NemoArcFSM.run`, with the live state assembly recorded per step.

    Mirrors `run` exactly -- probe(), inhibit, cue, step -- because the point
    is to observe the registered protocol, not a variant of it.
    """
    b = fsm.brain
    labels, snaps = [], []
    with b.probe():
        b.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(start)
        fsm._unfix_state()
        for sym in word:
            labels.append(fsm.step(sym))
            snaps.append(_snap(b, fsm.state_area))
    return labels, snaps


def _census(fsm, group, symbols):
    """E3: every (state, symbol) single-step, from the exact stored block.

    Returns per-pair records: correct?, margin (best minus second overlap),
    and the produced label. Runs inside probe() like everything else.
    """
    _states, _syms, transitions = word_problem_fsm(group)
    table = {(fr, sym): to for fr, sym, to in transitions}
    rows = []
    for st in fsm.states:
        for sym in symbols:
            got = fsm.run([sym], start_state=st)[0]
            live = _snap(fsm.brain, fsm.state_area)
            ovs = sorted((assembly_overlap(live.winners,
                           fsm.state_assembly(s2).winners), s2)
                         for s2 in fsm.states)
            (best_ov, _b), (second_ov, _s) = ovs[-1], ovs[-2]
            rows.append({
                "state": st, "symbol": sym, "expected": table[(st, sym)],
                "got": got, "ok": got == table[(st, sym)],
                "margin": float(best_ov - second_ov),
                "best_overlap": float(best_ov),
            })
    return rows, table


def _predict_first_bad(word, start, table, bad_pairs):
    """E4: first step whose (true state, symbol) is a censused defect."""
    prev = start
    for i, sym in enumerate(word):
        if (prev, sym) in bad_pairs:
            return i
        prev = table[(prev, sym)]
    return LONGEST


def worker(group_name, seed, _arm="trained"):
    group, fsm, symbols = build(group_name, seed, "trained")
    rng = random.Random(seed + 4242)          # the registered word, verbatim
    word = [rng.choice(symbols) for _ in range(LONGEST)]
    truth = true_trajectory(group, word)
    start = group.label(group.identity)

    # E1: determinism.
    labels1, snaps = _traced_run(fsm, word, start)
    labels2, _ = _traced_run(fsm, word, start)
    deterministic = labels1 == labels2

    first_bad = next((i for i, (a, b) in enumerate(zip(labels1, truth))
                      if a != b), LONGEST)

    # E2: on-block overlap of the live state with the block the readout chose,
    # pre-derailment; plus whether the post-derailment tail ever re-enters ANY
    # block exactly (the absorption claim).
    pre = [assembly_overlap(s.winners, fsm.state_assembly(l).winners)
           for l, s in zip(labels1[:first_bad], snaps[:first_bad])]
    tail_reentry = any(
        assembly_overlap(s.winners, fsm.state_assembly(l).winners) == 1.0
        for l, s in zip(labels1[first_bad + 1:], snaps[first_bad + 1:]))

    # E3 + E4.
    census, table = _census(fsm, group, symbols)
    bad = [(r["state"], r["symbol"]) for r in census if not r["ok"]]
    predicted = _predict_first_bad(word, start, table, set(bad))

    # E5: margins, split by censused correctness.
    ok_margin = [r["margin"] for r in census if r["ok"]]
    bad_rows = [r for r in census if not r["ok"]]

    return {
        "deterministic": bool(deterministic),
        "first_bad": int(first_bad),
        "predicted_first_bad": int(predicted),
        "prediction_exact": bool(predicted == first_bad),
        "pre_onblock_min": float(min(pre)) if pre else float("nan"),
        "pre_onblock_mean": float(np.mean(pre)) if pre else float("nan"),
        "tail_reentry": bool(tail_reentry),
        "n_defects": len(bad),
        "defects": [list(p) for p in bad],
        "defect_margins": [r["margin"] for r in bad_rows],
        "defect_best_overlap": [r["best_overlap"] for r in bad_rows],
        "ok_margin_min": float(min(ok_margin)) if ok_margin else float("nan"),
        "ok_margin_mean": float(np.mean(ok_margin)) if ok_margin else float("nan"),
    }


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== anatomy of the cliff: defect set or drift? ===")
    print(f"    same builds/seeds/words as the registered S5 study; "
          f"readouts only\n")

    cells = [(g, s, "trained") for g in GROUP_NAMES for s in seeds]
    r = run_tiered(cells, worker_fn=worker)

    out = {"seeds": seeds, "groups": {}}
    c1 = c2 = c3 = c4 = True
    print(f"\n    {'group':7s} {'seed':>4s} {'det':>4s} {'first_bad':>9s} "
          f"{'predicted':>9s} {'exact':>6s} {'defects':>8s} "
          f"{'onblock_min':>11s} {'reentry':>7s}")
    for g in GROUP_NAMES:
        rows = []
        for s in seeds:
            v = r[(g, s, "trained")]
            rows.append(v)
            c1 &= v["deterministic"]
            if v["first_bad"] > 0:
                c2 &= v["pre_onblock_min"] == 1.0
            c3 &= v["prediction_exact"]
            if g != "S5":
                c4 &= v["n_defects"] <= 2
            fb = v["first_bad"]
            print(f"    {g:7s} {s:4d} {str(v['deterministic'])[0]:>4s} "
                  f"{fb:9d} {v['predicted_first_bad']:9d} "
                  f"{str(v['prediction_exact'])[0]:>6s} {v['n_defects']:8d} "
                  f"{v['pre_onblock_min']:11.4f} "
                  f"{str(v['tail_reentry'])[0]:>7s}", flush=True)
        out["groups"][g] = rows

    print("\n=== BARS ===")
    for name, ok in (("C1 determinism", c1),
                     ("C2 pre-derailment states exactly on-block", c2),
                     ("C3 census predicts first_bad exactly, every seed", c3),
                     ("C4 defect count small (order-60 groups)", c4)):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = {"C1": c1, "C2": c2, "C3": c3, "C4": c4}

    path = os.path.join(_HERE, "seq_s5_cliff_anatomy_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
