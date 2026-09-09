"""The theorem regime: homeostasis where its preconditions actually hold.

Implements `research/notes/substrate/PREREG_theorem_regime.md`. Matched pair at
organ_p=0.5 (kp=35 clears 3 ln n), presentations=64 (clears (1/b)ln(n/k)),
w_max=None (the theorems have no clip; homeostasis is their boundedness):

    A'  synaptic_scaling=False   unbounded Hebbian, the honest control
    C'  synaptic_scaling=True    the theorems' substrate

Census is the primary instrument (exact@L descriptive, tie-fragile).
TR4 records stored-weight anatomy: A' must grow ~(1.1)^64 unbounded while
C' stays bounded -- the mechanical half of the claim that deep training
is only reachable under homeostasis.
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
ORGAN_P, PRESENTATIONS = 0.5, 64
REGISTERED_SOFT_TOTAL = 30          # substrate A at its own protocol (bb5195a)


def _ov(a, b):
    return assembly_overlap(np.asarray(a.winners), np.asarray(b.winners))


def worker(group_name, seed, arm):
    scaling = arm == "homeo"
    group, fsm, symbols = build(
        group_name, seed, "trained", norm_init=False,
        synaptic_scaling=scaling, organ_p=ORGAN_P,
        presentations=PRESENTATIONS, w_max=None)
    b = fsm.brain
    eng = b._engine
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
    soft, hard = [], []
    for st in fsm.states:
        for sym in symbols:
            with b.probe():
                b.inhibit_areas([fsm.arc_area, fsm.state_area])
                fsm._cue_state(st)
                fsm._unfix_state()
                label = fsm.step(sym)
                live = _snap(b, fsm.state_area)
            intended = fsm.state_assembly(table[(st, sym)])
            if label != table[(st, sym)]:
                hard.append((st, sym))
                continue
            got = set(np.asarray(live.winners).tolist())
            want = set(np.asarray(intended.winners).tolist())
            if got != want:
                soft.append({
                    "pair": [st, sym],
                    "overlap": len(got & want) / len(want),
                    "intruders": sorted(got - want),
                    "displaced": sorted(want - got),
                })

    # TR4: stored-weight anatomy on the arc->state fiber.
    conn = eng._area_conns[fsm.arc_area][fsm.state_area]
    w = conn.weights
    # Sanctioned materialized-count accessor (index-space ratchet): names the
    # quantity (neurons that exist) instead of the ambiguous `.w`, bounded by
    # the connectome's physical shape so it cannot overrun.
    rows = min(int(eng.materialized_count(fsm.arc_area)), w.shape[0])
    W = np.asarray(w.todense() if hasattr(w, "todense") else w,
                   dtype=np.float64)[:rows]
    cols = getattr(conn, "_log_cols", None) or W.shape[1]
    W = W[:, :cols]
    colsum = W.sum(axis=0)

    exact = {str(L): bool(labels[:L] == truth[:L]) for L in (10, 50, 100, 500)}
    return {
        "arm": arm, "first_bad": int(first_bad), "first_dev": first_dev,
        "n_soft": len(soft), "n_hard": len(hard), "soft": soft,
        "exact": exact,
        "w_stored_max": float(W.max()),
        "colsum_med": float(np.median(colsum)),
        "colsum_max": float(colsum.max()),
    }


def _print_arm(r, arm, seeds):
    total_soft = total_hard = 0
    exact500, wmaxes = {}, []
    print(f"\n    {'group':7s} {'seed':>4s} {'soft':>5s} {'hard':>5s} "
          f"{'first_bad':>9s} {'e@500':>5s} {'w_max':>8s} {'colsum_med':>11s}")
    for g in GROUP_NAMES:
        e500 = 0
        for s in seeds:
            v = r[(g, s, arm)]
            total_soft += v["n_soft"]
            total_hard += v["n_hard"]
            e500 += v["exact"]["500"]
            wmaxes.append(v["w_stored_max"])
            print(f"    {g:7s} {s:4d} {v['n_soft']:5d} {v['n_hard']:5d} "
                  f"{v['first_bad']:9d} {str(v['exact']['500'])[0]:>5s} "
                  f"{v['w_stored_max']:8.1f} {v['colsum_med']:11.0f}",
                  flush=True)
        exact500[g] = e500
    return total_soft, total_hard, exact500, wmaxes


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== theorem regime: organ_p=0.5, T=64, w_max=None ===\n")
    cells = [(g, s, a) for a in ("raw", "homeo")
             for g in GROUP_NAMES for s in seeds]
    r = run_tiered(cells, worker_fn=worker)

    print("\n--- A' (raw, unbounded Hebbian)")
    a_soft, a_hard, a_e500, a_w = _print_arm(r, "raw", seeds)
    print("\n--- C' (homeostatic)")
    c_soft, c_hard, c_e500, c_w = _print_arm(r, "homeo", seeds)

    print("\n=== BARS ===")
    tr1 = a_hard == 0 and c_hard == 0
    print(f"  {'PASS' if tr1 else 'FAIL'}  TR1 zero hard defects "
          f"(A' {a_hard}, C' {c_hard})")
    tr2 = c_soft < REGISTERED_SOFT_TOTAL
    print(f"  {'PASS' if tr2 else 'FAIL'}  TR2 C' soft {c_soft} < "
          f"registered A's {REGISTERED_SOFT_TOTAL}")
    tr3 = c_soft <= a_soft
    print(f"  {'PASS' if tr3 else 'FAIL'}  TR3 C' soft {c_soft} <= "
          f"A' soft {a_soft}")
    tr4 = max(c_w) < 60.0 and max(a_w) > 100.0
    print(f"  {'PASS' if tr4 else 'FAIL'}  TR4 boundedness: A' max stored "
          f"{max(a_w):.0f} (>100), C' max stored {max(c_w):.1f} (<60)")

    payload = {
        "seeds": seeds, "params": {"organ_p": ORGAN_P,
                                   "presentations": PRESENTATIONS,
                                   "w_max": None},
        "raw": {f"{g}/{s}": r[(g, s, "raw")]
                for g in GROUP_NAMES for s in seeds},
        "homeo": {f"{g}/{s}": r[(g, s, "homeo")]
                  for g in GROUP_NAMES for s in seeds},
        "totals": {"A_soft": a_soft, "C_soft": c_soft,
                   "A_exact500": a_e500, "C_exact500": c_e500},
        "verdicts": {"TR1": tr1, "TR2": tr2, "TR3": tr3, "TR4": tr4},
    }
    path = os.path.join(_HERE, "seq_s5_theorem_regime_results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
