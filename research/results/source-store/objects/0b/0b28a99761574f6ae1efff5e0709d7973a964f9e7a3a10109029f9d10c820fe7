"""A1 follow-up: decompose the sequence failure into per-STEP error.

A1 (seq_a1_fsm_parity.py) read P-CONJ 0.000/0.000 -- the arc is a perfect
conjunction, 33 disjoint assemblies -- while P-GOLD came in at 5/10 against a
bar of 8/10. So the (state, symbol) -> arc step works and the failure is in
arc -> next state, or in error accumulating along a sequence.

PREDICTION, STATED BEFORE THE NUMBERS. If single steps are independent and each
is correct with probability a, deciding BOTH sequences needs 6 + 4 = 10 correct
steps, so the observed 5/10 implies a ~ 0.5^(1/10) = 0.93. Therefore:

  * per-step accuracy in 0.88-0.97 -> failure IS per-step error accumulating,
    and the fix is whatever raises single-step reliability. The state area is
    the named suspect: its afferent kp from the arc is k*p = 14 against its own
    floor 3 ln 500 = 18.6, i.e. BELOW the regime floor the theory requires,
    while the arc sits at 28 against 25.6 and passes.
  * per-step accuracy > 0.99 -> error accumulation is NOT the story; look for a
    specific broken transition (the `end` symbol decides accept/reject and is
    the obvious candidate) rather than a regime problem.
  * per-step accuracy < 0.85 -> single steps are barely working and the
    sequence result is incidental.

Reported per symbol class, because `end` transitions are the ones that produce
the verdict and a failure concentrated there would look identical in aggregate.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from neural_assemblies.programs.mod3_fsm import END_SYMBOL, mod3_transition_table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_a1_fsm_parity import PRESENTATIONS, SEEDS, build


def step_accuracy(brain, fsm):
    """Accuracy of every single transition, taken one step at a time."""
    rows = []
    for from_state, symbol, to_state in mod3_transition_table():
        got = fsm.run([symbol], start_state=from_state)[0]
        rows.append({"from": from_state, "symbol": symbol, "to": to_state,
                     "got": got, "correct": got == to_state})
    return rows


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    all_rows, per_seed = [], []
    for seed in seeds:
        brain, fsm = build(seed, presentations=PRESENTATIONS)
        rows = step_accuracy(brain, fsm)
        acc = float(np.mean([r["correct"] for r in rows]))
        per_seed.append(acc)
        all_rows.extend({"seed": seed, **r} for r in rows)
        print(f"  seed {seed:2d}: per-step accuracy {acc:.3f} "
              f"({sum(r['correct'] for r in rows)}/{len(rows)})", flush=True)

    acc = float(np.mean(per_seed))
    sd = float(np.std(per_seed, ddof=1))
    digits = [r for r in all_rows if r["symbol"] != END_SYMBOL]
    ends = [r for r in all_rows if r["symbol"] == END_SYMBOL]

    print(f"\n  per-step accuracy      {acc:.3f} +/- {sd:.3f} over {len(per_seed)} seeds")
    print(f"    digit transitions    {np.mean([r['correct'] for r in digits]):.3f} "
          f"(n={len(digits)})")
    print(f"    'end' transitions    {np.mean([r['correct'] for r in ends]):.3f} "
          f"(n={len(ends)})")
    print(f"\n  implied P(decide both) = a^10 = {acc ** 10:.3f}; A1 observed 0.500")

    # Where does a wrong step land? A near-miss (adjacent residue) and a
    # collapse (always the same state) are different failures.
    wrong = [r for r in all_rows if not r["correct"]]
    if wrong:
        landing = {}
        for r in wrong:
            landing[r["got"]] = landing.get(r["got"], 0) + 1
        print("\n  wrong steps land on: "
              + ", ".join(f"{k}={v}" for k, v in sorted(landing.items(),
                                                        key=lambda kv: -kv[1])))

    from _results import write_result
    out = write_result(
        "sequence", "seq_a1_step_accuracy_results.json",
        {"per_seed": per_seed, "mean": acc, "sd": sd,
         "digit_accuracy": float(np.mean([r["correct"] for r in digits])),
         "end_accuracy": float(np.mean([r["correct"] for r in ends])),
         "rows": all_rows},
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
