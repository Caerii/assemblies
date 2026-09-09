"""GATE-3 of DESIGN_sequence_port.md: A1's horizon on the hashed organ at width.

The registered curve -- first divergence index over 2000 random digits of
the mod-3 machine -- on 20 brains at p = 0.3 and p = 0.4, against the five
numpy seeds in `seq_a1_horizon_results.json`. Same machine (N_ARC 5000,
N_STATE 500, K 70, beta 0.1, 15 presentations, norm_init off, the Brain's
clip), same digit strings per seed (`random.Random(seed * 7919)`), the
numpy seeds 1..5 among the hashed 1..20 so a paired reading is possible.

    BAR (GATE-3). At both p, every numpy first-error index (NEVER = censored
    at 2000) lies inside the hashed [5th, 95th] percentile.
    If it fails: the sampler is the first suspect (the numpy seeds ran on
    the sampled engine; the hashed organ equals the materialized one).

    python research/experiments/seq_a1_horizon_hashed.py [--brains 20] [--smoke]
"""
from __future__ import annotations

import argparse
import inspect
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

from neural_assemblies.core.brain import Brain                            # noqa: E402
from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM  # noqa: E402
from neural_assemblies.programs.mod3_fsm import (                         # noqa: E402
    ALL_STATES, ALL_SYMBOLS, mod3_transition_table)
from seq_a1_fsm_parity import BETA, K, N_ARC, N_STATE, PRESENTATIONS      # noqa: E402

LENGTH = 2000
P_VALUES = (0.3, 0.4)
CHECKPOINTS = (10, 50, 100, 500, 1000, 2000)
STRENGTH = 0.1
_HERE = os.path.dirname(os.path.abspath(__file__))


def digit_strings(seeds, length):
    out = torch.zeros(len(seeds), length, dtype=torch.int64)
    for b, seed in enumerate(seeds):
        rng = random.Random(seed * 7919)
        out[b] = torch.tensor([rng.randrange(10) for _ in range(length)])
    return out


def run_width(seeds, p, length=LENGTH):
    w_max = inspect.signature(Brain).parameters["w_max"].default
    fsm = HashedArcFSM(seeds, ALL_STATES, ALL_SYMBOLS, mod3_transition_table(),
                       n_arc=N_ARC, n_state=N_STATE, k=K, p=p, beta=BETA,
                       refracted_strength=STRENGTH, w_max=w_max, norm_init=False,
                       max_potentiations=PRESENTATIONS * 4 + 8, prefix="_mod3")
    t0 = time.perf_counter()
    fsm.train(PRESENTATIONS)
    fsm.check()
    digits = digit_strings(seeds, length).cuda()
    # ground truth: running residue per brain
    truth = torch.cumsum(digits, dim=1) % 3
    fsm.arc.inhibit()
    fsm.cue_state("0")
    got = torch.zeros_like(digits)
    exact = torch.zeros(len(seeds), dtype=torch.int64, device="cuda")
    for t in range(length):
        got[:, t] = fsm.step(digits[:, t])
        # exact: every winner inside the true block
        blk = fsm.state.winners // K
        exact += (blk == truth[:, t].view(-1, 1)).all(dim=1).to(torch.int64)
    correct = (got == truth).cpu().numpy()
    exact = (exact.cpu().numpy() / length)
    rows = []
    for b, seed in enumerate(seeds):
        wrong = np.flatnonzero(~correct[b])
        fe = int(wrong[0]) + 1 if len(wrong) else None
        rows.append({"seed": int(seed), "p": p, "length": length, "first_error": fe,
                     "accuracy": float(correct[b].mean()),
                     "exact_fraction": float(exact[b]),
                     "prefix_correct": {c: bool(correct[b, :c].all())
                                        for c in CHECKPOINTS if c <= length}})
    print(f"    p={p}: {len(seeds)} brains, {length} steps  "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    return rows


def gate3(hashed_rows, numpy_rows, p, length=LENGTH):
    """Numpy seeds inside the hashed [5, 95] percentile; NEVER is length + 1."""
    def val(r):
        return float(length + 1 if r["first_error"] is None else r["first_error"])
    h = np.array([val(r) for r in hashed_rows if r["p"] == p])
    lo, hi = np.percentile(h, 5), np.percentile(h, 95)
    verdicts = []
    for r in numpy_rows:
        if r["p"] != p:
            continue
        v = val(r)
        verdicts.append((r["seed"], r["first_error"], bool(lo <= v <= hi)))
    return lo, hi, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    length = 50 if args.smoke else LENGTH
    seeds = list(range(1, 1 + (3 if args.smoke else args.brains)))
    if args.smoke:
        print("*** SMOKE: API only. THESE NUMBERS ARE VOID. ***")
    with open(os.path.join(_HERE, "seq_a1_horizon_results.json")) as fh:
        numpy_rows = json.load(fh)
    print(f"=== GATE-3: the horizon at width ({len(seeds)} brains, {length} digits) ===")
    rows, out = [], {"brains": seeds, "length": length, "rows": []}
    for p in P_VALUES:
        got = run_width(seeds, p, length)
        rows += got
        never = sum(r["first_error"] is None for r in got)
        fes = sorted(r["first_error"] for r in got if r["first_error"] is not None)
        print(f"      first errors: {fes}  NEVER: {never}/{len(got)}")
        for c in CHECKPOINTS:
            if c <= length:
                n_ok = sum(r["prefix_correct"].get(c, False) for r in got)
                print(f"         first {c:>5} steps perfect: {n_ok}/{len(got)}")
        lo, hi, verdicts = gate3(got, numpy_rows, p, length)
        print(f"      hashed [5th, 95th] = [{lo:.0f}, {hi:.0f}]")
        for seed, fe, ok in verdicts:
            print(f"        numpy seed {seed}: first error {fe if fe else 'NEVER':>6}  "
                  f"{'inside' if ok else 'OUTSIDE'}")
        out[f"gate3_p{p}"] = {"lo": lo, "hi": hi,
                              "verdicts": [(s, fe, ok) for s, fe, ok in verdicts]}
    out["rows"] = rows
    allok = all(ok for p in P_VALUES for _, _, ok in out[f"gate3_p{p}"]["verdicts"])
    print(f"\n  GATE-3 {'PASS' if allok else 'FAIL'}"
          + ("" if not args.smoke else "  (SMOKE: VOID)"))
    if not args.smoke:
        path = os.path.join(_HERE, "seq_a1_horizon_results_hashed.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
