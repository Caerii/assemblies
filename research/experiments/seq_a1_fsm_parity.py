"""A1: the transition organ, end-to-end, on our Brain, against the golden.

Pre-registered in `research/notes/sequence/PREREG_seq_a1_fsm_parity.md` (commit 040f201,
amended before data with n_state=5000 and the reason). Design evidence is
`research/notes/sequence/the_arc_collapses_onto_whichever_conjunct_fires_more.md`
(d5e64f8). Enabling fixes: de7a32d (engine), 7588a44 (program).

BARS, restated here so a reader of the output does not have to fetch the note:

  P-GOLD   >= 8/10 seeds decide BOTH sequences (positive -> accept, negative ->
           reject), read out of the state ASSEMBLY. Reference: 3/3.
  P-CONJ   mean across-STATE and across-SYMBOL arc overlap BOTH < 0.15.
           Reference 0.000/0.000; #92 read 0.90-0.99; chance is k/n = 0.014.
           Both directions are required: one alone cannot tell a conjunction
           from a collapse onto the other conjunct.
  P-NULL   refraction off must FAIL: task <= 2/10 AND across-symbol > 0.5.
           If the null also passes P-GOLD, the whole result is VOID.
  P-DEGEN  zero-presentation and beta=0 arms both <= 2/10.
  P-PRE    pairwise overlap of the five state assemblies < 0.05, else the
           readout is confounded and P-GOLD is void.

Also reported regardless of outcome: the refraction-rule A/B (proportional vs
constant, on our Brain) and the max synaptic weight per arm, because our w_max
clamp has no counterpart in the reference and would make our null arm not the
reference's null.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from research.json_documents import write_new_document

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import ensemble
from neural_assemblies.programs.mod3_fsm import (
    ALL_STATES, DIGIT_SYMBOLS, build_mod3_fsm, run_digit_sequence,
    train_mod3_fsm,
)

SEEDS = tuple(range(1, 11))
N_ARC = 5000
N_STATE = 500
K = 70
P = 0.2
BETA = 0.1
PRESENTATIONS = 15
STRENGTH = 0.1
POSITIVE = (3, 0, 4, 7, 1, 10)
NEGATIVE = (6, 7, 3, 10)
RESIDUES = ("0", "1", "2")


@contextlib.contextmanager
def constant_refraction(enabled: bool):
    """Run under the pre-correction constant-increment rule (for the A/B)."""
    key = "ASSEMBLIES_CONSTANT_REFRACTION"
    old = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def build(seed, *, presentations=PRESENTATIONS, beta=BETA, strength=STRENGTH):
    """Train one mod-3 FSM. `strength=0` is the refraction-off null."""
    random.seed(seed)
    np.random.seed(seed)
    # norm_init=False pins the reference's substrate: `FSMNetwork` defaults to
    # raw Bernoulli(p) weights, and norm_init exists to stop SELF-recurrence
    # collapsing -- which this organ has none of, both areas being feed-forward.
    # See [[norm-init-substrate-vs-reference]]: parity reproductions pin False.
    brain = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse",
                  norm_init=False)
    fsm = build_mod3_fsm(brain, n=N_ARC, k=K, n_state=N_STATE, beta=beta,
                         refracted_strength=strength)
    if strength == 0.0:
        # The null. Disable rather than omit, so every other construction step
        # -- area sizes, stimulus draws, RNG consumption -- is identical.
        brain.set_refracted(fsm.arc_area, False)
    train_mod3_fsm(fsm, presentations=presentations)
    return brain, fsm


def arc_assembly(brain, fsm, state, symbol):
    """Arc winners for one (state, symbol), read without learning or growth."""
    with brain.probe():
        brain.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state(state)
        brain.project({fsm._sym_stim[symbol]: [fsm.arc_area]},
                      {fsm.state_area: [fsm.arc_area]})
        return _snap(brain, fsm.arc_area)


def collapse_overlaps(brain, fsm):
    """(across-state, across-symbol) mean pairwise arc overlap."""
    across_state = []
    for d in DIGIT_SYMBOLS:
        a = [arc_assembly(brain, fsm, q, d) for q in RESIDUES]
        across_state += [overlap(a[i], a[j])
                         for i in range(len(a)) for j in range(i + 1, len(a))]
    across_symbol = []
    for q in RESIDUES:
        a = [arc_assembly(brain, fsm, q, d) for d in DIGIT_SYMBOLS]
        across_symbol += [overlap(a[i], a[j])
                          for i in range(len(a)) for j in range(i + 1, len(a))]
    return float(np.mean(across_state)), float(np.mean(across_symbol))


def state_assembly_overlap(fsm):
    """P-PRE: mean pairwise overlap of the five stored state assemblies."""
    asms = [fsm.state_assembly(s) for s in ALL_STATES]
    return float(np.mean([overlap(asms[i], asms[j])
                          for i in range(len(asms))
                          for j in range(i + 1, len(asms))]))


def max_weight(brain, fsm):
    """Largest synaptic weight into the arc; our w_max clamp has no reference
    counterpart, so a clamped arm is not the reference's arm."""
    peak = 0.0
    for conns in (brain.connectomes.get(fsm.arc_area, {}),):
        for conn in conns.values():
            w = getattr(conn, "weights", None)
            if w is not None and np.size(w):
                peak = max(peak, float(np.max(np.asarray(w))))
    return peak


def trial(seed, **kwargs):
    t0 = time.time()
    brain, fsm = build(seed, **kwargs)
    pos, pos_traj = run_digit_sequence(fsm, list(POSITIVE))
    neg, _ = run_digit_sequence(fsm, list(NEGATIVE))
    a_state, a_symbol = collapse_overlaps(brain, fsm)
    return {
        "seed": seed,
        "positive_final": pos,
        "negative_final": neg,
        "decided": bool(pos == "accept" and neg == "reject"),
        "positive_trajectory": pos_traj,
        "across_state": a_state,
        "across_symbol": a_symbol,
        "state_assembly_overlap": state_assembly_overlap(fsm),
        "max_weight": max_weight(brain, fsm),
        "seconds": round(time.time() - t0, 1),
    }


def run_arm(name, seeds, *, constant=False, **kwargs):
    rows = []
    with constant_refraction(constant):
        for seed in seeds:
            row = trial(seed, **kwargs)
            rows.append(row)
            print(f"    seed {seed:2d}: pos={row['positive_final']:<6s} "
                  f"neg={row['negative_final']:<6s} "
                  f"{'DECIDED' if row['decided'] else '      .'}  "
                  f"arc across-state {row['across_state']:.3f} "
                  f"across-symbol {row['across_symbol']:.3f}  "
                  f"({row['seconds']}s)", flush=True)
    return {"arm": name, "constant_rule": constant, "params": kwargs, "rows": rows}


def summarize(arm):
    """Overlap statistics carry a 95% interval; decisions stay counts.

    P-CONJ and P-PRE compare a per-seed statistic against a threshold, and a
    mean over seeds with no interval cannot support that comparison. The
    decision bars (P-GOLD, P-NULL, P-DEGEN) are counts of seeds and need none.
    """
    rows = arm["rows"]
    seeds = [r["seed"] for r in rows]

    def ens(key):
        by_seed = {r["seed"]: r[key] for r in rows}
        return ensemble(lambda sd: by_seed[sd], seeds, f"{arm['arm']}:{key}")

    state, symbol, pre = (ens("across_state"), ens("across_symbol"),
                          ens("state_assembly_overlap"))
    return {
        "arm": arm["arm"], "n": len(rows),
        "decided": sum(r["decided"] for r in rows),
        "across_state": state.mean, "across_state_ci": state.ci,
        "across_symbol": symbol.mean, "across_symbol_ci": symbol.ci,
        "state_assembly_overlap": pre.mean, "state_assembly_overlap_ci": pre.ci,
        "max_weight": max(r["max_weight"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 2 presentations: API check only, numbers VOID")
    ap.add_argument("--p", type=float, default=None,
                    help="connection probability; the state area's regime floor "
                         "is 3 ln n_state / k = 0.266")
    ap.add_argument("--out", default="research/experiments/seq_a1_fsm_parity_results.json")
    args = ap.parse_args()

    if args.p is not None:
        globals()["P"] = args.p

    if args.smoke:
        print("SMOKE RUN -- checks API breakage only. Numbers are VOID.")
        seeds, presentations = SEEDS[:1], 2
    else:
        seeds, presentations = SEEDS[:args.seeds], PRESENTATIONS

    print(f"A1: mod-3 FSM through our Brain. n_arc={N_ARC} n_state={N_STATE} "
          f"k={K} p={P} beta={BETA} presentations={presentations}")
    print(f"    arc:   kp per fiber = {K * P:.0f}, both conjuncts = "
          f"{2 * K * P:.0f}, floor 3 ln {N_ARC} = {3 * np.log(N_ARC):.1f}")
    print(f"    state: afferent kp = {K * P:.0f}, "
          f"floor 3 ln {N_STATE} = {3 * np.log(N_STATE):.1f}")

    arms = []
    print("\n  [main] refraction on, drive-proportional")
    arms.append(run_arm("main", seeds, presentations=presentations))
    print("\n  [null] refraction OFF -- must FAIL, or the result is void")
    arms.append(run_arm("null", seeds, presentations=presentations, strength=0.0))
    print("\n  [degen] zero presentations")
    arms.append(run_arm("degen_untrained", seeds, presentations=0))
    print("\n  [degen] beta = 0")
    arms.append(run_arm("degen_beta0", seeds, presentations=presentations, beta=0.0))
    print("\n  [A/B] constant-increment refraction rule")
    arms.append(run_arm("constant_rule", seeds, constant=True,
                        presentations=presentations))

    summaries = [summarize(a) for a in arms]
    print("\n=== SUMMARY ===")
    print(f"  {'arm':<18s} {'decided':>9s} {'across-state':>13s} "
          f"{'across-symbol':>14s} {'state-asm ov':>13s} {'max w':>8s}")
    for s in summaries:
        print(f"  {s['arm']:<18s} {s['decided']:>6d}/{s['n']:<2d} "
              f"{s['across_state']:>13.3f} {s['across_symbol']:>14.3f} "
              f"{s['state_assembly_overlap']:>13.3f} {s['max_weight']:>8.2f}")

    main_s = summaries[0]
    null_s = summaries[1]
    n = main_s["n"]
    verdicts = {
        # UPPER bound below the threshold, so the claim survives seed variation.
        "P-PRE": (main_s["state_assembly_overlap"]
                  + main_s["state_assembly_overlap_ci"] < 0.05),
        "P-GOLD": main_s["decided"] >= 0.8 * n,
        "P-CONJ": (main_s["across_state"] + main_s["across_state_ci"] < 0.15
                   and main_s["across_symbol"]
                   + main_s["across_symbol_ci"] < 0.15),
        "P-NULL": (null_s["decided"] <= 0.2 * n
                   and null_s["across_symbol"] > 0.5),
        "P-DEGEN": all(s["decided"] <= 0.2 * n for s in summaries[2:4]),
    }
    print("\n=== BARS ===")
    for bar, ok in verdicts.items():
        print(f"  {bar:<8s} {'PASS' if ok else 'FAIL'}")
    if null_s["decided"] >= 0.8 * n:
        print("\n  *** VOID: the null arm passed P-GOLD. Refraction is not what "
              "is carrying the result. ***")
    if not verdicts["P-PRE"]:
        print("\n  *** P-GOLD is VOID: state assemblies overlap above 0.05, so "
              "the nearest-overlap readout is confounded. ***")

    write_new_document(Path(args.out), {
        "summaries": summaries, "arms": arms,
        "verdicts": verdicts,
        "params": {"n_arc": N_ARC, "n_state": N_STATE, "k": K,
                   "p": P, "beta": BETA,
                   "presentations": presentations,
                   "strength": STRENGTH},
    })
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
