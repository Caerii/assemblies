"""A3: does an INDUCED state help next-token prediction? [[SEQ-TRANSDUCER]]

Implements `research/notes/PREREG_seq_a3_transducer.md`. Parameters inherited
from #14 are FIXED and must not be tuned here; `n_arc` is swept as a design
axis and the WHOLE CURVE is reported, per the pre-registration.

Arms, in the order the prereg commits to running them:

  H5 null   beta = 0                        must be at or below unigram, 0.1178
  A3        the transducer, per n_arc cell
  CONTEXT   #14's recurrent accumulator, RE-RUN here rather than quoted, so
            the H1 comparison is paired on the same seeds and read through the
            same isolation

Bars (all judged on the CONFIDENCE BOUND, never the mean):

  H1  A3 - CONTEXT paired difference, lower bound > 0
  H2  A3 lower bound > 0.2074      (no-context model)
  H3  A3 lower bound > 0.2338      (bigram optimum)
  H4  cross-prefix state overlap, upper bound < 0.5, against #14's 0.7566

H4 is reported whatever H3 does. A degenerate-arm audit (score with the state
held empty) runs at the best cell unconditionally, which is more than the
prereg commits to -- it commits to running it only at or above 0.2338.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import (
    ensemble_from_values, format_report, paired_delta, regime_audit,
)
from neural_assemblies.programs.sequence_transducer import SequenceTransducer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "study4"))
sys.path.insert(0, _HERE)
import ntp  # noqa: E402
import ntp_ctx  # noqa: E402
from _parallel import run_cells  # noqa: E402

N, K, P, BETA = ntp.N, ntp.K, ntp.P, ntp.BETA
TRAIN_ROUNDS, GROUND_ROUNDS = ntp.TRAIN_ROUNDS, ntp.GROUND_ROUNDS
VOCAB_SIZE, N_TRAIN, N_TEST = 50, 200, 25
SEEDS = list(range(42, 52))

ORGAN_P = 0.20                      # kp = 200*0.2 = 40 vs floor 3 ln 1e4 = 27.6
N_ARC_SWEEP = [2000, 10000, 50000]

UNIGRAM, NO_CONTEXT, BIGRAM, CONTEXT_14 = 0.1178, 0.2074, 0.2338, 0.1046


def corpora(seed):
    words = ntp.vocabulary(VOCAB_SIZE)
    keep = set(words)
    tr = [[w for w in s if w in keep] for s in ntp.generate(N_TRAIN, seed)]
    te = [[w for w in s if w in keep] for s in ntp.generate(N_TEST, seed + 500)]
    return words, tr, te


def build(seed, *, n_arc, beta, organ_p=ORGAN_P):
    random.seed(seed)
    np.random.seed(seed)
    words, tr, te = corpora(seed)
    brain = Brain(p=P, seed=seed, engine="numpy_sparse")
    t = SequenceTransducer(brain, words, n=N, n_arc=n_arc, k=K, beta=beta,
                           organ_p=organ_p)
    t.ground(rounds=GROUND_ROUNDS)
    for s in tr:
        t.train_sentence(s, rounds=TRAIN_ROUNDS)
    return brain, t, te


def score(brain, t, corpus, *, tie_seed=0, collect_state=None,
          state_blind=False):
    """MRR of the true next word, read inside `probe()`.

    `probe()` rather than `frozen()`: plasticity-off still lets an area RECRUIT,
    and recruitment moved a trained FSM's trajectory between two identical runs
    ([[probe-isolation-required]]). #14 scored under plasticity-off only, so
    the CONTEXT arm is re-run here under the same context as A3 rather than
    compared against its published number through a different instrument.
    """
    rng = random.Random(tie_seed)
    rr, n = 0.0, 0
    with brain.probe():
        for s in corpus:
            t.reset()
            for pos, (a, truth) in enumerate(zip(s, s[1:])):
                if state_blind:
                    brain.inhibit_areas([t.state_area])
                t.tick(a, rounds=TRAIN_ROUNDS)
                if collect_state is not None:
                    collect_state.setdefault(pos, []).append(t.state())
                emitted = t.emit()
                rr += 1.0 / (t.rank(emitted, rng).index(truth) + 1)
                n += 1
    return rr / max(n, 1)


def cross_prefix_state_overlap(collected, cap=12):
    """Mean pairwise state overlap across DIFFERENT prefixes at the same position.

    #14's CONTEXT measured 0.7566 +/- 0.0958 here, i.e. one attractor.
    """
    ovs = []
    for _pos, arrs in collected.items():
        m = min(len(arrs), cap)
        for i in range(m):
            for j in range(i + 1, m):
                ovs.append(overlap(arrs[i], arrs[j]))
    return float(np.mean(ovs)) if ovs else float("nan")


def distinct_arc_assemblies(brain, t, corpus, threshold=0.5, cap=400):
    """Greedy count of distinct arc assemblies visited -- the load's numerator.

    [[REFRACTION-NEEDS-LOAD]] puts the arc's operating window in M*k/n, and for
    an induced state M is NOT KNOWN IN ADVANCE -- it is part of what is being
    asked. Counting it is the only way to report which cell of the n_arc sweep
    actually sat in the window.
    """
    seen = []
    with brain.probe():
        for s in corpus:
            t.reset()
            for a in s[:-1]:
                t.tick(a, rounds=TRAIN_ROUNDS)
                snap = _snap(brain, t.arc_area)
                if not any(overlap(snap, o) >= threshold for o in seen):
                    seen.append(snap)
                t.emit()
                if len(seen) >= cap:
                    return len(seen)
    return len(seen)


def a3_arm(seed, *, n_arc, beta=BETA, state_blind=False) -> float:
    brain, t, te = build(seed, n_arc=n_arc, beta=beta)
    return score(brain, t, te, state_blind=state_blind)


def a3_mechanism(seed, *, n_arc):
    """Score once more, collecting the state probe and the achieved arc load."""
    brain, t, te = build(seed, n_arc=n_arc, beta=BETA)
    collected: dict = {}
    score(brain, t, te, collect_state=collected)
    return (cross_prefix_state_overlap(collected),
            distinct_arc_assemblies(brain, t, te))


def context_arm(seed):
    """#14's CONTEXT arm, re-run so H1 is paired rather than quoted."""
    random.seed(seed)
    np.random.seed(seed)
    return ntp_ctx.run(seed, BETA, vocab_size=VOCAB_SIZE, n_train=N_TRAIN,
                       n_test=N_TEST, engine="numpy_sparse")


def worker(kind, seed, n_arc, beta):
    """ONE cell, run in its own process. Must stay top-level and picklable.

    Every cell reseeds numpy and random itself and shares nothing with the
    others, which is what makes the pool safe here -- the same property
    `_parallel` relies on. `PYTHONHASHSEED` is pinned by the caller because
    hash()-derived seeds differ across processes and no single-process test can
    catch that ([[pythonhashseed-nondeterminism]]).
    """
    if kind == "a3":
        return {"v": a3_arm(seed, n_arc=n_arc, beta=beta)}
    if kind == "blind":
        return {"v": a3_arm(seed, n_arc=n_arc, beta=beta, state_blind=True)}
    if kind == "context":
        return {"v": context_arm(seed)}
    if kind == "mechanism":
        ov, m = a3_mechanism(seed, n_arc=n_arc)
        return {"overlap": ov, "distinct_arcs": m}
    raise ValueError(f"unknown cell kind {kind!r}")


def _arm(results, kind, seeds, n_arc, beta, label, field="v"):
    return ensemble_from_values(
        [results[(kind, s, n_arc, beta)][field] for s in seeds],
        label, keys=seeds)


def report_regime(seed, n_arc):
    brain, t, _te = build(seed, n_arc=n_arc, beta=BETA)
    driven = {t.arc_area: [t.lex_area, t.state_area],
              t.state_area: [t.arc_area],
              t.out_area: [t.arc_area],
              t.lex_area: [t._s_stim[t.vocab[0]]]}
    rows = [r for r in regime_audit(brain, driven) if r.area.startswith("_seq")]
    print(format_report(rows), flush=True)


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    out = {"seeds": seeds, "organ_p": ORGAN_P, "n_arc_sweep": N_ARC_SWEEP}
    print("=== A3: transducer with an INDUCED state ===")
    print(f"    seeds {seeds[0]}..{seeds[-1]}  n={N} k={K} p={P} beta={BETA}")
    print(f"    organ_p={ORGAN_P}  references: unigram {UNIGRAM}, "
          f"no-context {NO_CONTEXT}, bigram {BIGRAM}, #14 CONTEXT {CONTEXT_14}")

    print("\n  [regime] organ areas at the first seed, n_arc=10000")
    report_regime(seeds[0], 10000)

    # THREE PHASES, and the split is forced by the pre-registration rather
    # than by the pool. H5 gates everything, and `best_n` is not known until
    # the sweep returns, so those are genuine barriers. Within a phase the
    # cells are independent and run concurrently -- 70 sequential cells at
    # roughly a minute each is two hours for work a 14-way pool finishes in
    # about ten minutes, and nothing about the measurement changes: each cell
    # reseeds itself and shares nothing.

    # -- H5, FIRST, as committed ------------------------------------------
    print("\n  [H5 null] beta = 0 -- must not beat the unigram baseline")
    r = run_cells(worker, [("a3", s, 10000, 0.0) for s in seeds])
    null = _arm(r, "a3", seeds, 10000, 0.0, "null(beta=0)")
    print(f"    {null}", flush=True)
    out["null"] = {"mean": null.mean, "ci": null.ci, "values": list(null.values)}
    h5 = null.high < UNIGRAM
    print(f"    H5 {'PASS' if h5 else 'FAIL'} "
          f"(upper {null.high:.4f} vs unigram {UNIGRAM})")
    if not h5:
        print("\n  H5 FAILED. The prereg says the study stops and becomes a "
              "bug hunt. Sweep NOT run.")
        _write(out)
        return

    # -- the n_arc curve ---------------------------------------------------
    print("\n  [sweep + CONTEXT] full n_arc curve, reported whole")
    r = run_cells(worker,
                  [("a3", s, na, BETA) for na in N_ARC_SWEEP for s in seeds]
                  + [("context", s, 0, BETA) for s in seeds])
    cells = {na: _arm(r, "a3", seeds, na, BETA, f"a3(n_arc={na})")
             for na in N_ARC_SWEEP}
    for na in N_ARC_SWEEP:
        print(f"    {cells[na]}", flush=True)
    out["sweep"] = {str(na): {"mean": e.mean, "ci": e.ci,
                              "values": list(e.values)}
                    for na, e in cells.items()}

    best_n = max(cells, key=lambda na: cells[na].mean)
    best = cells[best_n]
    print(f"\n    best cell by mean: n_arc={best_n}")

    # -- CONTEXT, paired ---------------------------------------------------
    print("\n  [CONTEXT] #14's accumulator, re-run on the same seeds")
    ctx = _arm(r, "context", seeds, 0, BETA, "context(#14)")
    print(f"    {ctx}", flush=True)
    out["context"] = {"mean": ctx.mean, "ci": ctx.ci, "values": list(ctx.values)}

    delta = paired_delta(best, ctx, label=f"a3(n_arc={best_n}) - context")
    print(f"    {delta}", flush=True)
    out["h1_delta"] = {"mean": delta.mean, "ci": delta.ci,
                       "values": list(delta.values)}

    # -- H4 mechanism, and the load actually achieved -----------------------
    print("\n  [H4 + audit] state overlap, achieved arc load, state-blind arm")
    r2 = run_cells(worker,
                   [("mechanism", s, best_n, BETA) for s in seeds]
                   + [("blind", s, best_n, BETA) for s in seeds])
    ovs = [r2[("mechanism", s, best_n, BETA)]["overlap"] for s in seeds]
    loads = [r2[("mechanism", s, best_n, BETA)]["distinct_arcs"] * K / best_n
             for s in seeds]
    for s, ov, ld in zip(seeds, ovs, loads):
        print(f"    seed {s}: state overlap {ov:.4f}  load {ld:.3f}")
    h4 = ensemble_from_values(ovs, "state_overlap", keys=seeds)
    load_e = ensemble_from_values(loads, "arc_load", keys=seeds)
    print(f"    {h4}\n    {load_e}")
    out["h4"] = {"mean": h4.mean, "ci": h4.ci, "values": ovs}
    out["load"] = {"mean": load_e.mean, "ci": load_e.ci, "values": loads}

    # -- degenerate-arm audit ----------------------------------------------
    print("\n  [audit] score with the state held EMPTY -- can the readout "
          "reach the bar without it?")
    blind = _arm(r2, "blind", seeds, best_n, BETA, "state-blind")
    print(f"    {blind}", flush=True)
    out["state_blind"] = {"mean": blind.mean, "ci": blind.ci,
                          "values": list(blind.values)}
    blind_delta = paired_delta(best, blind, label="a3 - state-blind")
    print(f"    {blind_delta}", flush=True)
    out["state_blind_delta"] = {"mean": blind_delta.mean,
                                "ci": blind_delta.ci,
                                "values": list(blind_delta.values)}

    # -- verdicts ----------------------------------------------------------
    print("\n=== BARS ===")
    verdicts = {
        "H1 beats #14 CONTEXT (paired)": delta.low > 0.0,
        "H2 beats no-context model": best.beats(NO_CONTEXT),
        "H3 beats bigram optimum": best.beats(BIGRAM),
        "H4 state does not collapse": h4.high < 0.5,
    }
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = verdicts
    out["best_n_arc"] = best_n
    _write(out)


def _write(out):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "seq_a3_transducer_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
