"""A3: does an INDUCED state help next-token prediction? [[SEQ-TRANSDUCER]]

Implements `research/notes/sequence/PREREG_seq_a3_transducer.md`. Parameters inherited
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
import time

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


CORPUS = "study4"      # or "chain" (PREREG_agreement_corpus.md)
CHAIN_GAP = 1
SUCCESSOR_GAIN = 1.0
STATE_MODE = "induced"  # or "copy" (PREREG_temporal_memory.md)
PREDICT_GAIN = 0.0
REGISTER = None         # (features, feature_of) for PREREG_feature_register.md
REGISTER_BLIND = False


def _gen():
    if CORPUS == "chain":
        import ntp_agree
        ntp_agree.use_chain(True, gap=CHAIN_GAP)
        return ntp_agree
    return ntp


def corpora(seed):
    g = _gen()
    words = g.vocabulary(VOCAB_SIZE)
    keep = set(words)
    tr = [[w for w in s if w in keep] for s in g.generate(N_TRAIN, seed)]
    te = [[w for w in s if w in keep] for s in g.generate(N_TEST, seed + 500)]
    return words, tr, te


def _corpora_study4(seed):
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


def _write(out, tag=""):
    from _results import results_path
    path = results_path("sequence", f"seq_a3_transducer_results{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


# ---------------------------------------------------------------------------
# the hashed organ (PREREG amendment of 2026-09-04): every seed its own
# corpus, brains in launches sized to memory, bars unchanged
# ---------------------------------------------------------------------------

HASHED_SEEDS = list(range(42, 62))
LAUNCH_BYTES = 5 << 30


def _schedules(per_brain, wi):
    """[(words, targets, starts)] per brain from its sentences, padded."""
    rows = []
    for sents in per_brain:
        W, T, St = [], [], []
        for sent in sents:
            for j, (a, nxt) in enumerate(zip(sent, sent[1:])):
                W.append(wi[a]); T.append(wi[nxt]); St.append(j == 0)
        rows.append((W, T, St))
    S = max(len(r[0]) for r in rows)
    import torch
    W = torch.full((len(rows), S), -1, dtype=torch.int64)
    T = torch.full((len(rows), S), -1, dtype=torch.int64)
    St = torch.zeros(len(rows), S, dtype=torch.bool)
    for b, (w, t, st) in enumerate(rows):
        W[b, :len(w)] = torch.tensor(w); T[b, :len(t)] = torch.tensor(t)
        St[b, :len(st)] = torch.tensor(st)
    return W, T, St


def _brains_per_launch(n_arc):
    # five organ fibers of int8 counts plus their presence masks, per brain
    per = 1 * (3 * N * n_arc + 2 * n_arc * N) + 4 * (N * (n_arc // 32 + 1) * 3 + n_arc * (N // 32 + 1) * 2)
    per += 2 * VOCAB_SIZE * N * 16
    return max(1, int(LAUNCH_BYTES // per))


def a3_hashed(seeds, *, n_arc, beta, state_blind=False, collect_state=False,
              tie_seed=0, strength=0.1, collect_margin=False, horizon=0,
              collect_arcs=False):
    """MRR per seed (and cross-prefix state overlap per seed when asked)."""
    if collect_arcs:
        raise ValueError("TM-9 mechanism collection is invalid: pooled positions include agreement words; "
                         "see research/notes/sequence/AUDIT_temporal_position_pooling.md")
    import torch
    from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer
    words = _gen().vocabulary(VOCAB_SIZE)
    wi = {w: i for i, w in enumerate(words)}
    per = [(s,) + corpora(s)[1:] for s in seeds]                     # (seed, train, test)
    mrr, ovl = {}, {}
    chunk = _brains_per_launch(n_arc)
    for g0 in range(0, len(per), chunk):
        group = per[g0:g0 + chunk]
        gseeds = [s for s, _, _ in group]
        t0 = time.perf_counter()
        t = HashedTransducer(gseeds, words, n=N, n_arc=n_arc, k=K, p=P, beta=beta,
                             organ_p=ORGAN_P, w_max=20.0, norm_init=True,
                             max_potentiations=64, refracted_strength=strength,
                             horizon=horizon, successor_gain=SUCCESSOR_GAIN,
                             state_mode=STATE_MODE, predict_gain=PREDICT_GAIN,
                             n_state=(n_arc if STATE_MODE == "copy" else None),
                             features=(REGISTER[0] if REGISTER else None),
                             feature_of=(REGISTER[1] if REGISTER else None))
        if REGISTER and REGISTER_BLIND:
            t.reg_blind = True
        margins = [[] for _ in group]
        t.ground(rounds=GROUND_ROUNDS)
        W, T, St = _schedules([tr for _, tr, _ in group], wi)
        t.train_schedules(W, T, St, rounds=TRAIN_ROUNDS)
        # scoring, frozen: each brain its own test sentences, positions aligned
        rng = random.Random(tie_seed)
        Wt, Tt, Stt = _schedules([te for _, _, te in group], wi)
        rr = [0.0] * len(group); cnt = [0] * len(group)
        states = [dict() for _ in group]
        pos = [0] * len(group)
        # TM-9: arc winners at each position, with the sentence's subject
        # number, per brain: {position: [(number, set(arc))]}
        arcs_by_pos = [dict() for _ in group]
        for step in range(Wt.shape[1]):
            live = Wt[:, step] >= 0
            if not bool(live.any()):
                break
            t.reset(Stt[:, step])
            if state_blind:
                t.state.inhibit_rows(live)
            t.tick(Wt[:, step], rounds=TRAIN_ROUNDS, freeze=True)
            if collect_margin:
                # the arc's MEMBER MARGIN: min net drive of a winner minus max
                # net drive of a non-winner, over the best outsider's drive
                raw = torch.zeros(len(group), t.n_arc, device="cuda")
                t.lex_arc.contribute(raw, t.lex.winners)
                t.state_arc.contribute(raw, t.state.winners)
                net = t.arc.apply_bias(raw)
                w = t.arc.winners.clamp_min(0)
                wmin = net.gather(1, w).min(1).values
                mask = torch.zeros_like(net, dtype=torch.bool).scatter_(1, w, True)
                omax = net.masked_fill(mask, -1e9).max(1).values
                for b in range(len(group)):
                    if bool(live[b]):
                        margins[b].append(float((wmin[b] - omax[b]) / omax[b].clamp_min(1e-6)))
            if collect_state:
                sw = t.state.winners.cpu()
            emitted = t.emit()
            ranked = t.rank(emitted, rng)
            for b in range(len(group)):
                if not bool(live[b]):
                    continue
                if bool(Stt[b, step]):
                    pos[b] = 0
                truth = words[int(Tt[b, step])]
                rr[b] += 1.0 / (ranked[b].index(truth) + 1); cnt[b] += 1
                if collect_state:
                    states[b].setdefault(pos[b], []).append(set(sw[b].tolist()))
                if collect_arcs:
                    w = words[int(Wt[b, step])]
                    num = _number_of(w)
                    arcs_by_pos[b].setdefault(pos[b], []).append(
                        (num, set(t.arc.winners[b].tolist())))
                pos[b] += 1
        for b, (seed, _, _) in enumerate(group):
            mrr[seed] = rr[b] / max(cnt[b], 1)
            if collect_arcs:
                ovl[("mechanism", seed)] = _distractor_overlaps(arcs_by_pos[b])
            if collect_margin:
                ovl[("margin", seed)] = float(np.mean(margins[b])) if margins[b] else float("nan")
            if collect_state:
                ovs = []
                for _p, sets_ in states[b].items():
                    m = min(len(sets_), 12)
                    for i in range(m):
                        for j in range(i + 1, m):
                            ovs.append(len(sets_[i] & sets_[j]) / K)
                ovl[seed] = float(np.mean(ovs)) if ovs else float("nan")
        print(f"      n_arc={n_arc} beta={beta} seeds {gseeds[0]}..{gseeds[-1]} "
              f"({len(group)} brains, {W.shape[1]} train steps)  "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        del t
        torch.cuda.empty_cache()
    return mrr, ovl


def main_hashed(seeds, cells=N_ARC_SWEEP, with_context=True):
    out = {"seeds": seeds, "organ_p": ORGAN_P, "n_arc_sweep": list(cells),
           "substrate": "hashed (DESIGN_sequence_port.md)"}
    print("=== A3 on the hashed organ (PREREG amendment 2026-09-04) ===")
    print(f"    seeds {seeds[0]}..{seeds[-1]} ({len(seeds)})  n={N} k={K} p={P} "
          f"organ_p={ORGAN_P}")
    print("\n  [H5 null] beta = 0 -- must not beat the unigram baseline")
    m, _ = a3_hashed(seeds, n_arc=10000, beta=0.0)
    null = ensemble_from_values([m[s] for s in seeds], "null(beta=0)", keys=seeds)
    print(f"    {null}", flush=True)
    out["null"] = {"mean": null.mean, "ci": null.ci, "values": list(null.values)}
    h5 = null.high < UNIGRAM
    print(f"    H5 {'PASS' if h5 else 'FAIL'} (upper {null.high:.4f} vs unigram {UNIGRAM})")
    if not h5:
        print("\n  H5 FAILED: the study stops (bug hunt). Sweep NOT run.")
        _write(out, "_hashed")
        return
    print("\n  [sweep] the n_arc curve, reported whole")
    cells_e = {}
    for na in cells:
        m, _ = a3_hashed(seeds, n_arc=na, beta=BETA)
        cells_e[na] = ensemble_from_values([m[s] for s in seeds], f"a3(n_arc={na})", keys=seeds)
        print(f"    {cells_e[na]}", flush=True)
    out["sweep"] = {str(na): {"mean": e.mean, "ci": e.ci, "values": list(e.values)}
                    for na, e in cells_e.items()}
    best_n = max(cells_e, key=lambda na: cells_e[na].mean)
    best = cells_e[best_n]
    print(f"\n    best cell by mean: n_arc={best_n}")
    if with_context:
        print("\n  [CONTEXT] #14's accumulator (numpy), the same seeds, in a pool")
        r = run_cells(worker, [("context", s, 0, BETA) for s in seeds])
        ctx = _arm(r, "context", seeds, 0, BETA, "context(#14)")
        print(f"    {ctx}", flush=True)
        out["context"] = {"mean": ctx.mean, "ci": ctx.ci, "values": list(ctx.values)}
        delta = paired_delta(best, ctx, label=f"a3(n_arc={best_n}) - context")
        print(f"    {delta}", flush=True)
        out["h1_delta"] = {"mean": delta.mean, "ci": delta.ci, "values": list(delta.values)}
    print("\n  [H4 + audit] state overlap at the best cell; state-blind arm")
    m, ov = a3_hashed(seeds, n_arc=best_n, beta=BETA, collect_state=True)
    h4 = ensemble_from_values([ov[s] for s in seeds], "state_overlap", keys=seeds)
    print(f"    {h4}", flush=True)
    out["h4"] = {"mean": h4.mean, "ci": h4.ci, "values": [ov[s] for s in seeds]}
    mb, _ = a3_hashed(seeds, n_arc=best_n, beta=BETA, state_blind=True)
    blind = ensemble_from_values([mb[s] for s in seeds], "state-blind", keys=seeds)
    print(f"    {blind}", flush=True)
    out["state_blind"] = {"mean": blind.mean, "ci": blind.ci, "values": list(blind.values)}
    bd = paired_delta(best, blind, label="a3 - state-blind")
    print(f"    {bd}", flush=True)
    out["state_blind_delta"] = {"mean": bd.mean, "ci": bd.ci, "values": list(bd.values)}
    out["load"] = "not measured on the hashed organ (amendment)"
    print("\n=== BARS ===")
    verdicts = {
        "H2 beats no-context model": best.beats(NO_CONTEXT),
        "H3 beats bigram optimum": best.beats(BIGRAM),
        "H4 state does not collapse": h4.high < 0.5,
    }
    if with_context:
        verdicts["H1 beats #14 CONTEXT (paired)"] = delta.low > 0.0
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = verdicts
    out["best_n_arc"] = best_n
    _write(out, "_hashed")


def main_strength(seeds, strength, n_arc=10000):
    """PREREG_seq_a3_transducer.md Amendment 2: the organ at a strength below
    beta, paired against the recorded beta cell; the state-blind audit and
    the arc's member margin at both."""
    print(f"=== A3 hashed, strength {strength} (Amendment 2) ===")
    from _results import results_path
    with open(results_path("sequence", "seq_a3_transducer_results_hashed.json")) as fh:
        base = json.load(fh)["sweep"][str(n_arc)]["values"]
    m, ov = a3_hashed(seeds, n_arc=n_arc, beta=BETA, strength=strength,
                      collect_state=True, collect_margin=True)
    cell = ensemble_from_values([m[s] for s in seeds], f"a3(s={strength})", keys=seeds)
    ref = ensemble_from_values(base, "a3(s=0.1)", keys=seeds)
    delta = paired_delta(cell, ref, label=f"a3(s={strength}) - a3(s=0.1)")
    h4 = ensemble_from_values([ov[s] for s in seeds], "state_overlap", keys=seeds)
    marg = ensemble_from_values([ov[("margin", s)] for s in seeds], "arc_margin", keys=seeds)
    mb, _ = a3_hashed(seeds, n_arc=n_arc, beta=BETA, strength=strength, state_blind=True)
    blind = ensemble_from_values([mb[s] for s in seeds], "state-blind", keys=seeds)
    bd = paired_delta(cell, blind, label="a3 - state-blind")
    _, ovb = a3_hashed(seeds, n_arc=n_arc, beta=BETA, strength=0.1, collect_margin=True)
    marg0 = ensemble_from_values([ovb[("margin", s)] for s in seeds], "arc_margin(s=0.1)", keys=seeds)
    for e in (cell, ref, delta, h4, blind, bd, marg, marg0):
        print(f"    {e}", flush=True)
    out = {"strength": strength, "n_arc": n_arc, "seeds": seeds,
           "mrr": {"mean": cell.mean, "ci": cell.ci, "values": list(cell.values)},
           "delta_vs_beta": {"mean": delta.mean, "ci": delta.ci, "values": list(delta.values)},
           "h4": {"mean": h4.mean, "ci": h4.ci},
           "state_blind": {"mean": blind.mean, "ci": blind.ci},
           "state_blind_delta": {"mean": bd.mean, "ci": bd.ci, "values": list(bd.values)},
           "arc_margin": {"mean": marg.mean, "ci": marg.ci},
           "arc_margin_beta": {"mean": marg0.mean, "ci": marg0.ci}}
    verdicts = {"A2-1 MRR above the beta cell (paired)": delta.low > 0.0,
                "A2-2 state informative (a3 - blind > 0)": bd.low > 0.0,
                "A2-3 state does not collapse": h4.high < 0.5}
    print("\n=== BARS ===")
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = verdicts
    _write(out, f"_hashed_s{strength}")


def main_successor(seeds, horizons=(0, 1, 2), n_arc=10000, gap=1, gain=1.0):
    """PREREG_successor_state.md on the chain corpus: h in {0, 1, 2}, the
    state-blind audit at the largest h, paired against each seed's own
    bigram and oracle (ntp_agree.oracle_gap)."""
    global CORPUS, CHAIN_GAP, SUCCESSOR_GAIN
    CORPUS, CHAIN_GAP, SUCCESSOR_GAIN = "chain", int(gap), float(gain)
    import ntp_agree
    ntp_agree.use_chain(True, gap=CHAIN_GAP)
    print(f"    gap {CHAIN_GAP}, successor gain {SUCCESSOR_GAIN}")
    print("=== successor state on the chain corpus (PREREG_successor_state.md) ===")
    base = {s: ntp_agree.oracle_gap(s) for s in seeds}
    bigram = ensemble_from_values([base[s][1] for s in seeds], "bigram", keys=seeds)
    oracle = ensemble_from_values([base[s][3] for s in seeds], "oracle", keys=seeds)
    print(f"    {bigram}\n    {oracle}", flush=True)
    out = {"corpus": "chain", "gap": CHAIN_GAP, "successor_gain": SUCCESSOR_GAIN,
           "seeds": seeds, "n_arc": n_arc,
           "bigram": {"mean": bigram.mean, "ci": bigram.ci, "values": list(bigram.values)},
           "oracle": {"mean": oracle.mean, "ci": oracle.ci, "values": list(oracle.values)}}
    cells = {}
    for h in horizons:
        m, ov = a3_hashed(seeds, n_arc=n_arc, beta=BETA, horizon=h, collect_state=True)
        cell = ensemble_from_values([m[s] for s in seeds], f"a3(h={h})", keys=seeds)
        d = paired_delta(cell, bigram, label=f"a3(h={h}) - bigram")
        h4 = ensemble_from_values([ov[s] for s in seeds], f"state_overlap(h={h})", keys=seeds)
        print(f"    {cell}\n    {d}\n    {h4}", flush=True)
        cells[h] = (cell, d, h4)
        out[f"h{h}"] = {"mrr": {"mean": cell.mean, "ci": cell.ci, "values": list(cell.values)},
                        "delta_bigram": {"mean": d.mean, "ci": d.ci, "values": list(d.values)},
                        "state_overlap": {"mean": h4.mean, "ci": h4.ci}}
    hmax = max(horizons)
    mb, _ = a3_hashed(seeds, n_arc=n_arc, beta=BETA, horizon=hmax, state_blind=True)
    blind = ensemble_from_values([mb[s] for s in seeds], f"state-blind(h={hmax})", keys=seeds)
    bd = paired_delta(cells[hmax][0], blind, label=f"a3(h={hmax}) - blind")
    m0b, _ = a3_hashed(seeds, n_arc=n_arc, beta=BETA, horizon=0, state_blind=True)
    blind0 = ensemble_from_values([m0b[s] for s in seeds], "state-blind(h=0)", keys=seeds)
    bd0 = paired_delta(cells[0][0], blind0, label="a3(h=0) - blind")
    print(f"    {blind}\n    {bd}\n    {blind0}\n    {bd0}", flush=True)
    out["state_blind_delta"] = {"mean": bd.mean, "ci": bd.ci, "values": list(bd.values)}
    out["state_blind_delta_h0"] = {"mean": bd0.mean, "ci": bd0.ci, "values": list(bd0.values)}
    print("\n=== BARS ===")
    verdicts = {
        "SR-0 h=0 within 0.03 of bigram and blind delta contains 0":
            abs(cells[0][1].mean) <= 0.03 and bd0.low <= 0.0 <= bd0.high,
        "SR-1 h=1 does not help (lower bound < 0.05)": (1 in cells) and cells[1][1].low < 0.05,
        f"SR-2 h={hmax} closes >= 40% of the gap (lower bound >= 0.10)": cells[hmax][1].low >= 0.10,
        f"SR-3 state informative at h={hmax}": bd.low > 0.0,
        f"SR-4 state does not collapse at h={hmax}": cells[hmax][2].high < 0.5,
    }
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = verdicts
    _write(out, f"_successor_chain_gap{CHAIN_GAP}_g{SUCCESSOR_GAIN}")


def main_temporal(seeds, gains=(0.0, 1.0, 4.0), n_arc=10000, gap=2, mechanism=False, tag=""):
    """PREREG_temporal_memory.md, cells A: state = previous arc, predicted
    neurons win at gain g, on the chain corpus; state-blind audit per g."""
    global CORPUS, CHAIN_GAP, STATE_MODE, PREDICT_GAIN
    CORPUS, CHAIN_GAP, STATE_MODE = "chain", int(gap), "copy"
    import ntp_agree
    ntp_agree.use_chain(True, gap=CHAIN_GAP)
    print(f"=== temporal memory on the chain corpus, gap {CHAIN_GAP} (PREREG_temporal_memory.md) ===")
    base = {s: ntp_agree.oracle_gap(s) for s in seeds}
    bigram = ensemble_from_values([base[s][1] for s in seeds], "bigram", keys=seeds)
    oracle = ensemble_from_values([base[s][3] for s in seeds], "oracle", keys=seeds)
    print(f"    {bigram}\n    {oracle}", flush=True)
    out = {"corpus": "chain", "gap": CHAIN_GAP, "state_mode": "copy", "seeds": seeds, "n_arc": n_arc,
           "bigram": {"mean": bigram.mean, "ci": bigram.ci, "values": list(bigram.values)},
           "oracle": {"mean": oracle.mean, "ci": oracle.ci, "values": list(oracle.values)}}
    verdicts = {}
    for g in gains:
        PREDICT_GAIN = float(g)
        m, ov = a3_hashed(seeds, n_arc=n_arc, beta=BETA, collect_state=True,
                          collect_arcs=mechanism)
        cell = ensemble_from_values([m[s] for s in seeds], f"tm(g={g})", keys=seeds)
        d = paired_delta(cell, bigram, label=f"tm(g={g}) - bigram")
        h4 = ensemble_from_values([ov[s] for s in seeds], f"arc_overlap(g={g})", keys=seeds)
        mb, _ = a3_hashed(seeds, n_arc=n_arc, beta=BETA, state_blind=True)
        blind = ensemble_from_values([mb[s] for s in seeds], f"blind(g={g})", keys=seeds)
        bd = paired_delta(cell, blind, label=f"tm(g={g}) - blind")
        print(f"    {cell}\n    {d}\n    {h4}\n    {blind}\n    {bd}", flush=True)
        out[f"g{g}"] = {"mrr": {"mean": cell.mean, "ci": cell.ci, "values": list(cell.values)},
                        "delta_bigram": {"mean": d.mean, "ci": d.ci, "values": list(d.values)},
                        "overlap": {"mean": h4.mean, "ci": h4.ci},
                        "blind_delta": {"mean": bd.mean, "ci": bd.ci, "values": list(bd.values)}}
        if mechanism:
            same = ensemble_from_values([ov[("mechanism", s_)]["same"] for s_ in seeds], f"same-number arc overlap(g={g})", keys=seeds)
            diff = ensemble_from_values([ov[("mechanism", s_)]["diff"] for s_ in seeds], f"different-number arc overlap(g={g})", keys=seeds)
            gapd = paired_delta(same, diff, label=f"same - different (g={g})")
            print(f"    {same}\n    {diff}\n    {gapd}", flush=True)
            out[f"g{g}"]["mechanism"] = {"same": same.mean, "diff": diff.mean,
                                         "delta": {"mean": gapd.mean, "ci": gapd.ci, "low": gapd.low}}
            verdicts[f"TM-9 g={g}: same-number minus different-number arc overlap "
                     f"{'>= 0.10' if g > 0 else 'within 0.02'}"] = (gapd.low >= 0.10) if g > 0 else (abs(gapd.mean) <= 0.02)
        if g == 0.0:
            verdicts["TM-1 copy-state alone: reported"] = True
        else:
            verdicts[f"TM-2 g={g} closes >= 40% of the gap (lower bound >= 0.085)"] = d.low >= 0.085
            verdicts[f"TM-3 g={g} state informative"] = bd.low > 0.0
    print("\n=== BARS ===")
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = verdicts
    _write(out, f"_temporal_chain_gap{CHAIN_GAP}{tag}")


def _number_of(word):
    """The chain corpus's number of a word's sentence: every word carries
    the subject's number except the distractor nouns, whose number is their
    own; the SUBJECT number of a sentence is read from its first word."""
    import ntp_agree
    return ntp_agree.CLASS[word].split("_")[1]


def _distractor_overlaps(arcs_by_pos):
    raise ValueError("TM-9 needs position-specific distractor selection; "
                     "see research/notes/sequence/AUDIT_temporal_position_pooling.md")


def _historical_pooled_arc_overlaps(arcs_by_pos):
    """Historical arithmetic for audit only; pools every noninitial position.

    Specification: research/notes/sequence/AUDIT_temporal_position_pooling.md
    This is not a distractor-specific mechanism measurement.
    """
    same, diff = [], []
    # the subject number per sentence: position 0's entries, in order
    subj = [num for num, _ in arcs_by_pos.get(0, [])]
    for p_, entries in arcs_by_pos.items():
        if p_ == 0 or len(entries) != len(subj):
            continue
        sets = [a for _, a in entries]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                k_ = max(len(sets[i]), 1)
                ov = len(sets[i] & sets[j]) / k_
                (same if subj[i] == subj[j] else diff).append(ov)
    return {"same": float(np.mean(same)) if same else float("nan"),
            "diff": float(np.mean(diff)) if diff else float("nan")}


def _feature_tables(words):
    """(gated, ungated) feature_of tables on the chain corpus: gated writes
    only from the agreeing classes; ungated from every number-marked word."""
    import ntp_agree
    gated, ungated = {}, {}
    for w in words:
        c = ntp_agree.CLASS[w]
        base, num = c.split("_")
        idx = {"sg": 0, "pl": 1}[num]
        ungated[w] = idx
        gated[w] = idx if base in ("AUX", "VERB", "PRON", "TAG") else -1
    return gated, ungated


def main_register(seeds, n_arc=10000, gap=2):
    """PREREG_feature_register.md: gated register, state-blind with the
    register, register-blind, and the ungated register."""
    global CORPUS, CHAIN_GAP, REGISTER, REGISTER_BLIND
    CORPUS, CHAIN_GAP = "chain", int(gap)
    import ntp_agree
    ntp_agree.use_chain(True, gap=CHAIN_GAP)
    words = ntp_agree.vocabulary(VOCAB_SIZE)
    gated, ungated = _feature_tables(words)
    print(f"=== feature register on the chain corpus, gap {CHAIN_GAP} (PREREG_feature_register.md) ===")
    base = {s: ntp_agree.oracle_gap(s) for s in seeds}
    bigram = ensemble_from_values([base[s][1] for s in seeds], "bigram", keys=seeds)
    oracle = ensemble_from_values([base[s][3] for s in seeds], "oracle", keys=seeds)
    print(f"    {bigram}\n    {oracle}", flush=True)
    out = {"corpus": "chain", "gap": CHAIN_GAP, "seeds": seeds, "n_arc": n_arc,
           "bigram": {"mean": bigram.mean, "ci": bigram.ci, "values": list(bigram.values)},
           "oracle": {"mean": oracle.mean, "ci": oracle.ci, "values": list(oracle.values)}}
    cells = {}
    for name, table, blind_reg, blind_state in (("gated", gated, False, False),
                                                ("gated_state_blind", gated, False, True),
                                                ("gated_register_blind", gated, True, False),
                                                ("ungated", ungated, False, False)):
        REGISTER, REGISTER_BLIND = (["sg", "pl"], table), blind_reg
        m, ov = a3_hashed(seeds, n_arc=n_arc, beta=BETA, state_blind=blind_state, collect_state=True)
        cell = ensemble_from_values([m[s] for s in seeds], f"reg({name})", keys=seeds)
        d = paired_delta(cell, bigram, label=f"reg({name}) - bigram")
        print(f"    {cell}\n    {d}", flush=True)
        cells[name] = d
        out[name] = {"mrr": {"mean": cell.mean, "ci": cell.ci, "values": list(cell.values)},
                     "delta_bigram": {"mean": d.mean, "ci": d.ci, "values": list(d.values)}}
    REGISTER, REGISTER_BLIND = None, False
    print("\n=== BARS ===")
    verdicts = {
        "FR-1 gated register closes the gap (lower bound >= 0.15)": cells["gated"].low >= 0.15,
        "FR-2 register carries it with the state empty (lower bound >= 0.15)": cells["gated_state_blind"].low >= 0.15,
        "FR-3 register-blind falls back (upper bound <= 0.05)": cells["gated_register_blind"].high <= 0.05,
        "FR-4 ungated register fails (upper bound <= 0.05)": cells["ungated"].high <= 0.05,
    }
    for name, ok in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    out["verdicts"] = verdicts
    _write(out, f"_register_chain_gap{CHAIN_GAP}")


if __name__ == "__main__":
    if "--register" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(HASHED_SEEDS)
        gap = int(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 2
        main_register(HASHED_SEEDS[:n_seeds], gap=gap)
    elif "--temporal" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(HASHED_SEEDS)
        gs = ([float(x) for x in sys.argv[sys.argv.index("--gains") + 1].split(",")]
              if "--gains" in sys.argv else (0.0, 1.0, 4.0))
        gap = int(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 2
        start = int(sys.argv[sys.argv.index("--seed-start") + 1]) if "--seed-start" in sys.argv else HASHED_SEEDS[0]
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
        main_temporal(list(range(start, start + n_seeds)), gains=tuple(gs), gap=gap,
                      mechanism="--mechanism" in sys.argv, tag=tag)
    elif "--successor" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(HASHED_SEEDS)
        hs = ([int(x) for x in sys.argv[sys.argv.index("--horizons") + 1].split(",")]
              if "--horizons" in sys.argv else (0, 1, 2))
        gap = int(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 1
        gain = float(sys.argv[sys.argv.index("--gain") + 1]) if "--gain" in sys.argv else 1.0
        main_successor(HASHED_SEEDS[:n_seeds], horizons=tuple(hs), gap=gap, gain=gain)
    elif "--strength" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(HASHED_SEEDS)
        main_strength(HASHED_SEEDS[:n_seeds], float(sys.argv[sys.argv.index("--strength") + 1]))
    elif "--engine" in sys.argv and sys.argv[sys.argv.index("--engine") + 1] == "hashed":
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else len(HASHED_SEEDS)
        cells = ([int(x) for x in sys.argv[sys.argv.index("--cells") + 1].split(",")]
                 if "--cells" in sys.argv else N_ARC_SWEEP)
        main_hashed(HASHED_SEEDS[:n_seeds], cells=cells,
                    with_context="--no-context" not in sys.argv)
    else:
        main()
