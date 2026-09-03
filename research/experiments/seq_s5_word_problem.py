"""Does a constant-size assembly organ solve an NC1-complete word problem?

Implements `research/notes/PREREG_s5_word_problem.md`, including Amendment 1
(`n_arc` matches arc LOAD, not size, so solvability is the only variable).

    Z60      order  60  abelian                TC0    -- easy for SSMs
    A4xZ5    order  60  solvable, non-abelian  TC0
    A5       order  60  NON-SOLVABLE           NC1-complete
    S5       order 120  NON-SOLVABLE           NC1-complete

CONTROLS RUN FIRST AND GATE THE REST. The retracted mod-3 golden certified a
DICTIONARY LOOKUP as a neural result: it passed untrained, at zero
presentations, and at beta=0. Any arm a degenerate machine also passes is not
evidence, so `untrained`, `beta0` and `shuffled` are run before the real arms
and must all sit at chance.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import (
    ensemble_from_values, format_report, regime_audit,
)
from neural_assemblies.programs.nemo_fsm import NemoArcFSM
from neural_assemblies.programs.word_problems import (
    GROUPS, true_trajectory, word_problem_fsm,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _parallel import run_cells  # noqa: E402

K, BETA, ORGAN_P, REFRACTED = 70, 0.10, 0.40, 0.10
PRESENTATIONS = 15
TARGET_LOAD = 0.42
LENGTHS = [10, 50, 100, 500]
SEEDS = list(range(42, 52))
GROUP_NAMES = ["Z60", "A4xZ5", "A5", "S5"]
ARMS = ["untrained", "beta0", "shuffled", "trained"]


def sizes(group, n_symbols):
    """Arc sized to hold LOAD at TARGET_LOAD; state exactly fits its blocks.

    `n_state` is `|G| * k`, which is precisely what the disjoint neuron-ID
    blocks occupy. It was `2 * |G| * k` in the first draft -- slack with no
    reason behind it, and expensive: the state area is FULLY MATERIALIZED
    because the blocks are assigned rather than grown, so arc -> state is a
    dense `n_arc x n_state` matrix that every projection traverses. Measured at
    the first draft's sizes, one Z60 cell took 731s of which construction and
    all 15 presentations were 34s; the rest was the L=500 run. Halving n_state
    halves that matrix. This is removing unjustified slack, not tuning a bar --
    TARGET_LOAD, which IS registered, is untouched.
    """
    m = group.order * n_symbols
    return int(round(m * K / TARGET_LOAD)), group.order * K


_W_MAX_DEFAULT = object()   # sentinel: None is a meaningful w_max (unclamped)


def build(group_name, seed, arm, norm_init=False, synaptic_scaling=False,
          organ_p=None, presentations=None, w_max=_W_MAX_DEFAULT,
          refracted_strength=None):
    """`norm_init` defaults to the REGISTERED substrate (False, A1 parity).

    The soft-transition census made the parity clause -- "norm_init exists
    only for self-fibers" -- an open question rather than a premise, so the
    intervention study passes True here. `synaptic_scaling=True` is
    substrate C (PREREG_substrate_c_homeostasis.md Amendment 1): per-round
    write-time homeostasis, the theorems' stated hypothesis.
    `organ_p` / `presentations` / `w_max` exist for PREREG_theorem_regime.md
    (kp and T floors, unclamped weights). ALL defaults reproduce the
    registered protocol byte-for-byte at every existing call site.
    """
    random.seed(seed)
    np.random.seed(seed)
    group = GROUPS[group_name]()
    states, symbols, transitions = word_problem_fsm(group)
    n_arc, n_state = sizes(group, len(symbols))
    beta = 0.0 if arm == "beta0" else BETA

    brain_kw = {}
    if w_max is not _W_MAX_DEFAULT:
        brain_kw["w_max"] = w_max
    brain = Brain(p=0.05, save_winners=True, seed=seed, engine="numpy_sparse",
                  norm_init=norm_init, synaptic_scaling=synaptic_scaling,
                  **brain_kw)
    fsm = NemoArcFSM(brain, states=states, symbols=symbols,
                     transitions=transitions, n=n_arc, k=K, n_state=n_state,
                     beta=beta, organ_p=ORGAN_P if organ_p is None else organ_p,
                     refracted_strength=(REFRACTED
                                         if refracted_strength is None
                                         else refracted_strength),
                     prefix="_wp")

    if arm != "untrained":
        table = [(sym, fr, to) for fr, sym, to in transitions]
        if arm == "shuffled":
            # Train on a PERMUTED table and test against the true one. Catches
            # any route by which the answer could arrive without the trained
            # weights carrying it -- the residual worry after the mod-3
            # retraction, which `untrained` alone does not close.
            rng = random.Random(seed + 9973)
            targets = [to for _s, _f, to in table]
            rng.shuffle(targets)
            table = [(s, f, t) for (s, f, _), t in zip(table, targets)]
        fsm.train_from_list(
            table,
            presentations=PRESENTATIONS if presentations is None
            else presentations)
    return group, fsm, symbols


def soft_hard_census(b, fsm, symbols, group):
    """Every (state, symbol) transition, probed: HARD = wrong label; SOFT =
    right label but the live state assembly is not the intended block.

    ONE owner for the census that `seq_s5_substrate_c.py` and
    `seq_s5_bar_tie.py` both judge -- the soft-pair SET is what the bar-tie
    test compares across readouts, so it must be produced by the same code
    that produced the registered counts. Returns (soft, hard); each soft
    record carries the pair, overlap, intruders and displaced neuron ids.
    """
    import numpy as np
    from neural_assemblies.assembly_calculus.ops import _snap
    from neural_assemblies.programs.word_problems import word_problem_fsm
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
    return soft, hard


def evaluate(group, fsm, symbols, seed, lengths=LENGTHS):
    """Exact-trajectory accuracy, plus the per-step readout it is too coarse for.

    THREE STATISTICS, and the second is the one that measures the machine.

    `exact`      500 consecutive correct steps. 10 binary outcomes over 10
                 seeds, so its CI is +/-0.23 at best -- it cannot resolve a
                 moderate effect, which is why S3 passed underpowered.
    `step`       TRANSITION accuracy: for each i, the expected next state is
                 re-derived from the OBSERVED previous state, not the true one.
                 Raw agreement with `truth` would conflate ONE derailment with
                 many errors, since a machine that leaves the correct state
                 stays wrong afterwards through no further fault of its own.
                 ~500 observations per cell instead of 1.
    `first_bad`  the step at which the trajectory first leaves ground truth,
                 or the length if it never does. Distinguishes "derails early
                 and drifts" from "runs clean then slips once".
    """
    rng = random.Random(seed + 4242)
    longest = max(lengths)
    word = [rng.choice(symbols) for _ in range(longest)]
    truth = true_trajectory(group, word)
    start = group.label(group.identity)
    got = fsm.run(word, start_state=start)

    _states, _symbols, transitions = word_problem_fsm(group)
    table = {(fr, sym): to for fr, sym, to in transitions}
    prev, correct = start, []
    for sym, obs in zip(word, got):
        correct.append(obs == table[(prev, sym)])
        prev = obs

    out = {}
    for L in lengths:
        out[str(L)] = bool(got[:L] == truth[:L])
        out[f"step{L}"] = float(np.mean(correct[:L]))
    diverge = next((i for i, (a, b) in enumerate(zip(got, truth)) if a != b),
                   longest)
    out["first_bad"] = int(diverge)
    return out


def worker(group_name, seed, arm):
    group, fsm, symbols = build(group_name, seed, arm)
    return evaluate(group, fsm, symbols, seed)


def report_regime(group_name, seed=42):
    group, fsm, symbols = build(group_name, seed, "untrained")
    brain = fsm.brain
    driven = {fsm.arc_area: [fsm.state_area, fsm._sym_stim[symbols[0]]],
              fsm.state_area: [fsm.arc_area]}
    n_arc, n_state = sizes(group, len(symbols))
    load = group.order * len(symbols) * K / n_arc
    print(f"    {group_name:7s} order={group.order:3d} "
          f"solvable={str(group.solvable):5s} n_arc={n_arc:6d} "
          f"n_state={n_state:6d} arc load={load:.3f}")
    print(format_report([r for r in regime_audit(brain, driven)
                         if r.area.startswith("_wp")]))


def _dense_gb(group_name):
    """Bytes the two organ fibers occupy DENSE, per worker.

    The state area is fully materialised (its assemblies are assigned blocks),
    so arc <-> state is stored dense at `n_arc x n_state` in each direction.
    """
    group = GROUPS[group_name]()
    n_arc, n_state = sizes(group, len(group.generators))
    return n_arc * n_state * 4 * 2 / 1e9


def run_tiered(cells, budget_gb=20.0, worker_fn=None):
    """Run cells in memory tiers, sized so a big group cannot exhaust RAM.

    The first attempt ran all 160 cells at the pool's default width and died:
    S5 needs 2.69 GB of dense fiber per worker, 37.6 GB across 14 workers, and
    growth REALLOCATES -- `_ensure_area_block_coverage` builds the new buffer
    while the old one is still live -- so the peak is about twice that. The
    order-60 groups are 0.67 GB each and were never the problem.

    Cells are grouped by footprint and each tier gets `budget_gb / footprint`
    workers. This is scheduling, not sizing: no cell's parameters change, so
    the measurement is identical to running them one at a time.
    """
    out = {}
    tiers = {}
    for cell in cells:
        tiers.setdefault(round(_dense_gb(cell[0]), 2), []).append(cell)
    for gb, group_cells in sorted(tiers.items()):
        # Capped by CORES as well as by memory -- the memory budget alone said
        # 29 workers for the order-60 tier on a 16-core box.
        #
        # PRICED AT THE PEAK, NOT THE NOMINAL. Growth REALLOCATES: the new
        # buffer is built while the old is live, so a worker's true peak is
        # ~2x its resident footprint. Budgeting the nominal put 7 S5 workers
        # (18.8 GB nominal, ~37 GB peak) against ~45 GB free -- it survived
        # twice and lost the race on the third run, mid-tier, on a 1.25 GiB
        # allocation. A scheduler that works twice and dies on the third
        # identical invocation is pricing the wrong quantity.
        workers = max(1, min(len(group_cells),
                             int(budget_gb // max(2.0 * gb, 0.01)),
                             max(1, (os.cpu_count() or 4) - 2)))
        print(f"    [tier {gb:.2f} GB/worker] {len(group_cells)} cells, "
              f"{workers} workers", flush=True)
        out.update(run_cells(worker_fn or worker, group_cells,
                             max_workers=workers))
    return out


def _acc(results, group_name, arm, seeds, length):
    return ensemble_from_values(
        [float(results[(group_name, s, arm)][str(length)]) for s in seeds],
        f"{group_name}/{arm}@L={length}", keys=seeds)


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== finite-group word problems on the A1 organ ===")
    print(f"    k={K} beta={BETA} organ_p={ORGAN_P} presentations="
          f"{PRESENTATIONS}, seeds {seeds[0]}..{seeds[-1]}")
    print(f"    n_arc set per group to hold load at {TARGET_LOAD} "
          f"(Amendment 1)\n")
    print("  [regime] per group, at the first seed")
    for g in GROUP_NAMES:
        report_regime(g, seeds[0])

    out = {"seeds": seeds, "lengths": LENGTHS, "target_load": TARGET_LOAD}

    # -- CONTROLS FIRST, as committed --------------------------------------
    print("\n  [controls] must all sit at chance before any real arm counts")
    ctrl_cells = [(g, s, a) for g in GROUP_NAMES for s in seeds
                  for a in ("untrained", "beta0", "shuffled")]
    r = run_tiered(ctrl_cells)
    ctrl_ok, ctrl_rows = True, {}
    for g in GROUP_NAMES:
        for a in ("untrained", "beta0", "shuffled"):
            e = _acc(r, g, a, seeds, LENGTHS[0])
            ctrl_rows[f"{g}/{a}"] = [e.mean, e.ci]
            bad = e.mean > 0.0
            ctrl_ok &= not bad
            print(f"    {g:7s} {a:10s} L={LENGTHS[0]:3d} exact "
                  f"{e.mean:.3f} +/- {e.ci:.3f}{'   <-- NOT AT CHANCE' if bad else ''}")
    out["controls"] = ctrl_rows
    out["controls_clean"] = ctrl_ok
    print(f"    S4 controls {'PASS' if ctrl_ok else 'FAIL'}")
    if not ctrl_ok:
        print("\n  A control produced a correct trajectory. The organ is not "
              "what is answering; the real arms are NOT run.")
        _write(out)
        return

    # -- the real arms ------------------------------------------------------
    print("\n  [trained] exact trajectory accuracy, full length curve")
    r.update(run_tiered([(g, s, "trained") for g in GROUP_NAMES
                         for s in seeds]))
    curve = {}
    header = "  ".join(f"L={L}" for L in LENGTHS)
    print(f"\n    {'group':8s} {'solvable':9s} {header}")
    for g in GROUP_NAMES:
        row = [_acc(r, g, "trained", seeds, L) for L in LENGTHS]
        curve[g] = {str(L): [e.mean, e.ci] for L, e in zip(LENGTHS, row)}
        solv = GROUPS[g]().solvable
        cells = "  ".join(f"{e.mean:.2f}+/-{e.ci:.2f}" for e in row)
        print(f"    {g:8s} {str(solv):9s} {cells}", flush=True)
    out["curve"] = curve

    print("\n  [per-step] TRANSITION accuracy -- what `exact` is too coarse for")
    print(f"\n    {'group':8s} {'solvable':9s} {'step@500':>20s} "
          f"{'first divergence':>20s}")
    per_step = {}
    for g in GROUP_NAMES:
        e = ensemble_from_values(
            [r[(g, s, "trained")]["step500"] for s in seeds],
            f"{g}/step", keys=seeds)
        fb = ensemble_from_values(
            [float(r[(g, s, "trained")]["first_bad"]) for s in seeds],
            f"{g}/first_bad", keys=seeds)
        per_step[g] = {"step500": [e.mean, e.ci], "first_bad": [fb.mean, fb.ci]}
        print(f"    {g:8s} {str(GROUPS[g]().solvable):9s} "
              f"{e.mean:.5f}+/-{e.ci:.5f}  {fb.mean:10.1f}+/-{fb.ci:.1f}",
              flush=True)
    out["per_step"] = per_step
    solv = [per_step[g]["step500"][0] for g in GROUP_NAMES
            if GROUPS[g]().solvable]
    hard = [per_step[g]["step500"][0] for g in GROUP_NAMES
            if not GROUPS[g]().solvable]
    gap = float(np.mean(solv) - np.mean(hard))
    print(f"    solvable {np.mean(solv):.5f}  non-solvable {np.mean(hard):.5f}"
          f"  gap {gap:+.5f}")
    out["solvability_gap_per_step"] = gap

    print("\n=== BARS ===")
    a5_100 = _acc(r, "A5", "trained", seeds, 100)
    s1 = sum(r[("A5", s, "trained")]["100"] for s in seeds) >= 9
    print(f"  {'PASS' if s1 else 'FAIL'}  S1 A5 exact at L=100 >= 9/10 seeds "
          f"(got {sum(r[('A5', s, 'trained')]['100'] for s in seeds)}/{len(seeds)},"
          f" mean {a5_100.mean:.3f})")

    s2 = all(_acc(r, g, "trained", seeds, 500).low
             >= _acc(r, g, "trained", seeds, 10).low - 1e-9
             or _acc(r, g, "trained", seeds, 500)
             .indistinguishable_from(_acc(r, g, "trained", seeds, 10).mean)
             for g in GROUP_NAMES)
    print(f"  {'PASS' if s2 else 'FAIL'}  S2 L=500 indistinguishable from L=10")

    solvable_mean = np.mean([_acc(r, g, "trained", seeds, 100).mean
                             for g in GROUP_NAMES if GROUPS[g]().solvable])
    hard = [_acc(r, g, "trained", seeds, 100) for g in GROUP_NAMES
            if not GROUPS[g]().solvable]
    s3 = all(e.indistinguishable_from(float(solvable_mean)) or e.low
             >= solvable_mean for e in hard)
    print(f"  {'PASS' if s3 else 'FAIL'}  S3 accuracy does not depend on "
          f"solvability (solvable mean {solvable_mean:.3f} at L=100)")
    print(f"  {'PASS' if ctrl_ok else 'FAIL'}  S4 controls at chance")
    out["verdicts"] = {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3),
                       "S4": bool(ctrl_ok)}
    _write(out)


def _write(out):
    path = os.path.join(_HERE, "seq_s5_word_problem_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
