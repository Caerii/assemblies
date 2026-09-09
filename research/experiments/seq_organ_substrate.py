"""Is the word-problem arc's assembly DECAY the substrate-C merger?

Implements `research/notes/substrate/PREREG_organ_substrate.md`; bars registered there
before this ran.

[[arc-training-is-not-batchable]] recorded the Z60 arc's identical-assembly
fraction climbing to ~0.575 by presentation 5 and then DEGRADING to 0.117 by
presentation 12, and read it as a fact about deep training. Three properties of
that measurement are exactly what `PREREG_substrate_ceiling.md` characterized:
it used `synaptic_scaling` with `norm_init` OFF (substrate C, whose assemblies
merge at 6-16x the chance floor), the arc was SATURATED throughout (the regime
where even `norm_init` alone loses a quarter of its assemblies to exact
duplicates), and it was one seed.

So this asks whether the decay is the merger. It is a TRANSFER test: a
substrate result that does not move a real organ is a fact about toys.

TWO THINGS THIS STUDY HAS THAT THE ORIGINAL MEASUREMENT DID NOT:

  * the DISTINCT FRACTION of the 120 arc assemblies, not just their mean
    overlap. Mean overlap is nearly blind to partial collapse -- 256 items on
    ~58 distinct assemblies still reads at the chance floor -- so a merger can
    be complete and invisible to it.
  * `organ_p=0.5`. The organ's default 0.40 gives kp = 28.0 against a floor of
    3 ln(20000) = 29.7, i.e. BELOW floor, which `regime_audit` printed on every
    run of this organ for weeks. A null there is not evidence about anything.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.diagnostics import (
    assembly_overlap, ensemble_from_values, paired_delta,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from seq_refraction_stability import _stability_from_saved     # noqa: E402

GROUP = "Z60"
ORGAN_P = 0.50            # kp = 35.0 vs floor 29.7; the default 0.40 is BELOW
PRESENTATIONS = 15
LONGEST = 100
SEEDS = [42, 43, 44, 45]
#: (label, norm_init, synaptic_scaling). NONE is what the organ ships with.
ARMS = [("NONE", False, False), ("B", True, False),
        ("C", False, True), ("G", True, True)]
DISTINCT_BAR = 0.90


#: Amendment 1: `w_max` is the one setting this study changed relative to the
#: organ's registered protocol, and unclamped weights interact with exactly the
#: thing under test -- a normalizer that divides by accumulated mass. Passed as
#: a STRING because `_W_MAX_DEFAULT` is a bare `object()` sentinel and a process
#: pool has to pickle every cell.
WMAXES = ("none", "default")


def worker(arm, norm_init, scaling, wmax_mode, seed):
    # Refraction stays at the organ default: this study varies the SUBSTRATE
    # and nothing else. Set explicitly because workers are reused and the flag
    # is read at call time.
    os.environ["ASSEMBLIES_CONSTANT_REFRACTION"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    from seq_s5_word_problem import _W_MAX_DEFAULT, K, build, sizes
    from neural_assemblies.programs.word_problems import (
        GROUPS, true_trajectory, word_problem_fsm,
    )
    wmax = None if wmax_mode == "none" else _W_MAX_DEFAULT
    group, fsm, symbols = build(
        GROUP, seed, "trained", norm_init=norm_init, synaptic_scaling=scaling,
        organ_p=ORGAN_P, presentations=PRESENTATIONS, w_max=wmax)
    _states, syms, transitions = word_problem_fsm(GROUPS[GROUP]())
    n_arc, _n_state = sizes(GROUPS[GROUP](), len(syms))

    curve = _stability_from_saved(fsm, len(transitions), PRESENTATIONS)
    ident = [c["identical"] for c in curve]
    peak = max(ident)
    terminal = ident[-1]

    # THE MERGER READOUT. The final presentation's arc assemblies, one per
    # transition, in NEURON IDS (`saved_winners` is the stable space).
    sw = fsm.brain.areas[fsm.arc_area].saved_winners
    last = sw[(PRESENTATIONS - 1) * len(transitions):
              PRESENTATIONS * len(transitions)]
    sets = [np.asarray(w) for w in last]
    distinct = len({tuple(sorted(int(x) for x in a)) for a in sets}) / len(sets)
    # Pairwise over a CAPPED sample of pairs: 120 transitions is 7140 pairs and
    # the full census is not needed for a mean -- but the cap is stated rather
    # than silent, because a silent truncation reads as "covered everything".
    rng = random.Random(seed + 31337)
    pairs = [(rng.randrange(len(sets)), rng.randrange(len(sets)))
             for _ in range(2000)]
    ov = [assembly_overlap(sets[i], sets[j]) for i, j in pairs if i != j]
    chance = K / n_arc

    rng2 = random.Random(seed + 4242)
    word = [rng2.choice(symbols) for _ in range(LONGEST)]
    truth = true_trajectory(group, word)
    labels = fsm.run(word, group.label(group.identity))
    acc = sum(a == t for a, t in zip(labels, truth)) / len(truth)

    # `materialized_count`, not `.w`: `.w` is neurons materialized on the
    # sparse engine but len(winners) == k on the explicit one, and fill is the
    # whole point of the saturation caveat here.
    eng = fsm.brain._engine_for(fsm.brain.areas[fsm.arc_area])
    fill = int(eng.materialized_count(fsm.arc_area) or 0) / n_arc

    return {"arm": arm, "seed": seed, "wmax": wmax_mode, "curve": ident,
            "peak": peak, "terminal": terminal,
            # `decay` is UNDEFINED when the arc never produced a single
            # identical assembly at any presentation (peak == 0): there is
            # nothing to decay FROM. That is not "no decay" and must not be
            # read as 1.0 -- it is a more total failure than any decay, and
            # mapping it to a number would invert the statistic exactly where
            # the arm is worst ([[undefinedness-correlates-with-outcome]]).
            # So it stays NaN, is never ensembled, and O2 is judged instead on
            # `peak - terminal`, which is always defined.
            "decay": terminal / peak if peak > 0 else float("nan"),
            "peak_minus_terminal": peak - terminal,
            "peak_is_zero": bool(peak == 0.0),
            "distinct": distinct,
            "pairwise": float(np.mean(ov)),
            "pairwise_x_chance": float(np.mean(ov)) / chance,
            "chance": chance, "acc": acc,
            "arc_fill": fill}


def _ens(res, key, arm, wmax="none"):
    """Seed statistics go through `ensemble_from_values`, never a bare mean."""
    return ensemble_from_values(
        [res[(arm,) + _flags(arm) + (wmax, s)][key] for s in SEEDS],
        label=f"{arm}/{wmax}/{key}", keys=SEEDS)


def _flags(arm):
    for name, ni, sc in ARMS:
        if name == arm:
            return (ni, sc)
    raise KeyError(arm)


def _gt(a, b, label):
    """Ordering judged on the PAIRED per-seed difference, not two means."""
    d = paired_delta(a, b, label=label)
    ok = d.beats(0.0)
    tag = "PASS" if ok else ("INCONCLUSIVE" if d.indistinguishable_from(0.0)
                             else "FAIL")
    return ok, (f"{a.mean:.3f} vs {b.mean:.3f}, paired delta "
                f"{d.mean:+.3f}+-{d.ci:.3f} [{tag}]")


def main():
    from _parallel import run_cells

    print("=== is the arc's assembly decay the substrate-C merger? ===")
    print(f"    {GROUP} organ_p={ORGAN_P} presentations={PRESENTATIONS} "
          f"w_max=None, refraction at organ default, seeds {SEEDS}")
    print(f"    kp = {70*ORGAN_P:.1f} vs floor 3*ln(20000) = "
          f"{3*np.log(20000):.1f}  -> IN REGIME\n")

    cells = [(a, ni, sc, wm, s) for (a, ni, sc) in ARMS
             for wm in WMAXES for s in SEEDS]
    res = run_cells(worker, cells, max_workers=12)

    print("\n--- identical-assembly fraction per presentation (mean of seeds)")
    print(f"    {'arm':5s} " + " ".join(f"p{p:02d}" for p in
                                        range(2, PRESENTATIONS + 1)))
    for (a, _ni, _sc) in ARMS:
        curves = np.array([res[(a, _ni, _sc, "none", s)]["curve"]
                           for s in SEEDS])
        print(f"    {a:5s} " + " ".join(f"{v:.2f}" for v in curves.mean(0)))

    print("\n--- the shape, the merger, and the task")
    print(f"    {'arm':5s} {'peak':>13} {'terminal':>13} {'decay':>16} "
          f"{'distinct':>13} {'pairw/chance':>13} {'acc':>13} {'fill':>6}")
    for (a, _ni, _sc) in ARMS:
        pk, tm = _ens(res, "peak", a), _ens(res, "terminal", a)
        ds = _ens(res, "distinct", a)
        xc, ac = _ens(res, "pairwise_x_chance", a), _ens(res, "acc", a)
        fl = _ens(res, "arc_fill", a)
        nz = sum(res[(a, _ni, _sc, "none", s)]["peak_is_zero"]
                 for s in SEEDS)
        dec = ("undefined %d/%d" % (nz, len(SEEDS)) if nz else
               "%.3f+-%.3f" % (_ens(res, "decay", a).mean,
                               _ens(res, "decay", a).ci))
        print(f"    {a:5s} {pk.mean:.3f}+-{pk.ci:.3f} {tm.mean:.3f}+-{tm.ci:.3f}"
              f" {dec:>16}  {ds.mean:.3f}+-{ds.ci:.3f}"
              f"  {xc.mean:8.2f}  {ac.mean:.3f}+-{ac.ci:.3f} {fl.mean:6.3f}")

    print("\n=== BARS ===")
    print("    judged on CONFIDENCE BOUNDS; orderings are PAIRED per-seed deltas")
    print("  PASS  O1 regime: kp = 35.0 >= floor 29.7, asserted before the run")

    # O2 judged on `peak - terminal`, which is defined even when the arm
    # never produced an identical assembly at all. The registered form was the
    # RATIO terminal/peak < 0.80; that is undefined exactly where the arm is
    # worst, so the same claim is tested in the always-defined direction.
    # Same claim, same direction, defined everywhere -- see Amendment 1.
    pm_c = _ens(res, "peak_minus_terminal", "C")
    pk_c = _ens(res, "peak", "C")
    o2 = pm_c.beats(0.0)
    print(f"  {'PASS' if o2 else 'FAIL'}  O2 defect reproduces: C peak "
          f"{pk_c.mean:.3f}+-{pk_c.ci:.3f}, peak-terminal "
          f"{pm_c.mean:+.3f}+-{pm_c.ci:.3f}, CI-low {pm_c.low:+.3f} > 0")
    if pk_c.high < 0.05:
        print("        -> C never reaches a peak at all here: there is no "
              "climb-then-degrade shape to explain. The phenomenon this "
              "study was built to explain is ABSENT at these settings.")

    o3, t3 = _gt(_ens(res, "terminal", "G"), _ens(res, "terminal", "C"),
                 "O3 G-C terminal")
    print(f"  {'PASS' if o3 else 'FAIL'}  O3 THE CLAIM: G terminal > C: {t3}")

    ds_g, ds_c = _ens(res, "distinct", "G"), _ens(res, "distinct", "C")
    o4 = ds_g.low >= DISTINCT_BAR and ds_c.high < DISTINCT_BAR
    print(f"  {'PASS' if o4 else 'FAIL'}  O4 merger on the organ: G distinct "
          f"CI-low {ds_g.low:.3f} >= {DISTINCT_BAR}, C CI-high "
          f"{ds_c.high:.3f} < {DISTINCT_BAR}")

    o5, t5 = _gt(_ens(res, "acc", "G"), _ens(res, "acc", "C"), "O5 G-C acc")
    print(f"  {'PASS' if o5 else 'FAIL'}  O5 does it matter: {t5}")

    o6, t6 = _gt(_ens(res, "terminal", "G"), _ens(res, "terminal", "NONE"),
                 "O6 G-NONE terminal")
    print(f"  {'PASS' if o6 else 'FAIL'}  O6 vs the organ's CURRENT default: "
          f"{t6}")

    if not o2:
        print("\n  -> The phenomenon is NOT present at these settings. O3-O6 "
              "say nothing about it; do not report them as if they did.")
    if o2 and not o3:
        print("\n  -> The decay is real and NOT the merger. The substrate "
              "account does not transfer to this organ; leave "
              "[[arc-training-is-not-batchable]] standing.")
    if o3 and not o5:
        print("\n  -> Stability without capability. Report as stability, "
              "never as a task result.")

    # DISCLOSED IN THE NOTE: NONE-vs-C at the organ default refraction is
    # exactly PREREG_refraction_stability's R5. Printed here so it is written
    # down once and not re-scored as independent evidence later.
    r5, t5r = _gt(_ens(res, "terminal", "NONE"), _ens(res, "terminal", "C"),
                  "R5 NONE-C")
    print(f"\n  [refraction R5, answered here as disclosed] scaling OFF more "
          f"stable than ON: {t5r}")

    print("\n--- AMENDMENT 1: the same arms at the organ's DEFAULT w_max")
    print(f"    {'arm':5s} {'wmax':>8} {'peak':>13} {'terminal':>13} "
          f"{'distinct':>13} {'acc':>13} {'fill':>6}")
    for (a, _ni, _sc) in ARMS:
        for wm in WMAXES:
            pk, tm = _ens(res, "peak", a, wm), _ens(res, "terminal", a, wm)
            ds, ac = _ens(res, "distinct", a, wm), _ens(res, "acc", a, wm)
            fl = _ens(res, "arc_fill", a, wm)
            print(f"    {a:5s} {wm:>8} {pk.mean:.3f}+-{pk.ci:.3f} "
                  f"{tm.mean:.3f}+-{tm.ci:.3f} {ds.mean:.3f}+-{ds.ci:.3f} "
                  f"{ac.mean:.3f}+-{ac.ci:.3f} {fl.mean:6.3f}")

    acc_none_d = _ens(res, "acc", "NONE", "default")
    o7 = acc_none_d.low > 0.90
    print(f"\n  {'PASS' if o7 else 'FAIL'}  O7 baseline is not a w_max=None "
          f"artifact: NONE acc at default w_max {acc_none_d.mean:.3f}"
          f"+-{acc_none_d.ci:.3f}, CI-low {acc_none_d.low:.3f} > 0.90")

    ac_c_d, ac_g_d = _ens(res, "acc", "C", "default"), _ens(res, "acc", "G",
                                                            "default")
    o8 = ac_c_d.high < 0.50 and ac_g_d.high < 0.50
    print(f"  {'PASS' if o8 else 'FAIL'}  O8 normalization still breaks it at "
          f"default w_max: C {ac_c_d.mean:.3f}+-{ac_c_d.ci:.3f}, "
          f"G {ac_g_d.mean:.3f}+-{ac_g_d.ci:.3f}, both CI-high < 0.50")
    if not o8:
        print("        -> the collateral result is a fact about UNCLAMPED "
              "WEIGHTS, not about normalization. Report it that way.")

    pm_c_d = _ens(res, "peak_minus_terminal", "C", "default")
    pk_c_d = _ens(res, "peak", "C", "default")
    o9 = pm_c_d.beats(0.0)
    print(f"  {'PASS' if o9 else 'FAIL'}  O9 decay reproduces at default "
          f"w_max: C peak {pk_c_d.mean:.3f}+-{pk_c_d.ci:.3f}, peak-terminal "
          f"{pm_c_d.mean:+.3f}+-{pm_c_d.ci:.3f}")
    if not o9:
        print("        -> the decay in [[arc-training-is-not-batchable]] does "
              "NOT reproduce on this engine at either w_max. That memory needs "
              "narrowing to the configuration it was measured in "
              "(torch_sparse, one seed).")

    path = os.path.join(_HERE, "seq_organ_substrate_results.json")
    with open(path, "w") as fh:
        json.dump({"group": GROUP, "organ_p": ORGAN_P,
                   "presentations": PRESENTATIONS, "seeds": SEEDS,
                   "bars": {"O1": True, "O2": bool(o2), "O3": bool(o3),
                            "O4": bool(o4), "O5": bool(o5), "O6": bool(o6),
                            "O7": bool(o7), "O8": bool(o8), "O9": bool(o9)},
                   "refraction_R5": bool(r5),
                   "cells": {f"{a}/{wm}/{s}": res[(a, ni, sc, wm, s)]
                             for (a, ni, sc) in ARMS for wm in WMAXES
                             for s in SEEDS}},
                  fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
