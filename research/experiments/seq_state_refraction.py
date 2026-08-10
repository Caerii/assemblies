"""Does refracting the STATE area stop the induced-state collapse?

Implements `research/notes/PREREG_state_refraction.md`. Parameters inherited
from A3 are FIXED; `state_refracted_strength` and `n_state` are swept as design
axes and the WHOLE GRID is reported.

THE TRAP THIS IS BUILT AROUND. Low state overlap is not the goal. A state area
that emitted a fresh random assembly every step would score a perfect
separation and carry nothing -- the [[fake-perfect-probe-signatures]] shape. So
each cell reports two statistics and must win BOTH:

    separation    1 - overlap across DIFFERENT prefixes at the same position
    determinism   overlap for the SAME prefix run TWICE

Collapse kills separation. Noise kills determinism. Only a real state code
holds both.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import (
    assembly_overlap, ensemble_from_values, read_assembly,
)
from neural_assemblies.programs.sequence_transducer import SequenceTransducer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "study4"))
sys.path.insert(0, _HERE)
import ntp  # noqa: E402
from _parallel import run_cells  # noqa: E402

N, K, P, BETA = ntp.N, ntp.K, ntp.P, ntp.BETA
TRAIN_ROUNDS, GROUND_ROUNDS = ntp.TRAIN_ROUNDS, ntp.GROUND_ROUNDS
VOCAB_SIZE, N_TRAIN, N_PROBE = 50, 50, 12
N_ARC, ORGAN_P = 10000, 0.20
SEEDS = list(range(42, 52))

STRENGTHS = [0.0, 0.02, 0.05, 0.10, 0.20]
N_STATES = [2000, 10000]

SEPARATION_BAR, DETERMINISM_BAR, G0_BAR = 0.5, 0.9, 0.2


def build(seed, *, n_state, strength):
    random.seed(seed)
    np.random.seed(seed)
    words = ntp.vocabulary(VOCAB_SIZE)
    keep = set(words)
    train = [[w for w in s if w in keep] for s in ntp.generate(N_TRAIN, seed)]
    probe = [[w for w in s if w in keep]
             for s in ntp.generate(N_PROBE, seed + 500)]
    brain = Brain(p=P, seed=seed, engine="numpy_sparse")
    t = SequenceTransducer(brain, words, n=N, n_arc=N_ARC, n_state=n_state,
                           k=K, beta=BETA, organ_p=ORGAN_P,
                           state_refracted_strength=strength)
    t.ground(rounds=GROUND_ROUNDS)
    for s in train:
        t.train_sentence(s, rounds=TRAIN_ROUNDS)
    return brain, t, probe


def _states_along(t, sentence, position):
    """State assembly after consuming `sentence[:position+1]`, in neuron IDs."""
    t.reset()
    for w in sentence[:position + 1]:
        t.tick(w, rounds=TRAIN_ROUNDS)
        t.emit()
    return read_assembly(t.brain, t.state_area)


def measure(brain, t, probe, position=1):
    """Separation across prefixes and determinism within one, both under probe.

    Reads run inside `probe()`, so no refraction bias is charged while
    measuring -- otherwise the first read would move the second and
    determinism would be measuring the instrument.
    """
    usable = [s for s in probe if len(s) > position + 1]
    if len(usable) < 3:
        return float("nan"), float("nan"), 0
    with brain.probe():
        firsts = [_states_along(t, s, position) for s in usable]
        repeats = [_states_along(t, s, position) for s in usable]
    cross = [assembly_overlap(firsts[i], firsts[j])
             for i in range(len(firsts)) for j in range(i + 1, len(firsts))]
    same = [assembly_overlap(a, b) for a, b in zip(firsts, repeats)]
    distinct = _distinct_count(firsts)
    return 1.0 - float(np.mean(cross)), float(np.mean(same)), distinct


def _distinct_count(assemblies, threshold=0.5):
    """Greedy count of distinct state assemblies -- the load's numerator."""
    seen = []
    for a in assemblies:
        if not any(assembly_overlap(a, o) >= threshold for o in seen):
            seen.append(a)
    return len(seen)


def worker(seed, n_state, strength):
    brain, t, probe = build(seed, n_state=n_state, strength=strength)
    sep, det, distinct = measure(brain, t, probe)
    return {"separation": sep, "determinism": det,
            "load": distinct * K / n_state}


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== does refracting the STATE area stop the collapse? ===")
    print(f"    n={N} k={K} n_arc={N_ARC} organ_p={ORGAN_P} beta={BETA}")
    print(f"    n_train={N_TRAIN} (mechanism scale), seeds "
          f"{seeds[0]}..{seeds[-1]}")
    print("    a cell must clear BOTH separation and determinism\n")

    # -- G0, FIRST, as committed -------------------------------------------
    print("  [G0] control must reproduce the collapse")
    r0 = run_cells(worker, [(s, N_STATES[0], 0.0) for s in seeds])
    ctrl_sep = ensemble_from_values(
        [r0[(s, N_STATES[0], 0.0)]["separation"] for s in seeds],
        "control separation", keys=seeds)
    ctrl_det = ensemble_from_values(
        [r0[(s, N_STATES[0], 0.0)]["determinism"] for s in seeds],
        "control determinism", keys=seeds)
    print(f"    {ctrl_sep}\n    {ctrl_det}")
    g0 = ctrl_sep.high < G0_BAR
    print(f"    G0 {'PASS' if g0 else 'FAIL'} "
          f"(separation upper {ctrl_sep.high:.4f} < {G0_BAR})")
    out = {"seeds": seeds, "n_train": N_TRAIN,
           "control": {"separation": ctrl_sep.mean, "determinism": ctrl_det.mean},
           "g0": g0}
    if not g0:
        print("\n  G0 FAILED -- the mechanism scale does not exhibit the "
              "collapse, so there is nothing here to rescue. Grid NOT run.")
        _write(out)
        return

    # -- the grid, reported whole ------------------------------------------
    print("\n  [grid] full sweep, every cell reported")
    cells = [(s, ns, st) for ns in N_STATES for st in STRENGTHS
             for s in seeds if not (ns == N_STATES[0] and st == 0.0)]
    r = dict(r0)
    r.update(run_cells(worker, cells))

    grid, winners = {}, []
    print(f"\n    {'n_state':>8s} {'strength':>9s} {'separation':>20s} "
          f"{'determinism':>20s} {'load':>7s}")
    for ns in N_STATES:
        for st in STRENGTHS:
            sep = ensemble_from_values(
                [r[(s, ns, st)]["separation"] for s in seeds], "sep", keys=seeds)
            det = ensemble_from_values(
                [r[(s, ns, st)]["determinism"] for s in seeds], "det", keys=seeds)
            load = ensemble_from_values(
                [r[(s, ns, st)]["load"] for s in seeds], "load", keys=seeds)
            ok = sep.low > SEPARATION_BAR and det.low > DETERMINISM_BAR
            if ok:
                winners.append((ns, st))
            grid[f"{ns}/{st}"] = {
                "separation": [sep.mean, sep.ci], "determinism": [det.mean, det.ci],
                "load": [load.mean, load.ci], "clears_both": ok,
            }
            print(f"    {ns:>8d} {st:>9.2f} "
                  f"{sep.mean:>10.4f} +/- {sep.ci:<6.4f} "
                  f"{det.mean:>10.4f} +/- {det.ci:<6.4f} "
                  f"{load.mean:>7.3f}  {'<-- BOTH' if ok else ''}", flush=True)
    out["grid"] = grid

    print("\n=== BARS ===")
    r1 = bool(winners)
    print(f"  {'PASS' if r1 else 'FAIL'}  R1 some cell clears BOTH bars"
          f"{'  ' + str(winners) if winners else ''}")
    print(f"  {'PASS' if g0 else 'FAIL'}  R2 control fails R1")
    print("  R3 (mechanism): separation against achieved load --")
    for key, v in grid.items():
        print(f"        load {v['load'][0]:.3f} -> separation "
              f"{v['separation'][0]:.4f}   [{key}]")
    out["verdicts"] = {"R1": r1, "R2": g0}
    out["winners"] = winners
    _write(out)


def _write(out):
    path = os.path.join(_HERE, "seq_state_refraction_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
