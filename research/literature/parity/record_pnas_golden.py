#!/usr/bin/env python3
"""Record PNAS extended parity metrics at pinned CI parameters (seed=42)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from neural_assemblies.assembly_calculus import (
    associate,
    chance_overlap,
    merge,
    overlap,
    pattern_complete,
    project,
    reciprocal_project,
    separate,
)
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "literature" / "parity" / "reference_pnas_golden.json"

SEED = 42
N = 5000
K = 80
P = 0.05
BETA = 0.1
# 20 formation rounds (up from 10): under norm_init, assemblies are robust
# attractors only with adequate reinforcement. At 10 rounds the associate
# binding is under-reinforced (cue overlap collapses to ~0.03) and project
# persistence sits at 0.95, so the pre-norm_init golden of 10 rounds pinned a
# hub-substrate artifact. 20 rounds restores the primitives (persistence 1.0,
# associate ~0.99, merge 1.0) while staying below the point (>=25 rounds) where
# the legacy no-norm_init reference over-consolidates its separate protocol.
ROUNDS = 20


def _brain(**kw) -> Brain:
    return Brain(p=P, save_winners=True, seed=SEED, engine="numpy_sparse", **kw)


def metrics_project_separate() -> dict:
    b = _brain()
    b.add_stimulus("s1", K)
    b.add_stimulus("s2", K)
    b.add_area("A", N, K, BETA)
    _, _, sep_ov = separate(b, "s1", "s2", "A", rounds=ROUNDS)

    b2 = _brain()
    b2.add_stimulus("s", K)
    b2.add_area("A", N, K, BETA)
    asm1 = project(b2, "s", "A", rounds=ROUNDS)
    asm2 = project(b2, "s", "A", rounds=ROUNDS)
    return {
        "project_persistence": overlap(asm1, asm2),
        "separate_overlap": sep_ov,
        "chance_overlap": chance_overlap(K, N),
    }


def metrics_associate() -> dict:
    b = _brain()
    b.add_stimulus("stimA", K)
    b.add_stimulus("stimB", K)
    b.add_area("A", N, K, BETA)
    b.add_area("B", N, K, BETA)
    b.add_area("C", N, K, BETA)
    project(b, "stimA", "A", rounds=ROUNDS)
    project(b, "stimB", "B", rounds=ROUNDS)
    associate(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)

    b_copy1 = copy.deepcopy(b)
    b_copy1.project({"stimA": ["A"]}, {"A": ["C"]})
    for _ in range(5):
        b_copy1.project({}, {"A": ["C"], "C": ["C"]})
    c1 = _snap(b_copy1, "C")

    b_copy2 = copy.deepcopy(b)
    b_copy2.project({"stimB": ["B"]}, {"B": ["C"]})
    for _ in range(5):
        b_copy2.project({}, {"B": ["C"], "C": ["C"]})
    c2 = _snap(b_copy2, "C")

    measured = overlap(c1, c2)
    ch = chance_overlap(K, N)
    return {
        "associate_cue_overlap": measured,
        "associate_above_chance_factor": measured / ch if ch else 0.0,
    }


def metrics_merge() -> dict:
    b = _brain()
    b.add_stimulus("stimA", K)
    b.add_stimulus("stimB", K)
    b.add_area("A", N, K, BETA)
    b.add_area("B", N, K, BETA)
    b.add_area("C", N, K, BETA)
    project(b, "stimA", "A", rounds=ROUNDS)
    project(b, "stimB", "B", rounds=ROUNDS)
    merged = merge(b, "A", "B", "C", stim_a="stimA", stim_b="stimB", rounds=ROUNDS)

    b_copy1 = copy.deepcopy(b)
    b_copy1.areas["A"].fix_assembly()
    b_copy1.project({}, {"A": ["C"]})
    for _ in range(5):
        b_copy1.project({}, {"A": ["C"], "C": ["C"]})
    c1 = _snap(b_copy1, "C")

    b_copy2 = copy.deepcopy(b)
    b_copy2.areas["B"].fix_assembly()
    b_copy2.project({}, {"B": ["C"]})
    for _ in range(5):
        b_copy2.project({}, {"B": ["C"], "C": ["C"]})
    c2 = _snap(b_copy2, "C")

    measured = overlap(c1, c2)
    ch = chance_overlap(K, N)
    return {
        "merge_cue_overlap": measured,
        "merge_above_chance_factor": measured / ch if ch else 0.0,
        "merge_merged_vs_cue_s1": overlap(merged, c1),
    }


def metrics_reciprocal() -> dict:
    # recurrent_projection=True to match the reference's formation loop for this
    # protocol, which projects `{"stimA": ["A"]}, {"A": ["A"]}` -- restoration
    # cannot exceed the consolidation of the assembly being restored. Kept in
    # sync with tests/test_cross_repo_parity.py's `b3`; see the note there.
    #
    # This golden was VERIFIED against the reference implementation rather than
    # only re-recorded from here: running
    # `simulations.fixed_assembly_recip_proj`'s protocol on
    # .reference/dmitropolsky-assemblies/brain.py at these parameters restores
    # 1.000, which is the value below. That check matters because this file
    # writes a file named `reference_pnas_golden.json` using OUR ops, so a
    # regression in the op would otherwise be recorded as the new "reference"
    # and the parity test would keep passing. It did exactly that: reciprocal
    # restoration read a perfect 1.0 for a while because the back-fiber was
    # never materialised and the projection was a silent no-op.
    b = _brain(recurrent_projection=True)
    b.add_stimulus("s", K)
    b.add_area("A", N, K, BETA)
    b.add_area("B", N, K, BETA)
    orig = project(b, "s", "A", rounds=ROUNDS)
    reciprocal_project(b, "A", "B", rounds=ROUNDS)
    reciprocal_project(b, "B", "A", rounds=ROUNDS)
    restored = _snap(b, "A")
    return {"reciprocal_restore_overlap": overlap(orig, restored)}


def metrics_pattern_complete() -> dict:
    b = _brain()
    b.add_stimulus("stim", K)
    b.add_area("A", N, K, BETA)
    project(b, "stim", "A", rounds=ROUNDS)
    _, recovery = pattern_complete(b, "A", fraction=0.5, rounds=5, seed=SEED)
    return {"pattern_complete_50pct_overlap": recovery}


def main() -> None:
    m = metrics_project_separate()
    m.update(metrics_associate())
    m.update(metrics_merge())
    m.update(metrics_reciprocal())
    m.update(metrics_pattern_complete())

    golden = {
        "protocol": "PNAS 2020 — extended ops parity",
        "source": "neural_assemblies package (seed=42)",
        "recorded": "2026-07-23",
        "parameters": {
            "seed": SEED,
            "n": N,
            "k": K,
            "p": P,
            "beta": BETA,
            "rounds": ROUNDS,
            "engine": "numpy_sparse",
        },
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()},
        "tolerance": 0.05,
        "thresholds": {
            "associate_above_chance_factor_min": 3.0,
            "merge_above_chance_factor_min": 2.0,
            "reciprocal_restore_overlap_min": 0.6,
            "pattern_complete_50pct_overlap_min": 0.6,
        },
        "notes": "Extended metrics use test_assembly_calculus cue protocols. Re-baselined under norm_init=True default (Brain) at 30 formation rounds; 10 rounds under-reinforces associate.",
    }
    OUT.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(golden["metrics"], indent=2))


if __name__ == "__main__":
    main()
