"""E3: sweep GAIN_MAX on the composed config -- clear 0.75 honestly, or map the ceiling.

PRE-REGISTERED (task #132), the step E2's decision rule prescribed: R1
missed its bar by 0.010 and R2's paired CI included zero on two tie seeds
-- a calibration and power question, not a mechanism question. GAIN_MAX=4.0
was chosen a priori in E2 and never tuned; this sweep is the tuning, done
in the open, at doubled seed count.

DESIGN (everything through the validated speed stack -- pool + pinned
BLAS, forked pre-stage checkpoints gated by fork_equivalence.py, cached
baselines):
  * configs: BOTH(g) for g in {2, 4, 6, 8}  (scaling {TENSE,NUMBER} + gain g)
             + OFF and SCALED baselines
  * seeds: 42..51 (E2's five plus five fresh; cached cells are free)
  * readouts and guards identical to E1/E2.

REGISTERED PREDICTIONS:
  S1 Some g* has mean number balanced >= 0.75 (E2's BOTH(4.0) read 0.740).
  S2 At that g*, paired BOTH(g*) - SCALED number delta excludes zero at
     n=10 (E2: +0.080 +/- 0.129 at n=5, no negative seed).
  S3 Tense at g* stays within +/-0.05 of OFF (E2's BOTH(4.0): +0.010).
  S4 Guards exact in every cell (roles 9/9, C1 identical, C2 <= 0.2).
  S5 Shape: we do NOT pre-commit to monotonicity above g=4 -- an
     inverted-U is mechanistically plausible (a large one-shot write for a
     rare form is also a large write for NOISE in that episode). The curve
     is reported whatever it is.

DECISION RULE: S1+S2 pass -> open the forcing-rate retirement A/B
(natural-rate corpus + mechanisms vs forced corpus). S1 fails at every g ->
the composed mechanism's ceiling is real at this n/k; measure whether the
ceiling moves with n before concluding anything stronger. S2 fails at n=10
-> the BOTH-over-SCALED margin is not established; scaling alone remains
the honest recommendation.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

SEEDS = list(range(42, 52))
GAINS = (2.0, 4.0, 6.0, 8.0)
CONFIGS = ["OFF", "SCALED"] + [f"BOTH:{g}" for g in GAINS]
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("GM_N", "3000"))

if os.environ.get("GM_SMOKE") == "1":
    SEEDS = [42]
    GAINS = (2.0,)
    CONFIGS = ["OFF", "BOTH:2.0"]
    N = int(os.environ.get("GM_N", "1500"))

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "gain_max_sweep_results.json")

from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets, score_recall  # noqa: E402

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the girl tells the food to the boy",
     {"girl": "AGENT", "food": "PATIENT", "boy": "GOAL"}),
    ("the boy sleeps in the house", {"boy": "AGENT", "house": None}),
]


def _flags(config: str) -> dict:
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )

    scaled = frozenset({TENSE, NUMBER})
    if config == "OFF":
        return {}
    if config == "SCALED":
        return {"synaptic_scaling": scaled}
    gain = float(config.split(":")[1])
    return {"synaptic_scaling": scaled, "novelty_gain_max": gain}


def _build(seed: int):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset

    return EmergentParser(n=N, k=30, seed=seed,
                          vocabulary=build_vocabulary_preset("core"),
                          fast_training=True)


def guards(parser):
    out = {"roles_ok": 0, "roles_total": 0}
    for text, expected in ROLE_PROBES:
        roles, _diag = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            out["roles_total"] += 1
            out["roles_ok"] += roles.get(w) == want
    _r1, d1 = parser.parse_roles_by_reconstruction(
        "the dog chases the cat".split())
    _r2, d2 = parser.parse_roles_by_reconstruction(
        "the cat is chased by the dog".split())
    out["c1_identical"] = bool(d1["winners"]) and d1["winners"] == d2["winners"]
    return out


def run_cell(config: str, seed: int) -> dict:
    """One (config, seed) cell: forked from the shared pre-stage checkpoint.

    Mechanism flags act only in the SENTENCES stage (fork_equivalence.py
    pinned exact equality of this construction with full training).
    """
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from _parallel import forked_parser

    flags = _flags(config)

    def build_and_train_pre():
        random.seed(seed)
        np.random.seed(seed)
        p = _build(seed)
        ct = CurriculumTrainer(p)
        for stage in PRE_STAGES:
            ct.train_stage(stage)
        return p

    def arm_setup(p):
        ss = flags.get("synaptic_scaling")
        if ss:
            p.brain._engine.synaptic_scaling = ss
            p.brain._synaptic_scaling = ss
        p.novelty_gain_max = float(flags.get("novelty_gain_max", 1.0))

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"gm-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)
    sets_ = attested_morph_sets(parser)
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})
    number = score_recall(parser.recall_number,
                          {k: sets_[k] for k in ("SG", "PL")})
    g = guards(parser)
    print(f"[{config} seed={seed}] num bal={number['_balanced']} "
          f"(PL={number['PL']['acc']}) | tense bal={tense['_balanced']} | "
          f"roles {g['roles_ok']}/{g['roles_total']} C1={g['c1_identical']}",
          flush=True)
    return {"tense": tense, "number": number, "guards": g}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell, [(c, s) for c in CONFIGS for s in SEEDS])
    results = defaultdict(dict)
    for (config, seed), res in cell_results.items():
        results[config][seed] = res

    with open(OUT_PATH, "w") as f:
        json.dump({c: {str(s): v for s, v in by.items()}
                   for c, by in results.items()}, f, indent=2)

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble, paired_delta

    print("\n=== registered predictions ===")
    ens = {}
    for feat in ("number", "tense"):
        for config in CONFIGS:
            e = ensemble(lambda s: results[config][s][feat]["_balanced"],
                         SEEDS, label=f"{feat}:{config}")
            ens[(feat, config)] = e
            print(e)
    best = max((c for c in CONFIGS if c.startswith("BOTH")),
               key=lambda c: ens[("number", c)].mean)
    print(f"\nS1 best config: {best} "
          f"(mean {ens[('number', best)].mean:.4f}, bar 0.75)")
    d = paired_delta(ens[("number", best)], ens[("number", "SCALED")],
                     label=f"S2 {best}-SCALED")
    print(f"{d}  (per-seed: {[round(x, 3) for x in d.values]})")
    dt = paired_delta(ens[("tense", best)], ens[("tense", "OFF")],
                      label=f"S3 tense {best}-OFF")
    print(dt)


if __name__ == "__main__":
    sys.exit(main())
