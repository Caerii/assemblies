"""E9: slow homeostasis x repetition -- fast Hebbian inside a slowly renormalized envelope.

PRE-REGISTERED (task #138), before deferred scaling ever trained at scale.

E8 (#137) split cleanly: repetition raised the RAW baseline monotonically
on both features (exposure confirmed as what reliability is made of), but
per-update scaling INTERFERED with repetition -- flat means, variance
exploding to +/-0.120, seed-bistable. Diagnosis: biological synaptic
scaling is SLOW (Turrigiano; hours-to-days, segregated from fast Hebbian
writes) while ours renormalized after EVERY projection, and repetition
multiplies exactly those collisions.

E9 is the INTERVENTION that converts that diagnosis into a demonstrated
cause: `synaptic_scaling_deferred` accumulates touched columns and
normalizes once per morph phase (flush at train_tense/train_number end).
Fast writes integrate freely; the slow step sees the time-averaged mass
and removes only its mean bias. If interference was the mechanism, slow
scaling restores composition AND collapses the variance. If instead the
E8 pattern persists (e.g. w_max saturation), slow scaling changes
nothing -- the arms separate the hypotheses.

CONFIGS: SLOW-SCALED x R in {1, 4}, 10 seeds, budget 50. References,
all measured: FAST-SCALED R1 = 0.650 +/- 0.058 (E3/E4), FAST-SCALED
R4 = 0.630 +/- 0.120 (E8, per-seed JSON for PAIRED deltas), OFF R4 =
0.585 +/- 0.048 (E8).

REGISTERED BARS:
  T1 SLOW R1 preserves the scaling margin: mean >= 0.60 with CI <= 0.08
     (deferral must not break what per-update scaling achieved at R=1).
  T2 SLOW R4 COMPOSES: paired per-seed delta vs FAST R4 (E8's values) > 0
     with the CI excluding zero, AND CI <= 0.08 (the variance collapse is
     itself a registered signature -- bistability was the interference).
  T3 SLOW R4 mean > 0.650 (beats the best fast-scaling configuration).
  T4 tense within -0.05 of references; guards exact.
  ASPIRATIONAL: mean >= 0.75 -- reported against the E-series bar.

DECISION RULE: T2 passes -> timescale separation demonstrated by
intervention; adoption discussion opens (deferred scaling as the
production form) plus the Zipf+scaling synthesis registration. T2 fails
with T1 passing -> interference was not (only) the mechanism; instrument
w_max saturation on the probed fibers next. T1 fails -> deferral broke
scaling itself; inspect flush placement before anything else.
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
REPS = (1, 4)
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("SH_N", "3000"))

if os.environ.get("SH_SMOKE") == "1":
    SEEDS = [42]
    REPS = (4,)
    N = int(os.environ.get("SH_N", "1500"))

FAST_R1_REF = (0.6500, 0.0584)
FAST_R4_JSON = os.path.join(os.path.dirname(__file__),
                            "repetition_recall_results.json")

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "slow_homeostasis_recall_results.json")

from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets, score_recall  # noqa: E402

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
]


def run_cell(reps: int, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from _parallel import forked_parser

    def build_and_train_pre():
        random.seed(seed)
        np.random.seed(seed)
        p = EmergentParser(n=N, k=30, seed=seed,
                           vocabulary=build_vocabulary_preset("core"),
                           fast_training=True)
        ct = CurriculumTrainer(p)
        for stage in PRE_STAGES:
            ct.train_stage(stage)
        return p

    def arm_setup(p):
        ss = frozenset({TENSE, NUMBER})
        p.brain._engine.synaptic_scaling = ss
        p.brain._synaptic_scaling = ss
        p.brain._engine.synaptic_scaling_deferred = True
        p.morph_repetitions = reps

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    parser = forked_parser(f"gm-n{N}", seed, build_and_train_pre,
                           arm_setup, train_final)
    sets_ = attested_morph_sets(parser)
    number = score_recall(parser.recall_number,
                          {k: sets_[k] for k in ("SG", "PL")})
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})
    g_ok = g_total = 0
    for text, expected in ROLE_PROBES:
        roles, _d = parser.parse_roles_by_reconstruction(text.split())
        for w, want in expected.items():
            g_total += 1
            g_ok += roles.get(w) == want
    print(f"[SLOW R{reps} seed={seed}] num bal={number['_balanced']} "
          f"(PL={number['PL']['acc']}) | tense bal={tense['_balanced']} | "
          f"roles {g_ok}/{g_total}", flush=True)
    return {"number": number, "tense": tense,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell, [(r, s) for r in REPS for s in SEEDS])
    results = defaultdict(dict)
    for (reps, seed), res in cell_results.items():
        results[f"R{reps}"][seed] = res

    with open(OUT_PATH, "w") as f:
        json.dump({c: {str(s): v for s, v in by.items()}
                   for c, by in results.items()}, f, indent=2)

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import Ensemble, ensemble, paired_delta

    print("\n=== registered bars ===")
    ens = {}
    for reps in REPS:
        e = ensemble(lambda s: results[f"R{reps}"][s]["number"]["_balanced"],
                     SEEDS, label=f"number:SLOW:R{reps}")
        ens[reps] = e
        print(e)
        et = ensemble(lambda s: results[f"R{reps}"][s]["tense"]["_balanced"],
                      SEEDS, label=f"tense:SLOW:R{reps}")
        print(f"  {et}")
    print(f"T1 SLOW R1 vs FAST R1 ref {FAST_R1_REF[0]:.4f}"
          f"+/-{FAST_R1_REF[1]:.4f}")

    with open(FAST_R4_JSON) as f:
        fast = json.load(f)
    fast_r4 = Ensemble(
        "number:FAST:R4",
        tuple(fast["R4:SCALED"][str(s)]["number"]["_balanced"]
              for s in SEEDS),
        float(np.mean([fast["R4:SCALED"][str(s)]["number"]["_balanced"]
                       for s in SEEDS])), 0.0)
    d = paired_delta(ens[4], fast_r4, label="T2 SLOW-FAST at R4")
    print(f"{d}  (per-seed: {[round(x, 3) for x in d.values]})")


if __name__ == "__main__":
    sys.exit(main())
