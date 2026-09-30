"""E8: repetition raises exposure at fixed diversity -- the last hypothesis.

PRE-REGISTERED (task #137), before any R > 1 ever trained at scale.

THE CHAIN: E7 (#136) showed uniform corpora PIN per-form exposure at
~1.5 at every corpus size (coverage widens as fast as the budget), so no
amount of data raises reliability -- only REPETITION does, which is how
real corpora (Zipf) and childhood (repeated utterances) supply it. E8
repeats the morph-training phases over the SAME stage corpus R times:
exposure becomes R x ~1.5 at unchanged diversity, unchanged rates,
unchanged everything else.

At FIXED budget 50 the attested exam does not grow with the
manipulation, so E7's reliability/load confound does not arise here --
full-set scoring is comparable to every E-series b50 number.

CONFIGS: R in {2, 4} x {OFF, SCALED}, budget 50, n=3000, seeds 42..51.
R=1 references are the measured E3/E4 ensembles (byte-identical corpus
and training at morph_repetitions=1). BOTH excluded: gain was exhausted
as a mechanism (E3/E4) and its arm would confound the exposure axis.

REGISTERED BARS:
  P1 SCALED number balanced rises with R (0.650 -> higher at R=2 -> R=4).
  P2 SCALED at some R reaches mean >= 0.75 -- the E-series bar.
  P3 tense within +/-0.05 of its R=1 reference (tense phases repeat too;
     tense forms get the same exposure boost, so tense may RISE -- the
     bar only guards against degradation).
  P4 guards exact (roles, C1 probes).
  P5 saturation check: w_max = 20 caps multiplicative growth
     (rounds-buy-convergence: retention <= 1-(1-p)^k, gain <= w_max), so
     R=4's gain over R=2 may be sublinear. Reported, not barred.

DECISION RULE: P2 passes -> the exposure account is CONFIRMED end to
end; the retirement A/B opens with repetition + scaling as the
learning-rule configuration, and the E9-shaped synthesis (Zipfian corpus
+ scaling: repetition without swamping) becomes the natural next
registration. P1 fails -> exposure was NOT the constraint and the
E-series' last hypothesis dies; instrument per-form margins before
theorizing further.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from research.json_documents import write_new_document

SEEDS = list(range(42, 52))
REPS = (2, 4)
MECHS = ("OFF", "SCALED")
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("RR_N", "3000"))

if os.environ.get("RR_SMOKE") == "1":
    SEEDS = [42]
    REPS = (4,)
    MECHS = ("SCALED",)
    N = int(os.environ.get("RR_N", "1500"))

R1_REFERENCE = {  # measured, E3/E4: morph_repetitions=1 byte-identical
    "OFF": (0.5300, 0.0384),
    "SCALED": (0.6500, 0.0584),
}
TENSE_R1_REFERENCE = {"OFF": 0.6778, "SCALED": 0.6415}

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "repetition_recall_results.json")

from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets, score_recall  # noqa: E402

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
]


def run_cell(reps: int, mech: str, seed: int) -> dict:
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
        if mech == "SCALED":
            ss = frozenset({TENSE, NUMBER})
            p.brain._engine.synaptic_scaling = ss
            p.brain._synaptic_scaling = ss
        p.morph_repetitions = reps

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    # Pre-checkpoints shared with E3/E5/E7: morph phases run only in the
    # final stage, so the checkpoint is repetition-blind.
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
    print(f"[R{reps} {mech} seed={seed}] num bal={number['_balanced']} "
          f"(PL={number['PL']['acc']}) | tense bal={tense['_balanced']} | "
          f"roles {g_ok}/{g_total}", flush=True)
    return {"number": number, "tense": tense,
            "guards": {"roles_ok": g_ok, "roles_total": g_total}}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell, [(r, m, s) for r in REPS for m in MECHS for s in SEEDS])
    results = defaultdict(dict)
    for (reps, mech, seed), res in cell_results.items():
        results[f"R{reps}:{mech}"][seed] = res

    write_new_document(Path(OUT_PATH), {
        c: {str(s): v for s, v in by.items()}
        for c, by in results.items()
    })

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    print("\n=== registered bars (vs measured R=1 references) ===")
    for mech in MECHS:
        ref_mean, ref_ci = R1_REFERENCE[mech]
        row = [f"R1 {ref_mean:.4f}+/-{ref_ci:.4f} (ref)"]
        for reps in REPS:
            e = ensemble(
                lambda s: results[f"R{reps}:{mech}"][s]["number"]
                ["_balanced"], SEEDS, label=f"number:{mech}:R{reps}")
            row.append(f"R{reps} {e.mean:.4f}+/-{e.ci:.4f}")
        print(f"{mech}: " + "  ->  ".join(row))
        tref = TENSE_R1_REFERENCE[mech]
        for reps in REPS:
            et = ensemble(
                lambda s: results[f"R{reps}:{mech}"][s]["tense"]
                ["_balanced"], SEEDS, label=f"tense:{mech}:R{reps}")
            print(f"  tense R{reps}: {et.mean:.4f}+/-{et.ci:.4f} "
                  f"(R1 ref {tref:.4f}, P3 guards degradation only)")


if __name__ == "__main__":
    sys.exit(main())
