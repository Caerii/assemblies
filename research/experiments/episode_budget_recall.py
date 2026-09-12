"""E7: the episode budget -- sweep FRAMES_PER_STAGE at unchanged rates.

PRE-REGISTERED (task #136), before any budget above 50 ever trained.

THE CHAIN THAT LEADS HERE. E5 (#134): the number-recall ceiling is
n-invariant under scaling -- not the substrate. E6 (#135): raising form
diversity at FIXED budget made everything worse -- form count and
per-form exposure are CONJUGATE, so the binding constraint is the
EPISODE BUDGET itself: generate_generic's 50-frame cap, held constant by
every experiment in this repository's history, and the least child-like
number in the pipeline. E7 raises ONLY that: rates, sampling, vocabulary
and mechanisms all unchanged. With-replacement sampling naturally widens
form coverage as draws increase, so diversity and strength rise TOGETHER
-- the move E6 proved impossible at fixed budget.

CONFIGS: budgets {100, 200} x {OFF, SCALED, BOTH(sqrt,4)}, n=3000, seeds
42..51. Budget-50 references are the measured E3/E4 ensembles (same
seeds, same protocol, byte-identical corpus at FRAMES_PER_STAGE=50).
Pre-stage checkpoints SHARED across all cells (the budget only affects
complexity>=3 stages; pre-stages return before the frame loop).

REGISTERED BARS:
  B1 number balanced rises MONOTONICALLY with budget for SCALED and
     BOTH (50 -> 100 -> 200).
  B2 some (budget, config) cell reaches mean >= 0.75 -- the bar the
     whole E-series has chased.
  B3 tense stays within +/-0.05 of its budget-50 reference per config
     (more episodes must not silently trade tense).
  B4 guards exact (roles, C1).
  B5 distinct attested PL forms rise with budget (the E6 census check,
     now expected to move WITH exposure rather than against it).

DECISION RULE: B2 passes -> the causal chain closes (budget was the
constraint) and the forcing-rate retirement A/B opens, run at the
winning budget. B1 fails (non-monotone) -> there is a competing cost to
corpus size (interference/crowding at fixed n) -- instrument image
separation and area health before anything else. B2 fails with B1
passing -> extrapolate the budget curve before deciding; 0.75 may simply
need more than 200.
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
BUDGETS = (100, 200)
MECHS = ("OFF", "SCALED", "BOTH")
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("EB_N", "3000"))

if os.environ.get("EB_SMOKE") == "1":
    SEEDS = [42]
    BUDGETS = (100,)
    MECHS = ("SCALED",)
    N = int(os.environ.get("EB_N", "1500"))

BUDGET50_REFERENCE = {  # measured, E3/E4, byte-identical corpus at 50
    "OFF": (0.5300, 0.0384),
    "SCALED": (0.6500, 0.0584),
    "BOTH": (0.7100, 0.0554),
}
TENSE50_REFERENCE = {
    "OFF": 0.6778, "SCALED": 0.6415, "BOTH": 0.6937,
}

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "episode_budget_recall_results.json")

from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets, score_recall  # noqa: E402

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
]


def _flags(mech: str) -> dict:
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )

    scaled = frozenset({TENSE, NUMBER})
    if mech == "OFF":
        return {}
    if mech == "SCALED":
        return {"synaptic_scaling": scaled}
    return {"synaptic_scaling": scaled, "novelty_gain_max": 4.0}


def run_cell(budget: int, mech: str, seed: int) -> dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        generation as generation_mod,
    )
    from _parallel import forked_parser

    flags = _flags(mech)

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
        # The budget is patched at FINAL-stage time only: pre-stages never
        # reach the frame loop, so the shared checkpoint is budget-blind.
        generation_mod.FRAMES_PER_STAGE = budget
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
    print(f"[b{budget} {mech} seed={seed}] num bal={number['_balanced']} "
          f"(PL={number['PL']['acc']} n={number['PL']['n']}) | "
          f"tense bal={tense['_balanced']} | roles {g_ok}/{g_total}",
          flush=True)
    return {"number": number, "tense": tense,
            "guards": {"roles_ok": g_ok, "roles_total": g_total},
            "n_pl_forms": number["PL"]["n"]}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell,
        [(b, m, s) for b in BUDGETS for m in MECHS for s in SEEDS])
    results = defaultdict(dict)
    for (budget, mech, seed), res in cell_results.items():
        results[f"b{budget}:{mech}"][seed] = res

    write_new_document(Path(OUT_PATH), {
        c: {str(s): v for s, v in by.items()}
        for c, by in results.items()
    })

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble

    print("\n=== registered bars (vs measured budget-50 references) ===")
    for mech in MECHS:
        ref_mean, ref_ci = BUDGET50_REFERENCE[mech]
        row = [f"b50 {ref_mean:.4f}+/-{ref_ci:.4f} (ref)"]
        for budget in BUDGETS:
            e = ensemble(
                lambda s: results[f"b{budget}:{mech}"][s]["number"]
                ["_balanced"], SEEDS, label=f"number:{mech}:b{budget}")
            row.append(f"b{budget} {e.mean:.4f}+/-{e.ci:.4f}")
        print(f"{mech}: " + "  ->  ".join(row))
        tref = TENSE50_REFERENCE[mech]
        for budget in BUDGETS:
            et = ensemble(
                lambda s: results[f"b{budget}:{mech}"][s]["tense"]
                ["_balanced"], SEEDS, label=f"tense:{mech}:b{budget}")
            print(f"  tense b{budget}: {et.mean:.4f}+/-{et.ci:.4f} "
                  f"(b50 ref {tref:.4f}, B3 |delta| <= 0.05)")
    pl = {b: min(results[f"b{b}:{m}"][s]["n_pl_forms"]
                 for m in MECHS for s in SEEDS) for b in BUDGETS}
    print(f"B5 min distinct PL forms by budget: {pl} (b50 reference: 10)")


if __name__ == "__main__":
    sys.exit(main())
