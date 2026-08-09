"""E4: the gain FORMULA is the lever -- linear mean/count, cap now live.

PRE-REGISTERED (task #133), before the linear form ever ran at scale.

E3 (#132) found the sweep axis VACUOUS: sqrt(mean_count/count) self-caps
at ~1.2 on this corpus (measured with the cap removed: max 1.215, mean
0.998, mean exposure 1.41), so GAIN_MAX never mattered and E2's "gain"
acted mostly as mild familiarity suppression. The formula, not the cap,
bounds the effect.

E4 changes ONE thing: novelty_gain_exp = 1.0 (linear). A form seen once
among 5x-seen neighbors now writes ~5x, still derived purely from the
learner's own exposure counts (label-free), and GAIN_MAX becomes a live
parameter for the first time.

CONFIGS: OFF, SCALED, LIN:2 and LIN:4 (scaling + linear gain, cap 2 / 4),
seeds 42..51, readouts and guards identical to E1-E3, forked pre-stage
checkpoints (SHARED with E3's -- pre-stages are arm-invariant).

REGISTERED BARS (task #133):
  L1 some LIN config reaches mean number balanced >= 0.75.
  L2 paired LIN-SCALED number delta excludes zero at n=10.
  L3 tense at that config within +/-0.05 of OFF.
  L4 guards exact everywhere.
  L5 LIN:2 vs LIN:4 must now DIFFER (the cap is live; identical results
     would mean the linear form saturates below 2, to be instrumented
     before anything else is concluded).

DECISION RULE: L1+L2 -> open the forcing-rate retirement A/B. Linear also
plateaus at ~0.71 -> the ceiling is not about write strength; run the
ceiling-vs-n branch. L5 fails -> instrument the raw gain trajectory first.
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
GAINS = (2.0, 4.0)
CONFIGS = ["OFF", "SCALED"] + [f"LIN:{g}" for g in GAINS]
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"
N = int(os.environ.get("LG_N", "3000"))

if os.environ.get("LG_SMOKE") == "1":
    SEEDS = [42]
    CONFIGS = ["OFF", "LIN:2.0"]
    N = int(os.environ.get("LG_N", "1500"))

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "linear_gain_recall_results.json")

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
    return {"synaptic_scaling": scaled, "novelty_gain_max": gain,
            "novelty_gain_exp": 1.0}


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
    """One (config, seed) cell, forked from the E3-shared pre-checkpoint."""
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
        p.novelty_gain_exp = float(flags.get("novelty_gain_exp", 0.5))

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    # Same tag as E3: pre-stages are arm-invariant, checkpoints shared.
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
    lin = [c for c in CONFIGS if c.startswith("LIN")]
    best = max(lin, key=lambda c: ens[("number", c)].mean)
    print(f"\nL1 best config: {best} "
          f"(mean {ens[('number', best)].mean:.4f}, bar 0.75)")
    d = paired_delta(ens[("number", best)], ens[("number", "SCALED")],
                     label=f"L2 {best}-SCALED")
    print(f"{d}  (per-seed: {[round(x, 3) for x in d.values]})")
    dt = paired_delta(ens[("tense", best)], ens[("tense", "OFF")],
                      label=f"L3 tense {best}-OFF")
    print(dt)
    if len(lin) == 2:
        same = all(
            results[lin[0]][s]["number"]["_balanced"]
            == results[lin[1]][s]["number"]["_balanced"] for s in SEEDS)
        print(f"L5 LIN configs differ: {not same}")


if __name__ == "__main__":
    sys.exit(main())
