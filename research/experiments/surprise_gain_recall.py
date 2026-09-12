"""E2: does surprise-modulated gain rescue rare-event EXPOSURE -- and does it
compose with scaling?

PRE-REGISTERED (task #131), before the gain mechanism was implemented.

BACKGROUND. #129: Hebbian mass follows token frequency (PL recall at chance,
SG 0.9-1.0, with every wire live). #130/E1: scoped homeostatic scaling
removes the accumulated-MASS component -- paired number delta +0.160 +/-
0.111, every seed positive -- but misses the 0.75 bar (0.660) and costs
tense -0.064. E1's residual is EXPOSURE: a plural form seen twice wrote two
updates however they are normalized. The biologically-matched lever for
exposure is GAIN -- neuromodulated plasticity (ACh/NE analog) writes BIGGER
updates for surprising events at the moment they occur.

THE HONESTY CONSTRAINT, stated up front: the surprise signal derives from
the LEARNER'S OWN HISTORY -- per-(feature-area, word-form) exposure counts
maintained by the training loop -- never from a linguistic label. A gain
keyed to "is plural" would be PLURAL_EVERY wearing a lab coat: corpus
knowledge smuggled in as biology. The registered form:

    gain(form) = min(GAIN_MAX, sqrt(mean_count / count(form)))

self-normalizing (a uniform corpus gives gain ~1 everywhere), label-free,
and computable online. GAIN_MAX caps the transient so a first exposure
cannot write an unbounded update.

MECHANISM PLUMBING: none new. The engines' `beta_by_source` behind the
`set_beta`/`get_beta` ABC is the authoritative per-fiber store (ENGINE.md;
the #88 lesson). train_tense/train_number bracket each episode's projection
with a transient set_beta on the core->feature fiber and restore after.
Default GAIN_MAX=1.0 is the exact current behavior.

ARMS (paired BRAIN seeds, natural rates, same harness as E1):
  OFF     production defaults.
  GAIN    novelty gain on TENSE/NUMBER episodes (GAIN_MAX=4.0), no scaling.
  SCALED  E1's arm: synaptic_scaling={TENSE, NUMBER}, no gain.
  BOTH    the composition cell -- the scientific payoff. E1 established the
          two mechanisms attack INDEPENDENT components (mass vs
          write-strength); if they compose, the sum should clear what each
          single missed.

READOUTS AND GUARDS: identical to E1 -- recall_tense/recall_number on the
canonical attested sets (evaluation/morph_features.py), image separation,
role probes + C1/C2 (scoped mechanisms must be invisible outside their
areas).

REGISTERED PREDICTIONS:
  R1 BOTH number balanced >= 0.75 (the bar OFF=0.500, SCALED=0.660 and,
     prediction, GAIN alone all miss).
  R2 Paired (BOTH - SCALED) number delta > 0 with the CI excluding zero --
     gain adds exposure on top of mass-normalization.
  R3 GAIN-alone lands BETWEEN OFF and SCALED on number (it strengthens rare
     writes but leaves the SG mass advantage intact, so it should rescue
     less than scaling does).
  R4 Tense in BOTH does not fall below OFF by more than E1's measured
     -0.064 (gain should not ADD a tense cost; the cost is scaling's).
  R5 Guards hold exactly in all arms (roles 9/9, C1 identical, C2 <= 0.2).

DECISION RULE: R1+R2 pass -> the frequency-imbalance program has a full
learning-rule answer, and the next unit A/Bs retiring PASSIVE_EVERY /
DITRANSITIVE_EVERY against it. R1 fails with R2 passing -> composition is
real but insufficient; sweep GAIN_MAX before concluding. R2 fails ->
exposure was the wrong residual diagnosis; re-open the mechanism question
with the per-form margin distributions from the diagnostics.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
# Sibling imports (_parallel) must resolve in spawned pool workers too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

SEEDS = [42, 43, 44, 45, 46]
ARMS = ("OFF", "GAIN", "SCALED", "BOTH")
STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES")
N = int(os.environ.get("SG_N", "3000"))
GAIN_MAX = float(os.environ.get("SG_GAIN_MAX", "4.0"))

if os.environ.get("SG_SMOKE") == "1":
    SEEDS = [42]
    N = int(os.environ.get("SG_N", "1500"))

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "surprise_gain_recall_results.json")

from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets, score_recall  # noqa: E402

ROLE_PROBES = [
    ("the dog chases the cat", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the cat is chased by the dog", {"dog": "AGENT", "cat": "PATIENT"}),
    ("the girl tells the food to the boy",
     {"girl": "AGENT", "food": "PATIENT", "boy": "GOAL"}),
    ("the boy sleeps in the house", {"boy": "AGENT", "house": None}),
]


def build_parser(seed: int, arm: str):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )

    kwargs = dict(n=N, k=30, seed=seed,
                  vocabulary=build_vocabulary_preset("core"),
                  fast_training=True)
    if arm in ("SCALED", "BOTH"):
        kwargs["synaptic_scaling"] = frozenset({TENSE, NUMBER})
    if arm in ("GAIN", "BOTH"):
        kwargs["novelty_gain_max"] = GAIN_MAX
    return EmergentParser(**kwargs)


def train(parser):
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )

    ct = CurriculumTrainer(parser)
    for stage in STAGES:
        ct.train_stage(stage)
    return parser


from research.experiments.role_guards import role_guards as _role_guards

def guards(parser):
    """Run role-probe controls for this study."""
    return _role_guards(parser, ROLE_PROBES)



def run_cell(arm: str, seed: int) -> dict:
    """One (arm, seed) cell -- top-level so the process pool can spawn it.

    The parser is fingerprint-cached: the OFF arm is byte-identical to
    E1's OFF arm and #129's DEFAULT arm, and re-runs after a crash resume
    from trained state (`_parallel.cached_parser` -- code changes
    invalidate the key).
    """
    from _parallel import cached_parser

    random.seed(seed)
    np.random.seed(seed)
    tag = f"sg-{arm}-n{N}-g{GAIN_MAX}"
    parser = cached_parser(tag, seed,
                           lambda: train(build_parser(seed, arm)))
    sets = attested_morph_sets(parser)
    tense = score_recall(parser.recall_tense,
                         {k: sets[k] for k in ("PRESENT", "PAST")})
    number = score_recall(parser.recall_number,
                          {k: sets[k] for k in ("SG", "PL")})
    g = guards(parser)
    print(f"[{arm} seed={seed}] "
          f"num bal={number['_balanced']} "
          f"(PL={number['PL']['acc']} SG={number['SG']['acc']}) | "
          f"tense bal={tense['_balanced']} | "
          f"roles {g['roles_ok']}/{g['roles_total']} "
          f"C1={g['c1_identical']} C2={g['c2']}",
          flush=True)
    return {"tense": tense, "number": number, "guards": g}


def main():
    from _parallel import run_cells

    cell_results = run_cells(
        run_cell, [(arm, seed) for arm in ARMS for seed in SEEDS])
    results = defaultdict(dict)
    for (arm, seed), res in cell_results.items():
        results[arm][seed] = res

    with open(OUT_PATH, "w") as f:
        json.dump({a: {str(s): v for s, v in by.items()}
                   for a, by in results.items()}, f, indent=2)

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    from neural_assemblies.diagnostics import ensemble, paired_delta

    print("\n=== registered predictions ===")
    ens = {}
    for feat in ("number", "tense"):
        for arm in ARMS:
            e = ensemble(lambda s: results[arm][s][feat]["_balanced"],
                         SEEDS, label=f"{feat}:{arm}")
            ens[(feat, arm)] = e
            print(e)
    for label, a, b in (("R2 number BOTH-SCALED", "BOTH", "SCALED"),
                        ("R3 number GAIN-OFF", "GAIN", "OFF"),
                        ("tense BOTH-OFF", "BOTH", "OFF")):
        d = paired_delta(ens[("number", a)] if "number" in label
                         else ens[("tense", a)],
                         ens[("number", b)] if "number" in label
                         else ens[("tense", b)], label=label)
        print(f"{d}  (per-seed: {[round(x, 3) for x in d.values]})")


if __name__ == "__main__":
    sys.exit(main())
