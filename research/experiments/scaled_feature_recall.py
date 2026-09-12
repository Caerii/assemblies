"""E1: does homeostatic synaptic scaling defeat the frequency-swamping law?

PRE-REGISTERED (task #130), before the scoped-scaling implementation ran.

BACKGROUND. Task #129 measured that with every wire live -- phases scheduled,
teacher two-class, readout built, image separation healthy (0.03-0.17) --
feature recall learns only what the corpus hammers: number balanced accuracy
is CHANCE (PL 0.0-0.2, SG 0.9-1.0) because every sentence trains several
singular noun tokens into NUMBER while plural forms ride ~30% of subjects.
Hebbian mass follows token frequency. The existing crutch (PASSIVE_EVERY-style
forcing rates) edits the WORLD; biology rebalances inside the learner.

THE MECHANISM UNDER TEST is not new machinery: `synaptic_scaling` has existed
in NumpySparseEngine since the norm_init work (`_normalize_area_columns`:
after each plasticity step, each just-touched postsynaptic column is rescaled
to the fixed setpoint rows*p -- Turrigiano-style multiplicative scaling). It
is OFF by default for a MEASURED reason: a per-fiber setpoint cancels the net
potentiation that makes an assembly a self-sustaining attractor (stability
0.01, completion 0.000 -- documented in the method's docstring).

WHY THE OBJECTION DOES NOT APPLY HERE, BY DESIGN: feature areas (TENSE,
NUMBER) are stimulus-anchored -- their assemblies are held by tense_*/number_*
stimulus drive, not by recurrent self-potentiation -- and recall needs the
afferent core->feature fiber to be DISCRIMINATIVE, not self-sustaining.
Column scaling is exactly the anti-swamping operation: a post-neuron whose
column accumulated mass from 50 singular tokens has its response to ANY input
diluted 50x, while a rarely-potentiated PL-image neuron keeps its per-synapse
advantage. The k-WTA competition at probe time then equalizes total
excitability across post-neurons.

IMPLEMENTATION UNDER TEST: `synaptic_scaling` scoped per TARGET AREA
(bool | collection of area names), threaded Brain -> engine -> EmergentParser.
Scoping is the unification move -- extend the existing flag, not a parallel
mechanism -- and it is also the safety boundary: role areas, core lexicons and
every attractor-bearing area stay unscaled, so the documented collapse mode is
out of scope by construction.

ARMS (paired BRAIN seeds, natural rates -- NO forcing knobs changed):
  OFF     current production defaults (synaptic_scaling=False).
  SCALED  synaptic_scaling={TENSE, NUMBER}.

READOUTS: recall_tense / recall_number per-class accuracy on the same
cleaned, corpus-attested test sets as #129 (raw-data scan, homographs and
zero-derivation pasts excluded), plus image separation, plus GUARDS that the
scaling did not reach beyond its scope:
  * role reconstruction on the 4-sentence probe set (both voices + ditransitive
    + locative) must stay exact;
  * C1 voice invariance (identical winners) and C2 <= 0.2 must hold in the
    SCALED arm.

REGISTERED PREDICTIONS:
  Q1 SCALED number balanced accuracy >= 0.75 (OFF measured 0.52 = chance),
     with PL accuracy specifically >= 0.6 (OFF: 0.0-0.2).
  Q2 Paired per-seed delta (SCALED - OFF) on number balanced > +0.2 on at
     least 4 of 5 seeds.
  Q3 SCALED tense balanced does not fall below OFF by more than 0.05
     (scaling may help tense's weak 0.65 or leave it; a drop means column
     scaling is destructive even for discriminative-only fibers).
  Q4 Guards hold exactly (role probes 4/4, C1 identical, C2 <= 0.2) in both
     arms -- scoped scaling must be invisible outside its areas.

If Q1/Q2 fail WITH guards holding and separation healthy, per-column scaling
is insufficient for rare-class rescue and the surprise-modulated-beta
alternative (neuromodulated gain) moves to the front. If Q4 fails, the scope
plumbing leaks and nothing else is interpretable.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

SEEDS = [42, 43, 44, 45, 46]
ARMS = ("OFF", "SCALED")
STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES")
N = int(os.environ.get("SF_N", "3000"))

if os.environ.get("SF_SMOKE") == "1":
    SEEDS = [42]
    N = int(os.environ.get("SF_N", "1500"))

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "scaled_feature_recall_results.json")

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
    if arm == "SCALED":
        kwargs["synaptic_scaling"] = frozenset({TENSE, NUMBER})
    return EmergentParser(**kwargs)


def train(parser):
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )

    ct = CurriculumTrainer(parser)
    for stage in STAGES:
        ct.train_stage(stage)
    return parser


from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets as test_sets  # noqa: E402
from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import score_recall as score  # noqa: E402


from research.experiments.role_guards import role_guards as _role_guards

def guards(parser):
    """Run role-probe controls for this study."""
    return _role_guards(parser, ROLE_PROBES)



def main():
    results = defaultdict(dict)
    for arm in ARMS:
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            parser = train(build_parser(seed, arm))
            sets = test_sets(parser)
            tense = score(parser.recall_tense,
                          {k: sets[k] for k in ("PRESENT", "PAST")})
            number = score(parser.recall_number,
                           {k: sets[k] for k in ("SG", "PL")})
            g = guards(parser)
            results[arm][seed] = {"tense": tense, "number": number,
                                  "guards": g}
            print(f"[{arm} seed={seed}] "
                  f"num bal={number['_balanced']} "
                  f"(PL={number['PL']['acc']} SG={number['SG']['acc']}) | "
                  f"tense bal={tense['_balanced']} | "
                  f"roles {g['roles_ok']}/{g['roles_total']} "
                  f"C1={g['c1_identical']} C2={g['c2']}",
                  flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({a: {str(s): v for s, v in by.items()}
                   for a, by in results.items()}, f, indent=2)

    # Seed summaries through `diagnostics.ensemble`/`paired_delta` -- Q2 IS
    # a paired test, and the confidence bound (not the mean) is the judge.
    from neural_assemblies.diagnostics import ensemble, paired_delta

    if len(SEEDS) < 3:
        print("\n(smoke mode: too few seeds for ensembles -- see JSON)")
        return

    print("\n=== registered predictions ===")
    ens = {}
    for feat in ("number", "tense"):
        for arm in ARMS:
            e = ensemble(lambda s: results[arm][s][feat]["_balanced"],
                         SEEDS, label=f"{feat}:{arm}")
            ens[(feat, arm)] = e
            print(e)
    d = paired_delta(ens[("number", "SCALED")], ens[("number", "OFF")],
                     label="Q2 number SCALED-OFF")
    print(f"{d}  (per-seed: {[round(x, 3) for x in d.values]})")


if __name__ == "__main__":
    sys.exit(main())
