"""Is fork-from-checkpoint training EXACTLY full training? Pin it or drop it.

Gate for `_parallel.forked_parser` (speed fix 2): arms may share pre-stage
training only if (build -> stages 1..3 -> checkpoint -> restore RNG -> arm
flags -> SENTENCES) produces the SAME parser as straight-through training.
"Same" is judged on every readout an experiment consumes: morph recall
scores per class, role probes, C1/C2 -- all deterministic given the seeding
fixes, so equality here is exact equality, not tolerance.

If this fails, the checkpoint is missing state that matters (an RNG stream,
a cache, a counter) and forked_parser must NOT be used until the missing
state is found and added -- a fork that silently diverges would put every
arm comparison on different substrates (the warm/cold split lesson).
"""
from __future__ import annotations

import os
import random
import sys

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

N = int(os.environ.get("FE_N", "1500"))
SEED = 42
PRE_STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD")
FINAL = "SENTENCES"


def build(arm_flags: dict):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset

    return EmergentParser(n=N, k=30, seed=SEED,
                          vocabulary=build_vocabulary_preset("core"),
                          fast_training=True, **arm_flags)


def readouts(parser) -> dict:
    from neural_assemblies.assembly_calculus.emergent.evaluation \
        .morph_features import attested_morph_sets, score_recall

    sets_ = attested_morph_sets(parser)
    tense = score_recall(parser.recall_tense,
                         {k: sets_[k] for k in ("PRESENT", "PAST")})
    number = score_recall(parser.recall_number,
                          {k: sets_[k] for k in ("SG", "PL")})
    roles, diag = parser.parse_roles_by_reconstruction(
        "the cat is chased by the dog".split())
    return {"tense": tense, "number": number,
            "roles": roles, "winners": diag["winners"]}


def main():
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        TENSE, NUMBER,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from _parallel import forked_parser

    arm_flags = {"synaptic_scaling": frozenset({TENSE, NUMBER}),
                 "novelty_gain_max": 4.0}

    # Arm A: straight-through training with construction-time flags.
    random.seed(SEED)
    np.random.seed(SEED)
    full = build(arm_flags)
    ct = CurriculumTrainer(full)
    for stage in PRE_STAGES + (FINAL,):
        ct.train_stage(stage)
    full_out = readouts(full)

    # Arm B: fork from a shared pre-stage checkpoint, flags applied post-hoc.
    def build_and_train_pre():
        random.seed(SEED)
        np.random.seed(SEED)
        p = build({})
        ct2 = CurriculumTrainer(p)
        for stage in PRE_STAGES:
            ct2.train_stage(stage)
        return p

    def arm_setup(p):
        p.brain._engine.synaptic_scaling = frozenset({TENSE, NUMBER})
        p.brain._synaptic_scaling = frozenset({TENSE, NUMBER})
        p.novelty_gain_max = 4.0

    def train_final(p):
        CurriculumTrainer(p).train_stage(FINAL)
        return p

    forked = forked_parser("fe", SEED, build_and_train_pre, arm_setup,
                           train_final)
    fork_out = readouts(forked)

    ok = full_out == fork_out
    print(f"full == fork: {ok}")
    if not ok:
        for key in full_out:
            if full_out[key] != fork_out[key]:
                print(f"  DIVERGES at {key}:")
                print(f"    full: {full_out[key]}")
                print(f"    fork: {fork_out[key]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
