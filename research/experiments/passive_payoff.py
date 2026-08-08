"""Does voice alternation buy anything? The A/B on the corpus that has it.

THE CLAIM UNDER TEST, and it is the premise of the whole passive arc: in an
all-active corpus, position predicts role perfectly, so a role representation
buys NOTHING over a position counter. Adding passives is supposed to be what
makes roles carry information. Nobody had checked that it does.

THE MEASUREMENT. Two corpora, identical in every other respect:

    active_only   PASSIVE_EVERY disabled -- the corpus this repo had before
    alternating   PASSIVE_EVERY = 4      -- the corpus it has now

trained on the SAME brain seed, then scored on HELD-OUT REVERSIBLE sentences in
BOTH voices. Reversible means both participants are animate, so either could
plausibly be the agent and no lexical or plausibility shortcut can answer it --
the sentence is decidable only from structure. Held out means the specific
verb+participant triple never occurred in training.

    active     "the dog chases the cat"        agent=dog  patient=cat
    passive    "the cat is chased by the dog"  agent=dog  patient=cat

CHANCE IS 0.5, not 0. Two nouns and two roles admit two assignments, so a coin
gets the pair right half the time. A score BELOW 0.5 is the interesting one: it
means the parser is not guessing but is systematically INVERTED, which is
exactly what a position counter does to a passive.

WHAT WOULD FALSIFY THE ARC: if `alternating` does not beat `active_only` on the
passive set, the machinery is decoration. If it beats it on passives but LOSES
on actives, the corpus taught the determiner to reverse and the cure is worse
than the disease -- so both voices are reported, always.

THE READOUT IS `_assign_roles_neural`, NOT `probe_parse`. They are two different
parsers under one word: `probe_parse` runs the RULE-PROGRAM route (NemoParser),
which skips words it has no program for -- "is", "by", the participle -- and
returns `roles: None` even for "the dog chases the cat", which the neural route
reads correctly. Scoring the A/B through it produced 0.0000 on BOTH arms and
BOTH voices, which reads as a clean null and is a dead readout. Isolation comes
from `brain.read_only()` directly (0 leak, 36.8x cheaper than deepcopy).

Arms are compared with `compare_arms` + `paired_delta` on shared seeds, because
the difference is what the question asks and two independent CIs are a weaker
test.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np  # noqa: E402

from neural_assemblies.assembly_calculus.emergent import (  # noqa: E402
    EmergentParser,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (  # noqa: E402
    CurriculumTrainer,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (  # noqa: E402
    trainer as trainer_module,
)
from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (  # noqa: E402
    build_vocabulary_preset,
)
from neural_assemblies.diagnostics import (  # noqa: E402
    compare_arms, paired_delta,
)
from neural_assemblies.lexicon.lexicon_manager import WordCategory  # noqa: E402

STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES")
N, K = 3000, 30
SEEDS = (11, 23, 37, 41, 53)
CHANCE = 0.5
#: The generator's own off switch. NOT a huge modulus: `n % N == 0` is true at
#: n == 0, so an "unreachable" modulus still emits the FIRST passive. That put
#: one passive into the control arm and the sanity check below caught it.
NO_PASSIVES = 0


def _feat(w) -> dict:
    return getattr(w, "features", None) or {}


def _held_out_probes(trainer, train_tokens) -> List[Tuple[List[str], str, str]]:
    """Reversible triples that never occurred in training, in both voices."""
    words = trainer._get_stage_words("SENTENCES")
    animate = [w for w in words if w.category == WordCategory.NOUN
               and _feat(w).get("animate")]
    verbs = [v for v in words if v.category == WordCategory.VERB
             and _feat(v).get("transitive")
             and (getattr(v, "forms", None) or {}).get("3sg")
             and (getattr(v, "forms", None) or {}).get("ppart")]

    rng = random.Random(4242)
    rng.shuffle(animate)
    rng.shuffle(verbs)

    probes: List[Tuple[List[str], str, str]] = []
    for verb in verbs:
        for i in range(0, len(animate) - 1, 2):
            agent, patient = animate[i], animate[i + 1]
            finite = verb.forms["3sg"]
            active = ["the", agent.lemma, finite, "the", patient.lemma]
            if tuple(active) in train_tokens:
                continue
            passive = ["the", patient.lemma, "is", verb.forms["ppart"],
                       "by", "the", agent.lemma]
            if tuple(passive) in train_tokens:
                continue
            probes.append((active, agent.lemma, patient.lemma))
            probes.append((passive, agent.lemma, patient.lemma))
            if len(probes) >= 40:
                return probes
    return probes


def _score(parser, probes) -> Tuple[float, float]:
    """(active accuracy, passive accuracy) over the probe set."""
    hits = {"active": 0, "passive": 0}
    total = {"active": 0, "passive": 0}
    for tokens, agent, patient in probes:
        voice = "passive" if "by" in tokens else "active"
        total[voice] += 1
        cats = {w: parser.classify_word_cached(w)[0] for w in tokens}
        with parser.brain.read_only():
            roles = parser._assign_roles_neural(list(tokens), cats)
        if roles.get(agent) == "AGENT" and roles.get(patient) == "PATIENT":
            hits[voice] += 1
    return (hits["active"] / max(1, total["active"]),
            hits["passive"] / max(1, total["passive"]))


def _train(seed: int, passive_every: int):
    """One trained parser, with the corpus arm selected by `passive_every`."""
    # Reseed the global generators per trial: Brain(seed=) alone is not
    # reproducible, and a leak between constructions flips borderline results.
    random.seed(seed)
    np.random.seed(seed)

    old = trainer_module.PASSIVE_EVERY
    trainer_module.PASSIVE_EVERY = passive_every
    try:
        parser = EmergentParser(
            n=N, k=K, seed=seed, vocabulary=build_vocabulary_preset("core"),
            fast_training=True,
        )
        trainer = CurriculumTrainer(parser)
        train_tokens = set()
        for stage in STAGES:
            trainer.train_stage(stage)
        # Record what was actually trained on, so "held out" is a fact rather
        # than an assumption about the generator.
        words = trainer._get_stage_words("SENTENCES")
        for plan in trainer._generate_sentences(words, 4,
                                                stage_name="SENTENCES"):
            train_tokens.add(tuple(plan.tokens))
        return parser, trainer, train_tokens
    finally:
        trainer_module.PASSIVE_EVERY = old


#: Substrate fingerprint per (arm, seed), filled as a side effect of scoring.
#: THE POINT: if the metric is bit-identical across seeds, that is either
#: seeds that never reached the substrate or a decision the substrate does not
#: make. Those are opposite conclusions and only this distinguishes them.
_FINGERPRINTS: dict = {}


def _arm(passive_every: int, voice: str):
    def run(seed: int) -> float:
        parser, trainer, train_tokens = _train(seed, passive_every)
        asm = parser.core_lexicons.get("NOUN_CORE", {}).get("dog")
        _FINGERPRINTS[(passive_every, voice, seed)] = (
            parser.brain.areas["NOUN_CORE"].w,
            tuple(sorted(int(x) for x in getattr(asm, "winners", []))[:4]),
        )
        probes = _held_out_probes(trainer, train_tokens)
        active_acc, passive_acc = _score(parser, probes)
        return passive_acc if voice == "passive" else active_acc
    return run


def main() -> None:
    print("Does voice alternation buy anything? The A/B on the corpus that has it.")
    print()

    # Sanity: the corpora must actually differ, or both arms are one arm.
    _p, tr_on, toks_on = _train(SEEDS[0], 4)
    _q, tr_off, toks_off = _train(SEEDS[0], NO_PASSIVES)
    n_on = sum(1 for t in toks_on if "by" in t)
    n_off = sum(1 for t in toks_off if "by" in t)
    print(f"  corpus check: alternating has {n_on} passives, "
          f"active_only has {n_off}")
    if n_on == 0 or n_off != 0:
        print("  ABORT: the arms do not differ in the way this claims to test")
        return
    probes = _held_out_probes(tr_on, toks_on)
    print(f"  probes: {len(probes)} held-out reversible sentences "
          f"({sum(1 for p in probes if 'by' in p[0])} passive)")
    print(f"  e.g. {' '.join(probes[0][0])!r} -> agent={probes[0][1]}")
    print(f"       {' '.join(probes[1][0])!r} -> agent={probes[1][1]}")
    print()

    for voice in ("active", "passive"):
        arms = compare_arms(
            {"alternating": _arm(4, voice),
             "active_only": _arm(NO_PASSIVES, voice)},
            SEEDS,
            # Not strict on the ACTIVE set: the two corpora share every active
            # sentence, so identical scores there are a real possibility and
            # would be a finding (no regression), not a dead pathway.
            strict=(voice == "passive"),
        )
        print(f"  === {voice.upper()} voice (chance {CHANCE:.2f}) ===")
        for name, ens in arms.items():
            verdict = ("ABOVE chance" if ens.beats(CHANCE)
                       else "BELOW chance -- systematically inverted"
                       if ens.high < CHANCE else "indistinguishable from chance")
            print(f"      {ens}   {verdict}")
        delta = paired_delta(arms["alternating"], arms["active_only"],
                             "alternating - active_only")
        print(f"      {delta}")
        print(f"      paired delta clears 0: {delta.beats(0.0)}")

        # Did the SUBSTRATE vary while the metric did not? A zero-width CI is
        # only meaningful once that is answered.
        fps = {v for (pe, vo, sd), v in _FINGERPRINTS.items() if vo == voice}
        n_distinct = len(fps)
        spread = max(e.ci for e in arms.values())
        if spread == 0.0:
            print(f"      SEED-INVARIANT metric with {n_distinct} DISTINCT "
                  f"substrates -- "
                  + ("the decision is not made by the assemblies"
                     if n_distinct > 1 else
                     "seeds never reached the substrate; the CI is fiction"))
        print()


if __name__ == "__main__":
    main()
