"""Does the neural term ever change a role decision? Count it, don't infer it.

`_assign_roles_neural` scores each candidate role as

    score = _STRUCTURAL_PRIOR * (0.5 ** rank)  +  (margin + eps) / total
            \\_______ 1.2 at rank 0, 0.6 at rank 1 _______/   \\_ in [0, 1] _/

so the LEXICAL (assembly-derived) term can only overturn the structural prior
when the normalised margin gap exceeds the rank gap of 0.6. #52 measured role
binding at 0.15-0.22 -- the crowding regime -- which predicts the term is inert.
`passive_payoff` then found the parse answer bit-identical across 10 distinct
substrates, which is what an inert term looks like from the outside.

This measures it from the inside, three ways, on the SAME probes:

    full         as shipped
    prior_only   lexical term forced to 0 -- keeps ONLY word order + gating
    neural_only  _STRUCTURAL_PRIOR = 0    -- keeps only assembly evidence

THE HEADLINE IS THE FLIP COUNT, not the accuracies. If `full` and `prior_only`
make the SAME decision on every item, the assemblies contribute nothing to
parsing, however good their retrieval looks in isolation.

WHAT `neural_only` DOES AND DOES NOT SHOW. With the prior at 0 and no lexical
evidence, every candidate scores alike and the argmax falls to the FIRST entry
of `role_order` -- which is still the structural answer. So `neural_only` is
"assembly evidence, tie-broken structurally", NOT a clean neural-only arm, and
its accuracy is an upper bound on what the assemblies alone could do. The flip
count is the honest statistic; this arm is context.
"""
from __future__ import annotations

import collections
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
from neural_assemblies.assembly_calculus.emergent.parser_mixins import (  # noqa: E402
    roles as roles_module,
)
from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (  # noqa: E402
    build_vocabulary_preset,
)
from neural_assemblies.lexicon.lexicon_manager import WordCategory  # noqa: E402

STAGES = ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES")
N, K = 3000, 30
SEEDS = (11, 23, 37)


class _NoLexicalEvidence:
    """Stands in for a Measured whose `.or_else` is always 0.

    Zeroing the margin this way keeps the SHAPE of the computation identical --
    same call, same candidate loop -- so the arm differs in the evidence, not
    in the code path taken.
    """

    def or_else(self, default: float) -> float:  # noqa: D401
        return 0.0


def _feat(w) -> dict:
    return getattr(w, "features", None) or {}


def _probes(trainer) -> List[Tuple[List[str], str, str]]:
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
    out: List[Tuple[List[str], str, str]] = []
    for verb in verbs:
        for i in range(0, len(animate) - 1, 2):
            a, p = animate[i], animate[i + 1]
            out.append((["the", a.lemma, verb.forms["3sg"], "the", p.lemma],
                        a.lemma, p.lemma))
            out.append((["the", p.lemma, "is", verb.forms["ppart"], "by",
                         "the", a.lemma], a.lemma, p.lemma))
            if len(out) >= 40:
                return out
    return out


def _decide(parser, probes, *, prior: float, lexical: bool):
    """Role decisions under one configuration, plus the observed margins."""
    old_prior = roles_module._STRUCTURAL_PRIOR
    old_margin = roles_module.RoleBindingMixin._role_binding_margin
    seen_margins: List[float] = []

    def traced(self, word, core_area, role_area):
        m = old_margin(self, word, core_area, role_area)
        seen_margins.append(float(m.or_else(0.0)))
        return m if lexical else _NoLexicalEvidence()

    roles_module._STRUCTURAL_PRIOR = prior
    roles_module.RoleBindingMixin._role_binding_margin = traced
    try:
        decisions = []
        for tokens, _a, _p in probes:
            cats = {w: parser.classify_word_cached(w)[0] for w in tokens}
            with parser.brain.read_only():
                r = parser._assign_roles_neural(list(tokens), cats)
            decisions.append(tuple(sorted(
                (w, v) for w, v in r.items() if v is not None)))
        return decisions, seen_margins
    finally:
        roles_module._STRUCTURAL_PRIOR = old_prior
        roles_module.RoleBindingMixin._role_binding_margin = old_margin


def _role_bias(parser) -> dict:
    """Each word's DOMINANT trained role, read from the role lexicons.

    The hypothesis the flip counts point at: the stored binding encodes a
    WORD-LEVEL role frequency ("dog is usually an agent"), not a SENTENCE-LEVEL
    assignment. If so the margin cannot see the current sentence at all, and
    every flip should run toward the word's trained majority role regardless of
    what this sentence says.
    """
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        ROLE_AGENT, ROLE_PATIENT,
    )
    bias = {}
    agent = set(parser.role_lexicons.get(ROLE_AGENT, {}))
    patient = set(parser.role_lexicons.get(ROLE_PATIENT, {}))
    for w in agent | patient:
        if w in agent and w not in patient:
            bias[w] = "AGENT"
        elif w in patient and w not in agent:
            bias[w] = "PATIENT"
        else:
            bias[w] = "BOTH"
    return bias


def _diagnose_flips(full, prior_only, probes, bias) -> tuple:
    """Do the flips run TOWARD each word's trained majority role?"""
    toward = against = unknown = 0
    examples = []
    for dec_full, dec_prior, (tokens, _a, _p) in zip(full, prior_only, probes):
        if dec_full == dec_prior:
            continue
        df, dp = dict(dec_full), dict(dec_prior)
        for w in set(df) | set(dp):
            if df.get(w) == dp.get(w):
                continue
            b = bias.get(w)
            if b in (None, "BOTH"):
                unknown += 1
            elif df.get(w) == b:
                toward += 1
                if len(examples) < 3:
                    examples.append(
                        f"{' '.join(tokens)!r}: {w} {dp.get(w)}->{df.get(w)} "
                        f"(trained bias {b})")
            else:
                against += 1
    return toward, against, unknown, examples


def _accuracy(decisions, probes) -> float:
    hits = 0
    for dec, (_tokens, agent, patient) in zip(decisions, probes):
        d = dict(dec)
        if d.get(agent) == "AGENT" and d.get(patient) == "PATIENT":
            hits += 1
    return hits / max(1, len(probes))


def main() -> None:
    print("Does the neural term ever change a role decision?")
    print()
    totals = collections.Counter()
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        parser = EmergentParser(
            n=N, k=K, seed=seed, vocabulary=build_vocabulary_preset("core"),
            fast_training=True)
        trainer = CurriculumTrainer(parser)
        for stage in STAGES:
            trainer.train_stage(stage)
        probes = _probes(trainer)

        full, margins = _decide(parser, probes,
                                prior=roles_module._STRUCTURAL_PRIOR,
                                lexical=True)
        prior_only, _ = _decide(parser, probes,
                                prior=roles_module._STRUCTURAL_PRIOR,
                                lexical=False)
        neural_only, _ = _decide(parser, probes, prior=0.0, lexical=True)

        flips = sum(1 for a, b in zip(full, prior_only) if a != b)
        nonzero = sum(1 for m in margins if m > 0.0)
        mx = max(margins) if margins else 0.0
        totals["flips"] += flips
        totals["items"] += len(probes)
        totals["nonzero"] += nonzero
        totals["margins"] += len(margins)

        print(f"  seed {seed}")
        print(f"      accuracy  full {_accuracy(full, probes):.4f}   "
              f"prior_only {_accuracy(prior_only, probes):.4f}   "
              f"neural_only {_accuracy(neural_only, probes):.4f}")
        print(f"      margins   {nonzero}/{len(margins)} non-zero, "
              f"max {mx:.4f}  (needs a normalised gap > 0.6 to overturn rank)")
        print(f"      FLIPS full vs prior_only: {flips}/{len(probes)}")
        bias = _role_bias(parser)
        toward, against, unknown, examples = _diagnose_flips(
            full, prior_only, probes, bias)
        print(f"      flip direction: {toward} toward the word's trained "
              f"majority role, {against} against, {unknown} unknown/both")
        for e in examples:
            print(f"          {e}")
        totals["toward"] += toward
        totals["against"] += against

    print()
    print(f"  TOTAL flips {totals['flips']}/{totals['items']} decisions; "
          f"{totals['nonzero']}/{totals['margins']} margins non-zero")
    if totals["flips"] == 0:
        print("  => the assembly evidence changes NOTHING. Parsing is decided")
        print("     entirely by word order and the gating rule.")
    else:
        print(f"  flips toward the trained majority role: {totals['toward']}, "
              f"against: {totals['against']}")


if __name__ == "__main__":
    main()
