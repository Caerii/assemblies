"""Can MUTUAL INHIBITION supply the lexical route that fiber-gating lacks?

WHERE THIS STARTS
-----------------
`nemo_substrate_sweep.py` established, at every substrate with zero variance:

    reversible   (word order suffices)      symbolic 1.000   gating 1.000
    irreversible (order is misleading)      symbolic 1.000   gating 0.000

Gating is a PERFECT POSITIONAL TEMPLATE. The symbolic parser matches it on
reversible items and beats it on irreversible ones purely because it also has a
lexical route -- but that route is `_score_role_binding`, a Python margin
comparison, not a neural mechanism. So the two-route claim rested on one route
being an algorithm rather than a dynamics.

THE QUESTION
------------
Is there an ASSEMBLY-CALCULUS-NATIVE mechanism that does the same job? The
audit (`research/PRIMITIVES_AUDIT.md`) found one already wired and never firing:
inter-area MUTUAL INHIBITION. `_apply_mutual_inhibition` is winner-take-all over
`total_activation` across a group of areas -- exactly a competition decided by
LEARNED WEIGHT. It never runs in production because it only triggers when two or
more group areas are targets of the SAME `project()` call, and the SVO rule
program never opens two role slots at once.

`competitive_verb_program` removes one rule -- it does not `INHIBIT ROLE_AGENT`
-- and `competitive_initial_open_areas` opens both slots from word 1. A noun
then projects into AGENT and PATIENT together, and MI arbitrates on the weight
the corpus built. Nothing is scored in Python.

PREDICTION, recorded before running
-----------------------------------
Competitive mode should RESCUE irreversible items, where a lexical preference
exists and only position argues against it, and should LOSE reversible items,
where the nouns are balanced by construction so removing the positional
tie-breaker leaves nothing to decide with. If both halves hold, gating and MI
are DISSOCIABLE LAYERS rather than rival implementations, and the two-route
architecture stands with BOTH routes in primitives.

RESULT (12 brain seeds, paired, zero variance throughout)
---------------------------------------------------------
Confirmed, and the failure mode is sharper than "loses reversible".

    kind            symbolic           gating       competitive
    reversible      1.000 +/-0.000     1.000 +/-0.000    0.667 +/-0.000
    irreversible    1.000 +/-0.000     0.000 +/-0.000    1.000 +/-0.000

Competitive is the EXACT COMPLEMENT of gating on irreversible items. And the
decision is GRADED, not a binary that happens to land right: `margins()` records
`activation_scores` at the moment `_apply_mutual_inhibition` compares them, and
the margin (PATIENT minus AGENT) orders with corpus role bias --

    word   corpus A/P   margin (seeds 42/43/44)   picks
    dog        53/2      -69.0  -98.8  -63.1      AGENT
    boy        40/36     -48.1  -24.9  -17.9      AGENT
    girl       40/36      -1.4   -9.9  -12.0      AGENT
    bird       40/40     +33.6  +40.0   +9.8      PATIENT
    book        0/21     +50.6  +55.3  +67.8      PATIENT
    ball        1/20     +76.0  +88.0  +82.2      PATIENT

The two words closest to a corpus tie (`girl`, `bird`) are the two closest to
the decision boundary, and the lexically extreme words sit farthest from it.
That is a dose-response relation between what the corpus taught and what the
dynamics decide, with nothing scored in Python.

OPEN, and deliberately not explained away: `bird` and `boy` have the SAME 40
agent tokens, yet boy's agent drive is roughly double bird's. Equal exposure is
not equal drive, and frequency alone does not account for it -- suppression by
the high-frequency agents (dog/cat, 53 each) would depress both equally. Worth
a separate measurement rather than a story.

Competitive is the EXACT COMPLEMENT of gating on irreversible items (0.000 ->
1.000). Its reversible score is not noise: because the winning slot is closed to
protect the binding, only the FIRST noun chooses freely and the second takes
what remains, so one word decides a whole sentence. Counting the corpus by its
own role labels:

    bird   agent 40   patient 40      <- an EXACT tie
    boy    agent 40   patient 36
    girl   agent 40   patient 36

Every reversible failure is a sentence with `bird` in first position. MI has no
positional information to break a 40/40 tie with, so it falls back on a
connectome asymmetry and picks PATIENT. On the 40/36 leaners it is correct. That
is not a broken mechanism -- it is a pure lexical-statistics arbiter behaving
exactly like one, and it is why word order has to exist as a SEPARATE layer.

TWO BUGS THIS MEASUREMENT FOUND, both worth keeping in mind
-----------------------------------------------------------
1. Reading `alive` as "role areas with winners" counted a slot won by an EARLIER
   noun, because closing an area does not clear it. 8 of 12 reversible items
   read out None. A closed slot is out of the competition BY DEFINITION.
2. The winner-protection has to be held on its own INDEX CHANNEL. Word programs
   act on channel 0, and the verb's `DISINHIBIT ROLE_PATIENT, 0` reopened a slot
   a noun had just won, after which `prepare_targets` wiped the binding. This is
   the concrete case that a boolean `inhibited` flag cannot express and NEMO's
   set-of-indices can.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Sequence

#: Seeds are BRAIN seeds, swept with paired deepcopy so all three conditions
#: see an identical parser -- see the `report-distributions-not-point-estimates`
#: lesson: determinism is not sample size.
SEEDS = tuple(range(42, 54))


def run(seeds: Sequence[int] = SEEDS) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from lesion_aphasia import build_corpus, test_items, train_parser
    from nemo_vs_symbolic import _mean_ci, _score
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )

    corpus = create_training_sentences() + build_corpus()
    transitive = infer_transitive_verbs(corpus)
    kinds = {
        kind: [(list(words), gold)
               for words, gold, k in test_items() if k == kind]
        for kind in ("reversible", "irreversible")
    }

    print(f"\n  {len(seeds)} brain seeds, paired.  gating = SVO slot-switch;")
    print("  competitive = both slots open, mutual inhibition arbitrates.\n")
    print(f"  {'kind':<14}{'symbolic':>17}{'gating':>17}{'competitive':>17}")

    for kind, rows in kinds.items():
        acc = {"sym": [], "gate": [], "comp": []}
        for seed in seeds:
            parser = train_parser(seed)
            # The symbolic path needs no copy (`parse()` is re-runnable); the
            # NemoParser paths mutate the brain, so each gets its own.
            gating_brain = copy.deepcopy(parser)
            comp_brain = copy.deepcopy(parser)
            tally = {k: [0, 0] for k in acc}
            for words, gold in rows:
                preds = (
                    ("sym", parser.parse(list(words))["roles"]),
                    ("gate", NemoParser(
                        gating_brain, transitive_verbs=transitive,
                    ).parse(list(words))),
                    ("comp", NemoParser(
                        comp_brain, transitive_verbs=transitive,
                        competitive=True,
                    ).parse(list(words))),
                )
                for key, pred in preds:
                    ok, total = _score(pred, gold)
                    tally[key][0] += ok
                    tally[key][1] += total
            for key in acc:
                acc[key].append(tally[key][0] / max(tally[key][1], 1))

        cells = "".join(
            f"{m:.3f} +/-{h:.3f}".rjust(17)
            for m, h in (_mean_ci(acc[k]) for k in ("sym", "gate", "comp"))
        )
        print(f"  {kind:<14}{cells}", flush=True)

    _report_role_counts(corpus)


def _report_role_counts(corpus) -> None:
    """The tie that explains the reversible score, counted not assumed.

    Reversible items are called balanced because they use nouns trained in both
    roles, but "both roles" is not "equally often", and the difference is the
    whole result. Counting uses the corpus's OWN role labels rather than
    position, so it cannot be confounded by word order.
    """
    import collections

    counts = collections.defaultdict(collections.Counter)
    for sentence in corpus:
        for word, role in zip(sentence.words, sentence.roles):
            counts[word][role] += 1

    print("\n  corpus role counts for the reversible-item nouns:")
    for word in ("bird", "boy", "girl"):
        c = counts[word]
        tie = "  <- exact tie" if c["agent"] == c["patient"] else ""
        print(f"    {word:<6} agent {c['agent']:>3}   patient {c['patient']:>3}{tie}")


def margins(seeds: Sequence[int] = (42, 43, 44),
            words: Sequence[str] = ("dog", "boy", "girl", "bird", "book", "ball"),
            ) -> None:
    """What MI actually compares, captured where it compares it.

    Do NOT try to read this off the connectomes directly: `compute_inputs` on
    `connectomes[NOUN_CORE][ROLE_AGENT]` returns all zeros before a parse,
    because those connectomes are lazily materialized. That probe measures
    nothing and reports it as 0.0 -- the same trap as the shape-(0,0)
    VERB_CORE -> ROLE_ACTION finding. The deciding quantity only exists inside
    `_apply_mutual_inhibition`, so record it from there.

    Each probe is `<noun> chases dog`, because with the winning slot closed to
    protect the binding, only the FIRST noun chooses freely.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from lesion_aphasia import build_corpus, train_parser
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        ROLE_AGENT, ROLE_PATIENT,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )

    transitive = infer_transitive_verbs(create_training_sentences() + build_corpus())

    for seed in seeds:
        base = train_parser(seed)
        print(f"\n  seed {seed}   activation_scores at the first noun's MI step")
        print(f"    {'first noun':<12}{'AGENT':>11}{'PATIENT':>11}{'margin':>11}"
              f"   wins")
        for noun in words:
            parser = copy.deepcopy(base)
            brain_cls = type(parser.brain)
            original = brain_cls._apply_mutual_inhibition
            captured = []

            def spy(self, scores, _orig=original, _cap=captured):
                if ROLE_AGENT in scores or ROLE_PATIENT in scores:
                    _cap.append(dict(scores))
                return _orig(self, scores)

            brain_cls._apply_mutual_inhibition = spy
            try:
                NemoParser(parser, transitive_verbs=transitive,
                           competitive=True).parse([noun, "chases", "dog"])
            finally:
                brain_cls._apply_mutual_inhibition = original

            if not captured:
                print(f"    {noun:<12}{'-- MI never fired --':>33}")
                continue
            first = captured[0]
            agent = float(first.get(ROLE_AGENT, 0.0))
            patient = float(first.get(ROLE_PATIENT, 0.0))
            print(f"    {noun:<12}{agent:>11.1f}{patient:>11.1f}"
                  f"{patient - agent:>11.1f}"
                  f"   {'PATIENT' if patient > agent else 'AGENT'}")


if __name__ == "__main__":
    if "--margins" in sys.argv:
        margins()
    else:
        run()
