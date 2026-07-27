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

CAN ONE COMPETITION HOLD BOTH ROUTES? NO -- the prior is too coarse
--------------------------------------------------------------------
`head_start_sweep()`, 6 seeds. The prior is paid in TIME (hold ROLE_PATIENT shut
for r rounds so the agent slot accumulates first), because inhibition is binary
and there is no graded weight to turn.

    head_start   reversible        irreversible
    0            0.667 +/-0.000    1.000 +/-0.000    <- pure competition
    1            1.000 +/-0.000    0.600 +/-0.101
    2            1.000 +/-0.000    0.600 +/-0.101
    3            0.500 +/-0.000    0.000 +/-0.000    <- degenerate, see below
    5            0.500 +/-0.000    0.000 +/-0.000    <- degenerate, see below

PREDICTION REFUTED. I expected a small head start to fix reversible while
leaving irreversible at 1.000, on the reasoning that lexical margins are large.
No value scores 1.000 on both, and the mechanism measurement says why. One round
of anticipation is not a nudge -- it is worth as much as the entire lexical
margin range:

    word   patient drive   AGENT hs=0 -> hs=1   anticipation gain   lexical margin
    dog        54.7          123.7 -> 201.8          +78.1              -69.0
    bird       91.1           57.6 -> 129.5          +72.0              +33.6
    book       72.8           22.2 ->  81.7          +59.5              +50.6
    ball       89.8           13.8 ->  49.3          +35.5              +76.0

Patient drive is untouched, and AGENT gains through RECURRENT SELF-DRIVE: an
open, active area self-projects in the derived map, so a pre-activated slot
sustains itself. That is a full k-winner assembly's worth of drive, which is why
the minimum step (+35 to +78) overshoots the resolution the task needs --
separating `bird` (+33.6, should flip to AGENT) from `book` (+50.6, must not)
requires finer grain than one round can express. At hs=1 `bird` flips correctly
and `ball` (+76) correctly resists, but `book` flips too; hence 0.600 with real
seed variance rather than a clean 0 or 1.

SCOPE OF THE NEGATIVE RESULT: this refutes a TIME-QUANTISED prior at one-round
granularity, not the composition idea in general. A weaker anticipatory drive
(projecting into ROLE_AGENT from a smaller source rather than for longer) is
untested and would be the next thing to try.

The hs>=3 rows are DEGENERATE and must not be read as "strong prior becomes the
SVO template". Traced: the SECOND noun reads None. Once the first noun's slot is
winner-protected, the second noun has only ROLE_PATIENT available, so shutting
it for most of the round budget leaves nothing to bind into. Those rows measure
a readout failure, not a prior.

Note: `head_start` closes the PATIENT AREA rather than the core->PATIENT fiber.
Measured behaviourally NEUTRAL (both variants give the table above), and kept
because area rules are how the SVO program expresses slot state.

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
                # BOTH, not either. Sampling on "either" catches calls only one
                # role area is a target of, and `.get(other, 0.0)` then reports
                # 0.0 for an area that was never in the call -- which reads
                # exactly like the competition destroying it. Under
                # `head_start` that is the first call every time. The
                # competition is the call where both are present.
                if ROLE_AGENT in scores and ROLE_PATIENT in scores:
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


def head_start_sweep(seeds: Sequence[int] = (42, 43, 44, 45, 46, 47),
                     values: Sequence[int] = (0, 1, 2, 3, 5)) -> None:
    """Can ONE competition hold both routes, with the prior paid in TIME?

    Gating and MI are currently ALTERNATIVES. Composing them needs a GRADED
    positional prior, and inhibition is binary -- a slot is open or shut, which
    is the all-or-nothing SVO template. What is gradable in assembly calculus is
    WHEN a fiber opens. `head_start=r` holds core->ROLE_PATIENT shut for r
    rounds so the agent slot accumulates drive first, then opens it and lets
    mutual inhibition compare. The prior is a number of rounds, not a
    coefficient fitted to the answer.

    PREDICTION, recorded before running
    -----------------------------------
    head_start=0 reproduces pure competition (0.667 / 1.000) and a large
    head_start reproduces the SVO template (1.000 / 0.000), because PATIENT
    never opens in time to compete. The interesting claim is that a SMALL head
    start gives 1.000 on BOTH: the lexical margins are large (`ball` drives
    PATIENT ~5x harder than AGENT) so one round of anticipation should not
    overturn them, while an exact 40/40 tie has nothing to resist it with.

    FALSIFIABLE: if NO value scores 1.000 on both, then the two routes cannot
    share a single competition, and the architecture genuinely needs two
    decision stages rather than one competition with a prior. That is a real
    result either way, so do not tune `values` until it works -- report the
    curve.
    """
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

    transitive = infer_transitive_verbs(create_training_sentences() + build_corpus())
    kinds = {
        kind: [(list(words), gold)
               for words, gold, k in test_items() if k == kind]
        for kind in ("reversible", "irreversible")
    }

    print(f"\n  {len(seeds)} seeds.  head_start = rounds ROLE_PATIENT stays shut")
    print("  while ROLE_AGENT accumulates.  0 = pure lexical competition.\n")
    print(f"  {'head_start':<12}{'reversible':>18}{'irreversible':>18}")

    # Train ONCE per seed and deepcopy per condition. Training dominates the
    # wall clock (~21s a seed), so training inside the head_start loop would
    # cost 60 trainings instead of 6 for identical numbers.
    scores = {(hs, kind): [] for hs in values for kind in kinds}
    for seed in seeds:
        trained = train_parser(seed)
        for hs in values:
            for kind, rows in kinds.items():
                brain = copy.deepcopy(trained)
                ok = total = 0
                for words, gold in rows:
                    pred = NemoParser(
                        brain, transitive_verbs=transitive,
                        competitive=True, head_start=hs,
                    ).parse(list(words))
                    a, b = _score(pred, gold)
                    ok += a
                    total += b
                scores[(hs, kind)].append(ok / max(total, 1))

    for hs in values:
        cells = "".join(
            f"{m:.3f} +/-{h:.3f}".rjust(18)
            for m, h in (_mean_ci(scores[(hs, kind)]) for kind in kinds)
        )
        print(f"  {hs:<12}{cells}", flush=True)


if __name__ == "__main__":
    if "--margins" in sys.argv:
        margins()
    elif "--head-start" in sys.argv:
        head_start_sweep()
    else:
        run()
