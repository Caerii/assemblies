"""Does a NEVER-SEEN subject+verb combination get a structured representation?

THE QUESTION, stated so it can fail
------------------------------------
`constituent_structure.py` establishes that a trained VP's representation tracks
its parts: overlap is graded by how many parents two constituents share, and the
gradient survives a training-proximity control. That is compositional STRUCTURE
in what the model has been trained on. It is not yet compositional
PRODUCTIVITY.

The productivity claim is the one that matters for the comparison with deep
sequence models, which fail systematic recombination (SCAN, COGS): a model
should represent `dog sees` correctly having seen `dog` with other verbs and
`sees` with other subjects, but never the two together. Assembly calculus has a
structural reason to succeed -- merge is a conjunctive bind over independently
formed parent assemblies, and nothing in it consults whether the pair was ever
co-experienced. This experiment asks whether that reasoning survives contact
with the substrate.

PROTOCOL
--------
* HOLD OUT a diagonal of (subject, verb) pairs. Every held-out subject still
  appears with other verbs and every held-out verb with other subjects, so only
  the COMBINATION is novel -- never a word.
* Train on the filtered corpus.
* At test, form the held-out VP by the SAME protocol `train_phrases` uses
  (project both parents, then merge), on a FORK so probes cannot contaminate
  each other.
* Score it against the trained constituents by shared parent, exactly as
  `constituent_structure.py` does: mean overlap with those sharing its subject,
  its verb, or nothing.
* Run the IDENTICAL procedure on pairs that WERE trained. That is the control,
  and it is what makes the contrast interpretable: same brain, same protocol,
  same scoring -- the only difference is whether the combination was ever seen.

PREDICTIONS, recorded before running
-------------------------------------
1. PRODUCTIVITY: the held-out gradient (share-parent minus share-nothing) is
   positive. A novel combination lands in the right region of VP space.
2. The held-out gradient is COMPARABLE to the trained gradient. If merge is
   genuinely a function of its parts, having seen the pair should add little.
3. A substantially SMALLER held-out gradient is the memorisation outcome and is
   a real possible result: it would mean the trained gradient is carried by
   pair-specific Hebbian traces rather than by composition.
4. The novel VP is not a duplicate of any trained one -- max overlap with the
   trained set stays below the level that would indicate it collapsed onto a
   memorised phrase. Reported so "generalisation" cannot be satisfied by
   returning a familiar assembly.

WHAT WOULD FALSIFY THE HEADLINE CLAIM
--------------------------------------
A flat held-out gradient with an intact trained gradient. That is memorisation
wearing the appearance of structure, and it is the outcome this design exists to
be able to see.
RE-BASELINED 2026-07-28, after synapse initialisation became content-addressed
------------------------------------------------------------------------------
That change moves every seeded weight, so any number here derived from one seed
was unverified until re-derived. Re-run over 5 seeds under BOTH disciplines by
``rebaseline.py``, which reports the reduced claim as mean +/- 95% CI:

    unseen_gradient     +0.2298 +/- 0.0264  (content)   +0.2436 +/- 0.0149  (stream)
    unseen_minus_seen   +0.0208 +/- 0.0068  (content)   +0.0168 +/- 0.0171  (stream)

Both survive, the two disciplines overlap, and ``unseen_minus_seen`` is
positive with its interval excluding zero under the current engine -- unseen
combinations score slightly ABOVE trained ones, not merely level with them.
Change a number here and change it there; ``rebaseline.py`` is what actually
gets re-run.
"""

from __future__ import annotations

import copy
import os
import statistics
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262}

#: One held-out pair per subject, cycling verbs so every verb also loses a
#: subject. Chosen so no word drops out of training entirely.
HELD_OUT: Tuple[Tuple[str, str], ...] = (
    ("dog", "chases"),
    ("cat", "finds"),
    ("bird", "sees"),
    ("boy", "chases"),
    ("girl", "finds"),
)


def _ci(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        return (values[0] if values else float("nan")), float("nan")
    t = _T95.get(n, 1.96)
    return statistics.mean(values), t * statistics.stdev(values) / (n ** 0.5)


def _subj_verb(sent) -> Tuple[str, str] | None:
    """The (agent, action) pair of a sentence, or None if it has no such pair."""
    subj = verb = None
    for word, role in zip(sent.words, sent.roles):
        if role == "agent":
            subj = word
        elif role == "action":
            verb = word
    return (subj, verb) if subj and verb else None


def build_filtered_corpus(held_out):
    """The full training corpus with the held-out COMBINATIONS removed."""
    from lesion_aphasia import build_corpus
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )

    base = list(create_training_sentences()) + list(build_corpus())
    kept, dropped = [], 0
    for sent in base:
        pair = _subj_verb(sent)
        if pair is not None and pair in held_out:
            dropped += 1
            continue
        kept.append(sent)
    return kept, dropped


def form_vp(parser, subj_word: str, verb_word: str):
    """Form a VP for (subj, verb) exactly as `train_phrases` does, on a fork.

    Forked because probing is not read-only: merge applies plasticity and
    recruits in VP, so forming one probe would otherwise change the next.
    """
    from neural_assemblies.assembly_calculus.ops import merge, project
    from neural_assemblies.assembly_calculus.emergent.core.areas import VERB_CORE, VP
    from neural_assemblies.assembly_calculus.emergent.parser_mixins.core import (
        MERGE_ROUNDS,
    )

    fork = copy.deepcopy(parser)
    subj_core = fork._word_core_area(subj_word)
    project(fork.brain, fork.stim_map[subj_word], subj_core, rounds=fork.rounds)
    project(fork.brain, fork.stim_map[verb_word], VERB_CORE, rounds=fork.rounds)
    return merge(fork.brain, subj_core, VERB_CORE, VP, rounds=MERGE_ROUNDS)


def profile(vp_asm, trained: Mapping[str, object], subj: str, verb: str):
    """Mean overlap of `vp_asm` with trained constituents, by shared parent."""
    from neural_assemblies.assembly_calculus.assembly import overlap

    bins: Dict[str, List[float]] = {"subject": [], "verb": [], "nothing": []}
    all_ov = []
    for key, asm in trained.items():
        ks, kv = key.split("_", 1)
        if ks == subj and kv == verb:
            continue  # the pair itself, when scoring a TRAINED control
        ov = overlap(vp_asm, asm)
        all_ov.append(ov)
        if ks == subj:
            bins["subject"].append(ov)
        elif kv == verb:
            bins["verb"].append(ov)
        else:
            bins["nothing"].append(ov)
    means = {b: (statistics.mean(v) if v else float("nan"))
             for b, v in bins.items()}
    means["max_any"] = max(all_ov) if all_ov else float("nan")
    return means


def run(seeds: Sequence[int] = (42, 7, 123), n: int = 1000, k: int = 50) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser

    held = set(HELD_OUT)
    corpus, dropped = build_filtered_corpus(held)
    print(f"\n  held out {len(held)} combinations, {dropped} sentences removed;"
          f" {len(corpus)} remain")
    print(f"  {len(seeds)} seeds, n={n} k={k}\n")

    grad_unseen, grad_seen = [], []
    unseen_rows, seen_rows = [], []

    for seed in seeds:
        parser = EmergentParser(n=n, k=k, p=0.05, beta=0.1, seed=seed, rounds=10)
        parser.train(corpus)
        trained = {kk: v for kk, v in parser.vp_assemblies.items()
                   if kk.count("_") == 1}

        # Sanity: the held-out pairs really are absent from what was learned.
        leaked = [f"{s}_{v}" for s, v in held if f"{s}_{v}" in trained]
        if leaked:
            print(f"  WARNING seed {seed}: held-out pairs present in "
                  f"training output: {leaked}")

        # Trained controls, drawn from the same vocabulary as the held-out set
        # so the two arms are comparable.
        subjects = {s for s, _ in held}
        verbs = {v for _, v in held}
        controls = [tuple(kk.split("_", 1)) for kk in trained
                    if kk.split("_", 1)[0] in subjects
                    and kk.split("_", 1)[1] in verbs]

        for _arm, pairs, rows, grads in (
            ("unseen", sorted(held), unseen_rows, grad_unseen),
            ("seen", sorted(controls), seen_rows, grad_seen),
        ):
            per_pair = []
            for subj, verb in pairs:
                if subj not in parser.stim_map or verb not in parser.stim_map:
                    continue
                vp = form_vp(parser, subj, verb)
                per_pair.append(profile(vp, trained, subj, verb))
            if not per_pair:
                continue
            agg = {b: statistics.mean(p[b] for p in per_pair
                                      if p[b] == p[b])
                   for b in ("subject", "verb", "nothing", "max_any")}
            shared = statistics.mean([agg["subject"], agg["verb"]])
            grads.append(shared - agg["nothing"])
            rows.append(agg)

    print(f"  {'arm':<10}{'share subj':>12}{'share verb':>12}"
          f"{'share none':>12}{'max any':>10}")
    for label, rows in (("unseen", unseen_rows), ("seen", seen_rows)):
        if not rows:
            continue
        m = {b: statistics.mean(r[b] for r in rows)
             for b in ("subject", "verb", "nothing", "max_any")}
        print(f"  {label:<10}{m['subject']:>12.3f}{m['verb']:>12.3f}"
              f"{m['nothing']:>12.3f}{m['max_any']:>10.3f}")

    gu, hu = _ci(grad_unseen)
    gs, hs = _ci(grad_seen)
    print(f"\n  GRADIENT (mean of share-subj/share-verb, minus share-none)")
    print(f"    unseen combinations   {gu:>7.3f} +-{hu:.3f}")
    print(f"    seen combinations     {gs:>7.3f} +-{hs:.3f}")

    print("\n  READING")
    productive = (gu - hu) > 0
    print(f"    novel combination is structured:  {productive}")
    if productive and gs > 0:
        ratio = gu / gs if gs else float("nan")
        print(f"    unseen/seen gradient ratio:       {ratio:.2f}")
        if ratio > 0.7:
            print("    -> PRODUCTIVE: a never-seen combination is represented")
            print("       about as compositionally as a trained one.")
        else:
            print("    -> PARTIAL: novel combinations are structured but")
            print("       measurably weaker; training the pair adds something,")
            print("       so composition is not the whole story.")
    elif not productive:
        print("    -> MEMORISATION: the trained gradient does not transfer to")
        print("       novel combinations. The headline claim FAILS.")


if __name__ == "__main__":
    run()
