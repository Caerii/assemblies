"""Choose calibration-frame words BY MEASURING the parser, not by remembering.

`the_calibration_frames_are_untrained.md`: the shipped frames were authored
against `create_training_sentences()` and are measured on a CDS-trained parser
where their main verb does not occur. Hand-authoring against a remembered
corpus is what failed; hand-authoring against a DIFFERENT remembered corpus
would fail the same way the next time the curriculum moves.

So select the words by asking this parser three things, in this order:

  1. is the word TRAINED -- does it have a core lexicon assembly? (the
     predicate `stim_map` was standing in for, and is not);
  2. does the parser CLASSIFY it as the category the frame needs? A trained
     word can still read as the wrong category -- `small` is a declared holdout
     AND classifies VERB, and `chases` is untrained AND classifies NOUN, so
     these are two independent failures and both must be excluded;
  3. does it OCCUR in the corpus that trained this depth, and how often? Not a
     filter -- reported, because a word trained from one occurrence is a
     different kind of evidence from one trained from eleven.

Emits a ranked candidate list per slot and a proposed area-matched frame set.
It proposes; it does not ship. The frames it suggests still have to beat the
current ones in a pre-registered study, because changing them moves calibrated
thresholds and every golden downstream.
"""
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    default_holdout_set,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)

SEEDS = [11, 42]


def _corpus_counts():
    """Word frequencies in the corpus that trains SENTENCES/COMPLEX_GRAMMAR.

    Read from the same modules `CurriculumTrainer._cds_corpus_sentences`
    imports, so this cannot drift from what training saw the way a
    hand-copied count does.
    """
    from neural_assemblies.lexicon.curriculum.stage4_sentences import (
        STAGE4_CORPUS,
    )
    from neural_assemblies.lexicon.curriculum.stage3_two_word import (
        STAGE3_CORPUS,
    )
    return Counter(
        w for line in list(STAGE4_CORPUS) + list(STAGE3_CORPUS)
        for w in line.split()
    )


def _profile(parser, holdouts, counts):
    """word -> (trained, category, corpus count) for everything registered."""
    trained = {w for lex in parser.core_lexicons.values() for w in lex}
    rows = {}
    for word in sorted(set(trained) | set(holdouts)):
        cat, _scores = parser.classify_word(word)
        rows[word] = (word in trained, cat, counts.get(word, 0))
    return rows


def main():
    counts = _corpus_counts()
    holdouts = set(default_holdout_set())

    profiles = {}
    for seed in SEEDS:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        profiles[seed] = _profile(parser, holdouts, counts)

    # A word is usable only if EVERY seed agrees on its category. A word whose
    # category flips between seeds is not a stable item, and averaging over an
    # unstable item is how a frame set stops meaning one thing.
    base = profiles[SEEDS[0]]
    stable = {}
    for word, (tr, cat, n) in base.items():
        cats = {profiles[s][word][1] for s in SEEDS}
        if len(cats) == 1:
            stable[word] = (tr, cat, n)

    unstable = sorted(set(base) - set(stable))
    print(f"profiled {len(base)} words over seeds {SEEDS}; "
          f"{len(unstable)} classify DIFFERENTLY across seeds: {unstable}")
    print()

    def pick(category, *, want_trained, top=12):
        rows = [(n, w) for w, (tr, cat, n) in stable.items()
                if cat == category and tr == want_trained
                and (w in holdouts) == (not want_trained)]
        rows.sort(reverse=True)
        return rows[:top]

    print("TRAINED nouns classified NOUN (corpus count, word):")
    print("  ", pick("NOUN", want_trained=True))
    print("TRAINED verbs classified VERB:")
    print("  ", pick("VERB", want_trained=True))
    print("TRAINED determiners classified DET:")
    print("  ", pick("DET", want_trained=True))
    print("HOLDOUT words classified NOUN (the novel arm needs these):")
    print("  ", pick("NOUN", want_trained=False))
    print("HOLDOUT words classified VERB:")
    print("  ", pick("VERB", want_trained=False))
    print()

    nouns = [w for _n, w in pick("NOUN", want_trained=True, top=6)]
    verbs = [w for _n, w in pick("VERB", want_trained=True, top=6)]
    novel = [w for _n, w in pick("NOUN", want_trained=False, top=3)]
    dets = [w for _n, w in pick("DET", want_trained=True, top=2)]

    print("PROPOSED area-matched frames (critical word always in OBJECT")
    print("position, so every arm expects the same slot):")
    if len(nouns) < 4 or len(verbs) < 4 or not novel or not dets:
        print("  INSUFFICIENT stable vocabulary -- "
              f"nouns={len(nouns)} verbs={len(verbs)} "
              f"novel={len(novel)} dets={len(dets)}")
        return
    d = dets[0]
    for i in range(3):
        subj, obj = nouns[i], nouns[i + 3]
        v = verbs[i]
        vio = verbs[i + 3]
        print(f'    ("grammatical",       "trained noun object {i+1}", '
              f'[{d!r}, {subj!r}, {v!r}, {obj!r}]),')
        print(f'    ("category_violation","verb as object {i+1}",      '
              f'[{d!r}, {subj!r}, {v!r}, {vio!r}]),')
        print(f'    ("novel_noun",        "holdout noun object {i+1}", '
              f'[{d!r}, {subj!r}, {v!r}, {novel[0]!r}]),')
    print()
    print("Each triple is a MINIMAL PAIR: same determiner, same subject, same")
    print("verb, differing only in the critical word. The three arms therefore")
    print("differ in the critical word ALONE, which is the contrast the labels")
    print("claim and the shipped set does not deliver.")


if __name__ == "__main__":
    main()
