"""Are the ERP calibration frames made of words the parser ever TRAINED?

`frames.py` already carries this exact lesson, written for one word:

    `verb as object 3` read ["she", "hits", "the", "eats"] until 2026-08-05.
    `hits` is in no curriculum sentence and no holdout -- it occurred ONLY in
    this file -- so the parser categorised it UNKNOWN and that item's MAIN VERB
    was unrecognised before its critical word was ever reached ... A third of
    the category-violation arm was therefore not a category violation.

The replacement was `chases`, "which is trained (16 curriculum occurrences)".
That count was taken from `create_training_sentences()`, which is the
FULL_TRAIN corpus. Every ERP result in this repo is measured on
`get_parser_cache().fork("SENTENCES")`, which is built by `CurriculumTrainer`
from the CDS corpus -- where `chases` occurs ZERO times.

THE GUARD THAT SHOULD HAVE CAUGHT IT TESTS THE WRONG THING.
`collect_frame_samples` keeps a word when `w in parser.stim_map`. That is
REGISTERED, not TRAINED: the vocabulary preset registers 222 words and the
parser ends up with 517 stimuli, while only ~123 words ever reach a core
lexicon. A registered-but-untrained word survives the filter, gets a phon
stimulus, gets a category from the distributional classifier, and produces a
number -- the totalizing substrate again, and `same-name-two-meanings` again.

This prints, for every word of every shipped frame: registered, trained (has a
core lexicon assembly), declared holdout, and the category the parser actually
assigns. A holdout SHOULD be untrained -- that is what makes the novel arm
novel. An untrained word that is NOT a holdout is the defect.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    AREA_MATCHED_CALIBRATION_FRAMES,
    DEFAULT_CALIBRATION_FRAMES,
    SWEEP_CALIBRATION_FRAMES,
    critical_position_for_frame,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    default_holdout_set,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)

DEPTH = os.environ.get("AUDIT_DEPTH", "SENTENCES")
SEED = 11

FRAME_SETS = {
    "DEFAULT_CALIBRATION_FRAMES": DEFAULT_CALIBRATION_FRAMES,
    "AREA_MATCHED_CALIBRATION_FRAMES": AREA_MATCHED_CALIBRATION_FRAMES,
    "SWEEP_CALIBRATION_FRAMES": SWEEP_CALIBRATION_FRAMES,
}


def _trained_words(parser):
    return {w for lex in parser.core_lexicons.values() for w in lex}


def main():
    parser = get_parser_cache().fork(DEPTH, seed=SEED)
    trained = _trained_words(parser)
    holdouts = set(default_holdout_set())

    print(f"depth={DEPTH} seed={SEED}")
    print(f"registered stimuli: {len(parser.stim_map)}   "
          f"words with a core assembly: {len(trained)}   "
          f"declared holdouts: {sorted(holdouts)}")
    print()

    for set_name, frames in FRAME_SETS.items():
        print(f"### {set_name}")
        bad_items = 0
        for label, desc, words in frames:
            known = [w for w in words if w in parser.stim_map]
            pos = critical_position_for_frame(
                label, known, holdout_words=holdouts,
            )
            pos = min(pos, len(known) - 1)
            critical = known[pos] if known else ""
            marks = []
            offenders = []
            for w in words:
                if w in trained:
                    marks.append(w)
                elif w in holdouts:
                    marks.append(f"{w}[holdout]")
                else:
                    marks.append(f"{w}[UNTRAINED]")
                    offenders.append(w)
            if offenders:
                bad_items += 1
            cats = {}
            for w in known:
                cat, _scores = parser.classify_word(w)
                cats[w] = cat
            print(f"  {label:<19} {desc:<24} {' '.join(marks)}")
            print(f"  {'':19} critical={critical!r} "
                  f"categories={ {w: cats[w] for w in known} }")
        print(f"  --> {bad_items}/{len(frames)} items contain a word that is "
              f"neither trained nor a declared holdout")
        print()

    print("A HOLDOUT IS SUPPOSED TO BE UNTRAINED -- that is what makes the")
    print("novel arm novel. An UNTRAINED non-holdout is the defect: the item")
    print("does not test what its label says, and nothing raises.")


if __name__ == "__main__":
    main()
