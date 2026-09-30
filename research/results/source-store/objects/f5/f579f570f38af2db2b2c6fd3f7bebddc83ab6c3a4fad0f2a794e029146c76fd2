"""RETRACTED CONTROL -- it was not condition-constant, and the retraction is
the finding.

WHAT THIS SCRIPT CLAIMED. That holding the CONDITION constant -- same
grammatical sentence, same trained noun in object position, "only the category
LABEL handed to the metric changed" -- still separates at AUC 0.78 / 0.89 /
0.89, and that this proved the P600 was reading SOURCE CORE IDENTITY rather
than integration difficulty. `expected_slot_source_core` was adopted on that
basis and the headline moved 0.7167 -> 0.6056.

WHY IT IS WRONG. `forced_category` is not a label. `_advance_incremental_word`
does:

    core_area = CATEGORY_TO_CORE.get(cat, self._word_core_area(word))
    project(self.brain, phon, core_area, rounds=rounds)

so forcing VERB PROJECTS THE NOUN INTO VERB_CORE, and the probe then reads
VERB_CORE -> ROLE_PATIENT: an untrained pathway. That is precisely what
`anchored_p600_live` documents a category violation to BE -- "a wrongly-typed
core routed through an untrained pathway". **The control manufactured a real
category violation and then reported the metric detecting it as proof of a
confound.**

So the 0.78-0.89 is the metric WORKING. The adoption was reverted.

WHAT THE NUMBERS STILL MEAN. Read the other way round, they are a decent
demonstration that the pathway mechanism is load-bearing: type a trained noun as
a verb and the P600 rises, on the same sentence, at the same position. Together
with the lesion measurement (matching the source costs 0.7167 -> 0.6056, because
the violation arm then reads a STALE SUBJECT assembly in NOUN_CORE instead of
the critical word), that is a lower bound on how much of the effect flows
through the designed route.

THE LESSON, which is the reason this file is kept rather than deleted: a control
is only condition-constant if you have checked what its knob DOES, not what its
parameter is called. `forced_category` reads like an annotation and is a
projection target. Same shape as [[same-name-two-meanings]], and it cost an
adoption.

THE CONTROL THIS STILL NEEDS: run the SAME frames on an untrained or shallow
parser, which changes no typing at all. If the separation survives there, it is
structural; if it collapses, it is learned.

--- original docstring below, retained so the retraction can be checked ---

Does the P600 read INTEGRATION, or just which core area the word came from?

THE 1.0000 THAT PROMPTED THIS. `TRAINED_AREA_MATCHED_CALIBRATION_FRAMES` --
items whose words are all trained and stably classified -- scores p600 AUC
exactly 1.0000 with ZERO variance across 10 cold seeds. Zero variance is this
codebase's signature for a CONSTANT (it is why `afferent_energy` was rejected),
so a perfect score is a reason to doubt, not to adopt.

THE HYPOTHESIS. `f79c4f5` area-matched the probe's TARGET area: with
`expected_slot` on, both arms probe ROLE_PATIENT. But
`measure_live_integration` still takes the SOURCE core from the OBSERVED
category:

    core = CATEGORY_TO_CORE.get(category)          # observed, not expected
    ... anchored_p600_live(parser, core, role_area, ...)

so the violation arm reads VERB_CORE -> ROLE_PATIENT and its grammatical control
reads NOUN_CORE -> ROLE_PATIENT. Different SOURCE areas, by construction, in
every frame set -- exactly what `structural_role_area` was for the target area,
one level over.

THE CONTROL, which is the same one that settled #108: hold the CONDITION
constant and vary ONLY the category, then see whether the score survives.

Here that is exact rather than approximate, because the forced-category probe
already exists in production: `trial_category_in_sentence(parser, words,
position, forced_category)` re-parses the SAME sentence at the SAME position and
forces the critical word's category. Same word, same context, same target area
-- the only thing that moves is the source core.

  * If AUC(VERB-forced vs NOUN-forced) is ~1.000 on GRAMMATICAL items, then
    source-core identity alone reproduces the headline, the sentences are
    irrelevant, and the trained frames measure part of speech.
  * If it is near chance, the 1.0000 is about the items after all and the
    frames deserve a second look.

NOTE THE ARM CONSTRUCTION. Every item here is GRAMMATICAL and every critical
word is a trained noun in object position. Neither forced reading is a category
violation of anything; `forced_category` is a label handed to the metric, not a
change to the sentence. That is what makes this a degenerate control rather than
a second version of the study it is checking.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.acquisition.wobbly.hypotheses import (  # noqa: E402
    trial_category_in_sentence,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    TRAINED_AREA_MATCHED_CALIBRATION_FRAMES,
    critical_position_for_frame,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    default_holdout_set,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                    # noqa: E402

SEEDS = [11, 12, 42]


def main():
    holdouts = default_holdout_set()
    grammatical = [
        (desc, words)
        for label, desc, words in TRAINED_AREA_MATCHED_CALIBRATION_FRAMES
        if label == "grammatical"
    ]

    print("CONDITION HELD CONSTANT: every item below is GRAMMATICAL and its")
    print("critical word is a trained noun in object position. Only the")
    print("category label handed to the metric changes.")
    print()

    for seed in SEEDS:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        as_noun, as_verb = [], []
        rows = []
        for desc, words in grammatical:
            known = [w for w in words if w in parser.stim_map]
            pos = critical_position_for_frame(
                "grammatical", known, holdout_words=holdouts,
            )
            pos = min(pos, len(known) - 1)
            got = {}
            for forced in ("NOUN", "VERB"):
                out = trial_category_in_sentence(parser, known, pos, forced)
                if out is None:
                    got[forced] = None
                    continue
                _combined, p600, _n400, _stab = out
                got[forced] = p600
            if got["NOUN"] is None or got["VERB"] is None:
                print(f"  seed {seed} {desc}: probe returned None, SKIPPED "
                      f"(this is a hole in the control, not a result)")
                continue
            as_noun.append(got["NOUN"])
            as_verb.append(got["VERB"])
            rows.append((desc, known[pos], got["NOUN"], got["VERB"]))

        print(f"=== seed {seed} ===")
        for desc, word, n, v in rows:
            print(f"  {desc:<24} critical={word:<6} "
                  f"as NOUN={n:.4f}  as VERB={v:.4f}  diff={v - n:+.4f}")
        if len(as_noun) >= 2:
            sep = separation(as_verb, as_noun, label="p600_forced_category")
            print(f"  AUC(VERB-forced ranked above NOUN-forced) = {sep.auc:.4f}"
                  f"   span={sep.span:.4f}   n_pairs={len(as_verb)*len(as_noun)}")
        print()

    print("READING IT. An AUC near 1.000 with the CONDITION held constant means")
    print("the metric separates on SOURCE CORE IDENTITY, and the 1.0000 in")
    print("erp_trained_frames_study is that fact wearing a sentence. Near 0.5")
    print("means the source core is not doing the work and the frames need a")
    print("different explanation.")


if __name__ == "__main__":
    main()
