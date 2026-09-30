"""Do frames made of TRAINED words beat the shipped ones?

THE DEFECT BEING TESTED. On the parser every ERP number here is measured on,
9 of 11 words in `DEFAULT_CALIBRATION_FRAMES` have no core lexicon entry.
`chases` -- the main verb of five of nine items -- occurs ZERO times in that
corpus and classifies NOUN, so "the dog chases cat" parses DET NOUN NOUN NOUN:
no verb, and a "category violation" that violates nothing that was parsed.
See research/notes/language/the_calibration_frames_are_untrained.md.

`TRAINED_AREA_MATCHED_CALIBRATION_FRAMES` fixes TWO things at once, and that is
a weakness of this design that has to be stated rather than hidden:

  1. every word is trained and stably classified (the defect above);
  2. every critical word is in object position, so all three arms expect the
     same slot (the area-matching lever from #108's family, which
     `AREA_MATCHED_CALIBRATION_FRAMES` also pulls).

So a difference here does NOT isolate either cause. `AREA_MATCHED` is included
as a third arm precisely to separate them: it shares lever 2 and NOT lever 1,
so AREA_MATCHED vs DEFAULT prices the area-matching alone, and
TRAINED_AREA_MATCHED vs AREA_MATCHED prices the vocabulary alone.

WHY THIS IS NOT A FREE CHANGE. `calibrate_erp_thresholds` consumes the frames
to TUNE thresholds, so the frame set moves calibrated thresholds and every
golden downstream of them. That is why it is measured rather than swapped.

Pre-registered bar, encoded in `Criteria` and not only in this docstring --
the mistake `erp_afferent_energy_study.py` made and recorded:
  * `above=0.5` and `on_every_seed=True`: the p600 must still separate the
    conditions, on every seed including 42;
  * `must_vary=True`: zero variance across seeds is this codebase's signature
    for a constant, i.e. a dead probe;
  * `allow_decrease=True`, and this one is deliberate. A DROP IS AN ACCEPTABLE
    OUTCOME. The shipped items are partly degenerate -- an item with no main
    verb is an easy discrimination for the wrong reason -- so removing that is
    expected to SHRINK the effect, exactly as area-matching did (0.9056 ->
    0.7167). Requiring an increase here would pre-register the wrong direction.
  * `delta_excludes_zero` is therefore NOT set on the AUC: this study is not
    claiming the new frames discriminate better. What it claims is that they
    measure the contrast the labels name, which is a fact about the ITEMS and
    is asserted by `test_shipped_frames_are_trained_on_the_parser`, not by an
    AUC.

WHAT WOULD MAKE THIS A FAILURE rather than a shrink: p600_auc at or below
chance, or constant across seeds. Both are encoded above.

RUN COLD -- `ASSEMBLIES_BACKBONE_CACHE=0` -- or it is evidence about cached
parsers only.

RE-RUN 2026-08-06 ON A DIFFERENT METRIC, and the reason is worth stating because
it is a trap this file walked into once.

The first run of this study returned p600 AUC exactly 1.0000 with ZERO variance
for `trained_area_matched`, which failed the `must_vary` bar. Asking WHY a score
was perfect found the cause: `expected_slot` corrected the probe's TARGET area
but not its SOURCE core, so the violation arm read VERB_CORE -> ROLE_PATIENT
against a control reading NOUN_CORE -> ROLE_PATIENT. That was measured, adopted
as `expected_slot_source_core` (0d70cc8), and moved the headline 0.7167 ->
0.6056.

**Which means the 1.0000 was measured on the metric that had the confound in
it -- the confound this study's own result was used to diagnose.** The adoption
invalidated the study that motivated it. So every number below the first run is
superseded, and the re-run is not a repetition: it is the first time clean ITEMS
and a clean METRIC are measured together.

It is also self-checking. A collapse toward ~0.6 confirms the source-core
diagnosis end to end. A 1.0000 that SURVIVES means there is a third confound and
the explanation was incomplete -- which is the more informative outcome and the
one to hope for last.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation import (   # noqa: E402
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    AREA_MATCHED_CALIBRATION_FRAMES,
    DEFAULT_CALIBRATION_FRAMES,
    TRAINED_AREA_MATCHED_CALIBRATION_FRAMES,
    unusable_frame_words,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    default_holdout_set,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from research.harness import Criteria, study                            # noqa: E402

SEEDS = [11, 12, 13, 42, 7, 19, 23, 31, 37, 101]

#: TWO studies, not one three-arm study, because `study()` compares exactly two
#: arms -- and that constraint is right here. Each pair isolates ONE lever, and
#: `area_matched` appears in both, which is what makes the two deltas
#: composable rather than merely adjacent.
CRITERIA = {
    "p600_auc_of_raw": Criteria(
        above=0.5, on_every_seed=True, must_vary=True, allow_decrease=True,
    ),
    "untrained_items": Criteria(allow_decrease=True),
}


def _arm(frames):
    def measure(seed: int):
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        report = calibrate_erp_thresholds(parser, frames=list(frames))
        q = report.p600_quantities()
        # Carried per-seed so the artefact records WHICH items were degenerate,
        # not merely that the arms differed. An arm whose untrained count moves
        # between seeds is not one arm.
        bad = unusable_frame_words(
            parser, list(frames), holdout_words=default_holdout_set(),
        )
        return {
            "p600_auc_of_raw": q.auc_of_raw,
            "p600_span_of_raw": q.span_of_raw,
            "untrained_items": float(len(bad)),
        }
    return measure


def main():
    print("STUDY A -- area-matching alone (both arms share the same untrained")
    print("vocabulary, so any difference is the item POSITIONS)")
    a = study(
        arms={
            "default": _arm(DEFAULT_CALIBRATION_FRAMES),
            "area_matched": _arm(AREA_MATCHED_CALIBRATION_FRAMES),
        },
        seeds=SEEDS, control="default", criteria=CRITERIA,
    )
    print(a)
    print()

    print("STUDY B -- vocabulary alone (both arms are area-matched already, so")
    print("any difference is that the words were actually TRAINED)")
    b = study(
        arms={
            "area_matched": _arm(AREA_MATCHED_CALIBRATION_FRAMES),
            "trained_area_matched": _arm(
                TRAINED_AREA_MATCHED_CALIBRATION_FRAMES),
        },
        seeds=SEEDS, control="area_matched", criteria=CRITERIA,
    )
    print(b)
    print()
    print("A + B compose: `area_matched` is the shared arm, so study A prices")
    print("the positions and study B prices the vocabulary, on one substrate.")
    print()
    print("`untrained_items` counts frames containing a word that is neither")
    print("trained nor a declared holdout. It should read 9 for both older")
    print("arms and 0 for the trained arm -- if it does not, the arms are not")
    print("what this study says they are.")
    print()
    prov = b.provenance
    if prov.cache_disk_hits > 0:
        print(f"CAVEAT: {prov.cache_disk_hits} parser(s) came from the DISK CACHE.")
        print("Re-run with ASSEMBLIES_BACKBONE_CACHE=0 -- backbone-fingerprint-gap.")
    elif prov.trained_fresh > 0:
        print(f"{prov.trained_fresh} parser(s) trained fresh in-process.")


if __name__ == "__main__":
    main()
