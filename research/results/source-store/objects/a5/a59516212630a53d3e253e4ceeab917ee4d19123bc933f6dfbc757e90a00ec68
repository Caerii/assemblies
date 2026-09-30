"""Completing the #108 fix: does taking the SOURCE core from the expected slot
remove the rest of the area-identity confound?

WHAT IS BEING FIXED. `expected_slot` reads "the category" three times --
`role_area` (probe TARGET), `phrase_category` (which phrase areas), and `core`
(probe SOURCE). It corrected the first two. `core` stayed on the OBSERVED
category, so the violation arm reads VERB_CORE -> ROLE_PATIENT against a control
reading NOUN_CORE -> ROLE_PATIENT. Area identity again, one level over, inside
the fix written to remove area identity -- rule 1 of [[one-canonical-way]],
committed by the fix itself.

WHY IT IS WORTH MEASURING RATHER THAN JUST SHIPPING. The degenerate control
(`erp_source_core_identity_control.py`) already shows the confound is large:
holding the CONDITION constant -- same grammatical sentence, same trained noun
in object position, only the category LABEL handed to the metric changed --
separates at AUC 0.78 / 0.89 / 0.89 with a span matching the real contrast's.
That says the confound EXISTS and is big. It does not say what is left after
removing it, and "what is left" is the actual scientific claim.

PRE-REGISTERED BAR, encoded in `Criteria` and not only here:
  * `above=0.5`, `on_every_seed=True` -- after the fix the P600 must still
    separate the conditions on every seed, 42 included.
  * `must_vary=True` -- zero variance is this codebase's signature for a
    constant, and it is what disqualified both `afferent_energy` (0.000) and
    the trained frames (1.000).
  * `allow_decrease=True`, and A LARGE DROP IS THE EXPECTED OUTCOME. Removing a
    confound shrinks an effect; that is what happened when the TARGET area was
    matched (0.9056 -> 0.7167). Requiring an increase would pre-register the
    wrong direction, and requiring "no decrease" would pre-register the
    confound.
  * `delta_excludes_zero` deliberately NOT set. The claim is not "this is
    bigger"; it is "this is what the number is once the arms are matched".

WHAT WOULD BE A FAILURE, and it is a real possibility worth naming in advance:
p600_auc at or below CHANCE after the fix. That would say the reported P600 was
area identity essentially all the way down, and the honest response is to
publish that rather than to keep the confounded number.

RUN COLD -- `ASSEMBLIES_BACKBONE_CACHE=0`.
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
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from research.harness import Criteria, study                            # noqa: E402

FLAG = "ERP_EXPECTED_SLOT_SOURCE_CORE"
SEEDS = [11, 12, 13, 42, 7, 19, 23, 31, 37, 101]


def _arm(enabled: bool):
    def measure(seed: int):
        # `calibrate_erp_thresholds` does not yet take an ErpProtocol -- the
        # #115 thread reaches `collect_frame_samples`, one level below it -- so
        # the environment is still how this arm is selected. Stated rather than
        # hidden: it is the remaining unthreaded link, not an oversight.
        prev = os.environ.get(FLAG)
        os.environ[FLAG] = "1" if enabled else "0"
        try:
            parser = get_parser_cache().fork("SENTENCES", seed=seed)
            report = calibrate_erp_thresholds(parser)
        finally:
            if prev is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = prev
        q = report.p600_quantities()
        return {"p600_auc_of_raw": q.auc_of_raw,
                "p600_span_of_raw": q.span_of_raw}
    return measure


def main():
    result = study(
        arms={"observed_source_core": _arm(False),
              "expected_source_core": _arm(True)},
        seeds=SEEDS,
        control="observed_source_core",
        criteria={
            "p600_auc_of_raw": Criteria(
                above=0.5, on_every_seed=True, must_vary=True,
                allow_decrease=True,
            ),
        },
    )
    print(result)
    print()
    print("The DEFAULT frames are used here on purpose, even though they are")
    print("known to contain untrained words: this study is about the METRIC,")
    print("and changing the items at the same time would confound the two")
    print("levers exactly as erp_trained_frames_study had to separate them.")
    print()
    prov = result.provenance
    if prov.cache_disk_hits > 0:
        print(f"CAVEAT: {prov.cache_disk_hits} parser(s) from the DISK CACHE.")
    elif prov.trained_fresh > 0:
        print(f"{prov.trained_fresh} parser(s) trained fresh in-process.")


if __name__ == "__main__":
    main()
