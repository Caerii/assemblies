"""Does AFFERENT drive beat self-recurrence, now that the arms are area-matched?

WHY THE OLD REJECTION IS VOID. `afferent_energy` was rejected on measurement:
p600 AUC exactly **0.000, with ZERO variance across four seeds**. That was read
as "the candidate is worse than nothing". It was measured under the OLD
dispatch, where the violation arm probed **VP** and the grammatical arm probed
**ROLE_PATIENT** -- so the number is a fact about two different brain areas, not
about the metric. And zero variance across seeds is the signature of a CONSTANT,
which is what a probe on the wrong area returns; see the VP dead-probe finding.

WHY THE QUESTION IS NOW DIFFERENT, not just re-run. `afferent_energy` existed
because it was the only quantity DEFINED FOR BOTH ARMS: `VP -> VP` is shape
(0,0) with zero synapses, so `_self_recurrent_energy` on the violation arm
returned a constant. With `expected_slot` adopted (f79c4f5) both arms probe
ROLE_PATIENT, which HAS a self-fiber -- so the shipped metric is now defined for
both arms too, and the candidate's original justification is dissolved.

That makes this a straight comparison of two well-defined metrics rather than a
rescue, and it admits a genuinely uninteresting answer: if afferent drive does
not beat self-recurrence, the right outcome is to KEEP THE SHIPPED ONE and stop
carrying the candidate.

PORTED FROM `erp_afferent_vs_recurrent.py`, which predates the harness and:
  * mutated `os.environ` with save/restore around each arm -- global state as an
    argument-passing mechanism, which cannot nest and leaks on exception;
  * ran both arms in ONE PROCESS off `get_parser_cache().fork()`, i.e. warm, and
    recorded nothing about the substrate.

Both are what `research/harness.py` and `ErpProtocol` exist to prevent. The
harness counterbalances arm order and stamps provenance into the artefact.

RUN COLD -- `ASSEMBLIES_BACKBONE_CACHE=0` -- or the result is evidence about
cached parsers only.

Pre-registered bar. Adoption requires the candidate to BEAT the shipped metric:
  * above chance on EVERY seed (42 included -- it is the seed that inverted the
    last structural change here, to 0.444);
  * not constant across seeds (zero variance was the original defect signature);
  * `allow_decrease=False`: unlike the expected-slot study, a DROP is not a pass
    here. There is no confound left for it to remove -- both arms are
    area-matched already -- so a drop just means it is the worse metric;
  * `delta_excludes_zero=True`, because "BEAT" is a claim about a DIFFERENCE.

THE LAST CRITERION WAS MISSING ON THE FIRST RUN, and that is the mistake worth
recording. The prose above said "beat"; the encoded `Criteria` said only "do not
decrease". The harness dutifully answered the question it was asked and printed
`VERDICT: PASS` for `p600_auc_of_raw   0.7167 -> 0.7500   delta +0.0333+/-0.0627`
-- a delta whose interval spans zero. A pre-registration is only as good as the
predicate it encodes; prose in a docstring is not a bar.
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

FLAG = "ERP_AFFERENT_ENERGY"
SEEDS = [11, 12, 13, 42, 7, 19, 23, 31, 37, 101]


def _arm(enabled: bool):
    def measure(seed: int):
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
        return {"p600_auc_of_raw": q.auc_of_raw, "p600_span_of_raw": q.span_of_raw}
    return measure


def main():
    result = study(
        arms={"self_recurrent": _arm(False), "afferent": _arm(True)},
        seeds=SEEDS,
        criteria={
            "p600_auc_of_raw": Criteria(
                above=0.5, on_every_seed=True, must_vary=True,
                allow_decrease=False, delta_excludes_zero=True,
            ),
        },
    )
    print(result)
    print()
    print("EXPECTED_SLOT is ON by default since f79c4f5, so BOTH arms here probe")
    print("ROLE_PATIENT. The shipped self-recurrent metric is therefore defined")
    print("on both, and `afferent_energy` no longer has a structural argument in")
    print("its favour -- only an empirical one, which is what this measures.")
    print()
    prov = result.provenance
    if prov.cache_disk_hits > 0:
        print(f"CAVEAT: {prov.cache_disk_hits} parser(s) came from the DISK CACHE.")
        print("Re-run with ASSEMBLIES_BACKBONE_CACHE=0 before reading it as an")
        print("adoption -- see backbone-fingerprint-gap.")
    elif prov.trained_fresh > 0:
        print(f"{prov.trained_fresh} parser(s) trained fresh in-process -- this is")
        print("the substrate that decides adoption.")


if __name__ == "__main__":
    main()
