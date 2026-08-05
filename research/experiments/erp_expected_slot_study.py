"""The expected-slot A/B, re-run through the canonical harness.

PORT OF `erp_expected_slot_ab.py`, which produced a result that did not survive.
Same question, same seeds, same metric -- the difference is protocol:

  * the old script ran BOTH ARMS IN ONE PROCESS, control first, so the candidate
    was only ever measured WARM while the suite measured it COLD;
  * it never recorded that every parser came from `get_parser_cache().fork()`,
    i.e. from a DISK-CACHED backbone. Measured afterwards: the dispatch passes
    on cached parsers and FAILS on freshly-trained ones. An A/B built on cached
    parsers is evidence about cached parsers only.

The harness counterbalances arm order and stamps substrate provenance into the
result, so both of those are visible in the artefact rather than reconstructed
weeks later from runtimes.

THIS IS EXPECTED TO REPORT disk_hits > 0 AND THEREFORE TO BE INSUFFICIENT ON ITS
OWN. That is the point of running it: the harness should make the limitation
legible instead of letting a clean-looking table imply more than it earned. Run
with `ASSEMBLIES_BACKBONE_CACHE=0` for the arm that actually decides adoption.

Pre-registered bar, unchanged from the original run:
  * above chance ON EVERY SEED (42 included -- it is the only seed that has ever
    caught a structural change here)
  * not constant across seeds
  * a DROP still passes: removing a confound should shrink an inflated effect.
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

FLAG = "ERP_EXPECTED_SLOT"
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
        # The SANCTIONED reader: all three "p600" quantities together, so
        # `auc_of_raw` cannot be mistaken for an AUC on the excess.
        q = report.p600_quantities()
        return {"p600_auc_of_raw": q.auc_of_raw, "p600_span_of_raw": q.span_of_raw}
    return measure


def main():
    result = study(
        arms={"observed_category": _arm(False), "expected_slot": _arm(True)},
        seeds=SEEDS,
        criteria={
            "p600_auc_of_raw": Criteria(
                above=0.5, on_every_seed=True, must_vary=True,
                allow_decrease=True,
            ),
        },
    )
    print(result)
    print()
    prov = result.provenance
    if prov.cache_disk_hits > 0:
        print(f"CAVEAT: {prov.cache_disk_hits} parser(s) came from the DISK CACHE.")
        print("This study is evidence about CACHED parsers. The same contrast")
        print("inverts on freshly-trained ones -- re-run with")
        print("ASSEMBLIES_BACKBONE_CACHE=0 before reading it as an adoption.")
    elif prov.trained_fresh > 0:
        print(f"{prov.trained_fresh} parser(s) trained fresh in-process -- this is")
        print("the substrate that decides adoption.")


if __name__ == "__main__":
    main()
