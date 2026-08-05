"""Does GLOBAL RNG STATE change an ERP calibration on an identical fork?

The ERP suite has order-dependent failures: 4 tests fail under `-k erp` and pass
alone, in pairs, and in a cold process (see
research/notes/p600_is_confounded_with_area_identity.md). `fork()` already
clones a pristine deepcopy (#103), so PARSER state is not the channel.

The remaining process-global that parsing touches is the RNG. If calibration
draws from the global numpy/random streams, then how much RNG earlier tests
consumed decides what a later calibration reads -- which is exactly an
order-dependent failure, and matches the recorded finding that Brain(seed=) is
not reproducible because the global stream leaks between constructions.

THE TEST. Two calibrations of IDENTICAL pristine forks, differing ONLY in how
much global RNG was consumed beforehand. Under the hypothesis they disagree;
if the channel is something else they are identical.

Prints the raw p600 vectors, because the AUC alone would hide a shift that
moves both arms the same way.
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from neural_assemblies.assembly_calculus.emergent.evaluation import (   # noqa: E402
    calibrate_erp_thresholds,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                   # noqa: E402

SEED = 11


def run(burn: int, label: str):
    """Calibrate a pristine fork after consuming `burn` global RNG draws."""
    if burn:
        np.random.random(burn)
        random.random()
    parser = get_parser_cache().fork("SENTENCES", seed=SEED)
    report = calibrate_erp_thresholds(parser)

    def vals(cond):
        return [round(float(s.p600), 4) for s in report.samples if s.label == cond]

    gram, catv = vals("grammatical"), vals("category_violation")
    auc = separation(catv, gram, "p600").auc if gram and catv else float("nan")
    print(f"  {label:22s} auc={auc:.3f}")
    print(f"  {'':22s} gram {gram}")
    print(f"  {'':22s} catv {catv}")
    return gram, catv


def main():
    print(f"seed={SEED}  ERP_EXPECTED_SLOT={os.environ.get('ERP_EXPECTED_SLOT', '<unset>')}")
    print("Identical pristine forks; ONLY prior global-RNG consumption differs.\n")
    a = run(0, "no RNG burned")
    b = run(10_000, "10k draws burned")
    c = run(0, "no RNG burned (again)")

    print()
    if a == b:
        print("  IDENTICAL across RNG states -- the global RNG is NOT the channel.")
    else:
        print("  DIFFERENT across RNG states -- global RNG leak CONFIRMED as a channel.")
    if a != c:
        print("  ALSO: two identically-prepared runs differ, so calibration is not")
        print("  even self-consistent within one process -- the cache or the parse")
        print("  is carrying state forward independently of the RNG.")


if __name__ == "__main__":
    main()
