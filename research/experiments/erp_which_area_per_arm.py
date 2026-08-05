"""Which AREA does each ERP arm actually probe, per calibration frame?

#108 / #23. The afferent-energy A/B returned p600_auc EXACTLY 0.000 on four
independent seeds with ZERO variance. That is not a weak effect with the wrong
sign -- zero variance across seeds is the signature of a CONSTANT STRUCTURAL
difference, i.e. the metric is ordering the AREAS, not the CONDITIONS.

Reading the dispatch says why, and it is not a regression of #23::

    runner.py:108    cat, ... = parser._advance_incremental_word(word, ...)
    adapters.py:381  role_area = structural_role_area(cat, verb_seen=verb_seen)
    adapters.py:295  VERB -> VP ;  NOUN/PRON -> ROLE_PATIENT (if verb_seen)

The probed area is dispatched on the OBSERVED word's category. But "category
violation" MEANS the observed category differs from the expected one. So for
every category-violation item the probe area differs from its grammatical
control BY CONSTRUCTION, in every frame set, at every seed -- no frame design
and no energy definition can remove it.

That also explains why #23 is marked completed and is not lying.
`AREA_MATCHED_CALIBRATION_FRAMES` (frames.py:64) area-matches the NOVEL-noun
arm: `cat`, `dog`, `bird` are all NOUNs in object position, so all three map to
ROLE_PATIENT. The grammatical/violation arm is untouched, because `finds`,
`runs`, `eats` are VERBs and map to VP. #23 fixed the arm it covered (#27).

THIS SCRIPT ASSERTS NOTHING AND CHANGES NOTHING. It prints the area each probe
read, grouped by condition. The prediction under the diagnosis above:

    grammatical        -> ROLE_PATIENT
    novel_noun         -> ROLE_PATIENT      (same area -- #23's contrast works)
    category_violation -> VP                (different area -- the dead arm)

If category_violation reads ROLE_PATIENT here, the diagnosis is WRONG and the
inversion is somewhere else. That is the point of running it.
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    AREA_MATCHED_CALIBRATION_FRAMES,
    DEFAULT_CALIBRATION_FRAMES,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.runner import (  # noqa: E402
    run_incremental_erp_probes,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)

FRAME_SETS = {
    "DEFAULT": DEFAULT_CALIBRATION_FRAMES,
    "AREA_MATCHED": AREA_MATCHED_CALIBRATION_FRAMES,
}


def main(depth="SENTENCES", seed=11):
    parser = get_parser_cache().fork(depth, seed=seed)

    for set_name, frames in FRAME_SETS.items():
        print(f"\n=== {set_name} (depth={depth}, seed={seed}) ===")
        by_cond = defaultdict(Counter)
        rows = []
        for label, name, words in frames:
            _result, probes = run_incremental_erp_probes(parser, list(words))
            # The CRITICAL word is the last one in every frame EXCEPT the
            # attributive-adjective item, where it is `small` and the sentence
            # continues. That frame is printed but excluded from the verdict.
            crit = probes[-1]
            attributive = "attributive" in name
            if not attributive:
                by_cond[label][crit.role_area] += 1
            for p in probes:
                rows.append((label, name, p.word, p.category, p.role_area,
                             p.p600, p.phrase_stability,
                             p is crit and not attributive))

        print(f"{'condition':19s} {'item':26s} {'word':7s} {'cat':5s} "
              f"{'AREA PROBED':14s} {'p600':>8s} {'stab':>8s}  crit")
        for label, name, word, cat, area, p600, stab, is_crit in rows:
            print(f"{label:19s} {name:26s} {word:7s} {cat:5s} "
                  f"{area:14s} {p600:8.4f} {stab:8.4f}  {'<<' if is_crit else ''}")

        print("\n  areas per condition:")
        for label, counts in by_cond.items():
            areas = ", ".join(f"{a} x{c}" for a, c in counts.items())
            print(f"    {label:19s} {areas}")

        gram = set(by_cond["grammatical"])
        viol = set(by_cond["category_violation"])
        novel = set(by_cond["novel_noun"])
        print(f"\n  grammatical vs violation : "
              f"{'MATCHED' if gram == viol else 'MISMATCHED -- ' + str(gram) + ' vs ' + str(viol)}")
        print(f"  grammatical vs novel     : "
              f"{'MATCHED' if gram == novel else 'MISMATCHED -- ' + str(gram) + ' vs ' + str(novel)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SENTENCES",
         int(sys.argv[2]) if len(sys.argv) > 2 else 11)
