"""How often does the N400 read an UNDEFINED value that looks like a finding?

#28 records the N400 as unexplainedly saturated and "bit-identical across parse
arms, i.e. not reading parse state at all". `measure_lexical_surprise` had FOUR
bare-float escapes, and one of them returned **1.0** -- MAXIMUM SURPRISE -- when
the word is absent from the prediction lexicon. That is exactly what a real
anomaly is supposed to look like, and a NOVEL or HELD-OUT word is precisely the
case that would be absent.

If that branch fires on the novel_noun arm, the "novel words are surprising"
result is partly an artefact of the word being missing from a dict. Same shape
as the VP dead-probe defect (#108), in the other half of the 2x2.

THIS SCRIPT ONLY COUNTS. It changes nothing and asserts nothing: it runs a real
calibration and reports, per condition, how many probes came back undefined and
which branch they took. `runner` now records these in `result["n400_undefined"]`.

Read the per-condition breakdown, not the total. An undefined count spread
evenly across conditions is a saturation problem; one CONCENTRATED in a single
arm is a confound, and the arm it concentrates in is the one whose result is
void.
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
    DEFAULT_CALIBRATION_FRAMES,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.runner import (  # noqa: E402
    run_incremental_erp_probes,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)


def main(seed=11):
    parser = get_parser_cache().fork("SENTENCES", seed=seed)

    per_condition = defaultdict(lambda: [0, 0])       # label -> [probes, undef]
    why_counts = Counter()
    examples = {}

    for label, name, words in DEFAULT_CALIBRATION_FRAMES:
        result, probes = run_incremental_erp_probes(parser, list(words))
        undef = result.get("n400_undefined", [])
        per_condition[label][0] += len(probes)
        per_condition[label][1] += len(undef)
        for rec in undef:
            key = rec["why"].split(",")[0][:60]
            why_counts[key] += 1
            examples.setdefault(key, (label, name, rec["word"], rec["legacy"]))

    print(f"seed={seed}   N400 undefined-branch census\n")
    print(f"{'condition':20s} {'probes':>7s} {'undefined':>10s} {'share':>7s}")
    for label in sorted(per_condition):
        n, u = per_condition[label]
        print(f"{label:20s} {n:7d} {u:10d} {u / n if n else 0:7.1%}")

    print("\nbranches taken:")
    for why, count in why_counts.most_common():
        label, name, word, legacy = examples[why]
        print(f"  {count:3d}x  legacy={legacy:.1f}  {why}")
        print(f"        e.g. {label}/{name}: {word!r}")

    total_probes = sum(n for n, _ in per_condition.values())
    total_undef = sum(u for _, u in per_condition.values())
    print(f"\ntotal {total_undef}/{total_probes} probes undefined "
          f"({total_undef / total_probes if total_probes else 0:.1%})")
    if total_undef == 0:
        print("CLEAN: every N400 in this frame set is a real measurement.")
    else:
        shares = {l: (u / n if n else 0) for l, (n, u) in per_condition.items()}
        worst = max(shares, key=lambda k: shares[k])
        if shares[worst] > 2 * (sum(shares.values()) / len(shares)):
            print(f"CONCENTRATED in {worst!r} -- that arm's N400 is confounded "
                  f"with dictionary membership, not surprise.")
        else:
            print("SPREAD across conditions -- a saturation problem, not a "
                  "per-arm confound.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
