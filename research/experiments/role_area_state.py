"""What did 36 bindings do to ROLE_PATIENT itself?

THE LAST UNTOUCHED SURFACE. Everything measured so far varied the SOURCE:

    index space          round trip 1.0000            excluded
    norm_init            1.23-1.35x, both directions  excluded
    target capacity      flat to M=32, readback 1.00  excluded
    consolidation        does not run at SENTENCES    excluded
    source crowding      NOUN_CORE overlap 0.126      excluded

None of them looked at what the role area became. The parser binds 36 words into
ROLE_PATIENT and 46 into ROLE_AGENT, and this repo has seen that shape collapse
before -- #31, "94 VPs, one assembly, unretrievable", and the reinforcement
tradeoff where deep replay yields a strong single attractor while MERGING a
multi-assembly area.

IF THE ROLE ASSEMBLIES HAVE MERGED, everything observed follows at once:

  * every word's binding points at the SAME target neurons, so the synapses a
    given word strengthened are the synapses every word strengthened;
  * drive into ROLE_PATIENT is then near-identical for bound and unbound words,
    which is the null `role_binding_writes_anything.py` found;
  * and the only thing left that can separate conditions is which AREA the drive
    is read from -- which is exactly what the P600 turned out to measure.

That would make the ERP result, the pathway null, and #31 one finding rather
than three.

TWO MEASUREMENTS.

1. SPREAD -- mean pairwise overlap among the stored assemblies of an area,
   against the chance floor k/n. NOUN_CORE is the built-in control: it reads
   0.126-0.132 on this same parser, so the protocol can distinguish "distinct"
   from "merged" and a high role-area number is not an artifact of how overlap
   is computed here.

2. RETRIEVAL -- activate each word's stored core assembly, project core -> role,
   and ask which stored role assembly the result best matches. Rank-1 accuracy
   against a 1/M chance baseline. Spread says whether the assemblies ARE
   distinct; retrieval says whether the pathway can FIND the right one, and a
   merged area fails both while a merely-crowded one can fail only the second.

Both read through `_substrate.read`/`similarity`, the sanctioned door between
neuron IDs and compact indices, and the projection runs under `brain.probe()` so
measuring cannot teach.
"""
import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from _substrate import read, similarity                                # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (  # noqa: E402
    NOUN_CORE, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.assembly_calculus.ops import activate_assembly  # noqa: E402

SEEDS = [11, 12, 42]


def _spread(assemblies, cap=30):
    """Mean pairwise overlap over a capped sample -- O(cap^2) not O(M^2)."""
    items = [np.asarray(a.winners, dtype=np.int64) for a in assemblies][:cap]
    pairs = list(combinations(items, 2))
    if not pairs:
        return float("nan"), 0
    return float(np.mean([similarity(x, y) for x, y in pairs])), len(items)


def main():
    for seed in SEEDS:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        brain = parser.brain
        core = parser.core_lexicons.get(NOUN_CORE, {})

        print(f"=== seed {seed} ===")
        print(f"{'area':<14} {'stored':>7} {'k':>5} {'n':>7} "
              f"{'chance k/n':>11} {'spread':>9} {'vs chance':>10}")
        print("-" * 68)
        for area_name, lex in (
            (NOUN_CORE, core),
            (ROLE_PATIENT, parser.role_lexicons.get(ROLE_PATIENT, {})),
            (ROLE_AGENT, parser.role_lexicons.get(ROLE_AGENT, {})),
        ):
            if not lex:
                print(f"{area_name:<14} {0:>7}   (empty)")
                continue
            sp, used = _spread(list(lex.values()))
            area = brain.areas.get(area_name)
            k = getattr(area, "k", 0) or 0
            n = getattr(area, "n", 0) or 0
            floor = (k / n) if n else float("nan")
            print(f"{area_name:<14} {len(lex):>7} {k:>5} {n:>7} "
                  f"{floor:>11.4f} {sp:>9.4f} "
                  f"{(sp / floor if floor else float('nan')):>9.1f}x")

        # RETRIEVAL: does core -> role find the right stored role assembly?
        for role_area in (ROLE_PATIENT, ROLE_AGENT):
            role_lex = parser.role_lexicons.get(role_area, {})
            shared = [w for w in role_lex if w in core]
            if len(shared) < 3:
                print(f"  {role_area}: only {len(shared)} words have both a core "
                      f"and a role assembly -- retrieval not measurable")
                continue
            stored = {w: np.asarray(role_lex[w].winners, dtype=np.int64)
                      for w in shared}
            hits, margins = 0, []
            for w in shared:
                with brain.probe():
                    activate_assembly(brain, core[w])
                    brain.project({}, {NOUN_CORE: [role_area]})
                    live = read(brain, role_area)
                scores = sorted(
                    ((similarity(live, a), other) for other, a in stored.items()),
                    reverse=True,
                )
                hits += int(scores[0][1] == w)
                if len(scores) > 1 and scores[1][0] > 0:
                    margins.append(scores[0][0] / scores[1][0])
            acc = hits / len(shared)
            mm = float(np.mean(margins)) if margins else float("nan")
            print(f"  {role_area}: retrieval {hits}/{len(shared)} = {acc:.3f} "
                  f"(chance {1/len(shared):.3f})  mean rank1/rank2 margin={mm:.4f}")
        print()

    print("READING IT. NOUN_CORE at ~0.13 is the built-in control: the protocol")
    print("CAN see distinct assemblies. A role-area spread near 1.0 means the")
    print("bindings merged, and then the pathway null, the P600's area-identity")
    print("result and #31 are one finding. Spread low but retrieval at chance")
    print("means distinct-but-unfindable, which is a different problem.")


if __name__ == "__main__":
    main()
