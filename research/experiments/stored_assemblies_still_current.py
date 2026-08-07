"""Do the stored core assemblies still mean what they meant when stored?

WHY. `role_binding_writes_anything.py` hands `input_drive` each word's STORED
`NOUN_CORE` assembly and reads the drive it delivers. That result came out
INVERTED (bound words drive ROLE_PATIENT less than unbound ones), and the
minimal-substrate reproduction says it is not a substrate property. So either
the probe is wrong or the parser's state is.

`activate_assembly` is NOT the obvious suspect it looked like. It maps stable
neuron IDs to compact indices through `_compact_index` and RAISES on a miss --
"silently dropping the missing neurons would inject a truncated, subtly wrong
assembly" is called out in its own docstring. Nothing raised during the probe,
so no neuron was missing.

BUT MEMBERSHIP IS NOT IDENTITY, and that gap is exactly what `assembly_is_current`
exists for:

    The mapping can move out from under a snapshot: consolidation
    (`consolidation.prepare_area_for_replay`) deliberately resets
    `compact_to_neuron_id` and RE-ISSUES neuron IDs, orphaning every
    pre-consolidation snapshot for that area.

If IDs were re-issued, an old ID can still be PRESENT in the map while pointing
at a DIFFERENT neuron. `activate_assembly` then succeeds, silently, and injects
the wrong assembly. `assembly_is_current` checks presence, not identity, so it
returns True in exactly that case. The SENTENCES curriculum runs
`consolidate_role_pathways` / `consolidate_vp_pathways`
(`training/schedule.py`), so this is a live possibility rather than a
hypothetical.

TWO CHECKS, and they fail differently:

  1. ROUND TRIP. Activate the stored assembly, read the area back, and compare
     to what was stored. This tests the ID <-> compact mapping is self-
     consistent RIGHT NOW. It cannot detect re-issued IDs, because after a
     re-issue the mapping is still internally consistent -- it just means
     something else. Expect 1.000; anything less is a live index-space bug.

  2. FRESHNESS. Re-project `phon -> NOUN_CORE` for the word and compare the
     result to the stored snapshot. This is the one that catches staleness: if
     the stored assembly no longer matches what the word actually produces, the
     probe has been reading assemblies that are not the word's.

     Read under `brain.probe()` so the re-projection cannot itself teach or
     recruit -- otherwise the check would change the thing it is checking.

A caveat stated in advance: a low freshness overlap does NOT by itself mean IDs
were re-issued. Assemblies drift with training, which `train_roles` already
knows -- it replays the stabilised snapshot precisely because "re-projecting
phon -> core carries plasticity, so the core assembly drifts between the moment
a role binding is stored and the moment it is read back, and retrieval then
misses." Drift and re-issue both show up here; they are separated by check 1
plus how the numbers are distributed, not by this measurement alone.
"""
import os
import sys
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
from neural_assemblies.assembly_calculus.ops import (                  # noqa: E402
    activate_assembly, assembly_is_current, project,
)

SEEDS = [11, 12, 42]


def main():
    for seed in SEEDS:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        brain = parser.brain
        core = parser.core_lexicons.get(NOUN_CORE, {})
        patient = set(parser.role_lexicons.get(ROLE_PATIENT, {}))
        agent = set(parser.role_lexicons.get(ROLE_AGENT, {}))

        print(f"=== seed {seed} ===  {len(core)} stored NOUN_CORE assemblies")
        n_current = sum(1 for a in core.values()
                        if assembly_is_current(brain, a))
        print(f"  assembly_is_current: {n_current}/{len(core)}")

        round_trip, freshness, rows = [], [], []
        for word, asm in sorted(core.items()):
            stored = np.asarray(asm.winners, dtype=np.int64)

            # (1) round trip -- does activating it read back as itself?
            try:
                with brain.probe():
                    activate_assembly(brain, asm)
                    back = read(brain, NOUN_CORE)
                rt = similarity(stored, back)
            except Exception as exc:                       # noqa: BLE001
                print(f"  {word}: activate RAISED {type(exc).__name__}: {exc}")
                continue

            # (2) freshness -- does the word still produce this assembly?
            phon = parser.stim_map.get(word)
            if phon is None:
                fr = float("nan")
            else:
                with brain.probe():
                    project(brain, phon, NOUN_CORE, rounds=parser.rounds)
                    fresh = read(brain, NOUN_CORE)
                fr = similarity(stored, fresh)

            round_trip.append(rt)
            if fr == fr:
                freshness.append(fr)
            grp = ("patient" if word in patient
                   else "agent" if word in agent else "unbound")
            rows.append((word, grp, rt, fr))

        if round_trip:
            print(f"  ROUND TRIP  mean={np.mean(round_trip):.4f} "
                  f"min={min(round_trip):.4f}  "
                  f"(1.000 expected; below it = live index-space bug)")
        if freshness:
            print(f"  FRESHNESS   mean={np.mean(freshness):.4f} "
                  f"min={min(freshness):.4f} max={max(freshness):.4f}")
            for grp in ("patient", "agent", "unbound"):
                vals = [f for _w, g, _r, f in rows if g == grp and f == f]
                if vals:
                    print(f"     {grp:<8} n={len(vals):<3} "
                          f"mean freshness={np.mean(vals):.4f}")
        worst = sorted((f, w, g) for w, g, _r, f in rows if f == f)[:6]
        print(f"  least fresh: {[(w, g, round(f, 3)) for f, w, g in worst]}")
        print()

    print("READING IT.")
    print("  round trip < 1.000  -> the ID<->compact mapping is broken NOW, and")
    print("                         every stored-assembly probe is void.")
    print("  round trip = 1.000 but freshness LOW -> the snapshots no longer")
    print("                         match what the words produce. The probe was")
    print("                         reading assemblies that are not the word's,")
    print("                         which is enough to void the inverted result")
    print("                         without any claim about WHY they drifted.")
    print("  both high            -> the stored assemblies are fine and the")
    print("                         inversion is about the role areas, not the")
    print("                         source.")


if __name__ == "__main__":
    main()
