"""Is the residual order-dependence the AREA or the FRAME? (task #32)

Under ``read_only()`` the order divergence does not vanish, but it splits
perfectly along the arm:

    frame                        forward   reversed    |diff|
    gram:the dog chases cat      0.02548    0.02548   0.00000
    gram:the cat sees dog        0.01036    0.01036   0.00000
    gram:she chases the cat      0.02548    0.02548   0.00000
    cate:the dog chases finds    0.00737    0.01566   0.00829
    cate:the cat sees runs       0.02101    0.02164   0.00064
    cate:she hits the eats       0.01477    0.03251   0.01774

3 of 3 grammatical frames are bit-identical; 3 of 3 violation frames are not.
That is aligned with the ARM, not with position in the sweep, so it is not
generic accumulation. Two hypotheses were already killed: ``_category_cache``
memoisation (the cache is fully warm from training and pre-warming changes
nothing) and reset-outside-the-guard placement (moving it inside changes
nothing, to the last decimal).

The arms differ in exactly two ways, and they are confounded in the shipped
probe: the violation arm reads VP where the grammatical arm reads ROLE_PATIENT,
and it drives from VERB_CORE where the other drives from NOUN_CORE.

  AREA    VP is the unstable term. VP is a merge-built phrase area, and task #31
          recorded it as degenerate -- 94 VPs collapsed onto one assembly. An
          area whose contents depend on merge history would read differently
          depending on what ran before, while ROLE_PATIENT stays put.

  FRAME   the violation frames are the unstable term -- a verb in object
          position drives the parse somewhere unusual, and the instability
          follows the sentence rather than the area being read.

Crossing them separates the two: read BOTH arms into BOTH areas.

  AREA  predicts divergence tracks the column -- VP diverges for grammatical
        frames too, ROLE_PATIENT is stable for violation frames.
  FRAME predicts divergence tracks the row -- violation frames diverge into
        ROLE_PATIENT as well, grammatical frames stay stable into VP.

Pre-registered. This decides whether #32 is an ERP bug or a VP bug.
"""

from __future__ import annotations

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.core.areas import (
    CATEGORY_TO_CORE,
    ROLE_PATIENT,
    VP,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    anchored_p600_live,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (
    DEFAULT_CALIBRATION_FRAMES,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    get_parser_cache,
)

ARMS = [
    (label, words)
    for label, _d, words in DEFAULT_CALIBRATION_FRAMES
    if label in ("grammatical", "category_violation")
]


def probe(parser, words, area):
    known = [w for w in words if w in parser.stim_map]
    parser._reset_context_state()
    circuit = parser._get_incremental_circuit(reset=True)
    verb_seen, noun_count, subject_core, cat = False, 0, None, ""
    with parser.brain.read_only():
        for i, word in enumerate(known):
            cat, verb_seen, noun_count = parser._advance_incremental_word(
                word, circuit, verb_seen, noun_count,
            )
            if cat in ("NOUN", "PRON") and i <= 1 and not verb_seen:
                subject_core = CATEGORY_TO_CORE.get(cat)
            elif verb_seen and cat in ("NOUN", "PRON") and subject_core is None:
                subject_core = CATEGORY_TO_CORE.get(cat)
        core = CATEGORY_TO_CORE.get(cat)
        if core is None:
            return None
        return 1.0 - anchored_p600_live(
            parser, core, area, subject_core=subject_core,
        )


def sweep(cache, area, order):
    p = cache.fork("SENTENCES", seed=42)
    out = {}
    for label, words in order:
        e = probe(p, words, area)
        if e is not None:
            out[(label, tuple(words))] = e
    return out


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cache = get_parser_cache()
    rev = list(reversed(ARMS))

    print(f"\n  {'frame':<30} {'area':<14} {'forward':>10} {'reversed':>10} "
          f"{'|diff|':>10}")
    tally = {}
    for area in (ROLE_PATIENT, VP):
        f, r = sweep(cache, area, ARMS), sweep(cache, area, rev)
        for k in sorted(set(f) & set(r)):
            d = abs(f[k] - r[k])
            tally.setdefault((k[0], area), []).append(d)
            tag = f"{k[0][:4]}:{' '.join(k[1])}"
            print(f"  {tag:<30} {area:<14} {f[k]:>10.5f} {r[k]:>10.5f} "
                  f"{d:>10.5f}")

    print(f"\n  {'arm':<20} {'area':<14} {'n diverging':>12} {'max |diff|':>12}")
    for (label, area), ds in sorted(tally.items()):
        print(f"  {label:<20} {area:<14} "
              f"{sum(1 for d in ds if d > 1e-9):>7}/{len(ds):<4} {max(ds):>12.5f}")

    def unstable(label, area):
        return any(d > 1e-9 for d in tally.get((label, area), []))

    g_rp = unstable("grammatical", ROLE_PATIENT)
    g_vp = unstable("grammatical", VP)
    c_rp = unstable("category_violation", ROLE_PATIENT)
    c_vp = unstable("category_violation", VP)

    print("\n  verdict")
    if g_vp and c_vp and not g_rp and not c_rp:
        print(f"    AREA. {VP} is order-dependent for BOTH arms and "
              f"{ROLE_PATIENT} for neither.")
        print("    #32's residual is a VP defect (cf. task #31), not an ERP one.")
    elif c_rp and c_vp and not g_rp and not g_vp:
        print("    FRAME. The violation frames are unstable in EVERY area --")
        print("    the parse state they leave is what varies, not the readout.")
    else:
        print("    Neither clean pattern; both terms contribute. Report as-is.")
        print(f"    gram/{ROLE_PATIENT}={g_rp}  gram/{VP}={g_vp}  "
              f"catv/{ROLE_PATIENT}={c_rp}  catv/{VP}={c_vp}")


if __name__ == "__main__":
    main()
