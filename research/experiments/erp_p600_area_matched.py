"""Does area-matching flip the P600 direction? (task #32, the decisive arm)

``erp_p600_zero_trace.py`` established WHAT the live pipeline measures. Three
things came out of it, and this file exists to test the one that is causal
rather than cosmetic.

Measured there, on the calibration frames:

    arm                  word    cat    stab    energy
    grammatical          cat     NOUN   0.0072  0.0254   -> ROLE_PATIENT
    category_violation   finds   VERB   0.0000  0.0291   -> VP

So the parser classifies the violation word as a VERB, and
``structural_role_area`` sends VERBs to VP and object-position NOUNs to
ROLE_PATIENT. The two arms read DIFFERENT AREAS. This resolves the caveat left
open in task #32: the earlier hand measurement passed the TRUE category, so it
was open whether the live parser might call "finds" a noun and land both arms in
the same area. It does not -- ``cat=VERB`` above is the parser's own output.

The violation delivers MORE energy than the grammatical control (0.0291 vs
0.0254), which is backwards from the adapter's stated premise ("a category
violation routes a wrongly-typed core through an untrained pathway and delivers
LESS"). Two explanations survive that observation:

  AREA      VP is simply a different area from ROLE_PATIENT -- different size,
            different in-degree, so different per-candidate normalization. The
            energy gap measures the areas, not the grammaticality.

  PATHWAY   the effect is real and the premise is wrong: VERB_CORE->VP is a
            heavily trained pathway (every verb in the curriculum uses it), so
            routing a verb into object position traverses a STRONGER pathway
            than a noun into ROLE_PATIENT.

These make opposite predictions once the area is held fixed. Probe BOTH arms
into ROLE_PATIENT -- the slot the object position expects, whatever the critical
word's own category:

  AREA   predicts the direction flips: gram energy > catv energy, because now
         the only difference is whether the core->ROLE_PATIENT pathway was
         trained for that core type.

  PATHWAY predicts the direction does NOT flip: the violation keeps its
         advantage even into the same area.

Pre-registered before running. Whichever way it lands, it decides whether
area-matching is a fix or a distraction.
"""

from __future__ import annotations

import os
import statistics

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.core.areas import (
    CATEGORY_TO_CORE,
    ROLE_PATIENT,
    VP,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    anchored_p600_live,
    structural_role_area,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (
    DEFAULT_CALIBRATION_FRAMES,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    get_parser_cache,
)

#: Only the two arms that form the grammaticality contrast. novel_noun is a
#: different question (lexical, not structural) and is measured elsewhere.
ARMS = [
    (label, words)
    for label, _desc, words in DEFAULT_CALIBRATION_FRAMES
    if label in ("grammatical", "category_violation")
]


def probe_frame(parser, words, area):
    """Energy into *area* at the final content word of *words*.

    Replays the incremental parse word by word exactly as ``run_incremental_erp_probes``
    does -- same ``_advance_incremental_word``, same subject_core bookkeeping --
    so the brain state at the critical word is the state the live pipeline sees.
    The ONLY departure is the target area, which is the manipulated variable.
    """
    known = [w for w in words if w in parser.stim_map]
    parser._reset_context_state()
    circuit = parser._get_incremental_circuit(reset=True)
    verb_seen = False
    noun_count = 0
    subject_core = None
    cat = ""

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
        native = structural_role_area(cat, verb_seen=verb_seen)
        target = native if area == "native" else area
        deficit = anchored_p600_live(
            parser, core, target, subject_core=subject_core,
        )
    return {
        "word": known[-1],
        "cat": cat,
        "core": core,
        "area": target,
        "energy": 1.0 - deficit,
    }


def run(parser, area, title):
    print(f"\n  {title}")
    print(f"    {'label':<20} {'word':<8} {'core':<12} {'area':<14} {'energy':>9}")
    by_label = {}
    for label, words in ARMS:
        r = probe_frame(parser, words, area)
        if r is None:
            continue
        by_label.setdefault(label, []).append(r["energy"])
        print(f"    {label:<20} {r['word']:<8} {r['core']:<12} "
              f"{r['area']:<14} {r['energy']:>9.5f}")

    g = by_label.get("grammatical", [])
    c = by_label.get("category_violation", [])
    if not (g and c):
        return None
    gm, cm = statistics.median(g), statistics.median(c)
    direction = "gram > catv (EXPECTED)" if gm > cm else "catv > gram (INVERTED)"
    print(f"    -> gram={gm:.5f}  catv={cm:.5f}  "
          f"drop={(gm - cm) / gm * 100:+.1f}%   {direction}")
    return gm, cm


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = get_parser_cache().fork("SENTENCES", seed=42)

    native = run(parser, "native", "AS SHIPPED -- each arm reads its own category's area")
    matched = run(parser, ROLE_PATIENT, f"AREA-MATCHED -- both arms read {ROLE_PATIENT}")
    vp = run(parser, VP, f"CONTROL -- both arms read {VP}")

    print("\n  verdict")
    if native and matched:
        n_flip = native[0] > native[1]
        m_flip = matched[0] > matched[1]
        if not n_flip and m_flip:
            print("    AREA. Direction inverted as shipped and CORRECT once the")
            print("    area is held fixed -- the shipped gap measured the areas.")
        elif not n_flip and not m_flip:
            print("    PATHWAY. The violation keeps its advantage into the SAME")
            print("    area, so area-matching is not the fix and the adapter's")
            print("    stated premise is what is wrong.")
        else:
            print("    Neither pre-registered prediction. Report as-is.")
    if vp:
        print(f"    {VP} control: gram={vp[0]:.5f} catv={vp[1]:.5f} "
              f"({'gram>catv' if vp[0] > vp[1] else 'catv>gram'})")


if __name__ == "__main__":
    main()
