"""Does read_only() roll back connectome MATERIALIZATION? (task #41)

The tell from task #32: NOUN_CORE->VP reads EXACTLY 0.00000 pre-kWTA energy on
a fresh fork, and 0.00860 once violation frames -- which drive VERB_CORE->VP --
have run first. A pathway that goes from delivering literally nothing to
delivering a real value because an unrelated probe ran first is not activity
state. Something structural was created and not put back.

``NumpySparseEngine.ensure_area_conn`` names the mechanism in its own docstring:

    Connectome columns are normally allocated as a side effect of the target
    recruiting new neurons. A fiber first used *after* its target has already
    grown therefore stays shaped (0, 0): it delivers zero input forever.

So a (0,0) block is exactly what "reads 0.00000" looks like, and allocating it
is a persistent structural change. ``read_only()`` snapshots winners, w,
fixed_assembly and the RNG bit_generator state, and it sets _no_recruitment --
none of which covers a block being allocated.

This measures block SHAPES directly rather than inferring from energies:
snapshot every (src, tgt) weight shape, run one probe under read_only(), diff.

PREDICTION: at least one block changes shape across a read_only() probe, and
NOUN_CORE->VP is among them. If nothing changes shape, the inference from
energies was wrong and the residual is somewhere else entirely.
"""

from __future__ import annotations

import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.core.areas import CATEGORY_TO_CORE
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    anchored_p600_live,
    structural_role_area,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    get_parser_cache,
)

VIOLATION = ["the", "dog", "chases", "finds"]


def block_shapes(brain):
    """{(src, tgt): (rows, cols)} over every materialized area->area block."""
    out = {}
    for engine in brain._all_engines():
        for src, targets in getattr(engine, "_area_conns", {}).items():
            for tgt, conn in targets.items():
                w = getattr(conn, "weights", None)
                out[(src, tgt)] = (
                    (0, 0) if w is None else tuple(getattr(w, "shape", (0, 0)))
                )
    return out


def probe(parser, words):
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
        core = CATEGORY_TO_CORE.get(cat)
        area = structural_role_area(cat, verb_seen=verb_seen)
        return 1.0 - anchored_p600_live(
            parser, core, area, subject_core=subject_core,
        )


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = get_parser_cache().fork("SENTENCES", seed=42)
    brain = parser.brain

    before = block_shapes(brain)
    energy = probe(parser, VIOLATION)
    after = block_shapes(brain)

    created = sorted(set(after) - set(before))
    grew = sorted(k for k in set(after) & set(before) if after[k] != before[k])

    print(f"\n  probe: {' '.join(VIOLATION)}  ->  energy {energy:.5f}")
    print(f"  blocks before={len(before)}  after={len(after)}")
    print(f"  newly created: {len(created)}")
    for k in created[:12]:
        print(f"    + {k[0]} -> {k[1]}  {after[k]}")
    print(f"  reshaped: {len(grew)}")
    for k in grew[:12]:
        print(f"    ~ {k[0]} -> {k[1]}  {before[k]} -> {after[k]}")

    zeros_before = sorted(k for k, s in before.items() if s == (0, 0))
    print(f"\n  (0,0) blocks before the probe: {len(zeros_before)}")
    for k in zeros_before[:12]:
        print(f"    0 {k[0]} -> {k[1]}")

    print("\n  verdict")
    if created or grew:
        print("    CONFIRMED. read_only() does not roll back materialization --")
        print(f"    {len(created)} blocks created and {len(grew)} reshaped by a")
        print("    single probe, and they persist after the guard exits.")
    else:
        print("    REFUTED for this probe: no block was created or reshaped.")
        print("    The energy inference was wrong; look elsewhere for the residual.")


if __name__ == "__main__":
    main()
