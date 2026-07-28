"""Is the ERP calibration path order-dependent? (task #32, the confound)

Two measurements of the SAME quantity disagree, and the disagreement is the
point.

``erp_p600_zero_trace.py`` read the energies out of ``calibrate_erp_thresholds``
and got catv ABOVE gram (0.0291 vs 0.0254) -- inverted. ``erp_p600_area_matched.py``
measured the same frames directly under ``brain.read_only()`` and got gram ABOVE
catv (0.0255 vs 0.0148) -- correct, and by 42%. Same parser fixture, same frames,
same adapter. Both cannot be right.

The candidate: the shipped ERP path probes under ``brain.frozen()``
(``runner.run_incremental_erp_probes``, ``frames._probe_at_critical_position_warm``,
``frames._ensure_parsed_through``). ``frozen()`` stops WEIGHT CHANGE but not
RECRUITMENT -- an area below its cap still grows ``w`` and still allocates new
neurons on every probe. That was established when ``read_only()`` was built:
frozen() alone left 29975 cells and 3 blocks of divergence between forward and
reversed probe order; ``read_only()`` left 0.

If that is what is happening here, calibration is not measuring the frames -- it
is measuring the frames PLUS however much the brain grew during the baseline
pass and the earlier frames. Calibration runs a baseline pass over 8 grammatical
sentences and then TWO full collect passes before the numbers it reports, so the
violation arm is read from a materially different brain than the grammatical one
it is compared against.

PRE-REGISTERED PREDICTIONS

  1. ORDER. Probing the frames forward vs reversed changes the energies under
     frozen() and does not under read_only().

  2. GROWTH. Total area ``w`` increases across a frozen() probe sweep and is
     unchanged across a read_only() sweep.

  3. SIGN. The frozen() sweep can reproduce the inverted direction; the
     read_only() sweep cannot.

Prediction 3 is the one that matters: if it holds, the "P600 is sign-inverted"
finding recorded against this project is an artifact of probe contamination
rather than a fact about the metric, and the fix is the probe discipline, not
the metric.
"""

from __future__ import annotations

import os
import statistics
from contextlib import contextmanager

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.core.areas import CATEGORY_TO_CORE
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

ARMS = [
    (label, words)
    for label, _d, words in DEFAULT_CALIBRATION_FRAMES
    if label in ("grammatical", "category_violation")
]


def total_w(brain) -> int:
    return sum(int(a.w) for a in brain.areas.values())


def probe_frame(parser, words, guard, reset_inside=False):
    """Energy at the final content word, under the supplied isolation *guard*.

    *reset_inside* moves ``_reset_context_state`` / ``_get_incremental_circuit``
    INSIDE the guard. As shipped they run outside it -- ``run_incremental_erp_probes``
    calls both before ``with parser.brain.frozen()`` -- so whatever they write to
    the brain escapes the rollback and carries into the next frame.
    """
    known = [w for w in words if w in parser.stim_map]
    circuit = None
    if not reset_inside:
        parser._reset_context_state()
        circuit = parser._get_incremental_circuit(reset=True)
    verb_seen, noun_count, subject_core, cat = False, 0, None, ""

    with guard():
        if reset_inside:
            parser._reset_context_state()
            circuit = parser._get_incremental_circuit(reset=True)
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
        area = structural_role_area(cat, verb_seen=verb_seen)
        deficit = anchored_p600_live(parser, core, area, subject_core=subject_core)
    return known[-1], 1.0 - deficit


def sweep(parser, guard, arms, reset_inside=False):
    """One full pass over *arms*; returns {(label, word): energy} and w growth.

    Keyed on (label, word) rather than word: two grammatical frames both end in
    "cat", and keying on the word alone would silently collapse them so that
    only whichever ran LAST survived -- an order artifact manufactured by the
    harness, in a file whose whole subject is order artifacts.
    """
    w0 = total_w(parser.brain)
    out = {}
    for label, words in arms:
        r = probe_frame(parser, words, guard, reset_inside=reset_inside)
        if r is not None:
            out[(label, tuple(words))] = r[1]
    return out, total_w(parser.brain) - w0


def direction(energies):
    by = {}
    for (label, _words), e in energies.items():
        by.setdefault(label, []).append(e)
    g, c = by.get("grammatical", []), by.get("category_violation", [])
    if not (g and c):
        return None
    return statistics.median(g), statistics.median(c)


def _verdict(gm, cm):
    return "gram>catv EXPECTED" if gm > cm else "catv>gram INVERTED"


def compare(cache, name, guard_of, *, reset_inside=False):
    """Forward vs reversed probe order, on two INDEPENDENT forks."""
    pf = cache.fork("SENTENCES", seed=42)
    fwd, grow_f = sweep(pf, guard_of(pf), ARMS, reset_inside=reset_inside)
    pr = cache.fork("SENTENCES", seed=42)
    rev, grow_r = sweep(pr, guard_of(pr), list(reversed(ARMS)),
                        reset_inside=reset_inside)

    shared = sorted(set(fwd) & set(rev))
    diffs = [abs(fwd[k] - rev[k]) for k in shared]
    print(f"\n  {name}")
    print(f"    {'frame':<34} {'forward':>10} {'reversed':>10} {'|diff|':>10}")
    for k in shared:
        tag = f"{k[0][:4]}:{' '.join(k[1])}"
        print(f"    {tag:<34} {fwd[k]:>10.5f} {rev[k]:>10.5f} "
              f"{abs(fwd[k] - rev[k]):>10.5f}")
    print(f"    order divergence: max={max(diffs):.5f} "
          f"n_differing={sum(1 for d in diffs if d > 1e-9)}/{len(shared)}")
    print(f"    w growth: forward={grow_f:+d}  reversed={grow_r:+d}")
    for lbl, energies in (("forward", fwd), ("reversed", rev)):
        d = direction(energies)
        if d:
            print(f"    direction ({lbl}): gram={d[0]:.5f} catv={d[1]:.5f} "
                  f"-> {_verdict(*d)}")
    return max(diffs)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cache = get_parser_cache()

    compare(cache, "frozen() -- as shipped", lambda p: p.brain.frozen)
    ro = compare(cache, "read_only()", lambda p: p.brain.read_only)

    # -- the residual channel -------------------------------------------------
    # read_only() rolls back BRAIN state, but the shipped probe calls
    # `_reset_context_state()` and `_get_incremental_circuit(reset=True)` BEFORE
    # entering the guard (`run_incremental_erp_probes` does exactly this). Those
    # write to the brain, so their effect escapes the rollback and accumulates
    # across frames: the state a frame starts from depends on how many frames
    # ran before it, which is order.
    #
    # Prediction: moving the reset INSIDE the guard drives the residual read_only
    # order divergence to 0. If it does not, this explanation is wrong too.
    inside = compare(cache, "read_only() + reset INSIDE the guard",
                     lambda p: p.brain.read_only, reset_inside=True)

    print("\n  verdict")
    print(f"    read_only, reset outside: max divergence {ro:.5f}")
    print(f"    read_only, reset inside:  max divergence {inside:.5f}")
    if inside <= 1e-9 < ro:
        print("    CONFIRMED. The residual was state written outside the guard.")
    elif inside < ro:
        print("    PARTIAL. Reset placement accounts for some of it, not all.")
    else:
        print("    REFUTED. Reset placement is not the residual channel.")


if __name__ == "__main__":
    main()
