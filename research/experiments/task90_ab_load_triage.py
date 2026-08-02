"""Which standing A/Bs did NOT share the sampler's error? A triage, not a re-run.

WHY
---
#90 established that a paired comparison on `numpy_sparse` is only trustworthy
when both arms sit at the same area LOAD, because the sampler's error is a steep
function of load (`recurrence_ceiling_on_exact_drive.md`). The known casualty is
norm_init on vs off, which reads an 8.0x capacity gain on the sampler and 1.0x
on exact drive.

The obvious next move is to re-derive every standing A/B on exact drive. That is
expensive and most of them are probably fine. This file does the cheap step
first: build both arms of each substrate-level A/B, MEASURE the load gap, and
rank. Re-derivation then goes to the flagged ones in order, instead of to
whichever result was most recently on someone's mind.

WHAT A FLAG MEANS, AND DOES NOT
-------------------------------
Flagged = the two arms did not share the instrument's error, so the MAGNITUDE
of the difference is not the model's. It is NOT a claim the direction is wrong;
in the norm_init case the direction survived (norm_init helps) and only the size
collapsed (8.0x -> 1.0x).

UNFLAGGED MEANS NOTHING. This was written as "the load-dependent channel is
ruled out" and that is FALSE, measured
(`task90_load_screen_sensitivity.py`). Two arms differing only in READOUT --
training bit-identical, load gap 0.000, screen passes -- read:

    numpy_sparse   acc 1.0000 vs 1.0000    delta +0.0000
    numpy_exact    acc 0.7292 vs 0.9948    delta -0.2656

The sampler reported NO effect where the substrate has a large one. So the
screen's specificity is zero in the only case that could be constructed, and
matched load is not sufficient for a trustworthy A/B: it matches the
RECRUITMENT channel of the error, and there is at least one other.

THIS FILE RANKS WHAT TO RE-RUN FIRST. It does not clear anything.

The threshold (0.05 of the area) is calibrated against seed noise, not theory:
two seeds of the same protocol separate by ~0.02, so 0.05 is above the null and
well below the 0.21 the known-bad case produces. Stated because it is a choice.

SCOPE
-----
Substrate-level manipulations that can be built in seconds. Deliberately NOT
the parser-level A/Bs (lesions, gating, mood) -- those need the emergent parser
on `numpy_exact`, which is blocked on the ABC surface it uses, and guessing
about them here would be worse than not listing them. They are named at the end
as the untriaged remainder rather than silently omitted.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies import diagnostics as dx  # noqa: E402

N, K, P, BETA, ROUNDS = 1000, 50, 0.05, 0.10, 6
AREA = "L"
M = 8
SEEDS = (42, 7, 123)
THRESHOLD = 0.05


def _brain(seed, norm_init=True, engine="numpy_sparse"):
    from neural_assemblies.core.brain import Brain
    b = Brain(p=P, seed=seed, norm_init=norm_init, engine=engine)
    b.add_area(AREA, N, K, beta=BETA)
    for m in range(M):
        b.add_stimulus(f"w{m}", K)
    return b


def _drive(b, recurrent=True, rounds=ROUNDS, beta=None):
    from neural_assemblies.assembly_calculus.ops import project
    if beta is not None:
        b.areas[AREA].beta = beta
        b._engine.set_beta(AREA, AREA, beta)
        for m in range(M):
            b._engine.set_beta(AREA, f"w{m}", beta)
    for m in range(M):
        if recurrent:
            project(b, f"w{m}", AREA, rounds=rounds, recurrent=True)
        else:
            for _ in range(rounds):
                b.project({f"w{m}": [AREA]}, {})
    return b


# Each entry: label -> {arm_name: build(seed) -> Brain}. The claim each one
# underwrites is named so a flag points somewhere.
def _cases():
    return {
        "norm_init ON vs OFF  [recurrence-needs-norm-init, "
        "self-recurrence-stability-window]": {
            "on": lambda s: _drive(_brain(s, norm_init=True)),
            "off": lambda s: _drive(_brain(s, norm_init=False)),
        },
        "recurrent vs feed-forward build  "
        "[recurrence-is-the-collapse-channel]": {
            "rec": lambda s: _drive(_brain(s), recurrent=True),
            "ff": lambda s: _drive(_brain(s), recurrent=False),
        },
        "beta 0.10 vs 0.00 (the null arm)  "
        "[beta-opposes-capacity-and-depth]": {
            "beta.10": lambda s: _drive(_brain(s), beta=0.10),
            "beta.00": lambda s: _drive(_brain(s), beta=0.00),
        },
        "rounds 6 vs 20 (reinforcement depth)  [reinforcement-tradeoff]": {
            "T6": lambda s: _drive(_brain(s), rounds=6),
            "T20": lambda s: _drive(_brain(s), rounds=20),
        },
        "CONTROL: same protocol, two seeds (must NOT flag)": {
            "a": lambda s: _drive(_brain(s)),
            "b": lambda s: _drive(_brain(s + 1000)),
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  A/B load triage -- did both arms share the sampler's error?")
    print(f"  n={N} k={K} beta={BETA} p={P}, {M} items, {len(SEEDS)} seeds, "
          f"flag above a gap of {THRESHOLD:.2f}\n")

    rows = []
    for label, arms in _cases().items():
        gaps = []
        loads = {a: [] for a in arms}
        for seed in SEEDS:
            built = {a: fn(seed) for a, fn in arms.items()}
            g = dx.load_audit(built, threshold=THRESHOLD)
            area = next(x for x in g if x.area == AREA)
            gaps.append(area.gap)
            for a, v in area.by_arm.items():
                loads[a].append(v)
        mean_gap = sum(gaps) / len(gaps)
        flag = mean_gap > THRESHOLD
        rows.append((mean_gap, flag, label, loads))

    for mean_gap, flag, label, loads in sorted(rows, reverse=True):
        tag = "CONFOUNDED" if flag else "ok        "
        arms = "   ".join(f"{a} {sum(v) / len(v):.3f}"
                          for a, v in sorted(loads.items()))
        print(f"  [{tag}] gap {mean_gap:.3f}   {label}")
        print(f"               load: {arms}\n")

    flagged = [r for r in rows if r[1] and "CONTROL" not in r[2]]
    control = next(r for r in rows if "CONTROL" in r[2])

    print("  READING\n")
    if control[1]:
        print("    THE CONTROL FLAGGED. Two seeds of one protocol should sit at")
        print("    the same load; if they do not, the threshold is below seed")
        print("    noise and every row above is uninterpretable. Fix that first.")
        return
    print(f"    control gap {control[0]:.3f} < {THRESHOLD:.2f}, so the threshold")
    print(f"    is above seed noise and the flags mean something.\n")
    print(f"    {len(flagged)} of {len(rows) - 1} substrate A/Bs are load-")
    print(f"    confounded on numpy_sparse. Their DIRECTIONS may well stand;")
    print(f"    their MAGNITUDES are not the model's until re-run on")
    print(f"    numpy_exact. In priority order:\n")
    for mean_gap, _, label, _ in sorted(flagged, reverse=True):
        print(f"      {mean_gap:.3f}  {label.splitlines()[0]}")

    print()
    print("    HOW MUCH A FLAG IS WORTH -- the two cases already re-derived on")
    print("    numpy_exact, which is the whole calibration set:")
    print()
    print("      norm_init on/off   flagged 0.204   8.0x -> 1.0x capacity gain")
    print("                                         CONCLUSION CHANGED")
    print("      rec vs ff          flagged 0.123   ff has no ceiling on both;")
    print("                                         rec ceiling 32 -> 16")
    print("                                         magnitude moved, CONCLUSION HELD")
    print()
    print("    So a flag is a SCREEN, not a verdict: 2 flagged, 2 re-derived,")
    print("    1 conclusion changed. And 4 of 4 firing means the screen is not")
    print("    very selective HERE, at load 0.87 in a nearly-full area, where")
    print("    almost any manipulation moves recruitment. Read the ordering,")
    print("    not the count. Its sensitivity is UNMEASURED -- there is no known")
    print("    case that passed the screen and still changed on exact drive,")
    print("    because no unflagged A/B has been re-derived yet. That is the")
    print("    experiment that would tell us whether this check can be trusted")
    print("    to clear anything, and it has not been run.")
    print()
    print("    UNTRIAGED, and named so the gap is visible: every parser-level")
    print("    A/B -- lesions, fiber gating, mood, ERP contrasts. Each changes")
    print("    which neurons win and so is a load-gap candidate.")
    print()
    print("    These were listed as BLOCKED on the emergent parser running on")
    print("    numpy_exact. That was asserted twice and never checked, and it")
    print("    is WRONG: the parser constructs, trains and parses on that")
    print("    engine, and returns the same categories. It does not request a")
    print("    non-default refractory_period, so nothing is rejected.")
    print()
    print("    Measured while checking (n=1000, k=50, one trained lexicon),")
    print("    load per core area, exact vs sparse:")
    print("        DET_CORE  0.376 / 0.876     PREP_CORE  0.225 / 0.506")
    print("        VERB_CORE 0.305 / 0.670     ADJ_CORE   0.197 / 0.464")
    print("        NOUN_CORE 0.230 / 0.530     PRON_CORE  0.154 / 0.338")
    print("    The sampler recruits ~2.3x the substrate in EVERY core area, so")
    print("    the whole parser sits at a systematically different load than")
    print("    the model it is meant to be. Nothing blocks re-deriving the")
    print("    parser-level results; it is now just work.")


if __name__ == "__main__":
    main()
