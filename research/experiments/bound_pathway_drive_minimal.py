"""Does a TRAINED pathway deliver MORE pre-kWTA drive, or LESS?

THE OBSERVATION TO EXPLAIN. On a real parser, nouns bound into ROLE_PATIENT
deliver LESS drive there than nouns never bound there -- AUC 0.4583 / 0.3482 /
0.3578 over three seeds, consistently below chance
(`role_binding_readback_is_inverted_UNRESOLVED.md`). That is neither
pre-registered outcome: "binding is dormant" predicts ~0.5, "binding is
readable" predicts >0.5. An inverted, seed-consistent result says something
structural is happening.

WHY THIS RUNS ON A TOY BRAIN. On the parser the result is entangled with
training history, classification, crowding, and the whole ERP path. Two areas
and a handful of stimuli can answer the question that matters on its own:
**after binding assembly A into area DST, does A deliver more or less
per-candidate drive to DST than an assembly that was never bound there?**

THE HYPOTHESIS, and it would matter well beyond role binding. `norm_init`
normalizes each postsynaptic neuron's incoming weights ONCE at initialization
by its in-degree. The neurons a bound assembly drives are exactly the ones that
RECEIVED the binding, so they carry the most incoming weight -- and if
normalization scales them down, a well-trained pathway can deliver LESS
per-candidate drive than an untrained one. That would invert every
drive-deficit readout in the repo.

It is not an idle worry. This package already moved the ERP readout from
post-kWTA churn to pre-kWTA energy BECAUSE the post-kWTA family reverses sign
under `norm_init` (see `erp/adapters.py` module docstring, Cohen's d -1.94). If
pre-kWTA drive inverts too, that fix relocated the sign flip rather than
removing it -- and the N400 is built the same way.

THE DESIGN.

    SRC, DST                two areas
    a0..aM                  M stimuli, one assembly each in SRC
    a0 is BOUND into DST    repeated SRC -> DST projection with plasticity
    a1..aM never bound

Then, for every assembly, read `input_drive(SRC -> DST)` with the stored
assembly handed in as the source. One target area, so the divisor is identical
for every reading and cancels -- which matters, because that divisor is broken
(`input_drive_normalization_is_disabled.md`).

FOUR ARMS, 2x2:

    norm_init x {True, False}   the hypothesis
    beta      x {0.10, 0.0}     the NULL

The beta=0 arm is the control that makes the rest interpretable: with no
plasticity, "bound" and "unbound" differ only in that one of them was projected
at, so the drive difference must be ~0 under BOTH norm_init settings. If beta=0
shows a difference, the protocol is measuring something other than learning and
nothing else here can be read.

PRE-REGISTERED:

  * bound > unbound at beta=0.10, both norm_init settings
        -> trained pathways deliver more drive; the parser result is NOT a
           substrate property and the probe or the parser state is at fault.
  * bound < unbound under norm_init=True and bound > unbound under False
        -> HYPOTHESIS CONFIRMED. `norm_init` inverts the sign of pre-kWTA drive
           for trained pathways, and every drive-deficit readout needs re-reading.
  * bound ~ unbound everywhere at beta=0.10
        -> binding writes nothing readable even on a clean substrate, which is a
           much larger claim about the calculus than about this repo.
  * any difference at beta=0.0 -> the protocol is broken; report that instead.
"""
import os
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _substrate import read                                            # noqa: E402
from neural_assemblies.assembly_calculus.assembly import Assembly      # noqa: E402
from neural_assemblies.assembly_calculus.binding import input_drive    # noqa: E402
from neural_assemblies.core.brain import Brain                         # noqa: E402

SRC, DST = "SRC", "DST"
N, K, P = 2000, 40, 0.05
M_UNBOUND = 8          # assemblies never bound into DST
BUILD_ROUNDS = 6
BIND_ROUNDS = 10
SEEDS = [42, 7, 123, 2024, 5]


def _build(brain, stim, area, rounds):
    """Feed-forward build, the idiom `_substrate.build` uses without recurrence."""
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read(brain, area)


def _set_beta(brain, beta):
    """Set every area-to-area beta, dict AND engine.

    Both, because `Area.update_beta_by_area` was a silent no-op for exactly this
    reason -- the engine reads a different dict (#88). This mirrors
    `CurriculumTrainer._set_global_beta`, which is the production idiom.
    """
    for area_name in brain.areas:
        area = brain.areas[area_name]
        for src in list(area.beta_by_area):
            area.beta_by_area[src] = beta
            brain._engine.set_beta(area_name, src, beta)


def trial(*, norm_init: bool, beta: float, seed: int):
    brain = Brain(p=P, seed=seed, norm_init=norm_init)
    brain.add_area(SRC, N, K, beta=beta)
    brain.add_area(DST, N, K, beta=beta)
    names = [f"a{i}" for i in range(M_UNBOUND + 1)]
    for nm in names:
        brain.add_stimulus(nm, K)

    stored = {}
    for nm in names:
        stored[nm] = _build(brain, nm, SRC, BUILD_ROUNDS)

    # MATERIALIZATION PASS, at beta=0 so it cannot teach anything. Every
    # assembly's SRC->DST rows get instantiated, so the bound/unbound
    # comparison below is about WEIGHTS and not about which columns exist.
    # Without this the beta=0 null read 1.678 / 2.182 instead of 1.0.
    _set_beta(brain, 0.0)
    for nm in names:
        brain.project({nm: [SRC]}, {SRC: [DST]})
    _set_beta(brain, beta)

    # BIND a0 into DST: drive SRC with a0 and project SRC -> DST repeatedly, so
    # the SRC->DST synapses from a0's neurons are the ones reinforced.
    for _ in range(BIND_ROUNDS):
        brain.project({"a0": [SRC]}, {SRC: [DST]})

    drives = {}
    for nm in names:
        d = input_drive(
            brain,
            sources=[SRC],
            target_areas=[DST],
            source_assemblies={SRC: Assembly(SRC, stored[nm])},
        )
        drives[nm] = float(d.get(DST, float("nan")))
    return drives


def main():
    print(f"n={N} k={K} p={P} build_rounds={BUILD_ROUNDS} "
          f"bind_rounds={BIND_ROUNDS} unbound={M_UNBOUND} seeds={SEEDS}")
    print()
    hdr = (f"{'norm_init':<10} {'beta':<6} {'bound a0':>12} "
           f"{'unbound mean':>14} {'ratio':>8} {'seeds bound>unbound':>21}")
    print(hdr)
    print("-" * len(hdr))

    raw = {}
    for norm_init in (True, False):
        for beta in (0.10, 0.0):
            per_seed = {s_: trial(norm_init=norm_init, beta=beta, seed=s_)
                        for s_ in SEEDS}
            raw[(norm_init, beta)] = per_seed
            mb = statistics.fmean(d["a0"] for d in per_seed.values())
            mu = statistics.fmean(
                statistics.fmean(d[f"a{i}"] for i in range(1, M_UNBOUND + 1))
                for d in per_seed.values())
            wins = sum(
                d["a0"] > statistics.fmean(
                    d[f"a{i}"] for i in range(1, M_UNBOUND + 1))
                for d in per_seed.values())
            print(f"{str(norm_init):<10} {beta:<6} {mb:>12.6f} {mu:>14.6f} "
                  f"{(mb/mu if mu else float('nan')):>8.3f} "
                  f"{wins:>18}/{len(SEEDS)}")

    print()
    print("The within-arm ratio above still contains MATERIALIZATION and")
    print("RECRUITMENT, which is why the beta=0 rows are not 1.0. The paired")
    print("gain below differences both out.")
    print()
    print(f"{'norm_init':<10} {'gain(a0)':>10} {'gain(unbound)':>15} "
          f"{'plasticity effect':>19} {'seeds a0 gains more':>21}")
    print("-" * 78)
    verdict = {}
    for norm_init in (True, False):
        learned, null = raw[(norm_init, 0.10)], raw[(norm_init, 0.0)]
        g0, gu, wins = [], [], 0
        for s_ in SEEDS:
            a0 = learned[s_]["a0"] / null[s_]["a0"]
            un = statistics.fmean(
                learned[s_][f"a{i}"] / null[s_][f"a{i}"]
                for i in range(1, M_UNBOUND + 1))
            g0.append(a0)
            gu.append(un)
            wins += int(a0 > un)
        m0, mu = statistics.fmean(g0), statistics.fmean(gu)
        verdict[norm_init] = m0 / mu if mu else float("nan")
        print(f"{str(norm_init):<10} {m0:>10.4f} {mu:>15.4f} "
              f"{verdict[norm_init]:>19.4f} {wins:>18}/{len(SEEDS)}")

    print()
    vt, vf = verdict[True], verdict[False]
    if vt < 1.0 and vf > 1.0:
        print("HYPOTHESIS CONFIRMED: norm_init INVERTS the sign of pre-kWTA")
        print("drive for a trained pathway. Every drive-deficit readout in the")
        print("repo -- P600 and N400 alike -- is reading the wrong direction.")
    elif vt > 1.0 and vf > 1.0:
        print("REFUTED. Plasticity RAISES a bound pathway's drive relative to")
        print("unbound ones under BOTH norm_init settings, so the parser-level")
        print("inversion is NOT a substrate property. Suspect the probe or the")
        print("parser state -- #120 step (a), the index-space check.")
    else:
        print("Neither clean pattern. Report the table; do not summarise it.")


if __name__ == "__main__":
    main()
