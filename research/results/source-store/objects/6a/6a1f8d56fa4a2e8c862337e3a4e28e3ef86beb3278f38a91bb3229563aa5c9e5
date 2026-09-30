"""How many bindings can one role area hold before a pathway stops being readable?

THE CONTRAST THIS EXPLAINS. Binding is readable on a clean substrate and not in
the parser:

    minimal substrate    1 assembly bound into DST     1.23-1.35x, 5/5 seeds
    parser ROLE_PATIENT  36 assemblies bound           AUC ~0.4, no effect

Source-side crowding is already excluded -- `NOUN_CORE` pairwise assembly
overlap is 0.126-0.132, so the sources ARE distinct. What differs is how many
assemblies share the TARGET.

THE DESIGN keeps everything except that number fixed. `SRC` always holds the
SAME 40 assemblies, built identically at every point of the sweep; only how many
of them get BOUND into `DST` varies. The 8 control assemblies are never bound at
any M, so the comparison group is literally the same items throughout. If
instead the population grew with M, source crowding would grow with it and the
sweep would confound the two.

    M in {1, 2, 4, 8, 16, 32}     bindings into DST
    8 controls                    never bound, identical at every M

THE STATISTIC IS THE PAIRED ONE, for the reason
`a_beta_zero_null_does_not_hold_the_connectome_still.md` records: a beta=0 arm
does NOT hold the connectome constant -- `project()` materialises columns and
recruits neurons whether or not it learns, and the naive within-arm ratio read
5.5x when the honest figure was 1.23x. So:

    gain(word)         = drive(word | beta=0.10) / drive(word | beta=0.0)
    plasticity effect  = mean gain(bound) / mean gain(control)

at the same seed, where both runs share wiring, build, materialisation pass and
recruitment history and differ only in whether the bind loop learned.

Also reported: READBACK RATE, the fraction of bound assemblies whose individual
gain beats the control mean. The aggregate can stay above 1.0 while most items
are already unreadable -- a mean is a poor description of a capacity, which is
the same reason the merge-capacity work reports margin rather than accuracy.

PRE-REGISTERED:

  * effect decays toward 1.0 as M grows, reaching it near M ~ 36
        -> target capacity explains the parser directly, and role binding is
           being run past its limit rather than being broken.
  * effect flat in M
        -> target capacity is REFUTED. The parser's failure is something
           neither this nor the minimal experiment has isolated, and the next
           suspect is the stored/fresh divergence (overlap 0.27) or the role
           area's own state.
  * effect already ~1.0 at M=1
        -> the harness stopped reproducing its own earlier result; fix that
           before reading anything else.
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
M_MAX = 32              # the largest number of bindings tested
N_CONTROL = 8           # never bound, at any M
BUILD_ROUNDS = 6
BIND_ROUNDS = 10
M_GRID = [1, 2, 4, 8, 16, 32]
SEEDS = [42, 7, 123]


def _set_beta(brain, beta):
    """Set every area-to-area beta, dict AND engine (the #88 idiom)."""
    for area_name in brain.areas:
        area = brain.areas[area_name]
        for src in list(area.beta_by_area):
            area.beta_by_area[src] = beta
            brain._engine.set_beta(area_name, src, beta)


def trial(*, m_bound: int, beta: float, seed: int):
    # Engine PINNED rather than left to "auto". Measured: auto resolves to
    # numpy_sparse at these sizes, so this is behaviour-preserving -- but an
    # implicit engine means a future default change silently reinterprets every
    # number already recorded from this file ([[pin-backend-not-global]]).
    brain = Brain(p=P, seed=seed, norm_init=True, engine="numpy_sparse")
    brain.add_area(SRC, N, K, beta=beta)
    brain.add_area(DST, N, K, beta=beta)

    # SRC population is CONSTANT across the sweep -- only how many get bound
    # changes. Controls are the last N_CONTROL and are never bound at any M.
    names = [f"a{i}" for i in range(M_MAX + N_CONTROL)]
    for nm in names:
        brain.add_stimulus(nm, K)

    stored = {}
    for nm in names:
        for _ in range(BUILD_ROUNDS):
            brain.project({nm: [SRC]}, {})
        stored[nm] = read(brain, SRC)

    # Materialisation pass at beta=0: instantiate EVERY assembly's SRC->DST
    # rows, so the comparison is about weights and not about which columns
    # exist. Without it the beta=0 null read 1.678 instead of 1.0.
    _set_beta(brain, 0.0)
    for nm in names:
        brain.project({nm: [SRC]}, {SRC: [DST]})
    _set_beta(brain, beta)

    bound = names[:m_bound]
    for nm in bound:
        for _ in range(BIND_ROUNDS):
            brain.project({nm: [SRC]}, {SRC: [DST]})

    drives = {}
    for nm in names:
        d = input_drive(
            brain, sources=[SRC], target_areas=[DST],
            source_assemblies={SRC: Assembly(SRC, stored[nm])},
        )
        drives[nm] = float(d.get(DST, float("nan")))
    return drives


def main():
    controls = [f"a{i}" for i in range(M_MAX, M_MAX + N_CONTROL)]
    print(f"n={N} k={K} p={P} SRC population={M_MAX + N_CONTROL} "
          f"(constant) controls={N_CONTROL} bind_rounds={BIND_ROUNDS} "
          f"seeds={SEEDS}")
    print()
    hdr = (f"{'M bound':>8} {'plasticity effect':>19} {'per-seed':>26} "
           f"{'readback rate':>15}")
    print(hdr)
    print("-" * len(hdr))

    table = {}
    for m in M_GRID:
        effects, rates = [], []
        for seed in SEEDS:
            learned = trial(m_bound=m, beta=0.10, seed=seed)
            null = trial(m_bound=m, beta=0.0, seed=seed)
            gain = {nm: learned[nm] / null[nm] for nm in learned
                    if null[nm]}
            bound = [f"a{i}" for i in range(m)]
            gb = [gain[nm] for nm in bound if nm in gain]
            gc = [gain[nm] for nm in controls if nm in gain]
            if not gb or not gc:
                continue
            mc = statistics.fmean(gc)
            effects.append(statistics.fmean(gb) / mc if mc else float("nan"))
            rates.append(sum(g > mc for g in gb) / len(gb))
        if not effects:
            continue
        table[m] = (statistics.fmean(effects), statistics.fmean(rates))
        print(f"{m:>8} {table[m][0]:>19.4f} "
              f"{str([round(e, 3) for e in effects]):>26} "
              f"{table[m][1]:>14.2f}")

    print()
    if 1 not in table or table[1][0] < 1.05:
        print("** M=1 did not reproduce the known ~1.23x effect. The harness is")
        print("not measuring what it measured before; fix that before reading")
        print("the sweep. **")
        return
    lo, hi = table[min(table)][0], table[max(table)][0]
    print(f"M={min(table)}: {lo:.4f}    M={max(table)}: {hi:.4f}")
    if hi < 1.05 <= lo:
        print("DECAYS. Target capacity explains the parser: role binding is")
        print("being run past its limit, not failing to write. The number to")
        print("extract next is WHERE it crosses, against the parser's 36.")
    elif hi >= 1.05:
        print("FLAT (or still above threshold at the largest M tested). Target")
        print("capacity does NOT explain the parser on this range. Next suspect")
        print("is the stored/fresh divergence (overlap 0.27) or the role area's")
        print("own state -- and the sweep should be extended past M=32 before")
        print("that is treated as settled.")
    else:
        print("Neither pattern. Report the table; do not summarise it.")


if __name__ == "__main__":
    main()
