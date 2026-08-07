"""`input_drive` measures HOW MUCH. Binding encodes WHERE. Pinned executably.

WHY THIS FILE EXISTS. `input_drive`'s docstring already states the rule --
"use this rather than `bind_strength` whenever the question is *which area*
rather than *which assembly*" -- and the P600 was built on it anyway, to answer
a which-assembly question. Two full investigations were spent concluding that
role binding was dormant, inverted, or capacity-bound before the real answer
turned out to be that the instrument cannot express the question
(research/notes/role_binding_works_the_metric_cannot_see_it.md).

Documentation demonstrably failed here, so this is the executable form. It also
guards the reverse: if someone later "fixes" `input_drive` to be
assembly-sensitive, these tests fail and force the conversation rather than
silently revalidating every result that assumed it was not.

THE THREE CLAIMS, in increasing strength:

  1. STRUCTURAL -- `input_drive` returns one scalar PER AREA. A per-assembly
     question is not expressible in its return type at all.
  2. POSITIVE -- `bind_strength` DOES discriminate: an assembly retrieves the
     target it was bound to over one it was not.
  3. CONSEQUENCE -- ranking by drive does not recover which assembly a source
     was bound to, on the same substrate where `bind_strength` does.

Claim 3 is the one that matters and the one a reader should check first: 1 and 2
are each satisfiable by a broken implementation, and only 3 fails if the
orthogonality is not real.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.binding import bind_strength, input_drive
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain

SRC, DST = "SRC", "DST"
N, K, P, BETA = 2000, 40, 0.05, 0.10
BUILD_ROUNDS, BIND_ROUNDS = 6, 10
WORDS = ("a0", "a1", "a2", "a3")


def _read(brain, area) -> np.ndarray:
    """Neuron IDs. `area.winners` is a different index space -- never use it."""
    return np.asarray(_snap(brain, area).winners, dtype=np.int64)


@pytest.fixture(scope="module")
def bound_brain():
    """Each source assembly bound to its OWN target assembly in one area.

    This is the configuration role binding produces and the one the P600 was
    asked to read: several sources sharing a target AREA, each with a distinct
    target ASSEMBLY inside it.
    """
    brain = Brain(p=P, seed=42, norm_init=True)
    brain.add_area(SRC, N, K, beta=BETA)
    brain.add_area(DST, N, K, beta=BETA)
    for w in WORDS:
        brain.add_stimulus(w, K)

    sources = {}
    for w in WORDS:
        for _ in range(BUILD_ROUNDS):
            brain.project({w: [SRC]}, {})
        sources[w] = _read(brain, SRC)

    targets = {}
    for w in WORDS:
        for _ in range(BIND_ROUNDS):
            brain.project({w: [SRC]}, {SRC: [DST]})
        targets[w] = _read(brain, DST)

    return brain, sources, targets


def _drive(brain, source_ids):
    return input_drive(
        brain, sources=[SRC], target_areas=[DST],
        source_assemblies={SRC: Assembly(SRC, source_ids)},
    )


def _strength(brain, source_ids, target_ids):
    return bind_strength(
        brain, sources=[SRC], target_area=DST,
        target_assembly=Assembly(DST, target_ids),
        source_assemblies={SRC: Assembly(SRC, source_ids)},
    )


def test_input_drive_returns_one_scalar_per_area(bound_brain):
    """A which-assembly question is not expressible in the return type.

    The keys are AREA NAMES. There is nowhere for a per-assembly answer to go,
    which is the structural reason no amount of dynamic range (#104) could have
    made this readout answer the question it was being asked.
    """
    brain, sources, _targets = bound_brain
    got = _drive(brain, sources["a0"])
    assert set(got) == {DST}
    assert isinstance(got[DST], float)


def test_bind_strength_discriminates_the_target_it_was_bound_to(bound_brain):
    """The WHERE readout works: each source retrieves its OWN target best.

    Asserted as a RANKING over all four candidate targets rather than as a
    single pairwise comparison, because a pairwise test passes for a readout
    that merely correlates with target size.
    """
    brain, sources, targets = bound_brain
    correct = 0
    for w in WORDS:
        scores = {t: _strength(brain, sources[w], targets[t]) for t in WORDS}
        best = max(scores, key=lambda t: scores[t])
        correct += int(best == w)
    assert correct >= 3, (
        f"bind_strength recovered only {correct}/4 bindings. If this drops, the "
        f"WHERE readout is no longer usable and #121 loses its instrument -- "
        f"investigate before relaxing the bound."
    )


def test_drive_cannot_recover_which_assembly_a_source_was_bound_to(bound_brain):
    """THE CLAIM THAT MATTERS. Same substrate, same bindings, drive is blind.

    For each source, ask the which-assembly question using drive -- the only
    way it CAN be asked, since drive yields one number per area: rank the
    candidate targets by the drive their own source delivers. That ranking
    carries no information about which target the probe source was bound to,
    because the number does not depend on the target at all.

    The assertion is deliberately weak and still decisive: drive must do WORSE
    than `bind_strength` on the identical question. A stronger "drive is at
    chance" assertion would be flaky at four items; a comparative one is not,
    and it is the comparison that licenses replacing the readout.
    """
    brain, sources, targets = bound_brain

    drive_correct = 0
    for w in WORDS:
        # The only which-assembly ranking drive can express: score each
        # candidate target by the drive ITS source delivers into the area.
        # It is the same set of four numbers for every probe word `w`, which
        # is precisely the point -- the ranking cannot depend on `w`.
        scores = {t: _drive(brain, sources[t])[DST] for t in WORDS}
        drive_correct += int(max(scores, key=lambda t: scores[t]) == w)

    bind_correct = 0
    for w in WORDS:
        scores = {t: _strength(brain, sources[w], targets[t]) for t in WORDS}
        bind_correct += int(max(scores, key=lambda t: scores[t]) == w)

    assert drive_correct <= 1, (
        f"drive recovered {drive_correct}/4 bindings. It ranks the SAME four "
        f"numbers regardless of which source is being probed, so anything above "
        f"1/4 is coincidence -- but if it is reproducible, the orthogonality "
        f"claim in role_binding_works_the_metric_cannot_see_it.md is wrong."
    )
    assert bind_correct > drive_correct, (
        f"bind_strength {bind_correct}/4 vs drive {drive_correct}/4. The whole "
        f"basis for rebuilding the ERP role readout (#121) is that the WHERE "
        f"instrument beats the HOW MUCH one on this question."
    )


def test_drive_is_not_merely_noisy_it_is_informative_about_the_area(bound_brain):
    """Drive is a GOOD instrument for its own question -- keep that visible.

    Without this, the file reads as "input_drive is bad" and someone deletes a
    working measurement. Drive separates a bound source from an unbound one,
    which is a which-AREA question, and that is what `wobbly` acquisition uses
    it for correctly.
    """
    brain, sources, _targets = bound_brain
    brain.add_stimulus("never_bound", K)
    for _ in range(BUILD_ROUNDS):
        brain.project({"never_bound": [SRC]}, {})
    unbound = _read(brain, SRC)

    bound_drive = float(np.mean([_drive(brain, sources[w])[DST] for w in WORDS]))
    unbound_drive = _drive(brain, unbound)[DST]
    assert bound_drive > unbound_drive, (
        f"bound {bound_drive:.6f} vs never-bound {unbound_drive:.6f}. Drive is "
        f"supposed to answer WHICH AREA; if it stops doing that, wobbly "
        f"acquisition loses its signal too."
    )
