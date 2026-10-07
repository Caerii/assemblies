"""Does ordered_recall ADVANCE, or only return the assembly it was cued with?

Registration: research/notes/sequence/PREREG_ordered_recall_reproduction.md.

`ordered_recall` fires the cue once and takes a snapshot, so `recalled[0]` is
the cue's own assembly; every later element is a step the chain took by
itself. The paper's claim (Dabagia, Papadimitriou and Vempala) is about those
later elements: inhibition removes the just-fired assembly from the
competition so the x_i -> x_{i+1} bridge can win.

`test_literature_parity.py::test_sequence_memorize_ordered_recall` asserts
`len(recalled) >= 1` and that `recalled[0]` overlaps `memorized[0]`. Both hold
when the chain never advances at all, because both describe the cue. This
module asserts the thing the paper claims, and it is `xfail(strict=True)`
because the implementation does not currently do it.
"""
from __future__ import annotations

import pytest

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import ordered_recall, sequence_memorize
from neural_assemblies.core.brain import Brain

N, K, P, BETA, W_MAX = 4000, 50, 0.05, 0.10, 20.0
LENGTH, ROUNDS, REPS, PHASE_B = 8, 8, 3, 0.5
PERIOD, STRENGTH = 3, 100.0
SEEDS = (42, 43, 44)
MATCH = 0.3


def _run(seed, *, materialize):
    brain = Brain(p=P, seed=seed, engine="numpy_sparse", w_max=W_MAX,
                  sampled_recurrence_policy="acknowledged")
    brain.add_area("SEQ", N, K, BETA)
    if materialize:
        brain.materialize_area("SEQ")
    names = [f"s{i}" for i in range(LENGTH)]
    for name in names:
        brain.add_stimulus(name, K)
    stored = sequence_memorize(brain, names, "SEQ", rounds_per_step=ROUNDS,
                               repetitions=REPS, phase_b_ratio=PHASE_B)
    brain.set_lri("SEQ", refractory_period=PERIOD, inhibition_strength=STRENGTH)
    recalled = ordered_recall(brain, "SEQ", names[0], max_steps=LENGTH + 4,
                              known_assemblies=list(stored))
    return list(stored), list(recalled)


def _steps_after_the_cue(stored, recalled):
    """Consecutive matches STRICTLY AFTER index 0, which is the cue."""
    got = 0
    for i in range(1, min(len(recalled), len(stored))):
        if overlap(recalled[i], stored[i]) >= MATCH:
            got += 1
        else:
            break
    return got


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "the chain does not advance: measured 0 steps after the cue on every seed, "
    "on both the sampled and the materialized connectome, across sequence "
    "lengths 3, 8 and 16, repetitions 3 and 10, and 64 knob combinations. "
    "See PREREG_ordered_recall_reproduction.md"))
def test_ordered_recall_advances_past_the_cue():
    advanced = [_steps_after_the_cue(*_run(seed, materialize=True)) for seed in SEEDS]
    assert min(advanced) >= 1, (
        f"recall never left the cued assembly: steps after the cue = {advanced}")


@pytest.mark.slow
def test_the_cue_itself_is_retrieved_far_better_on_the_sampled_connectome():
    """The sampler artifact, pinned. Our audit voids sampled sequence numbers,
    and the parity test for this paper does not materialize."""
    sampled, materialized = [], []
    for seed in SEEDS:
        stored, recalled = _run(seed, materialize=False)
        sampled.append(overlap(recalled[0], stored[0]))
        stored, recalled = _run(seed, materialize=True)
        materialized.append(overlap(recalled[0], stored[0]))
    assert min(sampled) > 0.7, sampled
    assert max(materialized) < 0.5, materialized
    assert min(sampled) > max(materialized) + 0.3, (sampled, materialized)


@pytest.mark.slow
def test_the_parity_assertions_pass_without_any_advancement():
    """Why this gap was invisible: the maintained parity assertions describe
    the CUE, and hold on a run that advances zero steps."""
    stored, recalled = _run(SEEDS[0], materialize=False)
    assert len(recalled) >= 1                          # parity assertion 1
    assert overlap(recalled[0], stored[0]) > 0.3       # parity assertion 2
    assert _steps_after_the_cue(stored, recalled) == 0  # and yet


@pytest.mark.slow
def test_ordered_recall_advances_when_each_element_is_written_as_a_transition():
    """The same operation, written with ONE stimulus-and-recurrence round per
    element at a write equal to the convergence threshold theta (1.775 at
    this cell), advances through the sequence (Amendment 2: 7 of 7 steps on
    17 of 20 registered seeds, 6 on the rest). Eight rounds per element make
    each element an attractor; one round writes only the transition."""
    import math
    theta = math.sqrt((1 - P) * math.log(N) / (P * K))
    advanced = []
    for seed in SEEDS:
        brain = Brain(p=P, seed=seed, engine="numpy_sparse", w_max=W_MAX,
                      sampled_recurrence_policy="acknowledged")
        brain.add_area("SEQ", N, K, round(theta, 4))
        brain.materialize_area("SEQ")
        names = [f"s{i}" for i in range(LENGTH)]
        for name in names:
            brain.add_stimulus(name, K)
        stored = sequence_memorize(brain, names, "SEQ", rounds_per_step=1,
                                   repetitions=1, phase_b_ratio=1.0)
        brain.set_lri("SEQ", refractory_period=PERIOD, inhibition_strength=STRENGTH)
        recalled = ordered_recall(brain, "SEQ", names[0], max_steps=LENGTH + 4,
                                  known_assemblies=list(stored))
        advanced.append(_steps_after_the_cue(list(stored), list(recalled)))
    assert min(advanced) >= LENGTH - 2, advanced
