"""A supplied state code decodes identically to the assigned one.

`PREREG_autonomous_chain.md` Amendment 2 replaces the teacher-forced disjoint
blocks with codes that can collide, which is the one difference between every
sequence result in this repository and the setting the papers work in. That
comparison is only meaningful if both arms are read out the SAME way.

The assigned path decodes by integer division (`w // k`), which cannot express
a neuron belonging to two states. The supplied path decodes by membership.
These tests pin that the two agree exactly on a disjoint contiguous code, so
the collidable arm and the disjoint arm sit on one instrument, and that the
supplied path refuses codes it cannot honestly represent.

Marked `gpu`, and skipped when the fused kernels cannot build: `HashedArea`
compiles them on construction, which needs the documented developer shell, so
the maintained gate skips these rather than failing them.
"""
from __future__ import annotations

import pytest


SEEDS = [42, 43]
K, N_ARC, N_STATE, P = 8, 256, 512, 0.3
STATES = [f"q{i}" for i in range(6)]
TICK = "tick"
TABLE = [(STATES[i], TICK, STATES[i + 1]) for i in range(len(STATES) - 1)]


@pytest.fixture
def fused(fused_kernels):
    """The shared device gate (tests/_devices.py)."""
    return fused_kernels


def _fsm(state_code=None):
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    return HashedArcFSM(SEEDS, STATES, [TICK], TABLE, n_arc=N_ARC, k=K, p=P,
                        n_state=N_STATE, beta=0.1, refracted_strength=0.1,
                        max_potentiations=32, device="cuda",
                        state_code=state_code)


def _disjoint_code(device="cuda"):
    from neural_assemblies.core._torch_ops import torch_ops
    return torch_ops.arange(len(STATES) * K, device=device,
                            dtype=torch_ops.int64).view(len(STATES), K)


def test_supplied_disjoint_code_decodes_exactly_as_integer_division(fused):
    """The two readouts are the same function on a disjoint contiguous code."""
    from neural_assemblies.core._torch_ops import torch_ops

    assigned, supplied = _fsm(), _fsm(state_code=_disjoint_code())
    assert assigned.membership is None and supplied.membership is not None
    for name in STATES:
        assigned.cue_state(name)
        supplied.cue_state(name)
        assert torch_ops.equal(assigned.state.winners, supplied.state.winners)
        assert torch_ops.equal(assigned.read_state(), supplied.read_state())


def test_supplied_code_survives_training_and_reads_the_same_chain(fused):
    """Not just the cue: the whole run agrees, which is what the arms compare."""
    from neural_assemblies.core._torch_ops import torch_ops

    steps = len(STATES) - 1
    got = []
    for code in (None, _disjoint_code()):
        fsm = _fsm(state_code=code)
        fsm.train(12)
        fsm.check()
        syms = torch_ops.zeros(len(SEEDS), steps, dtype=torch_ops.int64,
                               device="cuda")
        got.append(fsm.run(syms, STATES[0]).cpu().numpy().tolist())
    assert got[0] == got[1], "one readout, or the two arms are not comparable"


def test_a_collidable_code_is_allowed_and_the_area_is_not_widened(fused):
    """The point of the amendment: states that cannot be disjoint."""
    from neural_assemblies.core._torch_ops import torch_ops

    tight = 32                      # 6 states x k=8 = 48 > 32, so they MUST collide
    gen = torch_ops.Generator(device="cpu")
    gen.manual_seed(7)
    code = torch_ops.stack([torch_ops.randperm(tight, generator=gen)[:K]
                            for _ in STATES]).to("cuda").to(torch_ops.int64)
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    fsm = HashedArcFSM(SEEDS, STATES, [TICK], TABLE, n_arc=N_ARC, k=K, p=P,
                       n_state=tight, beta=0.1, refracted_strength=0.1,
                       max_potentiations=32, device="cuda", state_code=code)
    assert fsm.n_state == tight, "a supplied code must not widen the area"
    fsm.cue_state(STATES[0])
    assert int(fsm.read_state()[0]) in range(len(STATES))


def test_a_malformed_code_is_refused(fused):
    from neural_assemblies.core._torch_ops import torch_ops

    with pytest.raises(ValueError, match=r"must be \["):
        _fsm(state_code=torch_ops.zeros((len(STATES), K + 1),
                                        dtype=torch_ops.int64, device="cuda"))
    with pytest.raises(ValueError, match="outside the state area"):
        _fsm(state_code=_disjoint_code() + N_STATE)
    dup = _disjoint_code().clone()
    dup[0, 1] = dup[0, 0]
    with pytest.raises(ValueError, match="DISTINCT"):
        _fsm(state_code=dup)


# --- training ORDER, which is not neutral under refraction --------------------

def test_every_order_presents_every_transition_the_same_number_of_times():
    """The arms must differ in ORDER alone, never in dosage.

    Checked without a device: this is a property of the loop, and if the counts
    differ then a positional result is really a dosage result.
    """
    import random
    from collections import Counter
    table = {(f"q{i}", TICK): f"q{i+1}" for i in range(20)}
    for order in ("chain", "reversed", "shuffled"):
        seen = Counter()
        items = list(table.items())
        if order == "reversed":
            items = items[::-1]
        rng = random.Random(0)
        for _ in range(7):
            if order == "shuffled":
                items = list(table.items())
                rng.shuffle(items)
            for (fr, sym), to in items:
                seen[(fr, sym, to)] += 1
        assert set(seen.values()) == {7}, f"{order} does not present evenly"
        assert len(seen) == len(table)


def test_an_unknown_training_order_is_refused(fused):
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
    fsm = HashedArcFSM(SEEDS, STATES, [TICK], TABLE, n_arc=N_ARC, k=K, p=P,
                       n_state=N_STATE, beta=0.1, refracted_strength=0.1,
                       max_potentiations=32, device="cuda")
    with pytest.raises(ValueError, match="chain, reversed or shuffled"):
        fsm.train(2, order="backwards")


def test_reversed_order_trains_a_different_organ(fused):
    """If order were neutral the two organs would be identical; they are not."""
    from neural_assemblies.core._torch_ops import torch_ops
    from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM

    got = []
    for order in ("chain", "reversed"):
        fsm = HashedArcFSM(SEEDS, STATES, [TICK], TABLE, n_arc=N_ARC, k=K, p=P,
                           n_state=N_STATE, beta=0.1, refracted_strength=0.1,
                           max_potentiations=32, device="cuda")
        fsm.train(8, order=order)
        fsm.arc.inhibit()
        fsm.cue_state(STATES[0])
        fsm.step(TICK)
        got.append(torch_ops.sort(fsm.arc.winners, dim=1).values.clone())
    assert not torch_ops.equal(got[0], got[1]), (
        "the training order left the arc unchanged, so refraction is not "
        "accumulating across the sweep and the amendment has no treatment")
