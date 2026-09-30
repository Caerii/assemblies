"""Long range inhibition on the hashed substrate.

PROVENANCE, and it is not what the first draft of this file said. NEITHER
reference implements a refractory period: in `mdabagia-nemo/brain.py`
`inhibit()` is a RESET (`clear_input(); activations = []`), and in
`dmitropolsky-assemblies` "inhibit" means AREA GATING for control flow. Both
clones grep empty for refractory, steps_ago and recently-fired. LRI-with-decay
is THIS REPOSITORY'S OWN construction, the same status as `ordered_recall`.

It is kept because RECOVERY is the one ingredient the substrate lacks. With no
decay term anywhere, idle spacing is a no-op by construction
([[schedule-beats-mechanism]]) and the refraction bias is not a relative
refractory period in the biological sense. The hashed organ that carries the
fast sequence work had no recovering inhibition at all, so no sequence result
here has ever been measured with one.

LRI is NOT the refraction bias with a decay bolted on. The two are orthogonal:

    refraction   charges per WIN, in proportion to raw drive, never recovers,
                 gated on plasticity
    LRI          charges per STEP at a FIXED strength, recovers linearly over
                 `period`, and applies on frozen reads too

These tests pin the semantics against `_sparse.py`'s reference loop, and pin
first that switching it off changes nothing at all.

Marked `gpu`, skipped when the fused kernels cannot build.
"""
from __future__ import annotations

import pytest


SEEDS = [42, 43]
N, K, P = 512, 8, 0.3


@pytest.fixture
def fused(fused_kernels):
    """The shared device gate (tests/_devices.py)."""
    return fused_kernels


def _area(refracted=0.0):
    from neural_assemblies.core.torch_engine._hashed import HashedArea
    from neural_assemblies.core.torch_engine._hash import fnv1a_pair_seed
    seeds = [fnv1a_pair_seed(s, "A", "A") for s in SEEDS]
    return HashedArea(N, K, seeds, device="cuda", refracted_strength=refracted)


def test_off_by_default_and_a_zero_switches_it_off(fused):
    """The guard that lets every existing result stand unchanged."""
    from neural_assemblies.core._torch_ops import torch_ops
    a = _area()
    assert a.lri_period == 0 and a.lri_strength == 0.0
    assert a._lri_hist is None
    drive = torch_ops.ones(len(SEEDS), N, device="cuda")
    assert torch_ops.equal(a.apply_lri(drive), drive), "inert when unset"
    a.set_lri(4, 1.0)
    assert a._lri_hist is not None
    a.set_lri(0, 0.0)
    assert a._lri_hist is None
    assert torch_ops.equal(a.apply_lri(drive), drive), "a zero switches it off"


def test_a_fired_neuron_is_penalised_and_the_decay_is_linear(fused):
    """`strength * (1 - (age - 1) / period)`, exactly as _sparse.py computes."""
    from neural_assemblies.core._torch_ops import torch_ops
    a = _area()
    a.set_lri(4, 10.0)
    first = torch_ops.zeros(len(SEEDS), K, dtype=torch_ops.int64, device="cuda")
    first[:, :] = torch_ops.arange(K, device="cuda")
    a._push_lri(first)
    drive = torch_ops.zeros(len(SEEDS), N, device="cuda")
    got = a.apply_lri(drive)
    assert float(got[0, 0]) == pytest.approx(-10.0), "age 1 pays full strength"
    assert float(got[0, K]) == 0.0, "a neuron that never fired pays nothing"
    # age it: three more steps of DIFFERENT winners
    for step in range(1, 4):
        nxt = first + K * step
        a._push_lri(nxt)
    got = a.apply_lri(drive)
    # the original set is now age 4 of period 4 -> decay 1 - 3/4 = 0.25
    assert float(got[0, 0]) == pytest.approx(-2.5), "linear decay to 1/period"
    assert float(got[0, K * 3]) == pytest.approx(-10.0), "newest pays full"


def test_it_recovers_completely_after_the_period(fused):
    from neural_assemblies.core._torch_ops import torch_ops
    a = _area()
    a.set_lri(3, 5.0)
    first = torch_ops.arange(K, device="cuda").expand(len(SEEDS), K).contiguous()
    a._push_lri(first)
    for step in range(1, 4):
        a._push_lri(first + K * step)
    drive = torch_ops.zeros(len(SEEDS), N, device="cuda")
    assert float(a.apply_lri(drive)[0, 0]) == 0.0, (
        "older than `period` is forgotten entirely, not merely small")


def test_firing_twice_inside_the_window_pays_twice(fused):
    """The reference subtracts per SLOT, so a repeat accumulates."""
    from neural_assemblies.core._torch_ops import torch_ops
    a = _area()
    a.set_lri(4, 10.0)
    same = torch_ops.arange(K, device="cuda").expand(len(SEEDS), K).contiguous()
    a._push_lri(same)
    a._push_lri(same)
    drive = torch_ops.zeros(len(SEEDS), N, device="cuda")
    # ages 1 and 2 -> 10.0 + 7.5
    assert float(a.apply_lri(drive)[0, 0]) == pytest.approx(-17.5)


def test_the_history_is_per_brain(fused):
    from neural_assemblies.core._torch_ops import torch_ops
    a = _area()
    a.set_lri(2, 4.0)
    w = torch_ops.zeros(len(SEEDS), K, dtype=torch_ops.int64, device="cuda")
    w[0, :] = torch_ops.arange(K, device="cuda")
    w[1, :] = torch_ops.arange(K, device="cuda") + 100
    a._push_lri(w)
    got = a.apply_lri(torch_ops.zeros(len(SEEDS), N, device="cuda"))
    assert float(got[0, 0]) == pytest.approx(-4.0)
    assert float(got[0, 100]) == 0.0, "brain 0 is not penalised for brain 1"
    assert float(got[1, 100]) == pytest.approx(-4.0)
    assert float(got[1, 0]) == 0.0


def test_lri_and_refraction_compose_rather_than_replace(fused):
    """Orthogonal mechanisms: both subtract, neither cancels the other."""
    from neural_assemblies.core._torch_ops import torch_ops
    a = _area(refracted=0.5)
    a.set_lri(2, 3.0)
    assert a.bias is not None
    a.bias[:, 0] = 7.0
    w = torch_ops.arange(K, device="cuda").expand(len(SEEDS), K).contiguous()
    a._push_lri(w)
    drive = torch_ops.zeros(len(SEEDS), N, device="cuda")
    both = a.apply_lri(a.apply_bias(drive))
    assert float(both[0, 0]) == pytest.approx(-10.0), "bias 7 plus LRI 3"


def test_a_malformed_setting_is_refused(fused):
    a = _area()
    with pytest.raises(ValueError, match="refractory_period"):
        a.set_lri(-1, 1.0)
    with pytest.raises(ValueError, match="inhibition_strength"):
        a.set_lri(3, -1.0)
    with pytest.raises(ValueError, match="refractory_period"):
        a.set_lri(True, 1.0)
