"""Coin activity must preserve identities and dispatch to the area owner."""
from types import SimpleNamespace

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.pfa import RandomChoiceArea, _seed_winners
from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.core.brain import Brain


def fixture_brain():
    area = SimpleNamespace(n=10, explicit=False, winners=np.array([1], dtype=np.uint32))
    class Owner:
        def get_neuron_id_mapping(self, name):
            return self.mapping
        def set_winners(self, name, winners):
            self.winners = winners.copy()
    owner = Owner()
    owner.mapping = [8, 3, 7]
    owner.winners = area.winners.copy()
    class WrongEngine:
        def set_winners(self, *args):
            raise AssertionError("primary engine does not own this area")
    brain = SimpleNamespace(areas={"coin": area}, _engine=WrongEngine(),
                            _engine_for=lambda area: owner)
    return brain, area, owner


def test_seed_preserves_membership_and_uses_owner():
    brain, area, owner = fixture_brain()
    compact = _seed_winners(brain, "coin", [7, 8])
    np.testing.assert_array_equal(compact, [2, 0])
    np.testing.assert_array_equal(owner.winners, [2, 0])
    np.testing.assert_array_equal(area.winners, [2, 0])
    compact[0] = 1
    assert area.winners[0] == 2


@pytest.mark.parametrize("values", [[8, 9], [-1], [1.5], [[8]], [10]])
def test_invalid_seed_does_not_partially_activate(values):
    brain, area, owner = fixture_brain()
    with pytest.raises(ValueError):
        _seed_winners(brain, "coin", values)
    np.testing.assert_array_equal(area.winners, [1])
    np.testing.assert_array_equal(owner.winners, [1])


def test_legacy_index_bypass_is_rejected_before_mutation():
    brain, area, owner = fixture_brain()
    with pytest.raises(ValueError, match="Legacy coin seeding"):
        _seed_winners(brain, "coin", [8], remap=False)
    np.testing.assert_array_equal(area.winners, [1])
    np.testing.assert_array_equal(owner.winners, [1])



def bare_coin(brain):
    coin = RandomChoiceArea.__new__(RandomChoiceArea)
    coin.brain, coin.area_name, coin.k, coin.n = brain, "coin", 2, 10
    coin.construction = "attractor"
    return coin


@pytest.mark.parametrize("settings", [
    {"mode": "typo"}, {"bias": -0.1}, {"bias": 1.1}, {"bias": float("nan")},
    {"bias": float("inf")}, {"bias": True}, {"bias": "0.5"},
    {"rounds": -1}, {"rounds": 1.5}, {"rounds": True},
])
def test_bad_flip_options_fail_before_activity(settings):
    brain, area, owner = fixture_brain()
    with pytest.raises(ValueError):
        bare_coin(brain).flip(**settings)
    np.testing.assert_array_equal(area.winners, [1])
    np.testing.assert_array_equal(owner.winners, [1])


@pytest.mark.parametrize("settings", [
    {"construction": "typo"}, {"rounds_train": 0}, {"rounds_train": True},
    {"rounds_train": 2.5}, {"fires": -1}, {"fires": True}, {"fires": 2.5},
])
def test_bad_construction_controls_fail_before_registration(settings):
    # No brain methods exist: attempting registration before validation fails.
    with pytest.raises(ValueError):
        RandomChoiceArea(object(), **settings)


def test_uniform_seed_uses_owner_and_refuses_incomplete_population():
    brain, area, owner = fixture_brain()
    coin = bare_coin(brain)
    owner.materialized_count = lambda name: 3
    rng = np.random.default_rng(44)
    prior = repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match="complete materialized"):
        coin._seed_uniform(rng)
    assert repr(rng.bit_generator.state) == prior
    np.testing.assert_array_equal(area.winners, [1])
    np.testing.assert_array_equal(owner.winners, [1])
    owner.materialized_count = lambda name: 10
    coin._seed_uniform(rng)
    np.testing.assert_array_equal(owner.winners, area.winners)
    assert len(np.unique(area.winners)) == 2


@pytest.mark.parametrize("initial_fixed", [False, True])
def test_force_fire_restores_fixation_on_failure(initial_fixed):
    brain = Brain(p=.1, seed=1, engine="numpy_sparse")
    brain.add_area("coin", 10, 2, .1)
    coin = bare_coin(brain)
    coin.asm0 = Assembly("coin", [0, 1])
    coin.asm1 = Assembly("coin", [2, 3])
    area = brain.areas["coin"]
    area.fixed_assembly = initial_fixed
    def fail(*args, **kwargs):
        assert area.fixed_assembly
        raise RuntimeError("injected projection failure")
    brain.project = fail
    with pytest.raises(RuntimeError, match="injected projection"):
        coin._build_attractor(1)
    assert area.fixed_assembly is initial_fixed
