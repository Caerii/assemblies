"""Coin activity must preserve identities and dispatch to the area owner."""
from types import SimpleNamespace

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.pfa import _seed_winners


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
