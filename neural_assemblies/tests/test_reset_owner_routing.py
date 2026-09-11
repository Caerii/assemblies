"""Area reset calls must reach the owner of the named area."""

from unittest.mock import patch

import pytest

from neural_assemblies import Brain


def test_reset_routes_explicit_area_to_dense_owner():
    brain = Brain(engine="numpy_sparse", norm_init=False)
    brain.add_area("A", 20, 2, 0.1, explicit=True)
    owner = brain._engine_for(brain.areas["A"])
    assert owner is not brain._engine
    with patch.object(owner, "reset_area_connections") as reset:
        brain.reset_area_connections("A")
    reset.assert_called_once_with("A")


def test_reset_rejects_unknown_area_before_dispatch():
    brain = Brain(engine="numpy_sparse", norm_init=False)
    with pytest.raises(KeyError, match="unknown area"):
        brain.reset_area_connections("missing")


def test_stimulus_beta_updates_use_explicit_target_owner():
    brain = Brain(engine="numpy_sparse", norm_init=False)
    brain.add_stimulus("s", 2)
    brain.add_area("A", 20, 2, 0.1, explicit=True)
    owner = brain._engine_for(brain.areas["A"])
    with patch.object(owner, "set_beta") as set_beta, \
            patch.object(brain._engine, "set_beta") as primary_set_beta:
        brain.update_plasticities(stim_update_map={"A": [("s", 0.2)]})
    set_beta.assert_called_once_with("A", "s", 0.2)
    primary_set_beta.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"preserve_mapping": 1}, {"reset_count": None}])
def test_population_cursor_rejects_non_boolean_protocol_switches(kwargs):
    brain = Brain(engine="numpy_sparse", norm_init=False)
    brain.add_area("A", 20, 2, 0.1)
    with pytest.raises(TypeError, match="must be boolean"):
        brain.reset_area_population_cursor("A", **kwargs)
