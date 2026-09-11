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
