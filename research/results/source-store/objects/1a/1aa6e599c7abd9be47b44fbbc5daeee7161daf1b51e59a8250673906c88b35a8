"""Composition routes current winners, not historical population aliases."""

from types import SimpleNamespace

from neural_assemblies.programs.colt_mnist_brain_util import (
    area_has_active_winners,
)


def test_active_source_predicate_rejects_historical_population_as_activity():
    brain = SimpleNamespace(
        areas={
            "silent": SimpleNamespace(active_count=0, w=40),
            "active": SimpleNamespace(active_count=10, w=0),
        }
    )

    assert area_has_active_winners(brain, "silent") is False
    assert area_has_active_winners(brain, "active") is True
