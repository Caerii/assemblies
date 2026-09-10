"""Exact-engine masks preserve learned drive without advancing masked exponents."""
import copy

import numpy as np
import pytest

from neural_assemblies import Brain


def weights(engine, source):
    rows = np.arange(32, dtype=np.int64)
    block = engine._fiber_rows(source, "T", rows, 32)
    pot = engine._area_pot.get((source, "T"))
    if pot is not None:
        pot.apply_to(block, rows, engine.get_beta("T", source), engine.w_max)
    scale = engine._area_norm(source, "T")
    return block if scale is None else block * scale


@pytest.mark.parametrize("norm_init", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_mask_preserves_learned_drive_and_other_fiber_learning(norm_init, fixed):
    brain = Brain(engine="numpy_exact", p=.3, seed=97, norm_init=norm_init)
    for name in ("A", "B", "T"):
        brain.add_area(name, 32, 4, .5)
    brain.areas["A"].winners = [0, 1, 2, 3]
    brain.areas["B"].winners = [4, 5, 6, 7]
    routes = {"A": ["T"], "B": ["T"]}
    brain.project({}, routes)
    if fixed:
        brain.areas["T"].fix_assembly()
    reference = brain.clone()
    reference.disable_plasticity = True
    reference.project({}, routes)
    if not fixed:
        broken = reference.clone()
        broken._engine.set_beta("T", "A", 0)
        broken.project({}, routes)
        assert reference.last_activation_scores["T"] > broken.last_activation_scores["T"]
    a, b = weights(brain._engine, "A"), weights(brain._engine, "B")
    brain.set_fiber_plasticity("A", "T", False)
    brain.project({}, routes)
    np.testing.assert_array_equal(brain.areas["T"].winners, reference.areas["T"].winners)
    assert brain.last_activation_scores == reference.last_activation_scores
    np.testing.assert_array_equal(weights(brain._engine, "A"), a)
    assert np.any(weights(brain._engine, "B") > b)
    assert brain._engine.get_beta("T", "A") == .5
    brain.set_fiber_plasticity("A", "T", True)
    brain.project({}, routes)
    assert np.any(weights(brain._engine, "A") > a)


@pytest.mark.parametrize("norm_init", [False, True])
def test_masked_stimulus_retains_existing_potentiation(norm_init):
    brain = Brain(engine="numpy_exact", p=.3, seed=97, norm_init=norm_init)
    brain.add_stimulus("stim", 12)
    brain.add_area("T", 32, 4, .5)
    brain.project({"stim": ["T"]}, {})
    before = brain._engine._stim_pot["stim"]["T"].copy()
    reference = copy.deepcopy(brain._engine)
    expected = reference.project_into("T", ["stim"], [], plasticity_enabled=False)
    broken = copy.deepcopy(reference)
    broken.set_beta("T", "stim", 0)
    wrong = broken.project_into("T", ["stim"], [], plasticity_enabled=False)
    assert expected.total_activation > wrong.total_activation
    brain.set_fiber_plasticity("stim", "T", False)
    brain.project({"stim": ["T"]}, {})
    np.testing.assert_array_equal(brain._engine._stim_pot["stim"]["T"], before)
    np.testing.assert_array_equal(brain.areas["T"].winners, expected.winners)
    assert brain.last_activation_scores["T"] == expected.total_activation
    brain.set_fiber_plasticity("stim", "T", True)
    brain.project({"stim": ["T"]}, {})
    assert np.any(brain._engine._stim_pot["stim"]["T"] > before)
