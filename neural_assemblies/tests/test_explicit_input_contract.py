"""Legacy explicit calls must reject malformed inputs before numerical mutation."""
import copy

import numpy as np
import pytest

from neural_assemblies.core.numpy_engine import NumpyExplicitEngine


@pytest.fixture
def engine():
    engine = NumpyExplicitEngine(.1, seed=23)
    for name in ("S", "T"):
        engine.add_area(name, 4, 2, .2)
    engine.set_winners("S", np.array([0, 1], dtype=np.uint32))
    return engine


@pytest.mark.parametrize("drive", [[1], [[1, 2, 3, 4]], [0, 0, np.nan, 0],
                                   [1e100] * 4, [0, 0, np.inf, 0]])
@pytest.mark.parametrize("fixed", [False, True])
def test_invalid_drive_rejected_even_for_clamped_target(engine, drive, fixed):
    engine._areas["T"].fixed_assembly = fixed
    before = copy.deepcopy(engine)
    with pytest.raises(ValueError):
        engine.project_into("T", [], ["S"], external_drive=drive)
    np.testing.assert_array_equal(engine._area_conns["S"]["T"].weights,
                                  before._area_conns["S"]["T"].weights)
    np.testing.assert_array_equal(engine.get_winners("T"), before.get_winners("T"))
    assert engine._rng.bit_generator.state == before._rng.bit_generator.state


@pytest.mark.parametrize("winners", [[-1], [.5], [2**32], [[0]], [4], [0, 0]])
def test_invalid_winner_assignment_is_atomic(engine, winners):
    before = engine.get_winners("S")
    count = engine._areas["S"].w
    with pytest.raises(ValueError):
        engine.set_winners("S", winners)
    np.testing.assert_array_equal(engine.get_winners("S"), before)
    assert engine._areas["S"].w == count


@pytest.mark.parametrize("winners", [[-1], [.5], [4], [0, 0]])
def test_mutated_winner_buffer_cannot_bypass_projection_validation(engine, winners):
    engine._areas["S"].winners = np.array(winners)
    before = engine._area_conns["S"]["T"].weights.copy()
    with pytest.raises(ValueError):
        engine.project_into("T", [], ["S"])
    np.testing.assert_array_equal(engine._area_conns["S"]["T"].weights, before)


@pytest.mark.parametrize("sources", [["S", "S"], ["missing"]])
def test_bad_source_list_rejected(engine, sources):
    with pytest.raises(ValueError):
        engine.project_into("T", [], sources)


def test_valid_signed_additive_drive_and_empty_source(engine):
    engine.set_winners("S", [])
    result = engine.project_into("T", [], ["S"], external_drive=[-1, 4, 3, -2])
    assert list(result.winners) == [1, 2]


@pytest.mark.parametrize("targets", [("T",), ("T", "U")])
def test_brain_primary_explicit_engine_receives_drive(targets):
    from neural_assemblies import Brain
    brain = Brain(p=.1, seed=23, engine="numpy_explicit", norm_init=False)
    brain.add_area("S", 4, 2, .2)
    brain.areas["S"].winners = [0, 1]
    for target in targets:
        brain.add_area(target, 4, 2, .2)
        brain.connectomes["S"][target].weights[:] = 0
    brain.project({}, {"S": list(targets)},
                  external_drive={name: np.array([0, 0, 10, 9]) for name in targets})
    for target in targets:
        assert list(brain.areas[target].winners) == [2, 3]


@pytest.mark.parametrize("engine_name,drive_target", [("numpy_sparse", "T"),
                                                       ("numpy_explicit", "missing")])
def test_brain_rejects_unsupported_or_unscheduled_drive(engine_name, drive_target):
    from neural_assemblies import Brain
    brain = Brain(p=.1, seed=23, engine=engine_name, norm_init=False)
    brain.add_area("S", 4, 2, .2)
    brain.add_area("T", 4, 2, .2)
    brain.areas["S"].winners = [0, 1]
    with pytest.raises(ValueError):
        brain.project({}, {"S": ["T"]}, external_drive={drive_target: [0, 0, 10, 9]})
