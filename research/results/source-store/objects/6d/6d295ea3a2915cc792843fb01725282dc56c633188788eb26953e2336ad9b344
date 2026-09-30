"""The IR must execute through Brain bookkeeping and respect learning intent."""
import numpy as np
import pytest

from neural_assemblies import Brain
from neural_assemblies.diagnostics import read_assembly
from neural_assemblies.ir.projection import ExplicitProgram, ExplicitRound


@pytest.fixture(params=[False, True])
def brain(request):
    brain = Brain(p=.1, seed=41, engine="numpy_explicit" if not request.param else "numpy_sparse",
                  norm_init=False, save_winners=True)
    for name in ("S", "T"):
        brain.add_area(name, 4, 2, .5, explicit=request.param)
    brain.areas["S"].winners = [0, 1]
    brain.connectomes["S"]["T"].weights[:] = 0
    brain.connectomes["S"]["T"].weights[0] = [1, 4, 3, 2]
    return brain


@pytest.mark.parametrize("learning", [False, True])
def test_round_synchronizes_and_matches_ordinary_projection(brain, learning):
    direct = brain.clone()
    direct.disable_plasticity = not learning
    direct.project({}, {"S": ["T"]})
    winners = ExplicitRound("T", ["S"], learning).execute_on_brain(brain)
    assert set(winners) == {1, 2}
    np.testing.assert_array_equal(winners, direct.areas["T"].winners)
    np.testing.assert_array_equal(brain.connectomes["S"]["T"].weights,
                                  direct.connectomes["S"]["T"].weights)
    np.testing.assert_array_equal(brain.areas["T"].saved_winners[-1], winners)
    np.testing.assert_array_equal(brain._engine_for(brain.areas["T"]).get_winners("T"), winners)
    assert brain.areas["T"].saved_w == direct.areas["T"].saved_w
    assert brain.last_activation_scores == direct.last_activation_scores
    assert not brain.disable_plasticity
    winners[:] = 0
    assert set(read_assembly(brain, "T")) == {1, 2}


def test_external_only_round(brain):
    winners = ExplicitRound("T", [], False, [0, 0, 10, 9]).execute_on_brain(brain)
    assert list(winners) == [2, 3]
    np.testing.assert_array_equal(brain.areas["T"].saved_winners[-1], winners)


def test_program_preflights_all_brain_rounds_before_mutation(brain):
    program = ExplicitProgram((
        ExplicitRound("T", ["S"], True),
        ExplicitRound("T", ["missing"], False),
    ))
    before_weights = brain.connectomes["S"]["T"].weights.copy()
    before_winners = np.asarray(brain.areas["T"].winners).copy()
    with pytest.raises(ValueError, match="unregistered Brain area"):
        program.execute_on_brain(brain)
    np.testing.assert_array_equal(brain.connectomes["S"]["T"].weights, before_weights)
    np.testing.assert_array_equal(brain.areas["T"].winners, before_winners)
    assert not brain.areas["T"].saved_winners


@pytest.mark.parametrize("fault", ["clamp", "frozen", "indices", "mask"])
def test_incompatible_brain_state_rejected_before_learning(brain, fault):
    if fault == "clamp":
        brain.areas["T"].fixed_assembly = True
    elif fault == "frozen":
        brain.disable_plasticity = True
    elif fault == "indices":
        brain.areas["S"]._winners = np.array([.5])
    else:
        brain.plasticity_mask[("S", "T")] = False
    before = brain.connectomes["S"]["T"].weights.copy()
    with pytest.raises(ValueError):
        ExplicitRound("T", ["S"], True).execute_on_brain(brain)
    np.testing.assert_array_equal(before, brain.connectomes["S"]["T"].weights)
    assert not brain.areas["T"].saved_winners


def test_learning_flag_restored_after_backend_rejection(brain):
    with pytest.raises(ValueError):
        ExplicitRound("T", ["S"], False, [1]).execute_on_brain(brain)
    assert not brain.disable_plasticity


def test_public_drive_only_call_uses_the_same_schedule(brain):
    brain.project(external_drive={"T": [0, 0, 10, 9]})
    assert list(brain.areas["T"].winners) == [2, 3]
    assert len(brain.areas["T"].saved_winners) == 1


def test_inhibited_target_cannot_be_driven_through_public_api(brain):
    brain.inhibit_area("T")
    with pytest.raises(ValueError, match="inhibited"):
        brain.project(external_drive={"T": [0, 0, 10, 9]})
    assert not brain.areas["T"].saved_winners


def test_recurrent_brain_round_reads_the_last_public_cap(brain):
    brain.areas["T"].winners = [0]
    brain.connectomes["T"]["T"].weights[:] = [
        [0, 5, 4, 0], [6, 0, 0, 5], [6, 0, 0, 5], [0, 0, 0, 0]]
    instruction = ExplicitRound("T", ["T"], False)
    assert set(instruction.execute_on_brain(brain)) == {1, 2}
    assert set(instruction.execute_on_brain(brain)) == {0, 3}
    assert len(brain.areas["T"].saved_winners) == 2


@pytest.mark.parametrize("drive", [[1], [1e100, 0, 0, 0]])
def test_bad_round_preserves_unsynchronized_engine_state(brain, drive):
    engine = brain._engine_for(brain.areas["S"])
    engine.set_winners("S", np.array([3], dtype=np.uint32))
    before = engine.get_winners("S").copy()
    weights = brain.connectomes["S"]["T"].weights.copy()
    with pytest.raises(ValueError):
        ExplicitRound("T", ["S"], False, drive).execute_on_brain(brain)
    np.testing.assert_array_equal(engine.get_winners("S"), before)
    np.testing.assert_array_equal(brain.connectomes["S"]["T"].weights, weights)
    assert not brain.areas["T"].saved_winners
    assert not brain.disable_plasticity
