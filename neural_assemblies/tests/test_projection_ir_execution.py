"""Hand-computed outcomes and constructed failures for executable projection IR."""
import copy
import json

import numpy as np
import pytest

from neural_assemblies.core.numpy_engine import NumpyExplicitEngine
from neural_assemblies.ir import ExplicitRound, validate_explicit_round_document
from neural_assemblies.ir.protocol import schema_path


EXPLICIT_ROUND_CASES = json.loads(
    schema_path("explicit-round.cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", EXPLICIT_ROUND_CASES, ids=lambda case: case["name"])
def test_shared_explicit_round_wire(case):
    if "raw_json" in case:
        with pytest.raises(ValueError):
            ExplicitRound.from_document(json.loads(case["raw_json"]))
        return
    document = case["document"]
    assert (not validate_explicit_round_document(document)) == case["valid"]
    if case["valid"]:
        assert ExplicitRound.from_document(document).to_document() == document
    else:
        with pytest.raises(ValueError):
            ExplicitRound.from_document(document)


@pytest.fixture
def engine():
    engine = NumpyExplicitEngine(p=0.1, seed=17, w_max=20)
    engine.add_area("S", n=3, k=1, beta=1)
    engine.add_area("T", n=4, k=2, beta=1)
    engine.set_winners("S", np.array([1], dtype=np.uint32))
    weights = engine._area_conns["S"]["T"].weights
    weights[:] = 0
    weights[1] = [1, 4, 3, 2]
    return engine


@pytest.mark.parametrize("plasticity", [True, False])
def test_hand_computed_round_and_learning_null(engine, plasticity):
    instruction = ExplicitRound("T", ["S"], plasticity)
    decoded = ExplicitRound.from_document(instruction.to_document())
    assert decoded == instruction
    result = decoded.execute(engine)
    assert set(result.winners) == {1, 2}
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1] = [1, 8, 6, 2] if plasticity else [1, 4, 3, 2]
    np.testing.assert_array_equal(engine._area_conns["S"]["T"].weights, expected)
    assert result.total_activation == 7


def test_dead_fiber_changes_observation(engine):
    engine._area_conns["S"]["T"].weights[:] = 0
    result = ExplicitRound("T", ["S"], False, [9, 0, 0, 8]).execute(engine)
    assert set(result.winners) == {0, 3}
    # The intact circuit overcomes this bias; a dead probe cannot pass both.
    engine._area_conns["S"]["T"].weights[1] = [0, 20, 19, 0]
    result = ExplicitRound("T", ["S"], False, [9, 0, 0, 8]).execute(engine)
    assert set(result.winners) == {1, 2}


def test_recurrent_round_reads_previous_cap(engine):
    engine.set_winners("T", np.array([0], dtype=np.uint32))
    weights = engine._area_conns["T"]["T"].weights
    weights[:] = [[0, 5, 4, 0], [6, 0, 0, 5], [6, 0, 0, 5], [0, 0, 0, 0]]
    instruction = ExplicitRound("T", ["T"], False)
    assert set(instruction.execute(engine).winners) == {1, 2}
    assert set(instruction.execute(engine).winners) == {0, 3}


@pytest.mark.parametrize("change", [
    {"from_areas": ["S", "S"]}, {"from_areas": "S"},
    {"external_drive": [float("nan")]}, {"plasticity": 1},
    {"column_renorm": ["S"]}, {"profile": "numpy_exact"},
])
def test_bad_wire_is_rejected(change):
    document = ExplicitRound("T", ["S"], True).to_document()
    document.update(change)
    with pytest.raises(ValueError):
        ExplicitRound.from_document(document)


@pytest.mark.parametrize("fault", ["drive", "missing", "clamp", "indices", "duplicate",
                                    "weights", "beta", "plasticity", "overflow"])
def test_rejection_before_mutation(engine, fault):
    instruction = ExplicitRound("T", ["S"], True)
    if fault == "drive":
        instruction = ExplicitRound("T", ["S"], True, [1])
    elif fault == "missing":
        instruction = ExplicitRound("absent", ["S"], True)
    elif fault == "clamp":
        engine._areas["T"].fixed_assembly = True
    elif fault == "indices":
        engine._areas["S"].winners = np.array([-1])
    elif fault == "duplicate":
        engine._areas["S"].winners = np.array([1, 1])
    elif fault == "weights":
        engine._area_conns["S"]["T"].weights[0, 0] = np.nan
    elif fault == "beta":
        engine.set_beta("T", "S", float("nan"))
    elif fault == "plasticity":
        engine._plasticity_enabled_global = False
    elif fault == "overflow":
        instruction = ExplicitRound("T", ["S"], True, [1e100] * 4)
    weights = engine._area_conns["S"]["T"].weights.copy()
    winners = engine.get_winners("T")
    rng = copy.deepcopy(engine._rng.bit_generator.state)
    with pytest.raises(ValueError):
        instruction.execute(engine)
    np.testing.assert_array_equal(weights, engine._area_conns["S"]["T"].weights)
    np.testing.assert_array_equal(winners, engine.get_winners("T"))
    assert rng == engine._rng.bit_generator.state


def test_lowering_matches_direct_backend_state(engine):
    direct = copy.deepcopy(engine)
    instruction = ExplicitRound("T", ["S"], True, [0, 0, 0, 5])
    actual = instruction.execute(engine)
    expected = direct.project_into("T", [], ["S"], plasticity_enabled=True,
                                   external_drive=np.array([0, 0, 0, 5], dtype=np.float32))
    np.testing.assert_array_equal(actual.winners, expected.winners)
    np.testing.assert_array_equal(engine._areas["T"].ever_fired, direct._areas["T"].ever_fired)
    np.testing.assert_array_equal(engine._area_conns["S"]["T"].weights,
                                  direct._area_conns["S"]["T"].weights)
    assert engine._rng.bit_generator.state == direct._rng.bit_generator.state


def test_ties_and_clip_are_observable(engine):
    engine._area_conns["S"]["T"].weights[1] = [15, 15, 15, 15]
    result = ExplicitRound("T", ["S"], True).execute(engine)
    assert list(result.winners) == [0, 1]
    np.testing.assert_array_equal(engine._area_conns["S"]["T"].weights[1], [20, 20, 15, 15])


def test_instruction_copies_mutable_configuration():
    sources, drive = ["S"], [1, 2, 3, 4]
    instruction = ExplicitRound("T", sources, False, drive)
    sources.append("T")
    drive[0] = 99
    assert instruction.from_areas == ("S",)
    assert instruction.external_drive == (1, 2, 3, 4)


@pytest.mark.parametrize("fault", ["length", "overflow", "source_indices", "target_indices"])
def test_validate_rejects_numerically_invalid_round_without_execution(engine, fault):
    drive = [1] if fault == "length" else [1e100, 0, 0, 0] if fault == "overflow" else []
    if fault.endswith("indices"):
        name = "S" if fault == "source_indices" else "T"
        engine._areas[name].winners = np.array([.5])
    with pytest.raises(ValueError):
        ExplicitRound("T", ["S"], False, drive).validate(engine)
