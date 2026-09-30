"""A pre-k-WTA sum and its candidate count form one observation."""

import math

import pytest

from neural_assemblies import Brain, PreKwtaObservation
from neural_assemblies.assembly_calculus.binding import input_drive
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    _self_recurrent_energy,
)


def _brain_with_area():
    brain = Brain(engine="numpy_exact", p=0.2, seed=17, norm_init=False)
    brain.add_area("A", 100, 10, beta=0.1)
    return brain


def test_observation_is_absent_only_when_both_components_are_absent():
    brain = _brain_with_area()
    assert brain.pre_kwta_observation("A") is None

    brain.last_pre_kwta_totals["A"] = 12.0
    with pytest.raises(RuntimeError, match="incomplete pre-k-WTA observation"):
        brain.pre_kwta_observation("A")

    brain.last_pre_kwta_totals.clear()
    brain.last_pre_kwta_counts["A"] = 3
    with pytest.raises(RuntimeError, match="incomplete pre-k-WTA observation"):
        brain.pre_kwta_observation("A")


def test_observation_mean_uses_its_paired_count_not_area_state():
    brain = _brain_with_area()
    brain.last_pre_kwta_totals["A"] = 12.0
    brain.last_pre_kwta_counts["A"] = 3

    observed = brain.pre_kwta_observation("A")
    assert observed == PreKwtaObservation(total=12.0, candidate_count=3)
    assert observed.mean == 4.0

    brain.areas["A"].w = 99
    assert brain.pre_kwta_observation("A").mean == 4.0


@pytest.mark.parametrize(
    "total,count,error",
    [
        (math.nan, 2, "total must be finite"),
        (math.inf, 2, "total must be finite"),
        (True, 2, "total must be a finite real number"),
        ("1.0", 2, "total must be a finite real number"),
        (1.0, 0, "candidate_count must be a positive integer"),
        (1.0, -1, "candidate_count must be a positive integer"),
        (1.0, True, "candidate_count must be a positive integer"),
    ],
)
def test_observation_rejects_values_that_cannot_define_a_mean(total, count, error):
    with pytest.raises(ValueError, match=error):
        PreKwtaObservation(total=total, candidate_count=count)


def test_unknown_area_rejects_before_observation_lookup():
    with pytest.raises(KeyError, match="unknown area 'missing'"):
        _brain_with_area().pre_kwta_observation("missing")


def test_input_drive_uses_the_recorded_candidate_count():
    brain = Brain(engine="numpy_exact", p=0.2, seed=19, norm_init=False)
    brain.add_stimulus("S", 10)
    brain.add_area("SRC", 100, 10, beta=0.1)
    brain.add_area("DST", 100, 10, beta=0.1)
    brain.project({"S": ["SRC"]}, {})

    score = input_drive(brain, sources=["SRC"], target_areas=["DST"])["DST"]
    observation = brain.pre_kwta_observation("DST")

    assert observation is not None
    assert observation.candidate_count == 100
    assert brain.areas["DST"].w != observation.candidate_count
    assert score == observation.mean


def test_self_recurrent_energy_uses_the_same_observation_contract():
    brain = Brain(engine="numpy_exact", p=0.2, seed=23, norm_init=False)
    brain.add_stimulus("S", 10)
    brain.add_area("A", 100, 10, beta=0.1)
    brain.project({"S": ["A"]}, {})

    energy = _self_recurrent_energy(brain, "A")
    observation = brain.pre_kwta_observation("A")

    assert energy.defined
    assert observation is not None
    assert observation.candidate_count == 100
    assert brain.areas["A"].w != observation.candidate_count
    assert float(energy) == observation.mean
