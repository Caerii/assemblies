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


def _grown_pair(engine):
    """Two areas grown by their own stimuli; the B -> A fiber is never used."""
    brain = Brain(engine=engine, p=0.1, seed=5, norm_init=False)
    brain.add_stimulus("S", 20)
    brain.add_stimulus("T", 20)
    brain.add_area("A", 1000, 20, beta=0.1)
    brain.add_area("B", 1000, 20, beta=0.1)
    for _ in range(5):
        brain.project({"S": ["A"], "T": ["B"]}, {})
    return brain


def test_zero_signal_projection_is_a_measured_zero_not_zero_candidates():
    # An untrained cross-area fiber under a probe delivers exactly zero drive,
    # and the sparse engine takes its preserve-the-assembly shortcut. That
    # branch used to leave the count at its default, i.e. ZERO CANDIDATES; the
    # typed observation rejected it, and the ERP adapter read the rejection as
    # its legacy 0.0 deficit -- a PERFECT parse -- on every category
    # violation, so the P600 AUC read exactly 0.000.
    brain = _grown_pair("numpy_sparse")
    materialized = brain.population_counts("A").materialized

    drive = input_drive(brain, sources=["B"], target_areas=["A"])["A"]
    observation = brain.pre_kwta_observation("A")

    assert observation == PreKwtaObservation(total=0.0, candidate_count=materialized)
    assert drive == 0.0


@pytest.mark.requires_torch
def test_torch_reports_the_candidate_count_it_summed_over():
    # The torch engine recorded totals but never counts, so every one of its
    # observations read as zero candidates and was rejected.
    brain = _grown_pair("torch_sparse")

    drive = input_drive(brain, sources=["B"], target_areas=["A"])["A"]
    observation = brain.pre_kwta_observation("A")

    assert observation is not None
    assert observation.candidate_count > 0
    assert drive == observation.mean


def test_a_projection_that_sums_nothing_records_no_observation():
    # A FIXED target returns before its inputs are summed. Brain used to write
    # ProjectionResult's defaults down anyway -- 0.0 over zero candidates, a
    # malformed record -- where the contract says there is no observation.
    brain = _grown_pair("numpy_sparse")
    brain.areas["A"].fix_assembly()
    brain.record_activation = True
    try:
        brain.project({"S": ["A"]}, {})
    finally:
        brain.record_activation = False
    assert brain.pre_kwta_observation("A") is None
