"""Capacity migration preserves cell identity without requiring a GPU."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from research.experiments import seq_capacity_scaling as capacity


def test_capacity_configuration_is_immutable():
    config = capacity.CapacityProtocol()
    with pytest.raises(FrozenInstanceError):
        config.beta = 0.0


@pytest.mark.parametrize("kwargs", [
    {"checkpoints": (4, 4)}, {"checkpoints": (8, 4)}, {"p": float("nan")},
    {"rounds": 0}, {"stim_size": 0}, {"recall_sample": 0},
])
def test_invalid_configuration_fails_before_gpu_import(kwargs):
    with pytest.raises(ValueError):
        capacity.CapacityProtocol(**kwargs)


def test_same_n_different_k_retains_both_cells_and_seed_identity(monkeypatch):
    calls = []
    def fake_cell(n, k, protocol, seeds, rng, *, arm_settings, device):
        calls.append((n, k, arm_settings, device, tuple(seeds)))
        return {4: {"rank1": [0.9] * 3, "pairwise_x": [1.0] * 3,
                    "distinct": [1.0] * 3, "fill": [0.1] * 3}}
    monkeypatch.setattr(capacity, "run_cell", fake_cell)
    result = capacity.experiment({
        "seeds": [7, 13, 19], "mode": "smoke",
        "parameters": {"configuration": asdict(capacity.CapacityProtocol(checkpoints=(4,))),
                       "arms": ["B"], "nk": [[100, 10], [100, 20]],
                       "measurement_seed": 1234, "half_bar": 0.5,
                       "arm_settings": {"B": {"norm_init": False, "synaptic_scaling": True}},
                       "device": "cuda:0", "distinct_gate": 3., "distinct_low_bar": .9},
    })
    assert calls == [(100, k, {"norm_init": False, "synaptic_scaling": True},
                      "cuda:0", (7, 13, 19)) for k in (10, 20)]
    assert [(c["n"], c["k"]) for c in result["cells"]] == [(100, 10), (100, 20)]
    assert result["verdict"] == "VOID"
    assert result["cells"][0]["ensembles"][4]["rank1"]["keys"] == (7, 13, 19)
    # Exceeding the grid and filling the area are different censoring mechanisms.
    assert result["cells"][0]["ceiling"]["grid_censored"]
    assert not result["cells"][0]["ceiling"]["fill_censored"]


def test_original_capacity_cli_refuses_missing_tag(monkeypatch):
    calls = []
    monkeypatch.setattr(capacity, "run_experiment", lambda **kw: calls.append(kw))
    with pytest.raises(SystemExit) as exc:
        capacity.main([])
    assert exc.value.code == 2
    assert not calls


def test_repeated_invocations_do_not_retain_previous_configuration(monkeypatch):
    records = []
    monkeypatch.setattr(capacity, "run_experiment", lambda **kw: records.append(kw))
    common = ["--tag", "fixture", "--registration", "fixture.md"]
    capacity.main(common + ["--beta", "0.3", "--arms", "B"])
    capacity.main(common)
    assert records[0]["parameters"]["configuration"]["beta"] == 0.3
    assert records[1]["parameters"]["configuration"]["beta"] == capacity.BETA
    assert records[1]["parameters"]["arms"] == ["B", "G"]


def test_recorded_distinctness_bars_change_the_measured_gate():
    cell = {"rank1": [.9]*3, "pairwise_x": [2.]*3, "distinct": [.95]*3}
    assert capacity.gated(cell, [1,2,3], distinct_gate=3., distinct_low_bar=.9)[0] == .9
    assert capacity.gated(cell, [1,2,3], distinct_gate=1., distinct_low_bar=.9)[0] == 0
    assert capacity.gated(cell, [1,2,3], distinct_gate=3., distinct_low_bar=.99)[0] == 0


@pytest.mark.parametrize("change", [
    {"distinct_gate": float("nan")}, {"distinct_low_bar": 1.1},
    {"distinct_gate": True}, {"arm_settings": {}},
    {"arm_settings": {"B": {"norm_init": 1, "synaptic_scaling": False}}},
    {"device": ""},
])
def test_invalid_recorded_execution_settings_fail_before_measurement(monkeypatch, change):
    def fail(*args, **kwargs):
        pytest.fail("invalid settings reached measurement")
    monkeypatch.setattr(capacity, "run_cell", fail)
    parameters = {"distinct_gate": 3., "distinct_low_bar": .9,
                  "arms": ["B"], "device": "cuda",
                  "arm_settings": {"B": {"norm_init": True, "synaptic_scaling": False}}}
    parameters.update(change)
    with pytest.raises(ValueError):
        capacity.experiment({"parameters": parameters})


def test_cli_records_explicit_execution_and_measurement_options(monkeypatch):
    records = []
    monkeypatch.setattr(capacity, "run_experiment", lambda **kw: records.append(kw))
    capacity.main(["--tag", "fixture", "--registration", "fixture.md",
                   "--device", "cuda:0", "--distinct-gate", "2.5", "--distinct-low-bar", ".95"])
    parameters = records[0]["parameters"]
    assert parameters["device"] == "cuda:0"
    assert parameters["distinct_gate"] == 2.5 and parameters["distinct_low_bar"] == .95
