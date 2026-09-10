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
    def fake_cell(n, k, arm, protocol, seeds, rng):
        calls.append((n, k, arm, tuple(seeds)))
        return {4: {"rank1": [0.9] * 3, "pairwise_x": [1.0] * 3,
                    "distinct": [1.0] * 3, "fill": [0.1] * 3}}
    monkeypatch.setattr(capacity, "run_cell", fake_cell)
    result = capacity.experiment({
        "seeds": [7, 13, 19], "mode": "smoke",
        "parameters": {"configuration": asdict(capacity.CapacityProtocol(checkpoints=(4,))),
                       "arms": ["B"], "nk": [[100, 10], [100, 20]],
                       "measurement_seed": 1234, "half_bar": 0.5},
    })
    assert calls == [(100, 10, "B", (7, 13, 19)), (100, 20, "B", (7, 13, 19))]
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
