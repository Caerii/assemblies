"""Source-f0a0de8 replay and descriptive threshold controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from research.experiments.stability import test_phase_diagram as study
from research.experiments.base import summarize

CASES = [json.loads(line) for line in (Path(__file__).parent / "data/historical_phase_trials.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("expected", CASES, ids=lambda row: f"{row['seed']}-{row['beta']}")
def test_phase_retains_every_projection_winner_and_weight(monkeypatch, expected):
    brains = []
    class Trace(study.Brain):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = []
            brains.append(self)
        def project(self, *args, **kwargs):
            result = super().project(*args, **kwargs)
            self.trace.append(dict(args=list(args), kwargs=kwargs,
                                   winners={name: area.winners.tolist() for name, area in self.areas.items()}))
            return result
    monkeypatch.setattr(study, "Brain", Trace)
    result = study.run_phase_trial(study.PhaseConfig(60, 6, .2, expected["beta"], 20., train_rounds=3, test_rounds=3), expected["seed"])
    brain = brains[0]
    owner = brain._engine_for(brain.areas["A"])
    weights = {}
    for kind in ("_area_conns", "_stim_conns"):
        for source, targets in getattr(owner, kind).items():
            for target, conn in targets.items():
                weights[f"{kind}/{source}/{target}"] = hashlib.sha256(np.ascontiguousarray(conn.weights).tobytes()).hexdigest()
    assert result == expected["result"]
    assert brain.trace == expected["trace"]
    assert weights == expected["weights"]
    assert (expected["engine"], expected["owner"]) == ("numpy_sparse", "numpy_explicit")
    assert brain._engine.name == owner.name == "numpy_explicit"
    assert owner is brain._engine


def test_mean_above_threshold_does_not_become_resolved_stability():
    summary = summarize([1., 1., .9])
    assert summary["mean"] > .95
    assert study.persistence_interval_status(summary) == "unresolved"
    assert study.persistence_interval_status(summarize([.1, .2, .3])) == "below_threshold"
    assert study.persistence_interval_status(summarize([.99, .99, .99])) == "above_threshold"


def test_sampled_crossing_uses_lowest_beta_and_keeps_absence():
    rows = [dict(sparsity=.1, beta=.2, interval_status="above_threshold"),
            dict(sparsity=.2, beta=.1, interval_status="unresolved"),
            dict(sparsity=.1, beta=.1, interval_status="above_threshold")]
    result = study.sampled_threshold_crossings(rows)
    assert result["0.1"] == {"beta": .1, "status": "observed_in_sampled_grid"}
    assert result["0.2"] == {"beta": None, "status": "not_observed"}


def test_all_phase_cells_and_seed_values_are_retained(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "run_phase_trial", lambda cfg, seed: (seed-42)/10)
    result = study.PhaseDiagramExperiment(results_dir=tmp_path, verbose=False).run(n_seeds=3)
    assert result.raw_data["seeds"] == [42, 43, 44]
    assert len(result.raw_data["cells"]) == 40
    assert all(cell["values"] == [0., .1, .2] for cell in result.raw_data["cells"])
    assert "phase_boundary" not in result.metrics
    assert len(result.metrics["sampled_threshold_crossings"]) == 7
    assert all(row["beta"] is None for row in result.metrics["sampled_threshold_crossings"].values())


@pytest.mark.parametrize("kwargs", [{"n": 60}, {"n_seeds": 2}])
def test_invalid_phase_grid_fails_before_timer(kwargs):
    experiment = object.__new__(study.PhaseDiagramExperiment)
    experiment.seed = 42
    with pytest.raises(ValueError):
        experiment.run(**kwargs)


def test_resolved_grid_and_schedules_are_consumed_in_order(monkeypatch, tmp_path):
    calls = []
    def trial(cfg, seed):
        calls.append((cfg, seed))
        return seed / 10
    monkeypatch.setattr(study, "run_phase_trial", trial)
    result = study.PhaseDiagramExperiment(seed=999, results_dir=tmp_path, verbose=False).run(
        n=60, seed_ids=[9, 2, 7], sparsities=[.21, .1], betas=[.2, 0.],
        p_values=[.3, .1], p_effect_k=5, p_effect_beta=.3,
        train_rounds=3, test_rounds=4, initial_stimulus_rounds=2, persistence_threshold=.8)
    assert [seed for _, seed in calls] == [9, 2, 7]*6
    assert [(cfg.k, cfg.beta) for cfg, _ in calls[::3]] == [(12,.2),(12,0.),(6,.2),(6,0.),(5,.3),(5,.3)]
    for cfg, _ in calls:
        assert (cfg.train_rounds, cfg.test_rounds, cfg.initial_stimulus_rounds) == (3,4,2)
    assert result.parameters["resolved_assembly_sizes"] == [12,6]
    assert result.parameters["seed_ids"] == [9,2,7]
    assert result.metrics["sparsity_beta_grid"][0]["actual_sparsity"] == .2
    assert result.raw_data["cells"][0]["values"] == [.9,.2,.7]


@pytest.mark.parametrize("kwargs", [
    {"sparsities": [.101,.109]}, {"sparsities": [0.]}, {"betas": [True]},
    {"p_values": [float("nan")]}, {"p_values": [1.1]}, {"p_values": []},
    {"test_rounds": 0}, {"initial_stimulus_rounds": 0},
    {"persistence_threshold": 2}, {"seed_ids": [1,1,2]},
])
def test_bad_grid_fails_before_timer(kwargs):
    experiment = object.__new__(study.PhaseDiagramExperiment)
    experiment.seed = 42
    # n100 makes .101 and .109 both resolve to k10.
    with pytest.raises(ValueError):
        experiment.run(n=100, **kwargs)


def test_phase_cli_requires_tag_and_records_smoke(monkeypatch, capsys):
    from research.experiments import historical_phase as adapter
    calls = []
    monkeypatch.setattr(adapter, "run_experiment", lambda **kwargs: calls.append(kwargs) or "saved")
    with pytest.raises(SystemExit) as error:
        study.main([])
    assert error.value.code == 2
    assert "--tag" in capsys.readouterr().err
    study.main(["--quick", "--seeds", "9", "2", "7", "--tag", "fixture"])
    assert calls[0]["smoke"] is True
    assert calls[0]["seeds"] == [9,2,7]
    assert calls[0]["parameters"] == adapter.parameters(True)


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_grid_values_cannot_silently_become_zero(sign):
    from fractions import Fraction
    from research.experiment_config import resolve_real_grid
    with pytest.raises(ValueError, match="nonzero value to zero"):
        resolve_real_grid([Fraction(sign, 10**400)], name="plasticity")


def test_grid_overflow_is_a_configuration_error():
    from research.experiment_config import resolve_real_grid
    with pytest.raises(ValueError, match="finite float"):
        resolve_real_grid([10**400], name="plasticity")


def test_grid_preserves_zero_subnormal_and_resolved_uniqueness():
    from research.experiment_config import resolve_real_grid
    import math
    from fractions import Fraction
    subnormal = math.nextafter(0., 1.)
    assert resolve_real_grid([0., subnormal, Fraction(1, 3)], name="plasticity") == [0., subnormal, 1/3]
    with pytest.raises(ValueError, match="unique"):
        resolve_real_grid([2**53, 2**53+1], name="grid")


def test_phase_parameters_record_the_values_actually_consumed(monkeypatch, tmp_path):
    from fractions import Fraction
    from research.json_documents import encode_document
    calls = []
    def trial(cfg, seed):
        calls.append(cfg)
        assert type(cfg.p) is float and type(cfg.beta) is float and type(cfg.w_max) is float
        return .5
    monkeypatch.setattr(study, "run_phase_trial", trial)
    result = study.PhaseDiagramExperiment(results_dir=tmp_path, verbose=False).run(
        n=60, seed_ids=[1,2,3], sparsities=[.1], betas=[Fraction(1,3)],
        p_values=[.2], p=Fraction(1,5), p_effect_k=6, p_effect_beta=Fraction(1,4), w_max=20)
    assert result.parameters["base_p"] == calls[0].p == .2
    assert result.parameters["p_effect_beta"] == calls[-1].beta == .25
    assert type(result.parameters["base_wmax"]) is float
    encode_document(result.to_dict())
