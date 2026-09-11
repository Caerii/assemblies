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
    assert (brain._engine.name, owner.name) == (expected["engine"], expected["owner"])


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
