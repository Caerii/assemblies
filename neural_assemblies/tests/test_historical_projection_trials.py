"""Source-177dbbc replay controls; H4's old constant is a known defective observable."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from research.experiments.primitives import test_projection as study

CASES = [json.loads(line) for line in (Path(__file__).parent / "data/historical_projection_trials.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("expected", CASES, ids=lambda row: f"{row['trial']}-{row['seed']}")
def test_trial_dynamics_preserved_while_dead_weight_measurement_is_corrected(monkeypatch, expected):
    brains = []

    class TracedBrain(study.Brain):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = []
            brains.append(self)

        def project(self, *args, **kwargs):
            result = super().project(*args, **kwargs)
            self.trace.append(dict(args=list(args), kwargs=kwargs,
                                   winners={name: area.winners.tolist() for name, area in self.areas.items()}))
            if len(self.trace) == 3:
                self.pre_evaluation_weights = self.connectomes["A"]["A"].weights.copy()
                self.pre_evaluation_winners = self.areas["A"].winners.copy()
            return result

    monkeypatch.setattr(study, "Brain", TracedBrain)
    cfg = study.ProjConfig(60, 6, .2, .1, 20., train_rounds=3, test_rounds=3, max_train_rounds=8)
    name, seed = expected["trial"], expected["seed"]
    if name == "convergence":
        result = study.run_convergence_trial(cfg, seed)
    elif name in ("stim_self", "stim_only"):
        result = study.run_training_mode_trial(cfg, seed, name)
    elif name == "crossarea":
        result = study.run_crossarea_trial(cfg, seed)
    else:
        result = study.run_weight_dynamics_trial(60, 6, .2, .1, 20., 3, 3, seed)
    brain = brains[0]
    owner = brain._engine_for(brain.areas["A"])
    weights = {}
    for kind in ("_area_conns", "_stim_conns"):
        for source, targets in getattr(owner, kind).items():
            for target, connection in targets.items():
                weights[f"{kind}/{source}/{target}"] = hashlib.sha256(np.ascontiguousarray(connection.weights).tobytes()).hexdigest()
    assert brain.trace == expected["trace"]
    assert weights == expected["weights"]
    assert (brain._engine.name, owner.name) == (expected["engine"], expected["owner"])
    if name == "weights":
        assert expected["result"]["weight_ratio"] == 1.0  # The historical dead probe.
        matrix, selected = brain.pre_evaluation_weights, brain.pre_evaluation_winners
        numerator = sum(float(matrix[i, j]) for i in selected for j in selected) / len(selected)**2
        denominator = sum(float(value) for value in matrix.flat) / matrix.size
        assert result["weight_ratio"] == pytest.approx(numerator / denominator)
        assert result["weight_ratio"] != 1.0
        assert result["persistence"] == expected["result"]["persistence"]
    else:
        assert result == expected["result"]


def test_weight_ratio_moves_with_actual_matrix_and_includes_absent_edges():
    weights = np.array([[4., 0.], [0., 0.]])
    assert study.recurrent_weight_ratio(weights, [0]) == 4.
    assert study.recurrent_weight_ratio(weights, [1]) == 0.
    assert study.recurrent_weight_ratio(np.ones((2, 2)), [0]) == 1.
    assert study.recurrent_weight_ratio(weights * 7, [0]) == 4.


@pytest.mark.parametrize("weights,winners", [
    (np.zeros((2, 2)), [0]), (np.ones((2, 3)), [0]),
    (np.array([[np.nan]]), [0]), (np.array([[-1.]]), [0]),
    (np.ones((2, 2)), [0, 0]), (np.ones((2, 2)), [-1]),
    (np.ones((2, 2)), [2]), (np.ones((2, 2)), []),
    (np.ones((2, 2)), [0.5]), (np.ones((2, 2)), [True]),
])
def test_undefined_weight_measurement_never_defaults_to_one(weights, winners):
    with pytest.raises(ValueError):
        study.recurrent_weight_ratio(weights, winners)


def test_unknown_training_mode_fails_before_brain_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unknown mode created a brain")
    monkeypatch.setattr(study, "Brain", forbidden)
    with pytest.raises(ValueError, match="training mode"):
        study.run_training_mode_trial(study.ProjConfig(60, 6, .2, .1, 20.), 1, "stim_slef")


def test_crossarea_observation_does_not_depend_on_corrupted_b_cue(monkeypatch):
    original = np.random.default_rng
    cfg = study.ProjConfig(60, 6, .2, .1, 20., train_rounds=3, test_rounds=3)
    outcomes = []
    for replacement in (np.arange(6), np.arange(54, 60)):
        class Corruption:
            def choice(self, *args, **kwargs):
                return replacement.copy()

        def rng(seed=None):
            return Corruption() if seed == 77778 else original(seed)

        monkeypatch.setattr(np.random, "default_rng", rng)
        outcomes.append(study.run_crossarea_trial(cfg, 1))
    assert outcomes[0] == outcomes[1]
