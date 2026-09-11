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
    elif name == "convergence":
        assert result["training_rounds"] == expected["result"]["convergence_time"]
        assert result["persistence"] == expected["result"]["persistence"]
        history = [row["winners"]["A"] for row in brain.trace[:result["training_rounds"]]]
        stable = len(history) >= 4 and all(study.measure_overlap(history[-i-1], history[-i-2]) > .98 for i in range(3))
        assert result["converged"] is stable
        assert result["convergence_time"] == (result["training_rounds"] if stable else None)
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


def test_configured_study_consumes_grid_schedule_seeds_and_retains_raw(monkeypatch, tmp_path):
    calls = []

    def convergence(cfg, seed):
        calls.append(("h1", cfg, seed))
        return {"convergence_time": seed + cfg.n, "training_rounds": seed + cfg.n, "converged": True, "persistence": seed / 10}

    def mode(cfg, seed, mode):
        calls.append((mode, cfg, seed))
        return seed / 10

    def cross(cfg, seed):
        calls.append(("h3", cfg, seed))
        return seed / 10

    def weights(n, k, p, beta, w_max, train_rounds, test_rounds, seed):
        calls.append(("h4", study.ProjConfig(n, k, p, beta, w_max, train_rounds, test_rounds), seed))
        return {"weight_ratio": seed / 10, "persistence": seed / 10}

    monkeypatch.setattr(study, "run_convergence_trial", convergence)
    monkeypatch.setattr(study, "run_training_mode_trial", mode)
    monkeypatch.setattr(study, "run_crossarea_trial", cross)
    monkeypatch.setattr(study, "run_weight_dynamics_trial", weights)
    result = study.ProjectionExperiment(seed=999, results_dir=tmp_path, verbose=False).run(
        n=60, k=6, seed_ids=[9, 2, 7], h1_sizes=[60, 80], h3_sizes=[60],
        train_rounds=3, test_rounds=4, max_train_rounds=8, round_values=[2, 5])
    assert len(calls) == 21
    assert [seed for _, _, seed in calls] == [9, 2, 7] * 2 + [9, 9, 2, 2, 7, 7] + [9, 2, 7] * 3
    for name, cfg, seed in calls:
        assert cfg.test_rounds == 4
        if name != "h4":
            assert cfg.train_rounds == 3 and cfg.max_train_rounds == 8
    assert [cfg.train_rounds for name, cfg, _ in calls if name == "h4"] == [2]*3 + [5]*3
    assert result.raw_data["seeds"] == [9, 2, 7]
    assert len(result.raw_data["cells"]) == 6
    assert result.raw_data["cells"][0]["values"]["persistence"] == [.9, .2, .7]
    assert result.parameters["seed_ids"] == [9, 2, 7]
    assert result.parameters["h1_sizes"] == [60, 80]
    assert result.parameters["evaluation_learning"] is True


@pytest.mark.parametrize("kwargs", [
    {"seed_ids": [1, 1, 2]}, {"n_seeds": 2}, {"h1_sizes": [60]},
    {"h3_sizes": []}, {"round_values": [2, 2]}, {"test_rounds": 0},
])
def test_invalid_study_configuration_fails_before_timer(kwargs):
    experiment = object.__new__(study.ProjectionExperiment)
    experiment.seed = 42
    with pytest.raises(ValueError):
        experiment.run(**kwargs)


def test_legacy_cli_requires_tag_and_records_quick_configuration(monkeypatch, capsys):
    from research.experiments import historical_projection as adapter
    calls = []
    monkeypatch.setattr(adapter, "run_experiment", lambda **kwargs: calls.append(kwargs) or "saved")
    with pytest.raises(SystemExit) as error:
        study.main([])
    assert error.value.code == 2
    assert "--tag" in capsys.readouterr().err
    study.main(["--quick", "--seeds", "9", "2", "7", "--tag", "fixture"])
    assert calls[0]["smoke"] is True
    assert calls[0]["engine"] == "numpy_explicit"
    assert calls[0]["seeds"] == [9, 2, 7]
    assert calls[0]["parameters"] == adapter.parameters(True)


def test_constant_response_and_undefined_null_are_serializable(monkeypatch, tmp_path):
    from research.json_documents import encode_document
    monkeypatch.setattr(study, "run_convergence_trial", lambda cfg, seed: {"convergence_time": 4, "training_rounds": 4, "converged": True, "persistence": 1.})
    monkeypatch.setattr(study, "run_training_mode_trial", lambda *args: 1.)
    monkeypatch.setattr(study, "run_crossarea_trial", lambda *args: 1.)
    monkeypatch.setattr(study, "run_weight_dynamics_trial", lambda *args: {"weight_ratio": 2., "persistence": 1.})
    from research.experiments.historical_projection import parameters
    result = study.ProjectionExperiment(results_dir=tmp_path, verbose=False).run(seed_ids=[1, 2, 3], **parameters(True))
    assert result.metrics["scaling_fit"]["degenerate"] == "constant_response"
    assert result.metrics["scaling_fit"]["r_squared"] is None
    assert result.metrics["convergence_vs_size"][0]["test_vs_null"]["p"] is None
    encode_document(result.to_dict())


@pytest.mark.parametrize("final_winners,converged", [([0, 1], True), ([2, 3], False)])
def test_final_round_convergence_is_distinct_from_timeout(monkeypatch, final_winners, converged):
    from types import SimpleNamespace

    class ScriptedBrain:
        def __init__(self, **kwargs):
            self.areas = {"A": SimpleNamespace(winners=np.array([], dtype=int), explicit=True)}
            self.rounds = 0

        def add_area(self, *args, **kwargs):
            pass

        def add_stimulus(self, *args, **kwargs):
            pass

        def project(self, stimulus, fibers):
            self.rounds += 1
            self.areas["A"].winners = np.array([0, 1] if self.rounds < 4 else final_winners)

    monkeypatch.setattr(study, "Brain", ScriptedBrain)
    result = study.run_convergence_trial(study.ProjConfig(4, 2, .2, .1, 20., max_train_rounds=4, test_rounds=1), 1)
    assert result["training_rounds"] == 4
    assert result["converged"] is converged
    assert result["convergence_time"] == (4 if converged else None)


@pytest.mark.parametrize("kwargs", [
    {"convergence_window": 0}, {"convergence_window": True},
    {"convergence_threshold": float("nan")}, {"convergence_threshold": 1.1},
    {"convergence_threshold": True},
])
def test_invalid_stopping_rule_fails_before_timer(kwargs):
    experiment = object.__new__(study.ProjectionExperiment)
    experiment.seed = 42
    with pytest.raises(ValueError):
        experiment.run(**kwargs)


def test_censored_seed_is_retained_and_blocks_scaling_fit(monkeypatch, tmp_path):
    from research.experiments.historical_projection import parameters

    def convergence(cfg, seed):
        return {"training_rounds": 8, "converged": seed != 2,
                "convergence_time": 8 if seed != 2 else None, "persistence": seed / 10}

    monkeypatch.setattr(study, "run_convergence_trial", convergence)
    monkeypatch.setattr(study, "run_training_mode_trial", lambda cfg, seed, mode: seed / 10)
    monkeypatch.setattr(study, "run_crossarea_trial", lambda cfg, seed: seed / 10)
    monkeypatch.setattr(study, "run_weight_dynamics_trial", lambda *args: {"weight_ratio": 2., "persistence": .5})
    from research.experiments import _convergence
    monkeypatch.setattr(_convergence.stats, "linregress", lambda *args: pytest.fail("fit treated timeout as convergence"))
    result = study.ProjectionExperiment(results_dir=tmp_path, verbose=False).run(seed_ids=[1, 2, 3], **parameters(True))
    for cell in result.raw_data["cells"][:2]:
        assert cell["values"]["convergence_time"] == [8, None, 8]
        assert cell["values"]["converged"] == [True, False, True]
        assert cell["values"]["training_rounds"] == [8, 8, 8]
    assert result.metrics["scaling_fit"]["degenerate"] == "censored_observations"
    assert result.metrics["scaling_fit"]["slope"] is None
    assert result.metrics["convergence_vs_size"][0]["training_rounds"]["n"] == 3
