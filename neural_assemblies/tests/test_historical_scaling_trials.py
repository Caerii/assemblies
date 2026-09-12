"""Source-0b9909c scaling schedule replay and corrected stopping interpretation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from research.experiments.stability import test_scaling_laws as study
from research.experiments._convergence import convergence_scaling_fit, run_convergence_phase

CASES = [json.loads(line) for line in (Path(__file__).parent / "data/historical_scaling_trials.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("expected", CASES, ids=lambda row: f"{row['seed']}-{row['limit']}")
def test_scaling_retains_initial_activation_and_all_trial_dynamics(monkeypatch, expected):
    brains = []

    class TracedBrain(study.Brain):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = []
            brains.append(self)

        def project(self, *args, **kwargs):
            out = super().project(*args, **kwargs)
            self.trace.append(dict(args=list(args), kwargs=kwargs,
                                   winners={name: area.winners.tolist() for name, area in self.areas.items()}))
            return out

    monkeypatch.setattr(study, "Brain", TracedBrain)
    result = study.run_scaling_trial(study.ScalingConfig(60, 6, .2, .1, 20., max_train_rounds=expected["limit"], test_rounds=3), expected["seed"])
    brain = brains[0]
    owner = brain._engine_for(brain.areas["A"])
    weights = {}
    for kind in ("_area_conns", "_stim_conns"):
        for source, targets in getattr(owner, kind).items():
            for target, conn in targets.items():
                weights[f"{kind}/{source}/{target}"] = hashlib.sha256(np.ascontiguousarray(conn.weights).tobytes()).hexdigest()
    assert brain.trace == expected["trace"]
    assert brain.trace[0]["args"] == [{"s": ["A"]}, {}]
    assert weights == expected["weights"]
    assert (expected["engine"], expected["owner"]) == ("numpy_sparse", "numpy_explicit")
    assert brain._engine.name == owner.name == "numpy_explicit"
    assert owner is brain._engine
    assert result["training_rounds"] == expected["result"]["convergence_time"]
    assert result["persistence"] == expected["result"]["persistence"]
    history = [row["winners"]["A"] for row in brain.trace[1:1+result["training_rounds"]]]
    stable = len(history) >= 4 and all(study.measure_overlap(history[-i-1], history[-i-2]) > .98 for i in range(3))
    assert result["converged"] is stable
    assert result["convergence_time"] == (result["training_rounds"] if stable else None)


@pytest.mark.parametrize("coefficient", [1, 10, 100])
def test_fit_coefficient_never_infers_asymptotic_complexity(coefficient):
    result = convergence_scaling_fit([10, 100, 1000], [[coefficient*i]*3 for i in (1, 2, 3)])
    assert result["slope"] == pytest.approx(coefficient)
    assert "scaling_type" not in result
    assert "O(" not in str(result)


def test_shared_stopping_resets_streak_and_checks_each_pair_once(monkeypatch):
    from types import SimpleNamespace
    from research.experiments import _convergence as phase
    sequence = [[0, 1]]*3 + [[2, 3]] + [[0, 1]]*4
    brain = SimpleNamespace(areas={"X": SimpleNamespace(explicit=True, winners=np.array([], dtype=int))})
    projected = []

    def project(stimuli, fibers):
        assert stimuli == {"cue": ["X"]} and fibers == {"X": ["X"]}
        brain.areas["X"].winners = np.array(sequence[len(projected)])
        projected.append(1)

    brain.project = project
    overlaps = []
    original = phase.measure_overlap

    def counted(left, right):
        overlaps.append(1)
        return original(left, right)

    monkeypatch.setattr(phase, "measure_overlap", counted)
    result = run_convergence_phase(brain, stimulus="cue", area="X", max_rounds=8)
    assert result.converged is True and result.training_rounds == 8
    assert len(overlaps) == 7


def test_scaling_summary_keeps_timeouts_and_raw_seed_identities(monkeypatch, tmp_path):
    from research.json_documents import encode_document

    def trial(cfg, seed):
        return dict(training_rounds=100, converged=False, convergence_time=None, persistence=.5)

    monkeypatch.setattr(study, "run_scaling_trial", trial)
    result = study.ScalingLawsExperiment(seed=42, results_dir=tmp_path, verbose=False).run(n_seeds=3)
    assert result.raw_data["seeds"] == [42, 43, 44]
    assert len(result.raw_data["cells"]) == 6
    assert result.metrics["scaling_fit"]["degenerate"] == "censored_observations"
    for cell in result.raw_data["cells"]:
        assert cell["values"]["converged"] == [False]*3
        assert cell["values"]["convergence_time"] == [None]*3
    encode_document(result.to_dict())


def test_configuration_preserves_explicit_seed_order_and_consumes_every_schedule(monkeypatch, tmp_path):
    calls = []
    def trial(cfg, seed):
        calls.append((cfg, seed))
        return dict(training_rounds=cfg.max_train_rounds, converged=False,
                    convergence_time=None, persistence=seed / 10)
    monkeypatch.setattr(study, "run_scaling_trial", trial)
    result = study.ScalingLawsExperiment(seed=999, results_dir=tmp_path, verbose=False).run(
        seed_ids=[9, 2, 7], n_values=[80, 60], max_train_rounds=8, test_rounds=4,
        initial_stimulus_rounds=2, convergence_window=2, convergence_threshold=.9)
    assert [seed for _, seed in calls] == [9, 2, 7]*2
    assert [cfg.n for cfg, _ in calls] == [80]*3+[60]*3
    for cfg, _ in calls:
        assert (cfg.max_train_rounds, cfg.test_rounds, cfg.initial_stimulus_rounds,
                cfg.convergence_window, cfg.convergence_threshold) == (8, 4, 2, 2, .9)
    assert result.raw_data["seeds"] == [9, 2, 7]
    assert result.raw_data["cells"][0]["values"]["persistence"] == [.9, .2, .7]
    assert result.parameters["seed_ids"] == [9, 2, 7]
    assert result.parameters["initial_stimulus_rounds"] == 2


@pytest.mark.parametrize("kwargs", [
    {"n_seeds": 2}, {"seed_ids": [1, 1, 2]}, {"n_values": []},
    {"n_values": [60]}, {"n_values": [60, 60]}, {"n_values": [60, True]},
    {"test_rounds": 0}, {"initial_stimulus_rounds": 0}, {"convergence_window": 0},
    {"convergence_threshold": float("nan")},
])
def test_invalid_scaling_config_fails_before_timer(kwargs):
    experiment = object.__new__(study.ScalingLawsExperiment)
    experiment.seed = 42
    with pytest.raises(ValueError):
        experiment.run(**kwargs)


def test_old_cli_requires_tag_and_records_smoke_parameters(monkeypatch, capsys):
    from research.experiments import historical_scaling as adapter
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


@pytest.mark.parametrize("size", [True, 1.5, 0, -2, float("nan"), float("inf")])
def test_censoring_does_not_hide_invalid_population_sizes(size):
    with pytest.raises(ValueError):
        convergence_scaling_fit([size, 100], [[None, None, None], [4, 4, 4]])


@pytest.mark.parametrize("time", [True, 0, -1, 1.5, float("nan"), float("inf")])
def test_censoring_does_not_hide_invalid_event_times(time):
    with pytest.raises(ValueError):
        convergence_scaling_fit([10, 100], [[None, 4, 4], [time, 4, 4]])


@pytest.mark.parametrize("status", ["False", "True", 0, 1, None, np.bool_(True)])
def test_direct_stopping_record_rejects_ambiguous_status(status):
    from research.experiments._convergence import ConvergenceObservation
    from neural_assemblies.assembly_calculus.assembly import Assembly
    with pytest.raises(ValueError, match="boolean"):
        ConvergenceObservation(Assembly("A", [1, 2]), 4, status)


def test_direct_stopping_record_normalizes_integer_rounds_and_requires_snapshot():
    from research.experiments._convergence import ConvergenceObservation
    from neural_assemblies.assembly_calculus.assembly import Assembly
    observation = ConvergenceObservation(Assembly("A", [1, 2]), np.int64(4), False)
    assert observation.record() == {"training_rounds": 4, "converged": False, "convergence_time": None}
    assert type(observation.training_rounds) is int
    with pytest.raises(ValueError, match="snapshot"):
        ConvergenceObservation(None, 4, False)
    with pytest.raises(ValueError, match="positive integer"):
        ConvergenceObservation(Assembly("A", [1, 2]), 0, True)


def test_streak_matches_window_rule_for_all_eight_comparison_patterns():
    from itertools import product
    from types import SimpleNamespace

    # Exhaustive bounded equivalence, not a claim of a formal unbounded proof.
    for passes in product((False, True), repeat=8):
        sequence = [[0, 1]]
        for agrees in passes:
            sequence.append(sequence[-1] if agrees else ([2, 3] if sequence[-1] == [0, 1] else [0, 1]))
        for window in range(1, 5):
            brain = SimpleNamespace(areas={"A": SimpleNamespace(explicit=True, winners=np.array([], dtype=int))})
            count = 0
            def project(stimuli, fibers, brain=brain, sequence=sequence):
                nonlocal count
                brain.areas["A"].winners = np.array(sequence[count])
                count += 1
            brain.project = project
            expected = next((i+2 for i in range(window-1, 8) if all(passes[i-window+1:i+1])), None)
            observed = run_convergence_phase(brain, stimulus="s", area="A", max_rounds=9, window=window)
            assert observed.record() == {"training_rounds": expected or 9,
                                          "converged": expected is not None,
                                          "convergence_time": expected}
