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
    assert (brain._engine.name, owner.name) == (expected["engine"], expected["owner"])
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
