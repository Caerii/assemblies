"""Descriptive learning-on scaling protocol with recorded inputs and censored times."""
from pathlib import Path
from research.experiments.stability.test_scaling_laws import ScalingLawsExperiment
from research.runner import run_experiment
from research.experiments._historical import HistoricalStudy

REGISTRATION = "research/notes/memory/PREREG_historical_scaling_migration.md"


def parameters(smoke=False):
    return dict(p=.2 if smoke else .05, beta=.1, w_max=20.,
                n_values=[60, 80] if smoke else [100, 200, 500, 1000, 2000, 5000],
                max_train_rounds=8 if smoke else 100, test_rounds=3 if smoke else 20,
                initial_stimulus_rounds=1, convergence_window=3, convergence_threshold=.98)


STUDY = HistoricalStudy("memory.historical-scaling", "1", REGISTRATION, Path(__file__),
                        ScalingLawsExperiment, parameters, 'learning-on persistence and descriptive convergence fit; no asymptotic class')
experiment = STUDY.measure


def main(argv=None):
    return STUDY.main(argv, writer=run_experiment)


if __name__ == "__main__":
    main()
