"""Historical learning-on projection measurements with explicit run provenance."""
from pathlib import Path
from research.experiments.primitives.test_projection import ProjectionExperiment
from research.runner import run_experiment
from research.experiments._historical import HistoricalStudy

REGISTRATION = "research/notes/memory/PREREG_historical_projection_migration.md"


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, k=6 if smoke else 100,
                p=.2 if smoke else .05, beta=.1, w_max=20.,
                train_rounds=3 if smoke else 30, test_rounds=3 if smoke else 20,
                max_train_rounds=8 if smoke else 100, convergence_window=3, convergence_threshold=.98,
                h1_sizes=[60, 80] if smoke else [100, 200, 500, 1000, 2000, 5000],
                h3_sizes=[60] if smoke else [500, 1000, 2000],
                round_values=[1, 3] if smoke else [1, 5, 10, 20, 30, 50])


STUDY = HistoricalStudy("memory.historical-projection", "3", REGISTRATION, Path(__file__),
                        ProjectionExperiment, parameters, 'learning-on persistence and A-driven regeneration; not frozen completion')
experiment = STUDY.measure


def main(argv=None):
    return STUDY.main(argv, writer=run_experiment)


if __name__ == "__main__":
    main()
