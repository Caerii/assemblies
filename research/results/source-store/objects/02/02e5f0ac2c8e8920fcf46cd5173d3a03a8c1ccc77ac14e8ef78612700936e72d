"""Explicit historical merge schedules and descriptive parent overlaps."""
from pathlib import Path
from research.experiments.primitives.test_merge import MergeExperiment
from research.experiments._historical import HistoricalStudy
from research.runner import run_experiment

REGISTRATION = "research/notes/memory/PREREG_historical_merge_migration.md"


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, k=6 if smoke else 100, p=.2 if smoke else .05,
                beta=.1, w_max=20., establish_rounds=3 if smoke else 30,
                merge_rounds=3 if smoke else 30, test_rounds=3 if smoke else 20,
                round_values=[1,3] if smoke else [1,5,10,20,30,50],
                h4_sizes=[60] if smoke else [200,500,1000,2000])


STUDY = HistoricalStudy("memory.historical-merge", "1", REGISTRATION, Path(__file__),
                        MergeExperiment, parameters,
                        "sequential trained parent overlaps and learning-on driven recovery")
experiment = STUDY.measure


def main(argv=None):
    return STUDY.main(argv, writer=run_experiment)


if __name__ == "__main__":
    main()
