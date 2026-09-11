"""Preserved learning-on-recovery protocol through the shared runner."""
from pathlib import Path
from research.experiments.stability.test_noise_robustness import NoiseRobustnessExperiment
from research.runner import run_experiment
from research.experiments._historical import HistoricalStudy

REGISTRATION = 'research/notes/memory/PREREG_historical_noise_migration.md'


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, k=6 if smoke else 100,
                p=.2 if smoke else .05, beta=.1, w_max=20.,
                establish_rounds=3 if smoke else 30, recovery_rounds=3 if smoke else 20,
                noise_fracs=[0., .5, 1.] if smoke else [0., .1, .2, .3, .4, .5, .6, .8, 1.],
                h4_sizes=[60] if smoke else [200, 500, 1000, 2000],
                h4_noise_fracs=[0., .5, 1.] if smoke else [.3, .5, .7, 1.])


STUDY = HistoricalStudy("memory.historical-noise", "1", REGISTRATION, Path(__file__),
                        NoiseRobustnessExperiment, parameters, 'historical learning-on-recovery protocol; not frozen recovery')
experiment = STUDY.measure


def main(argv=None):
    return STUDY.main(argv, writer=run_experiment)


if __name__ == '__main__':
    main()
