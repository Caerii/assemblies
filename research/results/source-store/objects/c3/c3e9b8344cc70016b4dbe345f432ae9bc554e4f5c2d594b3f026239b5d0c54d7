"""Descriptive learning-on persistence grid with recorded inputs."""
from pathlib import Path
from research.experiments.stability.test_phase_diagram import PhaseDiagramExperiment
from research.runner import run_experiment
from research.experiments._historical import HistoricalStudy

REGISTRATION = "research/notes/memory/PREREG_historical_phase_migration.md"


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, p=.2 if smoke else .05, w_max=20.,
                sparsities=[.1,.2] if smoke else [.01,.02,.05,.10,.15,.20,.30],
                betas=[0.,.1] if smoke else [.01,.02,.05,.10,.20],
                p_values=[.1,.2] if smoke else [.01,.02,.05,.10,.20],
                p_effect_k=6 if smoke else 100, p_effect_beta=.1,
                train_rounds=3 if smoke else 30, test_rounds=3 if smoke else 20,
                initial_stimulus_rounds=1, persistence_threshold=.95)


STUDY = HistoricalStudy("memory.historical-phase", "1", REGISTRATION, Path(__file__),
                        PhaseDiagramExperiment, parameters, 'learning-on persistence and sampled crossings; not a physical phase boundary')
experiment = STUDY.measure


def main(argv=None):
    return STUDY.main(argv, writer=run_experiment)


if __name__ == "__main__":
    main()
