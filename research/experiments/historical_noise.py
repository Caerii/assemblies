"""Preserved learning-on-recovery protocol through the shared runner."""
from pathlib import Path
from research.experiments.stability.test_noise_robustness import NoiseRobustnessExperiment
from research.runner import ROOT, experiment_parser, run_experiment

REGISTRATION = 'research/notes/memory/PREREG_historical_noise_migration.md'


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, k=6 if smoke else 100,
                p=.2 if smoke else .05, beta=.1, w_max=20.,
                establish_rounds=3 if smoke else 30, recovery_rounds=3 if smoke else 20,
                noise_fracs=[0., .5, 1.] if smoke else [0., .1, .2, .3, .4, .5, .6, .8, 1.],
                h4_sizes=[60] if smoke else [200, 500, 1000, 2000],
                h4_noise_fracs=[0., .5, 1.] if smoke else [.3, .5, .7, 1.])


def experiment(record):
    """Specification: research/notes/memory/PREREG_historical_noise_migration.md"""
    if record['engine'] != 'numpy_explicit':
        raise ValueError('historical protocol requires the numpy_explicit area owner')
    study = NoiseRobustnessExperiment(seed=0, verbose=False,
                                     results_dir=ROOT / 'research/results/runs' / record['protocol'] / record['tag'])
    result = study.run(seed_ids=record['seeds'], **record['parameters'])
    return {'verdict': 'VOID' if record['mode'] == 'smoke' else 'UNADOPTED',
            'scope': 'historical learning-on-recovery protocol; not frozen recovery',
            'result': result.to_dict()}


def main(argv=None):
    parser = experiment_parser(__doc__, engines=('numpy_explicit',), default_seeds=tuple(range(42, 52)))
    parser.add_argument('--quick', action='store_true', dest='smoke', help='legacy alias for --smoke; scientific status VOID')
    args = parser.parse_args(argv)
    print(run_experiment(script=Path(__file__), protocol='memory.historical-noise', protocol_version='1',
                         registration=REGISTRATION, engine=args.engine, seeds=args.seeds, tag=args.tag,
                         smoke=args.smoke, parameters=parameters(args.smoke), measure=experiment))


if __name__ == '__main__':
    main()
