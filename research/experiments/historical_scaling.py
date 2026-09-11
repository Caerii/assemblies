"""Descriptive learning-on scaling protocol with recorded inputs and censored times."""
from pathlib import Path
from research.experiments.stability.test_scaling_laws import ScalingLawsExperiment
from research.runner import ROOT, experiment_parser, run_experiment

REGISTRATION = "research/notes/memory/PREREG_historical_scaling_migration.md"


def parameters(smoke=False):
    return dict(p=.2 if smoke else .05, beta=.1, w_max=20.,
                n_values=[60, 80] if smoke else [100, 200, 500, 1000, 2000, 5000],
                max_train_rounds=8 if smoke else 100, test_rounds=3 if smoke else 20,
                initial_stimulus_rounds=1, convergence_window=3, convergence_threshold=.98)


def experiment(record):
    """Specification: research/notes/memory/PREREG_historical_scaling_migration.md"""
    if record["engine"] != "numpy_explicit":
        raise ValueError("historical scaling requires the numpy_explicit area owner")
    study = ScalingLawsExperiment(seed=0, verbose=False,
                                  results_dir=ROOT / "research/results/runs" / record["protocol"] / record["tag"])
    result = study.run(seed_ids=record["seeds"], **record["parameters"])
    return {"verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
            "scope": "learning-on persistence and descriptive convergence fit; no asymptotic class",
            "result": result.to_dict()}


def main(argv=None):
    parser = experiment_parser(__doc__, engines=("numpy_explicit",), default_seeds=tuple(range(42, 52)))
    parser.add_argument("--quick", action="store_true", dest="smoke", help="alias for VOID smoke")
    args = parser.parse_args(argv)
    print(run_experiment(script=Path(__file__), protocol="memory.historical-scaling", protocol_version="1",
                         registration=REGISTRATION, engine=args.engine, seeds=args.seeds, tag=args.tag,
                         smoke=args.smoke, parameters=parameters(args.smoke), measure=experiment))


if __name__ == "__main__":
    main()
