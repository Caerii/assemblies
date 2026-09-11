"""Historical learning-on projection measurements with explicit run provenance."""
from pathlib import Path
from research.experiments.primitives.test_projection import ProjectionExperiment
from research.runner import ROOT, experiment_parser, run_experiment

REGISTRATION = "research/notes/memory/PREREG_historical_projection_migration.md"


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, k=6 if smoke else 100,
                p=.2 if smoke else .05, beta=.1, w_max=20.,
                train_rounds=3 if smoke else 30, test_rounds=3 if smoke else 20,
                max_train_rounds=8 if smoke else 100,
                h1_sizes=[60, 80] if smoke else [100, 200, 500, 1000, 2000, 5000],
                h3_sizes=[60] if smoke else [500, 1000, 2000],
                round_values=[1, 3] if smoke else [1, 5, 10, 20, 30, 50])


def experiment(record):
    """Specification: research/notes/memory/PREREG_historical_projection_migration.md"""
    if record["engine"] != "numpy_explicit":
        raise ValueError("historical projection requires the numpy_explicit area owner")
    study = ProjectionExperiment(seed=0, verbose=False,
                                 results_dir=ROOT / "research/results/runs" / record["protocol"] / record["tag"])
    result = study.run(seed_ids=record["seeds"], **record["parameters"])
    return {"verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
            "scope": "learning-on persistence and A-driven regeneration; not frozen completion",
            "result": result.to_dict()}


def main(argv=None):
    parser = experiment_parser(__doc__, engines=("numpy_explicit",), default_seeds=tuple(range(42, 52)))
    parser.add_argument("--quick", action="store_true", dest="smoke", help="alias for VOID smoke")
    args = parser.parse_args(argv)
    print(run_experiment(script=Path(__file__), protocol="memory.historical-projection", protocol_version="2",
                         registration=REGISTRATION, engine=args.engine, seeds=args.seeds, tag=args.tag,
                         smoke=args.smoke, parameters=parameters(args.smoke), measure=experiment))


if __name__ == "__main__":
    main()
