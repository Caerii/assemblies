"""Descriptive learning-on persistence grid with recorded inputs."""
from pathlib import Path
from research.experiments.stability.test_phase_diagram import PhaseDiagramExperiment
from research.runner import ROOT, experiment_parser, run_experiment

REGISTRATION = "research/notes/memory/PREREG_historical_phase_migration.md"


def parameters(smoke=False):
    return dict(n=60 if smoke else 1000, p=.2 if smoke else .05, w_max=20.,
                sparsities=[.1,.2] if smoke else [.01,.02,.05,.10,.15,.20,.30],
                betas=[0.,.1] if smoke else [.01,.02,.05,.10,.20],
                p_values=[.1,.2] if smoke else [.01,.02,.05,.10,.20],
                p_effect_k=6 if smoke else 100, p_effect_beta=.1,
                train_rounds=3 if smoke else 30, test_rounds=3 if smoke else 20,
                initial_stimulus_rounds=1, persistence_threshold=.95)


def experiment(record):
    """Specification: research/notes/memory/PREREG_historical_phase_migration.md"""
    if record["engine"] != "numpy_explicit":
        raise ValueError("historical phase requires the numpy_explicit area owner")
    study = PhaseDiagramExperiment(seed=0, verbose=False,
                                  results_dir=ROOT / "research/results/runs" / record["protocol"] / record["tag"])
    result = study.run(seed_ids=record["seeds"], **record["parameters"])
    return {"verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
            "scope": "learning-on persistence and sampled crossings; not a physical phase boundary",
            "result": result.to_dict()}


def main(argv=None):
    parser = experiment_parser(__doc__, engines=("numpy_explicit",), default_seeds=tuple(range(42, 52)))
    parser.add_argument("--quick", action="store_true", dest="smoke", help="alias for VOID smoke")
    args = parser.parse_args(argv)
    print(run_experiment(script=Path(__file__), protocol="memory.historical-phase", protocol_version="1",
                         registration=REGISTRATION, engine=args.engine, seeds=args.seeds, tag=args.tag,
                         smoke=args.smoke, parameters=parameters(args.smoke), measure=experiment))


if __name__ == "__main__":
    main()
