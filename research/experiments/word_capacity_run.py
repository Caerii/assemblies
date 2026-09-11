"""Immutable shared-runner entry point for the registered word-capacity study."""
from pathlib import Path

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments.word_capacity_protocol import (
    REGISTERED_PROTOCOL, SMOKE_PROTOCOL, WordCapacityProtocol,
)
from research.runner import experiment_parser, run_experiment

PROTOCOL = "aligner.word-capacity"
VERSION = "3.3"
REGISTRATION = "research/notes/aligner/PREREG_word_capacity.md"
REGISTERED_SEEDS = (42, *range(1, 20))


def parameters(smoke=False):
    """Resolve every result-changing study constant into the run record."""
    return (SMOKE_PROTOCOL if smoke else REGISTERED_PROTOCOL).to_parameters()


def _validate_parameters(raw):
    protocol = WordCapacityProtocol.from_parameters(raw)
    expected = REGISTERED_PROTOCOL.to_parameters()
    fixed = {key: expected[key] for key in expected.keys() - {
        "cells", "vocabulary_sizes", "feature_area",
    }}
    if any(raw[key] != value for key, value in fixed.items()):
        raise ValueError("this protocol version does not implement changed fixed constants")
    return protocol


def measure(record):
    """Execute only the model and parameters represented by the run record."""
    if record.get("protocol") != PROTOCOL or record.get("protocol_version") != VERSION:
        raise ValueError("word-capacity protocol identity mismatch")
    execution = ExecutionSemantics.normalize(record.get("execution_semantics"))
    if execution.kind is not ExecutionKind.ALIGNMENT:
        raise ValueError("word capacity requires alignment execution semantics")
    required = execution.profiles["default"].to_dict()
    protocol = _validate_parameters(record.get("parameters"))
    if record.get("engine") != "scheduled_aligner":
        raise ValueError("word-capacity protocol 3.3 requires scheduled_aligner")
    engine = "scheduled"
    curves = {
        name: capacity.run_cell(
            name, record["seeds"], protocol.vocabulary_sizes, engine=engine,
            protocol=protocol, aligner_semantics=required,
        )
        for name in protocol.cells
    }
    serial_curves = {
        name: {str(size): values for size, values in curve.items()}
        for name, curve in curves.items()
    }
    return {
        "verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
        "scope": "registered word-capacity curves and bars",
        "curves": serial_curves,
        "report": capacity.capacity_report(
            curves, record["seeds"], protocol=protocol,
        ),
    }


def main(argv=None):
    parser = experiment_parser(
        "Registered cross-situational word-capacity study",
        engines=("scheduled_aligner",),
        default_seeds=REGISTERED_SEEDS,
    )
    parser.add_argument("--cells", nargs="+", choices=tuple(REGISTERED_PROTOCOL.definitions))
    parser.add_argument("--vocabulary-sizes", nargs="+", type=int)
    parser.add_argument("--feature-area", nargs=2, type=int, metavar=("N", "K"))
    args = parser.parse_args(argv)
    base = SMOKE_PROTOCOL if args.smoke else REGISTERED_PROTOCOL
    protocol = base.select(
        cells=None if args.cells is None else tuple(args.cells),
        vocabulary_sizes=(None if args.vocabulary_sizes is None
                          else tuple(args.vocabulary_sizes)),
        feature_area=None if args.feature_area is None else tuple(args.feature_area),
    )
    resolved = protocol.to_parameters()
    # Reject the complete record before source capture or CUDA work.
    protocol = _validate_parameters(resolved)
    semantics = describe_hashed_aligner(
        p=protocol.connection_probability, beta=protocol.plasticity,
        rounds_word=protocol.rounds_per_pair, norm_init=True, scaling=True,
        w_max=None, stim_beta=0.0, stim_gain=None, store="present",
    )
    return run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, parameters=resolved, measure=measure,
        aligner_semantics=semantics,
    )


if __name__ == "__main__":
    main()
