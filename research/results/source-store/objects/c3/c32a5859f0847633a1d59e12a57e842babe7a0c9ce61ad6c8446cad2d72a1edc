"""Immutable shared-runner entry point for the registered word-capacity study."""
from pathlib import Path

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from neural_assemblies.core.registration import validate_area_registration
from research.experiments import word_capacity as capacity
from research.runner import experiment_parser, run_experiment

PROTOCOL = "aligner.word-capacity"
VERSION = "3.2"
REGISTRATION = "research/notes/aligner/PREREG_word_capacity.md"
REGISTERED_SEEDS = (42, *range(1, 20))


def parameters(smoke=False):
    """Resolve every result-changing study constant into the run record."""
    return {
        "cells": ["A"] if smoke else list(capacity.CELLS),
        "vocabulary_sizes": [8, 16] if smoke else list(capacity.VS_WIDE),
        "feature_area": [capacity.FEAT_N, capacity.FEAT_K] if smoke else [4000, 100],
        "cell_definitions": {name: list(values) for name, values in capacity.CELLS.items()},
        "connection_probability": capacity.U.P,
        "plasticity": capacity.U.BETA,
        "rounds_per_pair": capacity.ROUNDS_HASHED,
        "exposures_per_referent": capacity.EXPOSURES,
        "referents_per_scene": capacity.PER_SCENE,
        "category_count": capacity.CATS,
        "minimum_exposures": capacity.U.MIN_EXPOSURES,
        "threshold": capacity.THRESHOLD,
    }


def _validate_parameters(raw):
    expected = parameters(False)
    required = set(expected)
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("word-capacity parameters must contain the complete protocol")
    cells = raw["cells"]
    if (not isinstance(cells, list) or not cells
            or any(name not in capacity.CELLS for name in cells)
            or len(set(cells)) != len(cells)):
        raise ValueError("cells must be a nonempty unique subset of A-E")
    sizes = raw["vocabulary_sizes"]
    if (not isinstance(sizes, list) or len(sizes) < 2
            or any(type(value) is not int or value <= 1 for value in sizes)
            or sizes != sorted(set(sizes))):
        raise ValueError("vocabulary_sizes must be increasing unique integers above one")
    feature_area = raw["feature_area"]
    if not isinstance(feature_area, list) or len(feature_area) != 2:
        raise ValueError("feature_area must be [n, k]")
    feat_n, feat_k = validate_area_registration("FEAT", *feature_area)
    fixed = {key: expected[key] for key in required - {
        "cells", "vocabulary_sizes", "feature_area",
    }}
    if any(raw[key] != value for key, value in fixed.items()):
        raise ValueError("this protocol version does not implement changed fixed constants")
    return cells, sizes, (feat_n, feat_k)


def measure(record):
    """Execute only the model and parameters represented by the run record."""
    if record.get("protocol") != PROTOCOL or record.get("protocol_version") != VERSION:
        raise ValueError("word-capacity protocol identity mismatch")
    execution = ExecutionSemantics.normalize(record.get("execution_semantics"))
    if execution.kind is not ExecutionKind.ALIGNMENT:
        raise ValueError("word capacity requires alignment execution semantics")
    required = execution.profiles["default"].to_dict()
    cells, sizes, feature_area = _validate_parameters(record.get("parameters"))
    engine = {"hashed_aligner": "hashed", "scheduled_aligner": "scheduled"}.get(
        record.get("engine")
    )
    if engine is None:
        raise ValueError("word capacity requires an explicit alignment engine")
    if engine == "hashed" and feature_area != (capacity.FEAT_N, capacity.FEAT_K):
        raise ValueError("hashed_aligner only implements its fixed feature area")
    curves = {
        name: capacity.run_cell(
            name, record["seeds"], sizes, engine=engine, feat=feature_area,
            aligner_semantics=required,
        )
        for name in cells
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
            curves, record["seeds"], threshold=record["parameters"]["threshold"],
        ),
    }


def main(argv=None):
    parser = experiment_parser(
        "Registered cross-situational word-capacity study",
        engines=("scheduled_aligner", "hashed_aligner"),
        default_seeds=REGISTERED_SEEDS,
    )
    parser.add_argument("--cells", nargs="+", choices=tuple(capacity.CELLS))
    parser.add_argument("--vocabulary-sizes", nargs="+", type=int)
    parser.add_argument("--feature-area", nargs=2, type=int, metavar=("N", "K"))
    args = parser.parse_args(argv)
    resolved = parameters(args.smoke)
    if args.cells is not None:
        resolved["cells"] = args.cells
    if args.vocabulary_sizes is not None:
        resolved["vocabulary_sizes"] = args.vocabulary_sizes
    if args.feature_area is not None:
        resolved["feature_area"] = args.feature_area
    # Reject the complete record before source capture or CUDA work.
    _validate_parameters(resolved)
    semantics = describe_hashed_aligner(
        p=resolved["connection_probability"], beta=resolved["plasticity"],
        rounds_word=resolved["rounds_per_pair"], norm_init=True, scaling=True,
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
