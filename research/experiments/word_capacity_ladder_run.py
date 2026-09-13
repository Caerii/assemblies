"""Shared-runner entry point for the registered word-capacity FEAT ladder.

Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-word-capacity-ladder
Registration: research/notes/aligner/PREREG_word_capacity.md#part-1----the-feat-ladder
"""
from pathlib import Path

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments.word_capacity_protocol import (
    REGISTERED_PROTOCOL, WordCapacityProtocol,
)
from research.runner import (
    experiment_parser, run_experiment, validate_registered_seeds,
    validate_seed_identities,
)

PROTOCOL = "aligner.word-capacity-ladder"
VERSION = "3.3"
REGISTRATION = "research/notes/aligner/PREREG_word_capacity.md"
STUDY_SEEDS = (42, *range(1, 20))
LADDER_CELLS = ("A", "C")


def selected_protocol(*, smoke: bool) -> WordCapacityProtocol:
    """Return the complete registered selection for this execution mode."""
    if smoke:
        ladder = (REGISTERED_PROTOCOL.feature_ladder[0],)
        return REGISTERED_PROTOCOL.select(
            cells=("A",), vocabulary_sizes=(8, 16),
            feature_area=ladder[0], feature_ladder=ladder,
        )
    return REGISTERED_PROTOCOL.select(
        cells=LADDER_CELLS,
        feature_area=REGISTERED_PROTOCOL.feature_ladder[0],
    )


def _validate_parameters(raw: dict) -> WordCapacityProtocol:
    protocol = WordCapacityProtocol.from_parameters(raw)
    if any(name not in LADDER_CELLS for name in protocol.cells):
        raise ValueError("the FEAT ladder implements only registered cells A and C")
    registered = REGISTERED_PROTOCOL.to_parameters()
    variable = {"cells", "vocabulary_sizes", "feature_area", "feature_ladder"}
    if any(raw[key] != value for key, value in registered.items() if key not in variable):
        raise ValueError("this ladder protocol version does not implement changed constants")
    if protocol.feature_area != protocol.feature_ladder[0]:
        raise ValueError("feature_area must identify the first selected ladder rung")
    if any(area not in REGISTERED_PROTOCOL.feature_ladder for area in protocol.feature_ladder):
        raise ValueError("feature ladder contains an unregistered rung")
    indices = [REGISTERED_PROTOCOL.feature_ladder.index(area)
               for area in protocol.feature_ladder]
    if indices != sorted(set(indices)):
        raise ValueError("feature ladder must retain registered order without duplicates")
    return protocol


def measure(record: dict) -> dict:
    """Run only the ladder encoded by the immutable evidence record."""
    if record.get("protocol") != PROTOCOL or record.get("protocol_version") != VERSION:
        raise ValueError("word-capacity ladder protocol identity mismatch")
    execution = ExecutionSemantics.normalize(record.get("execution_semantics"))
    if execution.kind is not ExecutionKind.ALIGNMENT:
        raise ValueError("word-capacity ladder requires alignment semantics")
    if record.get("engine") != "scheduled_aligner":
        raise ValueError("word-capacity ladder 3.3 requires scheduled_aligner")
    raw_parameters = record.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("word-capacity ladder parameters must be a mapping")
    if record.get("mode", "study") == "study":
        validate_seed_identities(record["seeds"], STUDY_SEEDS)
    protocol = _validate_parameters(raw_parameters)
    profile = execution.profiles["default"].to_dict()
    curves, summaries = {}, {}
    for name in protocol.cells:
        for area in protocol.feature_ladder:
            selected = protocol.select(
                cells=(name,), feature_area=area, feature_ladder=(area,),
            )
            curve = capacity.run_cell(
                name, record["seeds"], selected.vocabulary_sizes,
                engine="scheduled", protocol=selected,
                aligner_semantics=profile,
            )
            key = f"{name}:{area[0]}x{area[1]}"
            curves[key] = {str(size): values for size, values in curve.items()}
            summaries[key] = capacity.capacity_report(
                {name: curve}, record["seeds"], protocol=selected,
            )["cells"][name]
    return {
        "verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
        "scope": "registered FEAT-ladder curves and per-rung ceilings; F1/F2 not readopted",
        "curves": curves,
        "rungs": summaries,
    }


def _area(value: str) -> tuple[int, int]:
    try:
        n, k = value.split(":", 1)
        return int(n), int(k)
    except ValueError as exc:
        raise ValueError("feature areas must use N:K") from exc


def main(argv=None):
    parser = experiment_parser(
        "Registered word-capacity FEAT ladder",
        engines=("scheduled_aligner",), default_seeds=STUDY_SEEDS,
    )
    parser.add_argument("--cells", nargs="+", choices=LADDER_CELLS)
    parser.add_argument("--vocabulary-sizes", nargs="+", type=int)
    parser.add_argument("--feature-areas", nargs="+", type=_area, metavar="N:K")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, STUDY_SEEDS)
    base = selected_protocol(smoke=args.smoke)
    ladder = None if args.feature_areas is None else tuple(args.feature_areas)
    protocol = base.select(
        cells=None if args.cells is None else tuple(args.cells),
        vocabulary_sizes=(None if args.vocabulary_sizes is None
                          else tuple(args.vocabulary_sizes)),
        feature_area=None if ladder is None else ladder[0],
        feature_ladder=ladder,
    )
    protocol = _validate_parameters(protocol.to_parameters())
    semantics = describe_hashed_aligner(
        p=protocol.connection_probability, beta=protocol.plasticity,
        rounds_word=protocol.rounds_per_pair, norm_init=True, scaling=True,
        w_max=None, stim_beta=0.0, stim_gain=None, store="present",
    )
    return run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, parameters=protocol.to_parameters(),
        measure=measure, aligner_semantics=semantics,
    )


if __name__ == "__main__":
    main()
