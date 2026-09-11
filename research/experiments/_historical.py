"""Shared execution adapter for explicitly specified historical protocols."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

from research.json_documents import decode_document
from research.runner import ROOT, experiment_parser, run_experiment
from neural_assemblies import describe_brain_model


@dataclass(frozen=True)
class HistoricalStudy:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-study-adapter"""
    protocol: str
    version: str
    registration: str
    script: Path
    factory: Callable
    parameters: Callable
    scope: str
    engine: str = "numpy_explicit"
    default_seeds: tuple[int, ...] = tuple(range(42, 52))

    def measure(self, record):
        for key, expected in (("protocol", self.protocol), ("protocol_version", self.version), ("engine", self.engine)):
            if record.get(key) != expected:
                raise ValueError(f"{self.protocol} requires {key}={expected!r}")
        if record.get("mode") not in ("smoke", "study"):
            raise ValueError("historical study mode must be smoke or study")
        producer = self.factory(seed=0, verbose=False,
                                results_dir=ROOT / "research/results/runs" / self.protocol / record["tag"])
        result = producer.run(seed_ids=record["seeds"], **record["parameters"])
        return {"verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
                "scope": self.scope, "result": result.to_dict()}

    def main(self, argv=None, *, writer=run_experiment):
        parser = experiment_parser(self.scope, engines=(self.engine,), default_seeds=self.default_seeds)
        parser.add_argument("--quick", action="store_true", dest="smoke", help="alias for VOID smoke")
        parser.add_argument("--parameters", type=Path,
                            help="repository-relative JSON overrides for protocol parameters")
        args = parser.parse_args(argv)
        parameters = dict(self.parameters(args.smoke))
        inputs = {}
        if args.parameters is not None:
            try:
                path = (ROOT / args.parameters).resolve()
                name = path.relative_to(ROOT).as_posix()
                data = path.read_bytes()
                overrides = decode_document(data.decode("utf-8"))
                if not isinstance(overrides, dict):
                    raise ValueError("parameter file must contain a JSON object")
                unknown = overrides.keys() - parameters.keys()
                if unknown:
                    raise ValueError(f"unknown or reserved parameters: {sorted(unknown)}")
                parameters.update(overrides)
                inputs = {"input_artifacts": (name,),
                          "expected_input_digests": {name: hashlib.sha256(data).hexdigest()}}
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        print(writer(script=self.script, protocol=self.protocol, protocol_version=self.version,
                     registration=self.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
                     smoke=args.smoke, parameters=parameters, measure=self.measure,
                     model_semantics=describe_brain_model(
                         self.engine, p=parameters.get("p", .05), seed=0,
                         w_max=parameters.get("w_max", 20.), norm_init=False,
                     ), **inputs))
