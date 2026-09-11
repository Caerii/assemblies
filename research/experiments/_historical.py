"""Shared execution adapter for explicitly specified historical protocols."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from research.runner import ROOT, experiment_parser, run_experiment


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
        args = parser.parse_args(argv)
        print(writer(script=self.script, protocol=self.protocol, protocol_version=self.version,
                     registration=self.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
                     smoke=args.smoke, parameters=self.parameters(args.smoke), measure=self.measure))
