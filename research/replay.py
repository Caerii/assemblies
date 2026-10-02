"""Replay a committed run on this checkout's code and demand identical observations.

Source-linked specification: research/README.md#recoverable-source

The gate for an engine change that claims to be exact (a faster kernel, a
batched launch, a fused op): rerun a study the repository has already
recorded, from the inputs its record captured, and compare the new
observations with the recorded ones byte for byte after a JSON round trip
(the record went through one). A study passes nothing it did not pass before
and fails nothing it did not fail; an exact change reproduces it entirely.

The run's measure function is found in the script its record names, as the
runner called it: ``measure`` if the module defines one, else ``experiment``.
With ``--cells``, only those cells of a study whose parameters list cells
(``n/k/p``) are rerun and compared -- a quick gate before the whole replay.

    python -m research.replay research/results/runs/memory.threshold-law/threshold-law-20261001/results.json
    python -m research.replay RECORD --cells 4000/60/0.5 4000/10/0.5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell_key(spec) -> str:
    return f"{spec['n']}/{spec['k']}/{spec['p']:g}"


def measure_of(script: Path):
    """The function the runner called for a run of ``script``."""
    spec = importlib.util.spec_from_file_location(f"_replayed_{script.stem}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    for name in ("measure", "experiment"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise LookupError(f"{script} defines neither measure nor experiment")


def replay(record_path, *, cells=None, root: Path = ROOT, measure=None) -> dict:
    """Rerun the run recorded at ``record_path``; returns the comparison."""
    from research.json_documents import snapshot_document
    recorded = json.loads(Path(record_path).read_text(encoding="utf-8"))
    run = dict(recorded["run"])
    if cells:
        listed = run["parameters"].get("cells")
        if not isinstance(listed, list) or not all(isinstance(c, dict) for c in listed):
            raise ValueError("--cells needs a study whose parameters list cells")
        known = {cell_key(c) for c in listed}
        missing = set(cells) - known
        if missing:
            raise ValueError(f"cells not in the record: {sorted(missing)}")
        run["parameters"] = {**run["parameters"],
                             "cells": [c for c in listed if cell_key(c) in set(cells)]}
    fn = measure if measure is not None else measure_of(root / run["script"])
    start = time.perf_counter()
    measured = fn(snapshot_document(run))
    seconds = time.perf_counter() - start
    measured = getattr(measured, "observations", measured)
    observed = json.loads(json.dumps(measured))
    want = recorded["observations"]
    if cells:
        observed, want = observed["cells"], {k: want["cells"][k] for k in observed["cells"]}
        differs = sorted(k for k in observed if observed[k] != want[k])
    else:
        differs = sorted(k for k in set(observed) | set(want) if observed.get(k) != want.get(k))
    return {"record": str(record_path), "seconds": seconds,
            "identical": not differs, "differs": differs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record")
    parser.add_argument("--cells", nargs="+", help="n/k/p cells to rerun (default: all)")
    args = parser.parse_args(argv)
    report = replay(args.record, cells=args.cells)
    print(json.dumps(report, indent=1))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
