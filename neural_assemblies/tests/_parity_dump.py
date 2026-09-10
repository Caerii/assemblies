"""Record each parity gate's worst relative error for the figure.

Set ``NEMO_PARITY_DUMP=1`` and run the parity suites; every gate then
writes its worst error under a label to
``research/results/substrate/parity_errors.json`` (the maximum per label
across parameter cases). ``research/experiments/figures_notes.py``
(``parity_gates``) draws the file. Without the variable this is a no-op,
so the tests stay pure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[2] / "research" / "results"
         / "substrate" / "parity_errors.json")


def record(label: str, err: float) -> None:
    if os.environ.get("NEMO_PARITY_DUMP") != "1":
        return
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        d = {}
    d[label] = max(float(err), float(d.get(label, 0.0)))
    _PATH.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
