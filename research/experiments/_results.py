"""Where an experiment's results file lives.

Results of the active lines are kept under ``research/results/<line>/`` so
the experiments folder holds code and the results folder holds evidence.
Scripts call ``results_path(line, name)`` for both reading and writing.
"""
import os
from pathlib import Path
import re

from research.json_documents import write_new_document

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(_ROOT, "research", "results")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _validate_component(label: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{label} must be a simple result name (letters, digits, dot, dash, underscore)"
        )


def results_path(line: str, name: str) -> Path:
    _validate_component("line", line)
    _validate_component("name", name)
    d = os.path.join(RESULTS, line)
    os.makedirs(d, exist_ok=True)
    return Path(d) / name


def write_result(line: str, name: str, value) -> Path:
    """Create one result document without overwriting prior evidence.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-evidence-json
    """
    path = results_path(line, name)
    write_new_document(path, value)
    return path
