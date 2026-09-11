"""Where an experiment's results file lives.

Results of the active lines are kept under ``research/results/<line>/`` so
the experiments folder holds code and the results folder holds evidence.
Scripts call ``results_path(line, name)`` for both reading and writing.
"""
import os
from pathlib import Path

from research.json_documents import write_new_document

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(_ROOT, "research", "results")


def results_path(line: str, name: str) -> Path:
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
