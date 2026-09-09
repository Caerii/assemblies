"""Where an experiment's results file lives.

Results of the active lines are kept under ``research/results/<line>/`` so
the experiments folder holds code and the results folder holds evidence.
Scripts call ``results_path(line, name)`` for both reading and writing.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(_ROOT, "research", "results")


def results_path(line: str, name: str) -> str:
    d = os.path.join(RESULTS, line)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
