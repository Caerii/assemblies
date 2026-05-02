"""Smoke tests for README and example entry points."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neural_assemblies.core.brain import Brain


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_readme_core_api_smoke() -> None:
    brain = Brain(p=0.05, engine="numpy_sparse")
    brain.add_stimulus("stim", size=32)
    brain.add_area("A", n=512, k=32, beta=0.05)
    brain.project({"stim": ["A"]}, {})
    assert len(brain.area_by_name["A"].winners) == 32


def test_readme_nemo_imports_smoke() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cupy")

    from neural_assemblies.nemo.language import LanguageLearner, SentenceGenerator

    learner = LanguageLearner(verbose=False)
    learner.hear_sentence(["dog", "chases", "cat"])
    generator = SentenceGenerator(learner)
    sentence = generator.generate_sentence(length=3)
    assert isinstance(sentence, list)
    assert len(sentence) == 3


def test_basic_example_script_runs() -> None:
    example = REPO_ROOT / "examples" / "01_basic_assembly_calculus.py"
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Merged assembly size:" in result.stdout


def test_example_notebooks_are_valid_json() -> None:
    notebook_dir = REPO_ROOT / "examples" / "notebooks"
    notebooks = sorted(
        path
        for path in notebook_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )
    assert notebooks
    assert not list(notebook_dir.glob("*.ipynb"))

    for notebook in notebooks:
        with notebook.open(encoding="utf-8") as handle:
            data = json.load(handle)
        assert data["nbformat"] >= 4
        assert data["cells"]
        assert data["cells"][0]["cell_type"] == "markdown"
