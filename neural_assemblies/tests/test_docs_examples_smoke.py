"""Smoke tests for README and example entry points."""

from __future__ import annotations

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
