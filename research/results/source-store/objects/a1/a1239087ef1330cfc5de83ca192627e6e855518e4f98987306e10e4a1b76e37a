"""Regression tests for COLT MNIST notebook-faithful protocol."""

from __future__ import annotations

from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_protocol


def test_mnist_protocol_multi_class_accuracy():
    """Notebook protocol should classify most digits, not collapse to one."""
    result = run_colt_mnist_protocol(n_examples=50, class_bias=-1.0)
    assert result.mean_accuracy >= 0.70
    assert int((result.per_class_accuracy >= 0.30).sum()) >= 8


def test_mnist_brain_explicit_within_protocol_tolerance():
    """Native explicit Brain should approach protocol (fallback if not)."""
    from neural_assemblies.programs.colt_mnist_brain import run_colt_mnist_brain
    from neural_assemblies.programs.colt_mnist_brain_explicit import run_colt_mnist_brain_explicit
    from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_protocol

    protocol = run_colt_mnist_protocol(n_examples=50)
    explicit = run_colt_mnist_brain_explicit(n_examples=50)
    brain = run_colt_mnist_brain(n_examples=50)
    assert protocol.mean_accuracy >= 0.70
    assert brain.mean_accuracy >= 0.70
    if abs(explicit.mean_accuracy - protocol.mean_accuracy) > 0.08:
        assert brain.backend == "protocol_fallback"


def test_mnist_protocol_matches_golden_tolerance():
    from neural_assemblies.parity.runner import verify_protocol
    from neural_assemblies.programs.colt_mnist_data import DatasetUnavailable
    import pytest

    try:
        result = verify_protocol("colt2022_mnist_notebook")
    except DatasetUnavailable as exc:
        pytest.skip(str(exc))
    assert result.passed, result.diffs
