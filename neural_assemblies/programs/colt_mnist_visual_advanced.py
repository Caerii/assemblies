"""
Public entry points for advanced MNIST visual models.

Simple illustration protocols (notebook port, two-layer, four-area Brain)
live in ``colt_mnist_protocol``, ``colt_mnist_brain_explicit``, and
``colt_mnist_hierarchical_brain``.

This module exposes ventral-stream models built on those baselines.  For
hypotheses explaining the ~64% vs ~81% hierarchical/recurrent gap, ventral
information structures, and a assembly-calculus roadmap toward >95%, see
``colt_mnist_ventral_theory``.

Example
-------
>>> from neural_assemblies.programs.colt_mnist_visual_advanced import (
...     run_recurrent_cortex_mnist,
... )
>>> result = run_recurrent_cortex_mnist(n_examples=50)
>>> result.mean_accuracy  # ~0.80 with real MNIST CSV
"""

from neural_assemblies.programs.colt_mnist_ventral_theory import (
    HYPOTHESES,
    ROADMAP_TO_95,
    VENTRAL_CODE_STRUCTURES,
    HypothesisId,
    ResearchTier,
    simple_vs_recurrent_summary,
)
from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
    ColtMnistVisualAdvancedResult,
    run_colt_mnist_visual_advanced,
    run_fuzzy_lexicon_mnist,
    run_recurrent_cortex_mnist,
    run_ventral_stream_mnist,
)

__all__ = [
    "ColtMnistVisualAdvancedResult",
    "HYPOTHESES",
    "HypothesisId",
    "ResearchTier",
    "ROADMAP_TO_95",
    "VENTRAL_CODE_STRUCTURES",
    "run_colt_mnist_visual_advanced",
    "run_ventral_stream_mnist",
    "run_recurrent_cortex_mnist",
    "run_fuzzy_lexicon_mnist",
    "simple_vs_recurrent_summary",
]
