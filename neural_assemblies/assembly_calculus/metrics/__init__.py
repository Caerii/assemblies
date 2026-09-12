"""Assembly Calculus ERP metric kernels (package layer).

Shared by research primitives and emergent parser adapters.
Parser-specific readiness, baselines, and violation typing live in
``emergent.evaluation.erp_metrics``.
"""

from .instability import (
    cosine_similarity,
    compute_anchored_instability,
    compute_jaccard_instability,
    jaccard_similarity,
    mean_jaccard_instability,
)
from .prediction import measure_n400

__all__ = [
    "compute_anchored_instability",
    "compute_jaccard_instability",
    "jaccard_similarity",
    "cosine_similarity",
    "mean_jaccard_instability",
    "measure_n400",
]
