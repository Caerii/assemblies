"""
Measurement Metrics for ERP Experiments

Reusable measurement functions for N400 (energy, settling, prediction)
and P600 (instability) metrics in Assembly Calculus.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research.experiments.metrics.energy import measure_pre_kwta_activation
from research.experiments.metrics.settling import measure_settling_dynamics
from research.experiments.metrics.prediction import (
    measure_prediction_error,
)
from research.experiments.metrics.instability import (
    measure_p600_settling,
    compute_jaccard_instability,
)

__all__ = [
    "measure_pre_kwta_activation",
    "measure_settling_dynamics",
    "measure_prediction_error",
    "measure_p600_settling",
    "compute_jaccard_instability",
]
