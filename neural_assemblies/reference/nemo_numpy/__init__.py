"""mdabagia/nemo numpy reference dynamics (parity backend)."""

from .areas import (
    FFArea,
    RecurrentArea,
    RefractedArea,
    ScaffoldNetwork,
    k_cap,
    positional_overlap,
)
from .fsm_network import FSMNetwork, run_mod3_fsm_numpy
from .scaffold_demo import ScaffoldRecallResult, compare_scaffold_vs_simple

__all__ = [
    "FFArea",
    "RecurrentArea",
    "RefractedArea",
    "ScaffoldNetwork",
    "FSMNetwork",
    "ScaffoldRecallResult",
    "compare_scaffold_vs_simple",
    "run_mod3_fsm_numpy",
    "k_cap",
    "positional_overlap",
]
