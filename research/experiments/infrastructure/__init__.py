"""
Experiment Infrastructure

Reusable parser setup, lexicon building, bootstrap connectivity,
and consolidation for N400/P600 experiments.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research.experiments.infrastructure.lexicon import build_core_lexicon
from research.experiments.infrastructure.bootstrap import (
    bootstrap_structural_connectivity,
)
from research.experiments.infrastructure.consolidation import (
    consolidate_role_connections,
    consolidate_vp_connections,
)

__all__ = [
    "build_core_lexicon",
    "bootstrap_structural_connectivity",
    "consolidate_role_connections",
    "consolidate_vp_connections",
]
