"""
P600 Pipeline Helpers

Standard setup sequences for P600 experiments: bootstrap + consolidate.
"""

from typing import List, Optional, Callable

from research.experiments.infrastructure.bootstrap import (
    bootstrap_structural_connectivity,
)
from research.experiments.infrastructure.consolidation import (
    consolidate_role_connections,
    consolidate_vp_connections,
    consolidate_number_role_connections,
    consolidate_number_vp_connections,
)


def setup_p600_pipeline(
    parser,
    training,
    structural_areas: List,
    source_areas: Optional[List] = None,
    consolidation_passes: int = 1,
    log_fn: Optional[Callable] = None,
):
    """Standard P600 pipeline: bootstrap + consolidate role + consolidate VP.

    Args:
        parser: Trained EmergentParser instance.
        training: Training sentences (for consolidation replay).
        structural_areas: Areas to bootstrap (e.g., [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]).
        source_areas: Additional source areas for bootstrap (default: None).
        consolidation_passes: Number of consolidation passes (default: 1).
        log_fn: Optional logging function.
    """
    bootstrap_structural_connectivity(
        parser, structural_areas,
        source_areas=source_areas,
        log_fn=log_fn,
    )
    consolidate_role_connections(
        parser, training, n_passes=consolidation_passes, log_fn=log_fn,
    )
    consolidate_vp_connections(
        parser, training, n_passes=consolidation_passes, log_fn=log_fn,
    )


def setup_number_p600_pipeline(
    parser,
    training,
    structural_areas: List,
    source_areas: Optional[List] = None,
    base_consolidation_passes: int = 1,
    number_consolidation_passes: int = 1,
    log_fn: Optional[Callable] = None,
):
    """Number-aware P600 pipeline: standard + number role + number VP.

    Calls setup_p600_pipeline for the base structural patterns, then
    adds number-specific consolidation passes.

    Args:
        parser: Trained EmergentParser instance.
        training: Training sentences (for consolidation replay).
        structural_areas: Areas to bootstrap.
        source_areas: Additional source areas for bootstrap (e.g., [NUMBER]).
        base_consolidation_passes: Passes for standard role/VP consolidation.
        number_consolidation_passes: Passes for number-specific consolidation.
        log_fn: Optional logging function.
    """
    setup_p600_pipeline(
        parser, training, structural_areas,
        source_areas=source_areas,
        consolidation_passes=base_consolidation_passes,
        log_fn=log_fn,
    )
    consolidate_number_role_connections(
        parser, training, n_passes=number_consolidation_passes, log_fn=log_fn,
    )
    consolidate_number_vp_connections(
        parser, training, n_passes=number_consolidation_passes, log_fn=log_fn,
    )
