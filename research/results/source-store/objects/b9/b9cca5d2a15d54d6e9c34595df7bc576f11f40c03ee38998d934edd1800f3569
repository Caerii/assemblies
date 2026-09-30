"""Materialize baseline structural connectivity on EmergentParser.

Creates random baseline weight matrices for core→structural area pairs
that were never projected during training (anatomical fibers before learning).

This is NOT POS inference (``emergent.acquisition.pos_inference``) or
wobbly episode replay (``emergent.acquisition.replay_wobbly_episodes``).

References:
  - Catani & Mesulam 2008: arcuate fasciculus connectivity
  - research/plans/P600_REANALYSIS.md
"""

from typing import List, Optional, Callable

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import NOUN_CORE, VERB_CORE


def materialize_structural_connectivity(
    parser: EmergentParser,
    structural_areas: List[str],
    source_areas: Optional[List[str]] = None,
    log_fn: Optional[Callable] = None,
) -> None:
    """Materialize weight matrices for all source->structural area pairs.

    After parser.train(), some source->structural pathways have never been
    projected through, leaving weight matrices empty (0x0 in sparse engine).
    This forces a projection through each pathway with plasticity OFF,
    materializing random binomial(p) baseline weights.

    Biologically: anatomical fibers exist between cortical areas before
    learning (arcuate fasciculus, thalamocortical projections). Training
    strengthens specific pathways; untrained pathways retain baseline
    connectivity.

    Args:
        structural_areas: Target areas to materialize connectivity into.
        source_areas: Source areas to project from. Defaults to
            [NOUN_CORE, VERB_CORE]. Pass additional areas (e.g., NUMBER)
            for number-aware experiments.
        log_fn: Optional callback for logging materialized pairs.

    For each empty (source, struct) pair:
      1. Project stimulus -> struct_area (gives struct winners, w > 0)
      2. Project stimulus -> source_area (gives source winners)
      3. Project stimulus+source -> struct (materializes source->struct weights)
    """
    engine = parser.brain._engine
    brain = parser.brain

    arb_stim = next(iter(parser.stim_map.values()))

    core_areas = source_areas if source_areas is not None else [NOUN_CORE, VERB_CORE]
    materialized = []

    brain.disable_plasticity = True

    for core_area in core_areas:
        for struct_area in structural_areas:
            conn = engine._area_conns.get(core_area, {}).get(struct_area)
            if conn is None:
                continue
            if (conn.weights.ndim == 2
                    and conn.weights.shape[0] > 0
                    and conn.weights.shape[1] > 0):
                continue

            brain.project({arb_stim: [struct_area]}, {})
            brain.project({arb_stim: [core_area]}, {})
            brain.project(
                {arb_stim: [struct_area]},
                {core_area: [struct_area]},
            )

            materialized.append((core_area, struct_area))

    brain.disable_plasticity = False

    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    if log_fn and materialized:
        log_fn(
            f"  Materialized {len(materialized)} connectivity pairs: "
            + ", ".join(f"{c}->{s}" for c, s in materialized),
        )


def bootstrap_structural_connectivity(
    parser: EmergentParser,
    structural_areas: List[str],
    source_areas: Optional[List[str]] = None,
    log_fn: Optional[Callable] = None,
) -> None:
    """Backward-compatible alias for ``materialize_structural_connectivity``."""
    materialize_structural_connectivity(
        parser,
        structural_areas,
        source_areas=source_areas,
        log_fn=log_fn,
    )
