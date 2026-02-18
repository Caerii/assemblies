"""
Bootstrap Structural Connectivity

Materializes random baseline weight matrices for all core->structural
area pairs that were never projected through during training. This models
the biological reality that anatomical fibers exist between cortical
areas before any learning occurs.

References:
  - Catani & Mesulam 2008: arcuate fasciculus connectivity
  - research/plans/P600_REANALYSIS.md: full design rationale
"""

from typing import List, Optional, Callable

from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import NOUN_CORE, VERB_CORE


def bootstrap_structural_connectivity(
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
        structural_areas: Target areas to bootstrap connectivity into.
        source_areas: Source areas to bootstrap from. Defaults to
            [NOUN_CORE, VERB_CORE] for backward compatibility. Pass
            additional areas (e.g., NUMBER) for number-aware experiments.

    The bootstrap works in 3 steps for each empty (source, struct) pair:
      1. Project stimulus -> struct_area (gives struct winners, w > 0)
      2. Project stimulus -> source_area (gives source winners)
      3. Project stimulus+source -> struct (triggers _expand_connectomes
         for the source->struct pair; stimulus provides non-zero signal
         to avoid the zero-signal early return)
    """
    engine = parser.brain._engine
    brain = parser.brain

    # Use an arbitrary stimulus for bootstrapping
    arb_stim = next(iter(parser.stim_map.values()))

    core_areas = source_areas if source_areas is not None else [NOUN_CORE, VERB_CORE]
    bootstrapped = []

    brain.disable_plasticity = True

    for core_area in core_areas:
        for struct_area in structural_areas:
            # Check if weights are already materialized
            conn = engine._area_conns.get(core_area, {}).get(struct_area)
            if conn is None:
                continue
            if (conn.weights.ndim == 2
                    and conn.weights.shape[0] > 0
                    and conn.weights.shape[1] > 0):
                continue  # already has materialized weights

            # Step 1: Give structural area winners via stimulus
            brain.project({arb_stim: [struct_area]}, {})

            # Step 2: Give core area winners via stimulus
            brain.project({arb_stim: [core_area]}, {})

            # Step 3: Project stimulus + core -> struct
            # Stimulus provides non-zero signal (avoids zero-signal early return)
            # Core in from_areas triggers _expand_connectomes for core->struct
            brain.project(
                {arb_stim: [struct_area]},
                {core_area: [struct_area]},
            )

            bootstrapped.append((core_area, struct_area))

    brain.disable_plasticity = False

    # Clear all areas after bootstrapping
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    if log_fn and bootstrapped:
        log_fn(f"  Bootstrapped {len(bootstrapped)} connectivity pairs: "
               + ", ".join(f"{c}->{s}" for c, s in bootstrapped))
