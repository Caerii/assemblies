"""Capacity: the assembly calculus cap and rate heterogeneity.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, SensitivityCheck, Status

CLAIMS: List[Result] = [
    Result(
        id="AC-CAP",
        engine="original capacity-note run: incompletely recorded (graded-similarity evidence compares explicit, materialized and sampled numpy_sparse); retained bracket: hashed_assembly_memory, the unrefracted Hebbian control arm of the paired capacity replay at (n, k) = (4000, 60)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/control/cells/B~14000~160/checkpoints/32/rank1",
            control_path="observations/conditions/control/cells/B~14000~160/checkpoints/128/rank1",
            relation="all-greater", minimum_effect=0.9,
            mechanism="Hebbian control arm: half-cue rank-1 recall at M = 32 (0.48 n/k, below the extensive ceiling) against M = 128 (1.9 n/k, above it); the ceiling lies between them on every brain",
        ),),
        sensitivity_gap="The retained bracket shows the Hebbian ceiling lies "
                        "between 0.48 and 1.9 n/k on twenty brains at one cell; "
                        "it does not pin the 1.15 constant, and the original "
                        "capacity-note run remains unidentified.",
        claim="Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct "
              "assemblies per area.",
        source="This repository (critical-load measurement).",
        evidence=("research/notes/categories/capacity_is_not_the_constraint_separation_is.md",
                  "research/notes/substrate/graded_similarity_and_sampler_load.md",
                  "refraction-paired-sensitivity-20260911, control arm (4000, 60), 20 brains: "
                  "rank-1 recall 1.000 on every brain at M = 32 and at M = 64 on 16 of 20 "
                  "(mean 0.81, min 0.03); at M = 128 the maximum over brains is 0.031"),
        evidence_refs=(
            EvidenceRef("research/notes/categories/capacity_is_not_the_constraint_separation_is.md", "analysis"),
            EvidenceRef("research/notes/substrate/graded_similarity_and_sampler_load.md", "analysis"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json", "artifact",
                        "Hebbian control arm only; brackets the extensive ceiling at one (n, k) cell"),
        ),
        provenance_gap="capacity-note run has no identifiable immutable artifact",
        caveat="The capacity-note run's engine provenance remains unresolved; "
               "this entry does not certify the 1.15 constant. What the "
               "retained control arm certifies is the extensive form at one "
               "cell: the Hebbian area holds every item at half n/k and none "
               "at twice n/k. "
               "This is why k and p are not interchangeable routes to a regime: "
               "raising k to reach kp spends capacity and forces n up with it.",
    ),

    Result(
        id="RATE-HETEROGENEITY",
        engine="numpy_explicit (materialized copied-fiber protocol)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json",
            sample_path="observations/rows/*/seed",
            treatment_path="observations/rows/*/cells/forward/ratio",
            control_path="observations/rows/*/cells/equal/ratio",
            relation="all-greater", minimum_effect=9.0,
            mechanism="per-fiber rate contrast versus equal-rate null",
        ),),
        claim="Learning rate is settable PER FIBER: copied fibers with identical "
              "pre/post activity at beta 0.06 and 0.005 diverge 14.35-fold after "
              "50 updates without reaching the weight clip.",
        source="This repository.",
        evidence=("20 seeds: fast geometric mean 18.4201012, slow 1.2832253, "
                  "ratio 14.3545340; swapped labels reproduce the ratio, equal-beta "
                  "null is 1.0, and maximum weight 18.4201012 is below w_max=20",),
        evidence_refs=(
            EvidenceRef("research/notes/memory/PREREG_per_fiber_plasticity.md", "registration"),
            EvidenceRef("research/experiments/per_fiber_plasticity.py", "producer"),
            EvidenceRef("research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json", "artifact"),
        ),
        implemented_by=("neural_assemblies.core.brain.Brain.update_plasticity",
                        "research.experiments.per_fiber_plasticity"),
        caveat="This isolates the multiplicative rate on copied materialized NumPy "
               "fibers; it does not measure assembly quality or biological "
               "heterogeneity, and it does not establish sampled or hashed backend "
               "equivalence. Near-zero intervals reflect the controlled arithmetic, "
               "not population certainty. The old saturated beta=0.5 number remains "
               "unreproduced and is not evidence for this entry.",
    ),
]
