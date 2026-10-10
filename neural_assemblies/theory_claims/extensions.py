"""Extensions: claims beyond what is proved or measured here.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, SensitivityCheck, Status

CLAIMS: List[Result] = [
    # ------------------------------------------------------------- extensions
    Result(
        id="DUAL-RATE",
        status=Status.EXTENSION,
        claim="Running fast and slow pathways at once is FUNCTIONALLY useful: "
              "a high-beta fiber binds in one shot (episodic) while a low-beta "
              "fiber accumulates statistics (semantic), and a system with both "
              "does something neither does alone.",
        source="Proposed here; not proved and not measured.",
        preconditions=("[[RATE-HETEROGENEITY]] -- the mechanism exists",
                       "[[SEQ-BETA-WINDOW]] -- each rate must sit inside its "
                       "own window, and the windows may not overlap"),
        caveat="UNTESTED. That the knob exists is measured; that turning it "
               "buys anything is not. Two known tensions to design against: "
               "beta trades capacity against depth (low favours capacity, high "
               "favours depth), and the afferent count flips the SIGN of beta's "
               "effect, so a rate that helps one fiber can hurt another at a "
               "different kp. Falsified if a dual-rate organ matches the better "
               "of its two single-rate controls.",
    ),
    Result(
        id="SEQ-ORGAN-EMBEDS",
        engine="numpy_sparse, arc materialized (retained runner replay, 10 seeds); the original organ-density experiment ran on the sampled arc",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.a1-local-regime/sens-local-regime-v3-20260912/results.json",
            sample_path="observations/arms/0/rows/*/seed",
            treatment_path="observations/arms/0/rows/*/both_correct",
            control_path="observations/arms/1/rows/*/both_correct",
            relation="all-greater", minimum_effect=1,
            mechanism="organ fibers at p = 0.4 inside an ambient p = 0.05 brain against the same organ left at the ambient density (kp 3.5 against a floor of 18.6), arc materialized",
        ),),
        sensitivity_gap="",
        claim="A sequence organ runs at its own regime INSIDE a brain whose "
              "ambient density is far lower, given per-fiber p.",
        source="This repository.",
        preconditions=("per-fiber p (`Brain.add_connectivity`, numpy_exact and "
                       "numpy_sparse)",
                       "EVERY per-fiber quantity scaled by the fiber's p, not "
                       "the global one -- see the caveat"),
        evidence=("research/experiments/seq_a1_local_regime.py: ambient p=0.05 "
                  "with organ fibers at p=0.4 gives 10/10 correct trajectories, "
                  "matching the uniform p=0.4 result, while the same organ left "
                  "at the ambient density gives 0/10",),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a1_local_regime_results.json", "artifact",
                        "sampled arc; sequence verdict void"),
            EvidenceRef("research/results/sequence/seq_a1_local_regime_results_materialized.json", "artifact"),
            EvidenceRef("research/results/runs/sequence.a1-local-regime/sens-local-regime-v3-20260912/results.json", "artifact",
                        "shared-runner replay with source archive; the ambient-only arm is the disabled-organ null"),
            EvidenceRef("research/experiments/seq_a1_local_regime.py", "producer"),
        ),
        provenance_gap="the two legacy artifacts lack runner records and source archives; the 2026-09-12 replay has both",
        caveat="The original organ-density experiment used the sampled numpy arc; "
               "its sequence-dynamics numbers are void under PREREG_sampler_audit.md "
               "until reproduced materialized or hashed. "
               "The heterogeneous path found a defect that the regime audit "
               "could NOT see: the stimulus weight clamp was scaled by the "
               "global p while the weights were drawn at the fiber's p, so a "
               "dense fiber in a sparse brain saturated at a sparse ceiling. "
               "The organ read 1/10 while every area reported comfortably "
               "in-regime. Any NEW per-fiber quantity is a candidate for the "
               "same class. The pooled candidate draw also remains "
               "moment-matched -- exact in the first two moments, an "
               "approximation beyond them.",
    ),
    Result(
        id="SEQ-STATE-CODE-EMERGENT",
        status=Status.EXTENSION,
        claim="The state alphabet can be INDUCED from data rather than assigned.",
        source="Not proved and not measured, here or in the source paper.",
        caveat="THE LOAD-BEARING GAP. Both our FSM and the reference assign "
               "state assemblies as disjoint blocks by hand. Every result under "
               "[[SEQ-FSM]] and [[SEQ-EXACT-RECOVERY]] is therefore about "
               "running a machine over a GIVEN alphabet, not about discovering "
               "one. Language needs the latter.",
    ),
]
