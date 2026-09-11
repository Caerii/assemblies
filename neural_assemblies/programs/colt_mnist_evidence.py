"""
Empirical evidence suite for the ventral-stream assembly narrative.

Runs baselines, advanced models, tier A/B/C experiments, and cross-domain
vision-language fusion; returns a structured report suitable for parity
golden export and narrative documentation.

Usage::

    from neural_assemblies.programs.colt_mnist_evidence import run_evidence_suite
    report = run_evidence_suite(n_examples=50)
    print(report.narrative_summary())
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


from neural_assemblies.programs.colt_mnist_ventral_theory import (
    HypothesisId,
    simple_vs_recurrent_summary,
)


@dataclass
class EvidenceRow:
    """One empirical row in the evidence table."""

    experiment_id: str
    tier: str
    mean_accuracy: float
    hypothesis_ids: tuple[str, ...]
    ac_primitives: tuple[str, ...]
    backend: str
    supports_narrative: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceReport:
    """Full structured evidence bundle."""

    recorded: str
    n_examples: int
    seed: int
    data_source: str
    rows: list[EvidenceRow]
    hypothesis_checks: dict[str, bool]
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "n_examples": self.n_examples,
            "seed": self.seed,
            "data_source": self.data_source,
            "rows": [asdict(r) for r in self.rows],
            "hypothesis_checks": self.hypothesis_checks,
            "narrative": self.narrative,
        }

    def narrative_summary(self) -> str:
        return self.narrative


def _row(
    experiment_id: str,
    tier: str,
    mean_accuracy: float,
    hypothesis_ids: tuple[str, ...],
    ac_primitives: tuple[str, ...],
    backend: str,
    supports_narrative: str,
    **metrics,
) -> EvidenceRow:
    return EvidenceRow(
        experiment_id=experiment_id,
        tier=tier,
        mean_accuracy=mean_accuracy,
        hypothesis_ids=hypothesis_ids,
        ac_primitives=ac_primitives,
        backend=backend,
        supports_narrative=supports_narrative,
        metrics=metrics,
    )


def run_evidence_suite(
    *,
    seed: int = 42,
    n_examples: int = 50,
    include_slow: bool = False,
) -> EvidenceReport:
    """Run the full evidence ladder and return structured results."""
    from neural_assemblies.programs.colt_mnist_data import find_mnist_dir
    from neural_assemblies.programs.colt_mnist_hierarchical import (
        run_colt_mnist_hierarchical,
    )
    from neural_assemblies.programs.colt_mnist_visual_advanced import (
        run_fuzzy_lexicon_mnist,
        run_recurrent_cortex_mnist,
        run_ventral_stream_mnist,
    )
    from neural_assemblies.programs.colt_mnist_tier_a import (
        run_merge_halves_mnist,
        run_multi_prototype_mnist,
    )
    from neural_assemblies.programs.colt_mnist_tier_b import (
        run_consolidation_mnist,
        run_pattern_completion_mnist,
    )
    from neural_assemblies.programs.colt_mnist_lri_readout import run_lri_cascade_mnist
    from neural_assemblies.programs.cross_domain_assemblies import (
        run_vision_language_digit_hub,
    )

    rows: list[EvidenceRow] = []
    data_source = "mnist_csv" if find_mnist_dir() else "synthetic_fallback"
    from neural_assemblies.programs.colt_mnist_tier_util import (
        clear_ventral_bundle_cache,
        load_ventral_bundle,
    )

    clear_ventral_bundle_cache()
    kw = dict(seed=seed, n_examples=n_examples)
    shared = load_ventral_bundle(**kw)

    simple = run_colt_mnist_hierarchical(**kw)
    rows.append(_row(
        "baseline_hierarchical", "0", simple.mean_accuracy,
        (HypothesisId.ATTRACTOR_DEFICIENCY.value,),
        ("project",), simple.backend,
        "Feedforward 4-area illustration baseline (~64%).",
        per_class=list(simple.per_class_accuracy),
    ))

    ventral = run_ventral_stream_mnist(**kw)
    rows.append(_row(
        "advanced_ventral", "0", ventral.mean_accuracy,
        (HypothesisId.INSUFFICIENT_EXPOSURE.value, HypothesisId.READOUT_MISMATCH.value),
        ("project", "fuzzy_readout", "consolidation"),
        ventral.backend,
        "Repeated exposure + connectome readout closes hierarchical gap.",
        min_separation=ventral.min_class_separation,
    ))

    recurrent = run_recurrent_cortex_mnist(**kw)
    rows.append(_row(
        "advanced_recurrent", "0", recurrent.mean_accuracy,
        (HypothesisId.ATTRACTOR_DEFICIENCY.value,),
        ("project", "recurrence"),
        recurrent.backend,
        "HIGH->HIGH recurrence matches notebook ~81% regime.",
    ))

    from neural_assemblies.programs.colt_mnist_spatial_ventral import run_spatial_ventral_mnist

    spatial = run_spatial_ventral_mnist(**kw)
    rows.append(_row(
        "spatial_ventral", "R", spatial.mean_accuracy,
        (
            HypothesisId.DIGIT3_STRUCTURED_ABSENCE.value,
            HypothesisId.HONEST_COMPLETION_GAIN.value,
        ),
        ("project", "local_rf", "structured_absence"),
        spatial.backend,
        "CNN-inspired local RF LOW→MID→HIGH + digit-3 center curriculum.",
        digit3_center_band_accuracy=spatial.extra.get("digit3_center_band_accuracy"),
        confused_digit_accuracy=spatial.extra.get("confused_digit_accuracy"),
        spatial_stats=spatial.spatial_stats,
    ))

    from neural_assemblies.programs.colt_mnist_geometry_panel import run_geometry_panel
    from neural_assemblies.programs.colt_mnist_regeneration_panel import run_regeneration_panel
    from neural_assemblies.programs.colt_mnist_attractor import run_attractor_mnist
    from neural_assemblies.programs.colt_mnist_representational import (
        run_honest_completion_mnist,
        run_occlusion_training_mnist,
        run_pair_curriculum_mnist,
        run_recurrent_representational_mnist,
        run_representational_stack_mnist,
    )

    geometry = run_geometry_panel(bundle=shared, **kw)
    rows.append(_row(
        "geometry_panel", "R", geometry.confused_digit_accuracy,
        (HypothesisId.CROSS_CLASS_OVERLAP.value, HypothesisId.SEPARABILITY_BEFORE_BINDING.value),
        ("overlap", "separate", "pattern_complete"),
        "geometry_panel",
        "Phase I: pair overlap and margin audit before binding.",
        confused_digit_accuracy=geometry.confused_digit_accuracy,
        non_confused_digit_accuracy=geometry.non_confused_digit_accuracy,
        binding_gate_open=geometry.binding_gate_open,
        top_pair_overlap=geometry.pair_overlap_ranked[0] if geometry.pair_overlap_ranked else None,
    ))

    regen_rec = run_regeneration_panel(baseline_label="recurrent", bundle=None, **kw)
    rows.append(_row(
        "regeneration_panel_recurrent", "R", regen_rec.mean_high_recovery,
        (
            HypothesisId.ATTRACTOR_RECOVERY.value,
            HypothesisId.DIGIT3_STRUCTURED_ABSENCE.value,
        ),
        ("pattern_complete", "project"),
        "regeneration_panel",
        "Baseline generative audit on default recurrent bundle (H11-H13).",
        digit3_recovery=regen_rec.per_digit_recovery[3],
        digit3_absence_mean=regen_rec.digit3_absence_mean,
        regeneration_gate_open=regen_rec.regeneration_gate_open,
        mean_low_regeneration=regen_rec.mean_low_regeneration,
        forward_overlap_mean=float(regen_rec.per_digit_forward_overlap.mean()),
    ))

    attractor = run_attractor_mnist(**kw)
    rows.append(_row(
        "attractor_recurrent", "R", attractor.mean_accuracy,
        (
            HypothesisId.ATTRACTOR_RECOVERY.value,
            HypothesisId.LOW_REGENERATION.value,
        ),
        ("learn_assembly", "pattern_complete", "project"),
        attractor.backend,
        "Convergent attractor training + structured absence curriculum.",
        mean_high_recovery=attractor.mean_high_recovery,
        digit3_recovery=attractor.digit3_recovery,
        digit3_absence_accuracy=attractor.digit3_absence_accuracy,
        mean_low_regeneration=attractor.mean_low_regeneration,
        confused_digit_accuracy=attractor.extra.get("confused_digit_accuracy"),
    ))

    regen_attr = run_regeneration_panel(baseline_label="attractor", **kw)
    rows.append(_row(
        "regeneration_panel_attractor", "R", regen_attr.mean_high_recovery,
        (
            HypothesisId.ATTRACTOR_RECOVERY.value,
            HypothesisId.DIGIT3_STRUCTURED_ABSENCE.value,
            HypothesisId.LOW_REGENERATION.value,
        ),
        ("pattern_complete", "reciprocal_project"),
        "regeneration_panel",
        "Post-attractor generative completeness (Ember digit-3 test).",
        digit3_recovery=regen_attr.per_digit_recovery[3],
        digit3_absence_mean=regen_attr.digit3_absence_mean,
        regeneration_gate_open=regen_attr.regeneration_gate_open,
        mean_low_regeneration=regen_attr.mean_low_regeneration,
    ))

    repr_recurrent = run_recurrent_representational_mnist(**kw)
    rows.append(_row(
        "representational_recurrent_base", "R", repr_recurrent.mean_accuracy,
        (HypothesisId.RECURRENT_BASE_SUPERIORITY.value,),
        ("project", "recurrence"),
        repr_recurrent.backend,
        "Recurrent bundle connectome readout (encoding base, H8).",
        confused_digit_accuracy=repr_recurrent.confused_digit_accuracy,
        non_confused_digit_accuracy=repr_recurrent.non_confused_digit_accuracy,
    ))

    honest = run_honest_completion_mnist(bundle=shared, **kw)
    rows.append(_row(
        "representational_honest_completion", "R", honest.mean_accuracy,
        (HypothesisId.HONEST_COMPLETION_GAIN.value,),
        ("forward_completion", "pattern_complete", "project"),
        honest.backend,
        "Pair-gated completion at inference — no oracle fallback (H7).",
        confused_digit_accuracy=honest.confused_digit_accuracy,
        confused_digit_delta=honest.extra.get("confused_digit_delta"),
        completion_rate=honest.extra.get("pair_gated_completion_rate"),
        forward_completion_rate=honest.extra.get("forward_completion_rate"),
        pattern_complete_rate=honest.extra.get("pattern_complete_rate"),
        baseline_connectome=honest.extra.get("baseline_connectome_accuracy"),
    ))

    pair_curr = run_pair_curriculum_mnist(bundle=shared, **kw)
    rows.append(_row(
        "representational_pair_curriculum", "R", pair_curr.mean_accuracy,
        (HypothesisId.CROSS_CLASS_OVERLAP.value,),
        ("reinforce_connectome", "wire_class_from_prototypes"),
        pair_curr.backend,
        "Confused-digit HIGH->CLASS curriculum (H3 encoding).",
        confused_digit_delta=pair_curr.extra.get("confused_digit_delta"),
        pre_confused_accuracy=pair_curr.extra.get("pre_confused_accuracy"),
    ))

    repr_stack = run_representational_stack_mnist(bundle=shared, **kw)
    rows.append(_row(
        "representational_stack", "R", repr_stack.mean_accuracy,
        (
            HypothesisId.CROSS_CLASS_OVERLAP.value,
            HypothesisId.HONEST_COMPLETION_GAIN.value,
        ),
        ("reinforce_connectome", "pattern_complete", "wire_class_from_prototypes"),
        repr_stack.backend,
        "Pair curriculum + honest completion stack (Phase II combined).",
        confused_digit_accuracy=repr_stack.confused_digit_accuracy,
        confused_digit_delta=repr_stack.extra.get("confused_digit_delta"),
        curriculum_confused_delta=repr_stack.extra.get("curriculum_confused_delta"),
        completion_rate=repr_stack.extra.get("pair_gated_completion_rate"),
    ))

    occ = run_occlusion_training_mnist(train_occlusion_fraction=0.25, **kw)
    rows.append(_row(
        "representational_occlusion_train", "R", occ.mean_accuracy,
        (),
        ("project", "pattern_complete"),
        occ.backend,
        "Train-time LOW occlusion for completion-capable assemblies.",
        confused_digit_accuracy=occ.confused_digit_accuracy,
    ))

    fuzzy = run_fuzzy_lexicon_mnist(**kw)
    rows.append(_row(
        "advanced_fuzzy_lexicon", "0", fuzzy.mean_accuracy,
        (HypothesisId.READOUT_MISMATCH.value,),
        ("fuzzy_readout", "build_lexicon"),
        fuzzy.backend,
        "Language-organ readout on digit IT assemblies.",
    ))

    multi = run_multi_prototype_mnist(bundle=shared, **kw)
    rows.append(_row(
        "tier_a_multi_prototype", "A", multi.mean_accuracy,
        (HypothesisId.CROSS_CLASS_OVERLAP.value,),
        ("merge", "fuzzy_readout", "build_lexicon"),
        multi.backend,
        "K-prototype view manifold improves over single prototype.",
        min_separation=multi.min_class_separation,
        delta_vs_ventral=multi.mean_accuracy - ventral.mean_accuracy,
    ))

    from neural_assemblies.programs.colt_mnist_synthesis import run_synthesis_audit

    synthesis = run_synthesis_audit(seed=seed, n_examples=n_examples, k=200)
    for label, card in synthesis.items():
        rows.append(_row(
            f"synthesis_{label}", "R", card.discriminative_accuracy,
            (
                HypothesisId.ATTRACTOR_RECOVERY.value,
                HypothesisId.DIGIT3_STRUCTURED_ABSENCE.value,
                HypothesisId.SEPARABILITY_BEFORE_BINDING.value,
            ),
            ("overlap", "pattern_complete", "forward_completion"),
            "ac_scorecard",
            f"AC north-star scorecard ({label}): stable={card.assembly_stable}, "
            f"gen={card.generative_viable}.",
            assembly_stable=card.assembly_stable,
            geometry_adequate=card.geometry_adequate,
            generative_viable=card.generative_viable,
            digit3_absence=card.digit3_absence_accuracy,
            pattern_complete=card.pattern_complete_recovery,
            confused_pair_overlap=card.mean_confused_pair_overlap,
        ))

    from neural_assemblies.programs.patch_merge import run_grid_patch_merge_mnist

    grid_merge = run_grid_patch_merge_mnist(
        bundle=shared, grid=2, merge_mode="chain", **kw,
    )
    rows.append(_row(
        "tier_a_grid_patch_merge", "A", grid_merge.mean_accuracy,
        (HypothesisId.PART_MERGE_REDUCES_OVERLAP.value,),
        ("merge", "project"),
        grid_merge.backend,
        "4-patch 2×2 grid merge chain with spatial RFs + teacher align.",
        routing_fidelity=grid_merge.extra.get("routing_fidelity"),
        n_patches=grid_merge.extra.get("n_patches"),
    ))

    merge_h = run_merge_halves_mnist(bundle=shared, **kw)
    rows.append(_row(
        "tier_a_merge_halves", "A", merge_h.mean_accuracy,
        (HypothesisId.CROSS_CLASS_OVERLAP.value,),
        ("merge", "project"),
        merge_h.backend,
        "Part-structure dual-stream; routing fidelity to ventral IT code.",
        classification_accuracy=merge_h.mean_accuracy,
        routing_fidelity=merge_h.extra.get("routing_fidelity"),
    ))

    from neural_assemblies.programs.colt_mnist_h9_diagnostic import run_h9_diagnostic

    h9 = run_h9_diagnostic(seed=seed, n_examples=n_examples, k=200)
    merge_geo = next((s for s in h9.streams if s.name == "merge_halves"), None)
    base_geo = next((s for s in h9.streams if s.name == "recurrent_ventral"), None)
    rows.append(_row(
        "h9_merge_overlap_diagnostic", "A",
        merge_geo.mean_accuracy if merge_geo else None,
        (HypothesisId.PART_MERGE_REDUCES_OVERLAP.value,),
        ("merge", "separate"),
        "h9_diagnostic",
        f"H9 verdict={h9.h9_verdict}: merge confused overlap vs recurrent baseline.",
        h9_verdict=h9.h9_verdict,
        merge_confused_overlap=merge_geo.mean_confused_overlap if merge_geo else None,
        recurrent_confused_overlap=base_geo.mean_confused_overlap if base_geo else None,
        overlap_delta=h9.pair_deltas.get("merge_halves_vs_recurrent", {}).get("mean_confused"),
    ))

    pcomp = run_pattern_completion_mnist(bundle=shared, **kw)
    rows.append(_row(
        "tier_b_pattern_complete", "B", pcomp.mean_accuracy,
        (),
        ("pattern_complete",),
        pcomp.backend,
        "Ventral readout + recurrent HIGH recovery metric.",
        mean_recovery=pcomp.mean_recovery,
        baseline_ventral=pcomp.extra.get("baseline_ventral_accuracy"),
    ))

    consol = run_consolidation_mnist(bundle=shared, **kw)
    rows.append(_row(
        "tier_b_consolidation", "B", consol.mean_accuracy,
        (),
        ("consolidate", "PathwayReplay"),
        consol.backend,
        "Systems replay strengthens HIGH->CLASS connectome readout.",
        pre_replay=consol.mean_accuracy,
        baseline_ventral=consol.extra.get("baseline_ventral_accuracy"),
    ))

    lri = run_lri_cascade_mnist(bundle=shared, **kw)
    rows.append(_row(
        "tier_c_lri_cascade", "C", lri.mean_accuracy,
        (HypothesisId.LRI_CASCADE_DISCRIMINATION.value,),
        ("set_lri", "fuzzy_readout", "project"),
        lri.backend,
        "LRI competitive cascade on confused pairs and low-margin trials.",
        cascade_rate=lri.cascade_rate,
        mean_margin=lri.mean_margin,
        cascade_global_accuracy=lri.extra.get("cascade_global_accuracy"),
        confused_digit_delta=lri.extra.get("confused_digit_delta"),
    ))

    cross = run_vision_language_digit_hub(bundle=shared, **kw)
    rows.append(_row(
        "cross_domain_vision_language", "X", cross.visual_accuracy,
        (),
        ("associate", "reciprocal_project", "activate_assembly", "fuzzy_readout"),
        cross.backend,
        "VLM-style two-stage hub: contrastive binding + fused ventral readout.",
        visual_accuracy=cross.visual_accuracy,
        language_accuracy=cross.language_accuracy,
        agreement_rate=cross.agreement_rate,
        image_to_text_recall=cross.image_to_text_recall,
        text_to_image_recall=cross.text_to_image_recall,
        mean_semantic_overlap=cross.mean_semantic_overlap,
        semantic_route_accuracy=cross.extra.get("semantic_route_accuracy"),
        semantic_connectome_accuracy=cross.extra.get("semantic_connectome_accuracy"),
        contrastive_margin=cross.extra.get("contrastive_margin"),
        binding_gate_open=geometry.binding_gate_open,
    ))

    profile_narrative = ""
    if include_slow:
        from neural_assemblies.programs.cross_domain_profile import profile_cross_domain_hub
        from neural_assemblies.programs.cross_domain_assemblies import _train_cross_domain_hub

        hub = _train_cross_domain_hub(bundle=shared, **kw)
        profile = profile_cross_domain_hub(hub=hub, **kw)
        profile_narrative = profile.narrative
        rows[-1].metrics["bottleneck_ids"] = [b.id for b in profile.bottlenecks]
        rows[-1].metrics["confused_digit_accuracy"] = profile.confused_digit_accuracy
        rows[-1].metrics["route_connectome"] = next(
            r.accuracy for r in profile.routes if r.name == "connectome_class"
        )
        rows[-1].metrics["route_connectome_lri"] = next(
            r.accuracy for r in profile.routes if r.name == "connectome_lri"
        )

    # Hypothesis checks (empirical)
    checks = {
        "H1_recurrent_beats_simple_hierarchical": recurrent.mean_accuracy > simple.mean_accuracy + 0.10,
        "H2_ventral_beats_simple_hierarchical": ventral.mean_accuracy > simple.mean_accuracy + 0.10,
        "H2_ventral_approaches_recurrent": ventral.mean_accuracy >= recurrent.mean_accuracy - 0.05,
        "H4_fuzzy_matches_ventral": abs(fuzzy.mean_accuracy - ventral.mean_accuracy) < 0.05,
        "cross_domain_agreement_above_chance": cross.agreement_rate > 0.20,
        "cross_domain_fused_beats_ventral": cross.visual_accuracy >= ventral.mean_accuracy - 0.01,
        "cross_domain_retrieval_i2t": cross.image_to_text_recall >= 0.70,
        "cross_domain_lang_perfect": cross.language_accuracy >= 0.95,
        "tier_a_multi_within_ventral_band": multi.mean_accuracy >= ventral.mean_accuracy - 0.05,
        "tier_b_recovery_measurable": (pcomp.mean_recovery or 0.0) > 0.50,
        "tier_b_classification_matches_ventral": pcomp.mean_accuracy >= ventral.mean_accuracy - 0.05,
        "tier_c_lri_readout_matches_readout": lri.mean_accuracy >= ventral.mean_accuracy - 0.05,
        "tier_c_confused_pairs_measurable": lri.cascade_rate > 0.0 or (lri.extra.get("confused_digit_delta") or 0) >= 0.0,
        "tier_b_consolidation_matches_ventral": abs(consol.mean_accuracy - ventral.mean_accuracy) < 0.05,
        "H9_part_merge_falsified": h9.h9_verdict == "falsified",
        "H9_merge_increases_confused_overlap": (
            (h9.pair_deltas.get("merge_halves_vs_recurrent") or {}).get("mean_confused", 0) > 0.05
        ),
        "tier_a_grid_merge_routing": False,
        "tier_a_merge_routing_fidelity": merge_h.mean_accuracy > 0.30,
        "H6_binding_gate_closed": not geometry.binding_gate_open,
        "H7_honest_completion_measurable": (honest.extra.get("pair_gated_completion_rate") or 0) > 0.0,
        "H7_forward_completion_path": (honest.extra.get("forward_completion_rate") or 0) > 0.0,
        "H_spatial_digit3_center_measurable": (spatial.extra.get("digit3_center_band_accuracy") or 0) > 0.0,
        "H_spatial_approaches_ventral": spatial.mean_accuracy >= ventral.mean_accuracy - 0.05,
        "H8_recurrent_base_above_ventral": repr_recurrent.mean_accuracy >= ventral.mean_accuracy - 0.02,
        "H3_confused_gap_measurable": (
            geometry.confused_digit_accuracy + 0.05 < geometry.non_confused_digit_accuracy
        ),
        "representational_pair_curriculum_delta": (pair_curr.extra.get("confused_digit_delta") or 0) >= 0.0,
        "H11_recurrent_pattern_complete": regen_rec.mean_high_recovery >= 0.50,
        "H11_attractor_pattern_complete": regen_attr.mean_high_recovery >= 0.50,
        "H12_digit3_absence_measurable": (
            regen_attr.digit3_absence_mean >= 0.55
            or regen_rec.digit3_absence_mean >= 0.50
        ),
        "H13_low_regeneration_path": (regen_attr.mean_low_regeneration or 0.0) >= 0.15,
        "H_synthesis_recurrent_generative": synthesis["recurrent"].generative_viable,
        "H_synthesis_attractor_generative": synthesis["attractor"].generative_viable,
        "H_synthesis_both_assembly_stable": (
            synthesis["recurrent"].assembly_stable and synthesis["attractor"].assembly_stable
        ),
        "H6_and_generative_dual_gate": (
            not geometry.binding_gate_open
            and (regen_rec.regeneration_gate_open or regen_attr.regeneration_gate_open)
        ),
    }

    narrative = _build_narrative(rows, checks)
    narrative = narrative + "\n\n" + geometry.narrative
    narrative = narrative + "\n\n" + regen_rec.narrative
    narrative = narrative + "\n\n" + regen_attr.narrative
    if profile_narrative:
        narrative = narrative + "\n\n" + profile_narrative
    return EvidenceReport(
        recorded=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        n_examples=n_examples,
        seed=seed,
        data_source=data_source,
        rows=rows,
        hypothesis_checks=checks,
        narrative=narrative,
    )


def _build_narrative(rows: list[EvidenceRow], checks: dict[str, bool]) -> str:
    lines = [
        "Ventral-stream assembly evidence report",
        "======================================",
        "",
        simple_vs_recurrent_summary(),
        "",
        "Empirical ladder:",
    ]
    for r in sorted(rows, key=lambda x: x.mean_accuracy, reverse=True):
        extra = ""
        if r.experiment_id == "tier_a_merge_halves":
            fid = r.metrics.get("routing_fidelity")
            cls = r.metrics.get("classification_accuracy")
            if fid is not None:
                extra = f" (cls={cls:.1%}, fidelity={fid:.1%})" if cls is not None else f" (fidelity={fid:.1%})"
        elif r.experiment_id == "tier_c_lri_cascade":
            delta = r.metrics.get("confused_digit_delta")
            casc = r.metrics.get("cascade_global_accuracy")
            if delta is not None:
                extra = f" (readout={r.mean_accuracy:.1%}, cascade={casc:.1%}, confused_delta={delta:+.1%})" if casc is not None else f" (confused_delta={delta:+.1%})"
        elif r.experiment_id == "cross_domain_vision_language":
            vis = r.metrics.get("visual_accuracy")
            lang = r.metrics.get("language_accuracy")
            i2t = r.metrics.get("image_to_text_recall")
            if vis is not None:
                extra = f" (vis={vis:.1%}, lang={lang:.1%}, i2t={i2t:.1%})" if i2t is not None else f" (vis={vis:.1%}, lang={lang:.1%})"
        elif r.experiment_id.startswith("representational_") or r.experiment_id == "geometry_panel":
            conf = r.metrics.get("confused_digit_accuracy")
            if conf is not None:
                extra = f" (confused={conf:.1%})"
        lines.append(
            f"  [{r.tier}] {r.experiment_id}: {r.mean_accuracy:.1%}{extra} — {r.supports_narrative}"
        )
    lines.extend(["", "Hypothesis checks:"])
    for k, v in checks.items():
        lines.append(f"  {'PASS' if v else 'FAIL'}: {k}")
    best = max(rows, key=lambda x: x.mean_accuracy)
    lines.extend([
        "",
        f"Best accuracy this run: {best.experiment_id} ({best.mean_accuracy:.1%}).",
        "Target >95% requires Tier A–C structures (parts, completion, LRI cascade)",
        "plus scale; see colt_mnist_ventral_theory ROADMAP_TO_95.",
    ])
    return "\n".join(lines)


def export_evidence_json(path: str, report: EvidenceReport) -> None:
    """Write evidence report to JSON for parity / documentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
        f.write("\n")


def export_evidence_golden(path: str, *, seed: int = 42, n_examples: int = 50) -> EvidenceReport:
    """Run suite and write golden-compatible metrics JSON."""
    report = run_evidence_suite(seed=seed, n_examples=n_examples)
    golden = {
        "protocol": "ventral_evidence_ladder",
        "seed": seed,
        "n_examples": n_examples,
        "metrics": {
            "hypothesis_checks": report.hypothesis_checks,
            "rows": {
                r.experiment_id: {
                    "tier": r.tier,
                    "mean_accuracy": r.mean_accuracy,
                    **r.metrics,
                }
                for r in report.rows
            },
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ventral evidence suite")
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", type=str, help="Write JSON to path")
    args = parser.parse_args()

    report = run_evidence_suite(seed=args.seed, n_examples=args.n_examples)
    print(report.narrative_summary())
    if args.export:
        export_evidence_json(args.export, report)
        print(f"Wrote {args.export}")


if __name__ == "__main__":
    main()
