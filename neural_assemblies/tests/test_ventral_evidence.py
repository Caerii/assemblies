"""Smoke tests for ventral theory tier experiments and evidence ladder."""

from __future__ import annotations

import pytest

from neural_assemblies.programs.colt_mnist_data import (
    DatasetUnavailable, require_mnist_dir,
)
from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache
from neural_assemblies.programs.colt_mnist_evidence import run_evidence_suite
from neural_assemblies.programs.colt_mnist_hierarchical import run_colt_mnist_hierarchical
from neural_assemblies.programs.colt_mnist_tier_a import (
    run_merge_halves_mnist,
    run_multi_prototype_mnist,
)
from neural_assemblies.programs.colt_mnist_tier_b import (
    run_consolidation_mnist,
    run_pattern_completion_mnist,
)
from neural_assemblies.programs.colt_mnist_lri_readout import run_lri_cascade_mnist
from neural_assemblies.programs.colt_mnist_visual_advanced import (
    run_recurrent_cortex_mnist,
    run_ventral_stream_mnist,
)
from neural_assemblies.programs.cross_domain_assemblies import run_vision_language_digit_hub
from neural_assemblies.programs.colt_mnist_ventral_theory import (
    experiment_modules,
    structures_not_yet_implemented,
)

# This whole module is the ventral-theory / MNIST evidence suite: research
# validation experiments, not dev-iteration coverage. Its tests dominate the
# suite wall-clock (test_evidence_suite_full_ladder alone is ~59 min; the
# cross-domain and tier probes run 1-5 min each), so the module is marked slow
# and excluded from the default `-m "not slow"` dev run. Run it in CI / full
# validation with `pytest -m slow` (or no marker filter).
pytestmark = pytest.mark.slow


N = 10  # fast smoke budget; full ladder uses n=50 in colt_mnist_evidence CLI


def _require_real_mnist_evidence() -> None:
    try:
        require_mnist_dir()
    except DatasetUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module", autouse=True)
def _clear_bundle_cache():
    clear_ventral_bundle_cache()
    yield
    clear_ventral_bundle_cache()


@pytest.fixture(scope="module")
def smoke_kw():
    return dict(seed=42, n_examples=N)


def test_tier_a_multi_prototype_runs(smoke_kw):
    r = run_multi_prototype_mnist(**smoke_kw)
    assert r.mean_accuracy >= 0.45
    assert r.tier == "A"


def test_tier_a_merge_halves_runs(smoke_kw):
    from neural_assemblies.programs.colt_mnist_tier_util import load_ventral_bundle
    bundle = load_ventral_bundle(**smoke_kw)
    r = run_merge_halves_mnist(bundle=bundle, **smoke_kw)
    assert r.mean_accuracy >= 0.25 or (r.extra.get("routing_fidelity") or 0) >= 0.05
    assert r.method == "merge_halves_stream"


def test_tier_b_pattern_completion_runs(smoke_kw):
    r = run_pattern_completion_mnist(**smoke_kw)
    assert r.mean_recovery is not None
    assert r.mean_recovery >= 0.0


def test_tier_b_consolidation_runs(smoke_kw):
    r = run_consolidation_mnist(**smoke_kw, replay_rounds=3)
    assert r.mean_accuracy >= 0.40
    assert r.method == "prototype_consolidation"


def test_tier_c_lri_cascade_runs(smoke_kw):
    r = run_lri_cascade_mnist(**smoke_kw)
    assert 0.0 <= r.cascade_rate <= 1.0
    assert r.mean_accuracy >= 0.45
    assert r.extra.get("confused_digit_delta") is not None


def test_cross_domain_profile_runs(smoke_kw):
    from neural_assemblies.programs.cross_domain_profile import profile_cross_domain_hub
    p = profile_cross_domain_hub(**smoke_kw)
    assert p.fused_accuracy >= 0.35
    assert len(p.routes) >= 5
    assert len(p.bottlenecks) >= 1
    assert p.narrative


def test_cross_domain_vision_language_hub(smoke_kw):
    r = run_vision_language_digit_hub(**smoke_kw)
    assert r.visual_accuracy >= 0.40
    assert r.language_accuracy >= 0.25
    assert r.agreement_rate >= 0.15
    assert r.image_to_text_recall >= 0.15
    assert r.text_to_image_recall >= 0.15
    assert r.extra.get("semantic_route_accuracy") is not None
    assert r.extra.get("contrastive_margin") is not None


def test_empirical_gap_hierarchical_vs_recurrent():
    """Structural gap is clearest at n=50 (see colt_mnist_ventral_theory)."""
    _require_real_mnist_evidence()
    kw = dict(seed=42, n_examples=50)
    simple = run_colt_mnist_hierarchical(**kw)
    ventral = run_ventral_stream_mnist(**kw)
    recurrent = run_recurrent_cortex_mnist(**kw)
    assert recurrent.mean_accuracy > simple.mean_accuracy + 0.08
    assert ventral.mean_accuracy > simple.mean_accuracy + 0.08
    assert ventral.mean_accuracy >= recurrent.mean_accuracy - 0.05


def test_geometry_panel_runs(smoke_kw):
    from neural_assemblies.programs.colt_mnist_geometry_panel import run_geometry_panel
    p = run_geometry_panel(**smoke_kw)
    assert p.prototype_overlap.shape == (10, 10)
    assert p.confused_digit_accuracy >= 0.0
    assert p.narrative


def test_representational_honest_completion(smoke_kw):
    from neural_assemblies.programs.colt_mnist_representational import run_honest_completion_mnist
    r = run_honest_completion_mnist(**smoke_kw)
    assert r.method == "honest_pair_gated_completion"
    assert r.extra.get("pair_gated_completion_rate") is not None
    assert r.extra.get("forward_completion_rate") is not None
    assert r.extra.get("pattern_complete_rate") is not None
    assert r.confused_digit_accuracy is not None


def test_spatial_ventral_runs(smoke_kw):
    from neural_assemblies.programs.colt_mnist_spatial_ventral import run_spatial_ventral_mnist
    r = run_spatial_ventral_mnist(**smoke_kw)
    assert r.method == "spatial_local_rf_ventral"
    assert r.extra.get("digit3_center_band_accuracy") is not None
    assert r.spatial_stats.get("mean_rf_fan_in", 0) > 0


def test_load_spatial_ventral_bundle(smoke_kw):
    from neural_assemblies.programs.colt_mnist_tier_util import load_spatial_ventral_bundle
    b = load_spatial_ventral_bundle(**smoke_kw)
    assert b.parameters.get("base") == "spatial_local_rf_ventral"
    assert b.high_outputs.shape[0] == 10


def test_engine_roadmap_registry():
    from neural_assemblies.programs.colt_mnist_engine_roadmap import (
        ENGINE_UPGRADES,
        engine_roadmap_narrative,
        engine_upgrades_for_tier,
        EngineTier,
    )
    assert len(ENGINE_UPGRADES) >= 8
    assert any(u.id.startswith("E1_") for u in ENGINE_UPGRADES)
    e2 = next(u for u in ENGINE_UPGRADES if u.id.startswith("E2_"))
    assert e2.status == "done"
    assert "E2" in engine_roadmap_narrative()
    assert len(engine_upgrades_for_tier(EngineTier.P0_BLOCKING)) >= 4


def test_representational_pair_curriculum(smoke_kw):
    from neural_assemblies.programs.colt_mnist_representational import run_pair_curriculum_mnist
    r = run_pair_curriculum_mnist(**smoke_kw)
    assert r.tier == "R"
    assert "confused_digit_delta" in r.extra


def test_representational_stack(smoke_kw):
    from neural_assemblies.programs.colt_mnist_representational import run_representational_stack_mnist
    r = run_representational_stack_mnist(**smoke_kw)
    assert r.method == "pair_curriculum_plus_honest_completion"
    assert r.extra.get("pair_gated_completion_rate") is not None


def test_regeneration_panel_runs(smoke_kw):
    from neural_assemblies.programs.colt_mnist_regeneration_panel import run_regeneration_panel
    p = run_regeneration_panel(**smoke_kw)
    assert p.per_digit_forward_overlap.shape == (10,)
    assert "top_half" in p.digit3_absence_by_protocol
    assert p.narrative


def test_attractor_mnist_runs(smoke_kw):
    from neural_assemblies.programs.colt_mnist_attractor import run_attractor_mnist
    r = run_attractor_mnist(**smoke_kw)
    assert r.method == "convergent_attractor_training"
    assert r.mean_high_recovery is not None
    assert r.digit3_recovery is not None


def test_theory_registry_covers_experiments():
    mods = experiment_modules()
    assert "geometry_panel" in mods
    assert "regeneration_panel" in mods
    assert "attractor_recurrent" in mods
    assert "spatial_ventral" in mods
    assert "engine_roadmap" in mods
    assert "representational_honest_completion" in mods
    assert "representational_stack" in mods
    assert "evidence_suite" in mods
    assert "cross_domain_vision_language" in mods
    remaining = structures_not_yet_implemented()
    assert all(s.name != "Cross-modal semantic hub" for s in remaining)


@pytest.mark.slow
def test_evidence_suite_full_ladder():
    _require_real_mnist_evidence()
    report = run_evidence_suite(seed=42, n_examples=50)
    assert len(report.rows) >= 10
    assert report.hypothesis_checks["H1_recurrent_beats_simple_hierarchical"]
    assert report.hypothesis_checks["H2_ventral_beats_simple_hierarchical"]
    assert report.narrative
