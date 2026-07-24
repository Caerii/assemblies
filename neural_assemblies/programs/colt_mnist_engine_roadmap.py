"""
Assembly Calculus engine upgrade roadmap for vision / generative completeness.

Documents gaps discovered empirically (pattern_complete ~0.5%, HIGH→HIGH
persistence failure, cross-engine winner indexing) and prioritized fixes.

Read alongside ``colt_mnist_ventral_theory`` Part VI–VII and the regeneration
panel (H11–H13).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class EngineTier(str, Enum):
    """Priority tiers for engine work."""

    P0_BLOCKING = "P0_blocking"       # blocks attractor / pattern_complete thesis
    P1_GENERATIVE = "P1_generative"   # blocks HIGH→LOW decoder quality
    P2_SCALE = "P2_scale"             # performance / scale
    P3_API = "P3_api"                 # ergonomics (FiberCircuit, batch)


@dataclass(frozen=True)
class EngineUpgrade:
    """One engine-level change with scientific motivation."""

    id: str
    tier: EngineTier
    title: str
    problem: str
    proposal: str
    acceptance_test: str
    modules: tuple[str, ...]
    status: Literal["open", "partial", "done"] = "open"


ENGINE_UPGRADES: tuple[EngineUpgrade, ...] = (
    EngineUpgrade(
        id="E1_attractor_persistence",
        tier=EngineTier.P0_BLOCKING,
        title="Area self-projection must preserve assemblies (attractor dynamics)",
        problem=(
            "Prior reports of ~0.5% HIGH→HIGH recovery on MNIST were traced to "
            "double ``/k`` normalization (E2). After fix, explicit and sparse "
            "engines recover ~99% on stabilized assemblies. Residual work: weight "
            "blow-up diagnostics, soft-attractor mode for partial cues under noise."
        ),
        proposal=(
            "1. Audit ``numpy_engine/_sparse.py`` winner selection when "
            "``prev_winner_inputs`` and recurrent W_HH both contribute — k-cap "
            "may re-sample unrelated neurons each round. "
            "2. Add ``AttractorProject`` mode: fix ≥(1-ε) fraction of prior winners "
            "during self-projection rounds (soft attractor). "
            "3. Add ``learn_assembly`` integration for *areas* (not only stimuli) "
            "with convergence criterion on consecutive ``_snap`` overlap. "
            "4. Optional: Oja / normalized Hebbian on W_HH to prevent weight blow-up "
            "(trained W_HH sum observed ~4e12 in diagnostics)."
        ),
        acceptance_test=(
            "On recurrent MNIST bundle: ``pattern_complete(HIGH, fraction=0.5)`` "
            "mean recovery ≥ 0.5 for ≥8/10 digits; digit-3 ≥ 0.4."
        ),
        modules=("core/numpy_engine/_sparse.py", "core/brain.py", "assembly_calculus/ops.py"),
        status="done",
    ),
    EngineUpgrade(
        id="E2_winner_index_contract",
        tier=EngineTier.P0_BLOCKING,
        title="Unified compact ↔ real neuron ID contract for Assembly overlap",
        problem=(
            "``_snap`` maps compact engine indices to neuron_id_pool; "
            "``set_kcap_winners`` sets compact winners; ``overlap(Assembly, Assembly)`` "
            "compares real IDs. Prototype overlap via Assembly can read ~0.5% when "
            "dot-product on binary vectors reads 50%+. Metrics and ``separate()`` "
            "diagnostics may be inconsistent across code paths."
        ),
        proposal=(
            "1. Document and enforce one canonical representation per area per phase "
            "(compact during project, real IDs in Assembly snapshots). "
            "2. Add ``Assembly.from_area(brain, area)`` helper always using ``_snap``. "
            "3. Regression tests: overlap(snap(a), snap(b)) == dot(binary_a, binary_b)/k."
        ),
        acceptance_test=(
            "Literature parity + MNIST: prototype overlap via Assembly equals "
            "numpy dot overlap within 1e-6."
        ),
        modules=("assembly_calculus/assembly.py", "assembly_calculus/ops.py", "programs/colt_mnist_brain_util.py"),
        status="done",
    ),
    EngineUpgrade(
        id="E3_structured_connectome_init",
        tier=EngineTier.P1_GENERATIVE,
        title="First-class spatial / block-sparse connectome initialization",
        problem=(
            "LOW→MID uses dense random sparse mask (784×2000). CNNs and V1 use "
            "local RFs. Program-level ``init_spatial_low_mid_weights`` exists but "
            "engine has no typed ``ConnectomeInit.local_rf`` / ``block_sparse`` API."
        ),
        proposal=(
            "1. ``Brain.init_connectome(src, dst, scheme='local_rf', rf_radius=2, grid=...)`` "
            "2. Optional weight sharing across tiled RFs (translation equivariance). "
            "3. ``FiberCircuit`` presets for ventral stream (LOW→MID, MID→HIGH, HIGH→HIGH)."
        ),
        acceptance_test=(
            "Spatial ventral ≥ feedforward ventral - 2% on confused digits; "
            "digit-3 center_band forward cls ≥ baseline + 10%."
        ),
        modules=("core/brain.py", "programs/colt_mnist_spatial_connectome.py", "assembly_calculus/fiber.py"),
    ),
    EngineUpgrade(
        id="E4_top_down_project",
        tier=EngineTier.P1_GENERATIVE,
        title="Bidirectional project with separate forward / backward connectomes",
        problem=(
            "HIGH→LOW for generative decode uses ad-hoc transpose init + Hebbian. "
            "No ``reciprocal_project`` training protocol; LOW regeneration ~0.1%."
        ),
        proposal=(
            "1. ``brain.add_reciprocal_pair(A, B, init='transpose_forward')`` "
            "2. ``consolidate_pair(A, B, assembly_a, assembly_b, rounds)`` for sleep replay. "
            "3. Generative loss proxy: maximize overlap(project(HIGH→LOW), LOW_ref) "
            "without breaking forward LOW→HIGH (forward-only plasticity flag per fiber)."
        ),
        acceptance_test="Mean LOW regeneration overlap ≥ 0.15 on MNIST digits 0–9.",
        modules=("core/brain.py", "assembly_calculus/ops.py", "programs/colt_mnist_forward_completion.py"),
        status="partial",
    ),
    EngineUpgrade(
        id="E5_partial_input_drive",
        tier=EngineTier.P1_GENERATIVE,
        title="Partial stimulus drive during project (occlusion-native)",
        problem=(
            "Occlusion implemented by zeroing pixels before ``set_kcap_winners``. "
            "Engine does not distinguish 'missing' vs 'inactive' — no top-down fill-in."
        ),
        proposal=(
            "1. ``external_drive`` on LOW during partial cues (predictive residual). "
            "2. ``missing_mask`` per area suppresses k-cap on zeroed inputs but allows "
            "MID→LOW expectations to activate masked sites in later rounds."
        ),
        acceptance_test=(
            "Digit-3 center_band classification ≥ 70% with engine-native partial drive "
            "vs ≥ 55% with pixel-zero only."
        ),
        modules=("core/brain.py", "core/numpy_engine/_sparse.py"),
    ),
    EngineUpgrade(
        id="E6_plasticity_gating",
        tier=EngineTier.P0_BLOCKING,
        title="Per-fiber plasticity gating (forward vs backward vs recurrent)",
        problem=(
            "Global ``disable_plasticity`` caused measurement bugs; backward pass "
            "Hebbian during eval corrupted connectomes. Need fiber-level LTP/LTD control."
        ),
        proposal=(
            "``brain.plasticity_mask[(src,dst)] = True/False``; "
            "default eval freezes all; training enables forward fibers only; "
            "sleep replay enables selected pairs (HIGH→LOW, HIGH→HIGH)."
        ),
        acceptance_test=(
            "Regeneration panel metrics identical with plasticity on/off during eval; "
            "training-only fibers update."
        ),
        modules=("core/brain.py", "assembly_calculus/fiber.py"),
        status="partial",
    ),
    EngineUpgrade(
        id="E7_batch_explicit_mixed",
        tier=EngineTier.P2_SCALE,
        title="Robust explicit+sparse mixed projections for recurrent MNIST",
        problem=(
            "Brain routes explicit HIGH targets through sparse engine for cross-area "
            "drive; ``external_drive`` and ``high_bias`` interaction is fragile "
            "(lost high_bias caused 10% accuracy bug)."
        ),
        proposal=(
            "Single ``project_step`` trace API returning per-area input breakdown; "
            "golden tests for LOW→HIGH + HIGH→HIGH one step vs COLT notebook."
        ),
        acceptance_test="Recurrent MNIST parity golden ±0.5% vs reference executor.",
        modules=("core/brain.py", "parity/executors.py"),
    ),
    EngineUpgrade(
        id="E8_learn_assembly_area",
        tier=EngineTier.P0_BLOCKING,
        title="``learn_assembly`` for explicit areas with external LOW drive",
        problem=(
            "``learn_assembly(stim, area)`` works for stimuli; MNIST uses "
            "``external_inputs`` on LOW area. No convergence loop wrapper for areas."
        ),
        proposal=(
            "``learn_assembly_from_pattern(brain, src_area, pattern, dst, rounds, τ)`` "
            "wrapping set_kcap_winners + project until persistence ≥ τ."
        ),
        acceptance_test="≥ 0.85 persistence on isolated 784→2000 random patterns.",
        modules=("assembly_calculus/ops.py", "programs/colt_mnist_attractor.py"),
        status="done",
    ),
)


def engine_upgrades_for_tier(tier: EngineTier) -> tuple[EngineUpgrade, ...]:
    return tuple(u for u in ENGINE_UPGRADES if u.tier == tier)


def engine_roadmap_narrative() -> str:
    """Human-readable engine upgrade summary."""
    lines = [
        "Assembly Calculus — Engine Upgrade Roadmap",
        "=" * 44,
        "",
        "Empirical trigger: pattern_complete recovery was misreported (~0.5%) due to "
        "double /k normalization (E2); explicit MNIST recurrent brain recovers ~99%.",
        "forward LOW→HIGH generative path works; HIGH→HIGH attractor path does not.",
        "",
    ]
    for tier in EngineTier:
        items = engine_upgrades_for_tier(tier)
        if not items:
            continue
        lines.append(f"{tier.value}:")
        for u in items:
            lines.append(f"  [{u.id}] {u.title} ({u.status})")
            lines.append(f"    Problem: {u.problem[:120]}...")
            lines.append(f"    Test: {u.acceptance_test}")
        lines.append("")
    lines.extend([
        "Interim program strategy (no engine blockers):",
        "  • Use forward completion (partial LOW → HIGH) for H7/H12",
        "  • Spatial RF connectomes at program layer (colt_mnist_spatial_connectome)",
        "  • Digit-3 center_band curriculum at program layer",
        "  • Defer pattern_complete-gated inference until E1 passes",
        "",
        "Engine work order: E2 → E1 → E6 → E8 → E4 → E3 → E5 → E7",
    ])
    return "\n".join(lines)


def main() -> None:
    print(engine_roadmap_narrative())


if __name__ == "__main__":
    main()
