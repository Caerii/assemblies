"""
Attractor training for MNIST HIGH assemblies (generative completeness).

Closes the gap between discriminative encoding (~80%) and actual attractor
dynamics (pattern_complete recovery ~1% on default recurrent bundle).

Protocol (per digit):
1. Repeated LOW+HIGH exposure (ventral-style experience)
2. ``learn_high_attractor`` — joint LOW+HIGH project until persistence >= τ
3. ``consolidate_high_self_attractor`` — HIGH→HIGH only until self-persistence
4. Optional structured-absence curriculum on confused digits (Ember 3-dropout)
5. Optional top-down HIGH→LOW reciprocal consolidation (H13)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import _snap, pattern_complete
from neural_assemblies.programs.colt_mnist_absence import (
    ABSENCE_CURRICULUM_DIGITS,
    apply_absence_mask,
)
from neural_assemblies.programs.colt_mnist_advanced_util import (
    wire_class_from_prototypes,
)
from neural_assemblies.programs.colt_mnist_brain_util import (
    clear_area_winners,
    renorm_connectome_columns,
    set_kcap_winners,
)
from neural_assemblies.programs.colt_mnist_data import find_mnist_dir, load_mnist_arrays
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    LOW,
    MID,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
)
from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples
from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict
from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
    _min_pairwise_prototype_overlap,
)


@dataclass
class AttractorResult(ColtMnistHierarchicalBrainResult):
    tier: str = "R"
    method: str = ""
    mean_high_recovery: float | None = None
    digit3_recovery: float | None = None
    digit3_absence_accuracy: float | None = None
    mean_low_regeneration: float | None = None
    extra: dict = field(default_factory=dict)


def _assembly_persistence(a: Assembly, b: Assembly, k: int) -> float:
    return float(overlap(a, b))


def learn_high_attractor_from_low(
    brain,
    low_pattern: np.ndarray,
    high_bias: np.ndarray,
    k: int,
    *,
    max_epochs: int = 15,
    project_rounds: int = 8,
    stability_window: int = 2,
    convergence: float = 0.90,
) -> tuple[Assembly, int, float]:
    """LOW+HIGH joint learning until assembly persistence converges (learn_assembly analogue)."""
    history: list[Assembly] = []
    for epoch in range(1, max_epochs + 1):
        clear_area_winners(brain, HIGH)
        winners = set_kcap_winners(brain, LOW, low_pattern)
        for _ in range(project_rounds):
            brain.project(
                external_inputs={LOW: winners},
                projections={LOW: [HIGH], HIGH: [HIGH]},
                external_drive={HIGH: high_bias},
            )
        snap = _snap(brain, HIGH)
        history.append(snap)
        if len(history) >= stability_window:
            pairs = [
                _assembly_persistence(history[i], history[i + 1], k)
                for i in range(len(history) - stability_window, len(history) - 1)
            ]
            if all(p >= convergence for p in pairs):
                return snap, epoch, min(pairs)
    final_pers = (
        _assembly_persistence(history[-2], history[-1], k) if len(history) > 1 else 0.0
    )
    return history[-1], max_epochs, final_pers


def apply_learn_assembly_curriculum(
    brain,
    high_bias: np.ndarray,
    examples: np.ndarray,
    high_outputs: np.ndarray,
    k: int,
    n_examples: int,
    *,
    max_epochs: int = 10,
    project_rounds: int = 6,
    convergence: float = 0.90,
    consolidate_per_digit: bool = True,
    freeze_low_high: bool = True,
) -> dict[str, float]:
    """
    Convergent LOW+HIGH ``learn_assembly`` pass on all training exemplars (Tier A / E8).

    When ``freeze_low_high`` is True (default for post-discriminative use), only
    HIGH->HIGH weights update — preserves the LOW->HIGH encoding while deepening
    attractor basins.
    """
    saved = brain.disable_plasticity
    brain.disable_plasticity = False
    if freeze_low_high:
        brain.set_fiber_plasticity(LOW, HIGH, False)
        brain.set_fiber_plasticity(HIGH, HIGH, True)
    persistences: list[float] = []
    n_high = brain.areas[HIGH].n

    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            if freeze_low_high:
                clear_area_winners(brain, HIGH)
                set_kcap_winners(brain, HIGH, high_outputs[digit, j])
                pers = consolidate_high_self_attractor(
                    brain, high_outputs[digit, j], k,
                    max_rounds=max_epochs * 2, convergence=convergence,
                )
                persistences.append(pers)
            else:
                _, _, pers = learn_high_attractor_from_low(
                    brain,
                    examples[digit, j],
                    high_bias,
                    k,
                    max_epochs=max_epochs,
                    project_rounds=project_rounds,
                    convergence=convergence,
                )
                persistences.append(pers)
        if not freeze_low_high:
            renorm_connectome_columns(brain, LOW, HIGH)
        renorm_connectome_columns(brain, HIGH, HIGH)

        if consolidate_per_digit:
            class_mean = high_outputs[digit].mean(axis=0)
            top = class_mean.argsort()[-k:]
            vec = np.zeros(n_high, dtype=np.float32)
            vec[top] = 1.0
            consolidate_high_self_attractor(
                brain, vec, k, max_rounds=20, convergence=0.88,
            )
            renorm_connectome_columns(brain, HIGH, HIGH)

    if freeze_low_high:
        brain.set_fiber_plasticity(LOW, HIGH, True)
    brain.disable_plasticity = saved
    return {
        "mean_persistence": float(np.mean(persistences)) if persistences else 0.0,
        "min_persistence": float(np.min(persistences)) if persistences else 0.0,
    }


def refresh_high_outputs_from_forward(
    brain,
    high_bias: np.ndarray,
    examples: np.ndarray,
    high_outputs: np.ndarray,
    n_examples: int,
) -> None:
    """Re-encode all exemplars through LOW->HIGH after weight updates."""
    from neural_assemblies.programs.colt_mnist_forward_completion import forward_high_from_low

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = forward_high_from_low(
                brain, examples[digit, j], high_bias,
            )
    brain.disable_plasticity = saved


def consolidate_high_self_attractor(
    brain,
    high_vec: np.ndarray,
    k: int,
    *,
    max_rounds: int = 30,
    warmup_rounds: int = 3,
    stability_window: int = 2,
    convergence: float = 0.92,
) -> float:
    """HIGH-only recurrence — trains the basin used by ``pattern_complete``."""
    clear_area_winners(brain, HIGH)
    set_kcap_winners(brain, HIGH, high_vec)
    for _ in range(warmup_rounds):
        brain.project({}, {HIGH: [HIGH]})
    history = [_snap(brain, HIGH)]
    for _ in range(max_rounds):
        brain.project({}, {HIGH: [HIGH]})
        history.append(_snap(brain, HIGH))
        if len(history) >= stability_window + 1:
            pairs = [
                _assembly_persistence(history[i], history[i + 1], k)
                for i in range(len(history) - stability_window, len(history) - 1)
            ]
            if all(p >= convergence for p in pairs):
                break
    renorm_connectome_columns(brain, HIGH, HIGH)
    return _assembly_persistence(history[-1], history[0], k)


def consolidate_low_from_high(
    brain,
    high_vec: np.ndarray,
    low_reference: np.ndarray,
    k: int,
    *,
    rounds: int = 10,
) -> float:
    """Top-down HIGH→LOW reciprocal pass; return overlap with reference LOW assembly."""
    clear_area_winners(brain, HIGH)
    clear_area_winners(brain, LOW)
    set_kcap_winners(brain, HIGH, high_vec)
    for _ in range(rounds):
        brain.project({}, {HIGH: [LOW]})
    ref = Assembly(LOW, np.flatnonzero(low_reference > 0).astype(np.uint32))
    recovered = _snap(brain, LOW)
    return float(overlap(recovered, ref))


def measure_recovery_on_bundle(
    bundle,
    *,
    fraction: float = 0.5,
    rounds: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Per-digit mean pattern_complete recovery (fraction of k)."""
    brain = bundle.brain
    per_digit = np.zeros(NUM_DIGITS)
    has_rec = float(np.sum(brain.connectomes[HIGH][HIGH].weights)) > 0.0
    for digit in range(NUM_DIGITS):
        recs: list[float] = []
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            set_kcap_winners(brain, HIGH, hv)
            if has_rec:
                for _ in range(3):
                    brain.project({}, {HIGH: [HIGH]})
            _, ov = pattern_complete(
                brain, HIGH, fraction=fraction, rounds=rounds,
                seed=seed + digit * 100 + j,
                observation_mode="plastic",
            )
            recs.append(float(ov))
        per_digit[digit] = float(np.mean(recs))
    return per_digit


def _renorm_low_high_path(brain) -> None:
    """Renormalize fibers along the active LOW→HIGH forward path."""
    if MID in brain.areas:
        renorm_connectome_columns(brain, LOW, MID)
        renorm_connectome_columns(brain, MID, MID)
        renorm_connectome_columns(brain, MID, HIGH)
    else:
        renorm_connectome_columns(brain, LOW, HIGH)
    renorm_connectome_columns(brain, HIGH, HIGH)


def apply_forward_generative_curriculum(
    brain,
    high_bias: np.ndarray,
    examples: np.ndarray,
    n_examples: int,
    *,
    seed: int = 42,
    digit3_center_passes: int = 4,
    absence_curriculum_prob: float = 0.35,
    absence_exposure_passes: int = 1,
    attractor_rehearsal_rounds: int = 3,
    enable_top_down_low: bool = False,
    top_down_rounds: int = 4,
    high_outputs: np.ndarray | None = None,
    patch_graph=None,
    absence_protocols: tuple[str, ...] | None = None,
) -> None:
    """
    Post-discriminative exposure for structured partial LOW → HIGH alignment.

    Digit-3 center-band passes teach the forward path to complete occluded strokes;
    confused-digit absence exposure generalizes partial-view encoding.
    """
    from neural_assemblies.programs.colt_mnist_forward_completion import forward_high_from_low

    saved = brain.disable_plasticity
    brain.disable_plasticity = False
    occ_rng = np.random.default_rng(seed + 777)
    protocols = absence_protocols or ("top_half", "center_band", "bottom_half")
    if patch_graph is not None and patch_graph.absence_protocols:
        center_proto = (
            "center_patch"
            if "center_patch" in patch_graph.absence_protocols
            else next(iter(patch_graph.absence_protocols))
        )
    else:
        center_proto = "center_band"

    def _mask(pat: np.ndarray, proto: str) -> np.ndarray:
        if patch_graph is not None and proto in patch_graph.absence_protocols:
            return patch_graph.apply_absence(pat, proto, rng=occ_rng)
        return apply_absence_mask(pat, proto, rng=occ_rng)

    for _ in range(digit3_center_passes):
        for j in range(n_examples):
            pat = _mask(examples[3, j], center_proto)
            forward_high_from_low(brain, pat, high_bias)
        _renorm_low_high_path(brain)

    for _ in range(absence_exposure_passes):
        for digit in ABSENCE_CURRICULUM_DIGITS:
            for j in range(n_examples):
                pat = examples[digit, j]
                if occ_rng.random() < absence_curriculum_prob:
                    pat = _mask(
                        pat, protocols[int(occ_rng.integers(0, len(protocols)))],
                    )
                forward_high_from_low(brain, pat, high_bias)
                if MID not in brain.areas:
                    for _ in range(max(attractor_rehearsal_rounds // 2, 1)):
                        brain.project({}, {HIGH: [HIGH]})
            _renorm_low_high_path(brain)

    if enable_top_down_low and high_outputs is not None:
        if LOW in brain.areas:
            has_hl = (
                HIGH in brain.connectomes
                and LOW in brain.connectomes.get(HIGH, {})
                and float(np.sum(brain.connectomes[HIGH][LOW].weights)) > 0
            )
            if not has_hl:
                brain.init_reciprocal_connectome(LOW, HIGH, init="transpose_forward")
                brain.set_fiber_plasticity(LOW, HIGH, True)
                brain.set_fiber_plasticity(HIGH, LOW, True)
        brain.set_fiber_plasticity(HIGH, LOW, True)
        brain.set_fiber_plasticity(LOW, HIGH, False)
        for digit in range(NUM_DIGITS):
            for j in range(min(n_examples, 15)):
                set_kcap_winners(brain, HIGH, high_outputs[digit, j])
                for _ in range(top_down_rounds):
                    brain.project({}, {HIGH: [LOW]})
            renorm_connectome_columns(brain, HIGH, LOW)

    brain.disable_plasticity = saved


def capture_generative_prototypes(
    brain,
    high_bias: np.ndarray,
    examples: np.ndarray,
    n_examples: int,
    k: int,
    n_high: int,
) -> np.ndarray:
    """Re-forward full digits after curriculum; aggregate into generative prototypes."""
    from neural_assemblies.programs.colt_mnist_forward_completion import forward_high_from_low

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    generative_high = np.zeros((NUM_DIGITS, n_examples, n_high), dtype=np.float32)
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            hv = forward_high_from_low(brain, examples[digit, j], high_bias)
            generative_high[digit, j] = hv
    brain.disable_plasticity = saved

    generative_prototypes = np.zeros((NUM_DIGITS, n_high), dtype=np.float32)
    for digit in range(NUM_DIGITS):
        support = generative_high[digit].sum(axis=0)
        generative_prototypes[digit, support.argsort()[-k:]] = 1.0
    return generative_prototypes


def train_attractor_brain(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    p: float = 0.1,
    beta: float = 1.0,
    n_low: int = 784,
    n_high: int = 2000,
    n_class: int = 2000,
    class_bias: float = -1.0,
    class_passes: int = 5,
    class_beta: float = 3.0,
    n_rounds: int = 5,
    attractor_rehearsal_rounds: int = 6,
    attractor_rehearsal_passes: int = 0,
    absence_curriculum_prob: float = 0.35,
    absence_exposure_passes: int = 1,
    digit3_center_passes: int = 4,
    enable_top_down_low: bool = True,
    top_down_rounds: int = 4,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Extend the visual_advanced recurrent bundle with attractor strengthening.

    Phase 1 — standard recurrent encoding (preserves ~80% discriminative accuracy).
    Phase 2 — HIGH-only rehearsal on stored assemblies (attractor basin training).
    Phase 3 — structured absence re-exposure on confused digits (Ember curriculum).
    Phase 4 — optional HIGH→LOW top-down pairing (generative decoder path).
    """
    from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
        _attach_class_to_recurrent,
    )

    brain, high_outputs, high_bias = _attach_class_to_recurrent(
        seed=seed,
        n_low=n_low,
        n_high=n_high,
        n_class=n_class,
        k=k,
        beta=beta,
        n_rounds=n_rounds,
        n_examples=n_examples,
        p=p,
        class_bias=class_bias,
        class_passes=class_passes,
        class_beta=class_beta,
    )
    discriminative_high = high_outputs.copy()

    train_imgs, train_labels, _, _ = load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )

    if enable_top_down_low and LOW in brain.areas:
        has_hl = (
            HIGH in brain.connectomes
            and LOW in brain.connectomes.get(HIGH, {})
            and float(np.sum(brain.connectomes[HIGH][LOW].weights)) > 0
        )
        if not has_hl:
            brain.init_reciprocal_connectome(LOW, HIGH, init="transpose_forward")
            brain.set_fiber_plasticity(LOW, HIGH, True)
            brain.set_fiber_plasticity(HIGH, LOW, True)

    saved = brain.disable_plasticity
    brain.disable_plasticity = False

    for _pass in range(attractor_rehearsal_passes):
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                set_kcap_winners(brain, HIGH, high_outputs[digit, j])
                for _ in range(attractor_rehearsal_rounds):
                    brain.project({}, {HIGH: [HIGH]})
            renorm_connectome_columns(brain, HIGH, HIGH)

    apply_forward_generative_curriculum(
        brain,
        high_bias,
        examples,
        n_examples,
        seed=seed,
        digit3_center_passes=digit3_center_passes,
        absence_curriculum_prob=absence_curriculum_prob,
        absence_exposure_passes=absence_exposure_passes,
        attractor_rehearsal_rounds=attractor_rehearsal_rounds,
        enable_top_down_low=enable_top_down_low,
        top_down_rounds=top_down_rounds,
        high_outputs=high_outputs,
    )

    brain.disable_plasticity = saved

    generative_prototypes = capture_generative_prototypes(
        brain, high_bias, examples, n_examples, k, n_high,
    )

    high_outputs = discriminative_high
    prototypes = np.zeros((NUM_DIGITS, n_high))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0

    return brain, high_outputs, prototypes, high_bias, k, examples, generative_prototypes


def run_attractor_mnist(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    **kwargs,
) -> AttractorResult:
    """Train attractor-consolidated recurrent cortex and evaluate generative metrics."""
    result = train_attractor_brain(
        seed=seed, n_examples=n_examples, k=k, **kwargs,
    )
    brain, high_outputs, prototypes, high_bias, k, examples, generative_prototypes = result
    data_source = "mnist_csv" if find_mnist_dir() else "synthetic_fallback"

    from neural_assemblies.programs.colt_mnist_tier_util import VentralBundle

    bundle = VentralBundle(
        brain=brain,
        high_outputs=high_outputs,
        prototypes=prototypes,
        high_bias=high_bias,
        k=k,
        n_examples=n_examples,
        examples=examples,
        data_source=data_source,
        parameters={"seed": seed, "n_examples": n_examples, "k": k, "base": "attractor_recurrent", **kwargs},
    )

    wire_class_from_prototypes(brain, prototypes, HIGH, CLASS, k)
    saved_plast = brain.disable_plasticity
    brain.disable_plasticity = True
    recovery = measure_recovery_on_bundle(bundle, seed=seed)

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(high_outputs[digit, j], brain, k=k) == digit
            for j in range(n_examples)
        )
        correct[digit] = hits / n_examples

    gen_brain = brain
    wire_class_from_prototypes(gen_brain, generative_prototypes, HIGH, CLASS, k)

    low_regen: list[float] = []
    if kwargs.get("enable_top_down_low", True):
        for digit in range(NUM_DIGITS):
            low_regen.append(
                consolidate_low_from_high(
                    brain, generative_prototypes[digit], examples[digit, 0], k, rounds=5,
                )
            )

    from neural_assemblies.programs.colt_mnist_absence import digit3_absence_battery, apply_absence_mask
    from neural_assemblies.programs.colt_mnist_forward_completion import (
        forward_high_from_low,
    )

    d3_hits = d3_total = 0
    for proto in digit3_absence_battery():
        for j in range(n_examples):
            pat = apply_absence_mask(examples[3, j], proto)
            hv = forward_high_from_low(brain, pat, high_bias)
            d3_total += 1
            d3_hits += connectome_predict(hv, gen_brain, k=k) == 3
    brain.disable_plasticity = saved_plast

    digit3_abs = d3_hits / max(d3_total, 1)
    confused = float(np.mean([correct[d] for d in (2, 3, 5, 6, 7, 8, 9)]))

    return AttractorResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=data_source,
        parameters=bundle.parameters,
        backend="attractor_recurrent",
        tier="R",
        method="convergent_attractor_training",
        mean_high_recovery=float(recovery.mean()),
        digit3_recovery=float(recovery[3]),
        digit3_absence_accuracy=digit3_abs,
        mean_low_regeneration=float(np.mean(low_regen)) if low_regen else None,
        extra={
            "per_digit_recovery": recovery.tolist(),
            "confused_digit_accuracy": confused,
            "min_class_separation": _min_pairwise_prototype_overlap(prototypes, k),
        },
    )
