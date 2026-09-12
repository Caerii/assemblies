"""
Phase II — Representational learning interventions.

Encoding-first experiments (no oracle readout, no benchmark-maxxing):

* Recurrent ventral base (H8)
* Honest pair-gated pattern completion at inference (H7)
* Confused-pair HIGH→CLASS curriculum (H3)
* Train-time LOW occlusion (Tier B completion training)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.ops import pattern_complete
from neural_assemblies.programs.colt_mnist_absence import apply_absence_mask
from neural_assemblies.programs.colt_mnist_forward_completion import forward_high_from_low
from neural_assemblies.programs.colt_mnist_advanced_util import (
    read_class_connectome_scores,
    refined_confused_prototypes,
    wire_class_from_prototypes,
    wire_class_mixed_prototypes,
)
from neural_assemblies.programs.colt_mnist_brain_util import (
    reinforce_class_slot,
    renorm_connectome_columns,
    set_kcap_winners,
)
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    LOW,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
)
from neural_assemblies.programs.colt_mnist_lri_readout import connectome_lri_predict
from neural_assemblies.programs.colt_mnist_tier_util import (
    CONFUSED_DIGITS,
    CONFUSED_PAIRS,
    connectome_predict,
    load_recurrent_bundle,
    load_ventral_bundle,
)
from neural_assemblies.programs.colt_mnist_visual_advanced import run_recurrent_cortex_mnist


@dataclass
class RepresentationalResult(ColtMnistHierarchicalBrainResult):
    tier: str = "R"
    method: str = ""
    confused_digit_accuracy: float | None = None
    non_confused_digit_accuracy: float | None = None
    extra: dict = field(default_factory=dict)


def _confused_pair_in_top2(order: np.ndarray) -> bool:
    a, b = int(order[0]), int(order[1])
    return (a, b) in CONFUSED_PAIRS or (b, a) in CONFUSED_PAIRS


def apply_pair_representation_curriculum(
    brain,
    bundle,
    *,
    passes: int = 3,
) -> None:
    """Extra Hebbian HIGH→CLASS on stroke-confusable digits (H3 curriculum)."""
    apply_pairwise_contrast_curriculum(brain, bundle, pair_passes=passes)


def apply_pairwise_contrast_curriculum(
    brain,
    bundle,
    *,
    pair_passes: int = 5,
    pair_beta: float = 3.0,
) -> None:
    """
    Interleaved pairwise HIGH→CLASS reinforcement on confused pairs (H3/H10).

    For each stroke-confusable pair (a, b), alternate exemplars from both
    digits so connectome slots sharpen relative to the confusable neighbour.
    """
    for _ in range(pair_passes):
        for a, b in CONFUSED_PAIRS:
            for j in range(bundle.n_examples):
                set_kcap_winners(brain, HIGH, bundle.high_outputs[a, j])
                reinforce_class_slot(
                    brain, HIGH, CLASS, a, bundle.k, beta=pair_beta,
                )
                set_kcap_winners(brain, HIGH, bundle.high_outputs[b, j])
                reinforce_class_slot(
                    brain, HIGH, CLASS, b, bundle.k, beta=pair_beta,
                )
            renorm_connectome_columns(brain, HIGH, CLASS)
    refined = refined_confused_prototypes(
        bundle.prototypes, bundle.high_outputs, CONFUSED_DIGITS, bundle.k,
    )
    wire_class_mixed_prototypes(
        brain, bundle.prototypes, refined, CONFUSED_DIGITS, HIGH, CLASS, bundle.k,
    )


def honest_pair_gated_predict(
    high_vec: np.ndarray,
    readout_brain,
    recurrent_brain,
    bundle,
    *,
    low_pattern: np.ndarray | None = None,
    mp_lex: dict | None = None,
    completion_fraction: float = 0.5,
    completion_rounds: int = 5,
    seed: int = 0,
    use_lri_on_pair: bool = True,
    use_forward_completion: bool = True,
    forward_protocols: tuple[str, ...] = ("full", "center_band", "top_half"),
) -> tuple[int, dict]:
    """
    Honest inference: when top-2 is a confused pair and margin is low, try
    forward encoding from structured LOW absence cues (generative path).

    Falls back to ``pattern_complete`` on HIGH only if forward is disabled
    or ``low_pattern`` is unavailable (engine-limited ~0.5% recovery).
    """
    scores = read_class_connectome_scores(
        high_vec, readout_brain, HIGH, CLASS, bundle.k, NUM_DIGITS,
    )
    order = np.argsort(scores)[::-1]
    baseline = int(order[0])
    margin = float(scores[order[0]] - scores[order[1]])
    meta = {
        "baseline": baseline,
        "margin": margin,
        "completion_applied": False,
        "forward_completion_applied": False,
        "lri_applied": False,
    }

    if not _confused_pair_in_top2(order):
        return baseline, meta

    if margin >= 0.08:
        return baseline, meta

    encode_brain = recurrent_brain if LOW in recurrent_brain.areas else readout_brain
    best_pred = baseline
    best_margin = margin
    best_hv = high_vec

    if use_forward_completion and low_pattern is not None:
        rng = np.random.default_rng(seed)
        for _pi, proto in enumerate(forward_protocols):
            pat = (
                low_pattern if proto == "full"
                else apply_absence_mask(low_pattern, proto, rng=rng)
            )
            hv = forward_high_from_low(encode_brain, pat, bundle.high_bias)
            scores_f = read_class_connectome_scores(
                hv, readout_brain, HIGH, CLASS, bundle.k, NUM_DIGITS,
            )
            order_f = np.argsort(scores_f)[::-1]
            margin_f = float(scores_f[order_f[0]] - scores_f[order_f[1]])
            if margin_f > best_margin:
                best_margin = margin_f
                best_pred = int(order_f[0])
                best_hv = hv
                meta["forward_protocol"] = proto
        if best_margin > margin:
            meta["forward_completion_applied"] = True
            meta["completion_applied"] = True
            meta["margin_after"] = best_margin

    if not meta["forward_completion_applied"]:
        set_kcap_winners(recurrent_brain, HIGH, high_vec)
        if float(np.sum(recurrent_brain.connectomes[HIGH][HIGH].weights)) > 0:
            for _ in range(3):
                recurrent_brain.project({}, {HIGH: [HIGH]})
        recovered, recovery = pattern_complete(
            recurrent_brain, HIGH, fraction=completion_fraction,
            rounds=completion_rounds, seed=seed,
            observation_mode="plastic",
        )
        meta["completion_applied"] = True
        meta["recovery"] = float(recovery) / bundle.k
        hv = np.zeros_like(high_vec)
        if recovered.winners.size:
            hv[np.asarray(recovered.winners, dtype=int)] = 1.0
        scores_after = read_class_connectome_scores(
            hv, readout_brain, HIGH, CLASS, bundle.k, NUM_DIGITS,
        )
        order_after = np.argsort(scores_after)[::-1]
        margin_after = float(scores_after[order_after[0]] - scores_after[order_after[1]])
        if margin_after > best_margin:
            best_pred = int(order_after[0])
            best_margin = margin_after
            best_hv = hv
            meta["margin_after"] = margin_after

    if best_margin <= margin:
        return baseline, meta

    completed_pred = best_pred
    hv = best_hv

    if use_lri_on_pair and mp_lex:
        lri_pred, _, nc = connectome_lri_predict(
            hv, readout_brain, mp_lex, k=bundle.k, margin_threshold=0.04,
        )
        if nc > 0:
            meta["lri_applied"] = True
            completed_pred = lri_pred

    return completed_pred, meta


def _split_confused_accuracy(per_digit: np.ndarray) -> tuple[float, float]:
    confused = float(np.mean([per_digit[d] for d in CONFUSED_DIGITS]))
    other = float(np.mean([per_digit[d] for d in range(NUM_DIGITS) if d not in CONFUSED_DIGITS]))
    return confused, other


def run_recurrent_representational_mnist(
    *,
    bundle=None,
    **kwargs,
) -> RepresentationalResult:
    """Recurrent encoding base via visual_advanced recurrent cortex (H8)."""
    base = run_recurrent_cortex_mnist(**kwargs)
    per_digit = base.per_class_accuracy
    confused, other = _split_confused_accuracy(per_digit)
    return RepresentationalResult(
        per_class_accuracy=per_digit,
        mean_accuracy=float(base.mean_accuracy),
        data_source=base.data_source,
        parameters={**base.parameters, "base": "visual_advanced_recurrent"},
        backend="representational_recurrent_base",
        tier="R",
        method="recurrent_connectome",
        confused_digit_accuracy=confused,
        non_confused_digit_accuracy=other,
        extra={"recurrent_cortex_reference": base.mean_accuracy},
    )


def run_honest_completion_mnist(
    *,
    completion_fraction: float = 0.5,
    bundle=None,
    **kwargs,
) -> RepresentationalResult:
    """Honest pair-gated pattern completion — no oracle (H7)."""
    readout_bundle = bundle or load_ventral_bundle(use_cache=True, **kwargs)
    rec_bundle = load_recurrent_bundle(use_cache=True, **kwargs)
    brain = readout_bundle.brain
    wire_class_from_prototypes(brain, readout_bundle.prototypes, HIGH, CLASS, readout_bundle.k)

    from neural_assemblies.programs.colt_mnist_tier_a import build_multi_prototype_lexicon
    mp_lex = build_multi_prototype_lexicon(
        readout_bundle.high_outputs, k=readout_bundle.k, prototypes_per_digit=3,
    )
    readout_bundle.mp_lex = mp_lex  # type: ignore[attr-defined]

    correct = np.zeros(NUM_DIGITS)
    baseline = np.zeros(NUM_DIGITS)
    forward_rate = 0
    pattern_rate = 0
    n_eval = NUM_DIGITS * readout_bundle.n_examples

    for digit in range(NUM_DIGITS):
        b_hits = h_hits = 0
        for j in range(readout_bundle.n_examples):
            hv = readout_bundle.high_outputs[digit, j]
            b_pred = connectome_predict(hv, brain, k=readout_bundle.k)
            b_hits += b_pred == digit

            pred, meta = honest_pair_gated_predict(
                hv, brain, rec_bundle.brain, readout_bundle,
                low_pattern=readout_bundle.examples[digit, j],
                mp_lex=mp_lex,
                completion_fraction=completion_fraction,
                seed=kwargs.get("seed", 42) + digit * 100 + j,
            )
            if meta.get("forward_completion_applied"):
                forward_rate += 1
            elif meta.get("completion_applied") and not meta.get("forward_completion_applied"):
                pattern_rate += 1
            h_hits += pred == digit
        correct[digit] = h_hits / readout_bundle.n_examples
        baseline[digit] = b_hits / readout_bundle.n_examples

    confused, other = _split_confused_accuracy(correct)
    b_conf, _ = _split_confused_accuracy(baseline)
    return RepresentationalResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=readout_bundle.data_source,
        parameters={
            **readout_bundle.parameters,
            "completion_fraction": completion_fraction,
            "honest": True,
            "recurrent_base": "visual_advanced_recurrent",
        },
        backend="representational_honest_completion",
        tier="R",
        method="honest_pair_gated_completion",
        confused_digit_accuracy=confused,
        non_confused_digit_accuracy=other,
        extra={
            "baseline_connectome_accuracy": float(baseline.mean()),
            "baseline_confused_accuracy": b_conf,
            "confused_digit_delta": confused - b_conf,
            "pair_gated_completion_rate": (forward_rate + pattern_rate) / n_eval,
            "forward_completion_rate": forward_rate / n_eval,
            "pattern_complete_rate": pattern_rate / n_eval,
        },
    )


def run_pair_curriculum_mnist(
    *,
    curriculum_passes: int = 5,
    bundle=None,
    **kwargs,
) -> RepresentationalResult:
    """Confused-pair HIGH→CLASS curriculum on feedforward ventral bundle (H3)."""
    bundle = bundle or load_ventral_bundle(use_cache=True, **kwargs)
    brain = bundle.brain
    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, bundle.k)

    pre = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        pre[digit] = sum(
            connectome_predict(bundle.high_outputs[digit, j], brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        ) / bundle.n_examples

    apply_pairwise_contrast_curriculum(brain, bundle, pair_passes=curriculum_passes)

    post = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        post[digit] = sum(
            connectome_predict(bundle.high_outputs[digit, j], brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        ) / bundle.n_examples

    confused, other = _split_confused_accuracy(post)
    pre_conf, _ = _split_confused_accuracy(pre)
    return RepresentationalResult(
        per_class_accuracy=post,
        mean_accuracy=float(post.mean()),
        data_source=bundle.data_source,
        parameters={**bundle.parameters, "curriculum_passes": curriculum_passes},
        backend="representational_pair_curriculum",
        tier="R",
        method="pair_curriculum",
        confused_digit_accuracy=confused,
        non_confused_digit_accuracy=other,
        extra={
            "pre_curriculum_accuracy": float(pre.mean()),
            "pre_confused_accuracy": pre_conf,
            "confused_digit_delta": confused - pre_conf,
        },
    )


def run_representational_stack_mnist(
    *,
    curriculum_passes: int = 5,
    completion_fraction: float = 0.5,
    bundle=None,
    **kwargs,
) -> RepresentationalResult:
    """
    Combined Phase II stack: pair curriculum + honest completion on recurrent base.

    Applies pairwise contrast on feedforward HIGH codes, then evaluates with
    pair-gated pattern completion using visual_advanced recurrent dynamics (H7+H8).
    """
    readout_bundle = bundle or load_ventral_bundle(use_cache=True, **kwargs)
    rec_bundle = load_recurrent_bundle(use_cache=True, **kwargs)
    brain = readout_bundle.brain
    wire_class_from_prototypes(brain, readout_bundle.prototypes, HIGH, CLASS, readout_bundle.k)

    pre = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        pre[digit] = sum(
            connectome_predict(readout_bundle.high_outputs[digit, j], brain, k=readout_bundle.k) == digit
            for j in range(readout_bundle.n_examples)
        ) / readout_bundle.n_examples

    apply_pairwise_contrast_curriculum(
        brain, readout_bundle, pair_passes=curriculum_passes,
    )

    from neural_assemblies.programs.colt_mnist_tier_a import build_multi_prototype_lexicon
    mp_lex = build_multi_prototype_lexicon(
        readout_bundle.high_outputs, k=readout_bundle.k, prototypes_per_digit=3,
    )

    correct = np.zeros(NUM_DIGITS)
    baseline = np.zeros(NUM_DIGITS)
    completion_rate = 0
    n_eval = NUM_DIGITS * readout_bundle.n_examples

    for digit in range(NUM_DIGITS):
        b_hits = h_hits = 0
        for j in range(readout_bundle.n_examples):
            hv = readout_bundle.high_outputs[digit, j]
            b_pred = connectome_predict(hv, brain, k=readout_bundle.k)
            b_hits += b_pred == digit

            pred, meta = honest_pair_gated_predict(
                hv, brain, rec_bundle.brain, readout_bundle,
                low_pattern=readout_bundle.examples[digit, j],
                mp_lex=mp_lex,
                completion_fraction=completion_fraction,
                seed=kwargs.get("seed", 42) + digit * 100 + j,
            )
            if meta.get("forward_completion_applied"):
                completion_rate += 1
            elif meta["completion_applied"]:
                completion_rate += 1
            h_hits += pred == digit
        correct[digit] = h_hits / readout_bundle.n_examples
        baseline[digit] = b_hits / readout_bundle.n_examples

    confused, other = _split_confused_accuracy(correct)
    b_conf, _ = _split_confused_accuracy(baseline)
    pre_conf, _ = _split_confused_accuracy(pre)
    return RepresentationalResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=readout_bundle.data_source,
        parameters={
            **readout_bundle.parameters,
            "curriculum_passes": curriculum_passes,
            "completion_fraction": completion_fraction,
            "stack": True,
            "recurrent_base": "visual_advanced_recurrent",
        },
        backend="representational_stack",
        tier="R",
        method="pair_curriculum_plus_honest_completion",
        confused_digit_accuracy=confused,
        non_confused_digit_accuracy=other,
        extra={
            "pre_curriculum_accuracy": float(pre.mean()),
            "pre_confused_accuracy": pre_conf,
            "post_curriculum_baseline": float(baseline.mean()),
            "baseline_confused_accuracy": b_conf,
            "confused_digit_delta": confused - b_conf,
            "curriculum_confused_delta": b_conf - pre_conf,
            "pair_gated_completion_rate": completion_rate / n_eval,
        },
    )


def run_occlusion_training_mnist(
    *,
    train_occlusion_fraction: float = 0.25,
    **kwargs,
) -> RepresentationalResult:
    """Ventral training with random LOW dropout (completion-capable codes)."""
    bundle = load_ventral_bundle(
        train_occlusion_fraction=train_occlusion_fraction, use_cache=True, **kwargs,
    )
    brain = bundle.brain
    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, bundle.k)

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        )
        correct[digit] = hits / bundle.n_examples

    confused, other = _split_confused_accuracy(correct)
    return RepresentationalResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=bundle.data_source,
        parameters={
            **bundle.parameters,
            "train_occlusion_fraction": train_occlusion_fraction,
        },
        backend="representational_occlusion_train",
        tier="R",
        method="occlusion_training",
        confused_digit_accuracy=confused,
        non_confused_digit_accuracy=other,
    )


def run_representational_suite(
    *,
    seed: int = 42,
    n_examples: int = 50,
    **kwargs,
) -> dict:
    """Run Phase II ladder and return structured results."""
    from neural_assemblies.programs.colt_mnist_geometry_panel import run_geometry_panel

    clear = kwargs.pop("clear_cache", True)
    if clear:
        from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache
        clear_ventral_bundle_cache()

    kw = dict(seed=seed, n_examples=n_examples, **kwargs)
    return {
        "geometry": run_geometry_panel(**kw),
        "recurrent_base": run_recurrent_representational_mnist(**kw),
        "honest_completion": run_honest_completion_mnist(**kw),
        "pair_curriculum": run_pair_curriculum_mnist(**kw),
        "representational_stack": run_representational_stack_mnist(**kw),
        "occlusion_train": run_occlusion_training_mnist(**kw),
    }
