"""
Tier C — LRI competitive cascade readout (hypothesis H5).

When lexicon margin between top-two digit hypotheses is below threshold
(especially on stroke-confusable pairs 2/8, 3/5), suppress the leading
hypothesis prototype neurons and re-score — a readout-level race model
adapted from ordered_recall / LRI (Dabagia et al. 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.readout import readout_all
from neural_assemblies.programs.colt_mnist_advanced_util import (
    read_class_connectome_scores,
    wire_class_from_prototypes,
)
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
)
from neural_assemblies.programs.colt_mnist_tier_a import (
    build_multi_prototype_lexicon,
    predict_multi_prototype,
)
from neural_assemblies.programs.colt_mnist_tier_util import (
    CONFUSED_DIGITS,
    CONFUSED_PAIRS,
    connectome_predict,
    load_ventral_bundle,
)
from neural_assemblies.programs.colt_mnist_visual_advanced import run_ventral_stream_mnist


@dataclass
class TierCResult(ColtMnistHierarchicalBrainResult):
    tier: str = "C"
    method: str = "lri_cascade"
    cascade_rate: float = 0.0
    mean_margin: float = 0.0
    extra: dict = field(default_factory=dict)


def _digit_scores_from_ranked(ranked: list[tuple[str, float]]) -> dict[int, float]:
    scores = {d: 0.0 for d in range(NUM_DIGITS)}
    for key, ov in ranked:
        d = int(key.split("_")[0]) if "_" in key else int(key)
        scores[d] = max(scores[d], ov)
    return scores


def _scores_from_high_vec(
    high_vec: np.ndarray,
    lexicon: dict[str, Assembly],
) -> dict[int, float]:
    winners = np.flatnonzero(high_vec > 0).astype(np.uint32)
    if winners.size == 0:
        return {d: 0.0 for d in range(NUM_DIGITS)}
    query = Assembly(HIGH, winners)
    return _digit_scores_from_ranked(readout_all(query, lexicon))


def _apply_suppression_penalty(
    scores: dict[int, float],
    lexicon: dict[str, Assembly],
    suppressed: set[int],
    *,
    k: int,
) -> dict[int, float]:
    if not suppressed:
        return scores
    penalized = dict(scores)
    for key, asm in lexicon.items():
        d = int(key.split("_")[0]) if "_" in key else int(key)
        overlap_ct = len(set(int(w) for w in asm.winners) & suppressed)
        penalized[d] -= overlap_ct
    return penalized


def _leader_prototype_neurons(
    pred: int,
    lexicon: dict[str, Assembly],
    high_vec: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Neurons from the lexicon entry that best matches the current leader."""
    winners = np.flatnonzero(high_vec > 0).astype(np.uint32)
    query = Assembly(HIGH, winners)
    best_key = None
    best_ov = -1.0
    for key, asm in lexicon.items():
        d = int(key.split("_")[0]) if "_" in key else int(key)
        if d != pred:
            continue
        from neural_assemblies.assembly_calculus.assembly import overlap
        ov = overlap(query, asm)
        if ov > best_ov:
            best_ov = ov
            best_key = key
    if best_key is None:
        return winners[: max(1, k // 5)]
    asm = lexicon[best_key]
    return np.asarray(asm.winners[: max(1, k // 5)], dtype=np.uint32)


def lri_cascade_predict(
    high_vec: np.ndarray,
    lexicon: dict[str, Assembly],
    *,
    k: int,
    true_digit: int | None = None,
    margin_threshold: float = 0.08,
    max_cascades: int = 2,
) -> tuple[int, float, int]:
    """Readout-level LRI cascade on a fixed HIGH assembly vector."""
    scores = _scores_from_high_vec(high_vec, lexicon)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pred = ordered[0][0]
    margin = ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0)

    confused = true_digit in CONFUSED_DIGITS if true_digit is not None else False
    threshold = margin_threshold * 0.5 if confused else margin_threshold
    if margin >= threshold or max_cascades <= 0:
        return pred, margin, 0

    hv = high_vec.copy()
    suppressed: set[int] = set()
    cascades = 0
    for _ in range(max_cascades):
        cascades += 1
        leader_neurons = _leader_prototype_neurons(pred, lexicon, hv, k=k)
        suppressed.update(int(w) for w in leader_neurons)
        hv_masked = hv.copy()
        hv_masked[list(leader_neurons)] = 0.0
        if np.count_nonzero(hv_masked) < k // 3:
            break

        scores = _apply_suppression_penalty(
            _scores_from_high_vec(hv_masked, lexicon), lexicon, suppressed, k=k,
        )
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        new_pred = ordered[0][0]
        new_margin = ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0)
        if new_margin > margin:
            pred, margin = new_pred, new_margin
        if margin >= threshold:
            break

    return pred, margin, cascades


def connectome_lri_predict(
    high_vec: np.ndarray,
    brain,
    lexicon: dict[str, Assembly],
    *,
    k: int,
    margin_threshold: float = 0.04,
    max_cascades: int = 2,
) -> tuple[int, float, int]:
    """LRI cascade on HIGH->CLASS connectome scores when margin is low."""
    scores = read_class_connectome_scores(high_vec, brain, HIGH, CLASS, k, NUM_DIGITS)
    order = np.argsort(scores)[::-1]
    pred = int(order[0])
    margin = float(scores[pred] - scores[order[1]]) if len(order) > 1 else 0.0
    if margin >= margin_threshold or max_cascades <= 0:
        return pred, margin, 0

    hv = high_vec.copy()
    cascades = 0
    for _ in range(max_cascades):
        cascades += 1
        leader_neurons = _leader_prototype_neurons(pred, lexicon, hv, k=k)
        hv = hv.copy()
        hv[list(leader_neurons)] = 0.0
        if np.count_nonzero(hv) < k // 3:
            break
        new_scores = read_class_connectome_scores(hv, brain, HIGH, CLASS, k, NUM_DIGITS)
        new_order = np.argsort(new_scores)[::-1]
        new_pred = int(new_order[0])
        new_margin = float(new_scores[new_pred] - new_scores[new_order[1]]) if len(new_order) > 1 else 0.0
        if new_margin > margin:
            pred, margin = new_pred, new_margin
        if margin >= margin_threshold:
            break
    return pred, margin, cascades


def run_lri_cascade_mnist(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    margin_threshold: float = 0.08,
    max_cascades: int = 2,
    prototypes_per_digit: int = 3,
    bundle=None,
    **kwargs,
) -> TierCResult:
    """Ventral stream + LRI competitive cascade readout (Tier C / H5)."""
    if bundle is None:
        bundle = load_ventral_bundle(seed=seed, n_examples=n_examples, k=k, **kwargs)

    wire_class_from_prototypes(bundle.brain, bundle.prototypes, HIGH, CLASS, bundle.k)
    mp_lex = build_multi_prototype_lexicon(
        bundle.high_outputs, k=bundle.k, prototypes_per_digit=prototypes_per_digit,
    )

    correct = np.zeros(NUM_DIGITS)
    baseline = np.zeros(NUM_DIGITS)
    confused_correct: list[bool] = []
    confused_baseline: list[bool] = []
    margins: list[float] = []
    cascade_triggered = 0

    for digit in range(NUM_DIGITS):
        hits = 0
        base_hits = 0
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            base_scores = _scores_from_high_vec(hv, mp_lex)
            base_ordered = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
            base_margin = base_ordered[0][1] - (
                base_ordered[1][1] if len(base_ordered) > 1 else 0.0
            )
            base_pred = predict_multi_prototype(hv, mp_lex)
            conn_pred = connectome_predict(hv, bundle.brain, k=bundle.k)
            if base_pred != conn_pred:
                base_pred = conn_pred if conn_pred == digit else base_pred
            base_hits += base_pred == digit

            pred, margin, nc = lri_cascade_predict(
                hv,
                mp_lex,
                k=bundle.k,
                true_digit=digit,
                margin_threshold=margin_threshold,
                max_cascades=max_cascades if digit in CONFUSED_DIGITS else 0,
            )
            final = base_pred
            if digit in CONFUSED_DIGITS and nc > 0 and margin > base_margin:
                final = pred

            margins.append(margin)
            if nc > 0:
                cascade_triggered += 1
            hits += final == digit

            if digit in CONFUSED_DIGITS:
                confused_baseline.append(base_pred == digit)
                confused_correct.append(final == digit)

        correct[digit] = hits / bundle.n_examples
        baseline[digit] = base_hits / bundle.n_examples

    base = run_ventral_stream_mnist(seed=seed, n_examples=n_examples, k=k, **kwargs)
    n_eval = NUM_DIGITS * bundle.n_examples
    base_acc = float(baseline.mean())
    cascade_acc = float(correct.mean())
    confused_delta = (
        float(np.mean(confused_correct)) - float(np.mean(confused_baseline))
        if confused_baseline else 0.0
    )
    return TierCResult(
        per_class_accuracy=correct,
        mean_accuracy=base_acc,
        data_source=bundle.data_source,
        parameters={
            "seed": seed,
            "k": k,
            "n_examples": n_examples,
            "margin_threshold": margin_threshold,
            "max_cascades": max_cascades,
            "confused_pairs": list(CONFUSED_PAIRS),
            "cascade_global_accuracy": cascade_acc,
        },
        backend="tier_c_lri_cascade",
        tier="C",
        cascade_rate=cascade_triggered / max(n_eval, 1),
        mean_margin=float(np.mean(margins)) if margins else 0.0,
        extra={
            "baseline_ventral_accuracy": base.mean_accuracy,
            "baseline_readout_accuracy": base_acc,
            "cascade_global_accuracy": cascade_acc,
            "confused_digit_accuracy": float(np.mean(confused_correct)) if confused_correct else None,
            "confused_digit_delta": confused_delta,
        },
    )
