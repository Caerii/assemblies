"""
Ventral push panel — Tier A/C stack toward 88%+ on recurrent baseline.

Phase 1: readout tricks (multi-prototype, LRI).
Phase 2: encoding + curriculum (learn_assembly, confused-pair contrast, stack).

Run::

    python -m neural_assemblies.programs.colt_mnist_ventral_push --n-examples 50
    python -m neural_assemblies.programs.colt_mnist_ventral_push --phase 2 --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_tier_util import (
    CONFUSED_DIGITS,
    VentralBundle,
    connectome_predict,
    refresh_bundle_representations,
)


@dataclass
class VentralPushRow:
    name: str
    mean_accuracy: float
    confused_digit_accuracy: float
    non_confused_digit_accuracy: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VentralPushPanel:
    rows: list[VentralPushRow]
    phase: str
    narrative: str


def _split_acc(per_digit: np.ndarray) -> tuple[float, float]:
    conf = [per_digit[d] for d in CONFUSED_DIGITS]
    other = [per_digit[d] for d in range(NUM_DIGITS) if d not in CONFUSED_DIGITS]
    return float(np.mean(conf)), float(np.mean(other))


def evaluate_bundle_accuracy(bundle: VentralBundle) -> tuple[np.ndarray, float, float, float]:
    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes

    wire_class_from_prototypes(bundle.brain, bundle.prototypes, HIGH, CLASS, bundle.k)
    per_digit = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], bundle.brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        )
        per_digit[digit] = hits / bundle.n_examples
    conf, other = _split_acc(per_digit)
    return per_digit, float(per_digit.mean()), conf, other


def _row_from_bundle(name: str, bundle: VentralBundle, **details) -> VentralPushRow:
    _, mean_acc, conf, other = evaluate_bundle_accuracy(bundle)
    return VentralPushRow(
        name=name,
        mean_accuracy=mean_acc,
        confused_digit_accuracy=conf,
        non_confused_digit_accuracy=other,
        details=details,
    )


def run_ventral_push_phase1(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
) -> list[VentralPushRow]:
    from neural_assemblies.programs.colt_mnist_lri_readout import run_lri_cascade_mnist
    from neural_assemblies.programs.colt_mnist_tier_a import run_multi_prototype_mnist
    from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_recurrent_bundle

    clear_ventral_bundle_cache()
    bundle = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )

    _, mean_acc, conf, other = evaluate_bundle_accuracy(bundle)
    rows = [
        VentralPushRow(
            name="recurrent_connectome",
            mean_accuracy=mean_acc,
            confused_digit_accuracy=conf,
            non_confused_digit_accuracy=other,
        ),
    ]

    mp = run_multi_prototype_mnist(bundle=bundle, seed=seed, n_examples=n_examples, k=k)
    mp_conf, mp_other = _split_acc(mp.per_class_accuracy)
    rows.append(VentralPushRow(
        name="multi_prototype_lexicon",
        mean_accuracy=mp.mean_accuracy,
        confused_digit_accuracy=mp_conf,
        non_confused_digit_accuracy=mp_other,
        details={"readout_best": mp.extra.get("readout_best")},
    ))

    lri = run_lri_cascade_mnist(bundle=bundle, seed=seed, n_examples=n_examples, k=k)
    lri_conf, lri_other = _split_acc(lri.per_class_accuracy)
    rows.append(VentralPushRow(
        name="lri_cascade_readout",
        mean_accuracy=float(lri.per_class_accuracy.mean()),
        confused_digit_accuracy=lri_conf,
        non_confused_digit_accuracy=lri_other,
        details={
            "cascade_rate": lri.cascade_rate,
            "confused_digit_delta": lri.extra.get("confused_digit_delta"),
        },
    ))
    return rows


def run_ventral_push_phase2(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    pair_passes: int = 5,
    learn_assembly_epochs: int = 10,
) -> list[VentralPushRow]:
    from neural_assemblies.programs.colt_mnist_attractor import apply_learn_assembly_curriculum
    from neural_assemblies.programs.colt_mnist_representational import apply_pairwise_contrast_curriculum
    from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_recurrent_bundle

    clear_ventral_bundle_cache()
    baseline = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )
    rows = [_row_from_bundle("recurrent_baseline", baseline)]

    pair_bundle = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )
    apply_pairwise_contrast_curriculum(
        pair_bundle.brain, pair_bundle, pair_passes=pair_passes,
    )
    rows.append(_row_from_bundle(
        "confused_pair_curriculum",
        pair_bundle,
        pair_passes=pair_passes,
    ))

    learn_bundle = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )
    la_stats = apply_learn_assembly_curriculum(
        learn_bundle.brain,
        learn_bundle.high_bias,
        learn_bundle.examples,
        learn_bundle.high_outputs,
        learn_bundle.k,
        learn_bundle.n_examples,
        max_epochs=learn_assembly_epochs,
    )
    refresh_bundle_representations(learn_bundle)
    rows.append(_row_from_bundle(
        "learn_assembly_curriculum",
        learn_bundle,
        **la_stats,
    ))

    stack = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )
    stack_la = apply_learn_assembly_curriculum(
        stack.brain,
        stack.high_bias,
        stack.examples,
        stack.high_outputs,
        stack.k,
        stack.n_examples,
        max_epochs=learn_assembly_epochs,
    )
    refresh_bundle_representations(stack)
    apply_pairwise_contrast_curriculum(
        stack.brain, stack, pair_passes=pair_passes,
    )
    rows.append(_row_from_bundle(
        "learn_assembly_plus_pair_curriculum",
        stack,
        pair_passes=pair_passes,
        **stack_la,
    ))

    return rows


def run_ventral_push_panel(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    phase: Literal["1", "2", "all"] = "all",
) -> VentralPushPanel:
    rows: list[VentralPushRow] = []
    if phase in ("1", "all"):
        rows.extend(run_ventral_push_phase1(seed=seed, n_examples=n_examples, k=k))
    if phase in ("2", "all"):
        if phase == "all":
            rows.append(VentralPushRow(
                name="--- phase 2 ---",
                mean_accuracy=0.0,
                confused_digit_accuracy=0.0,
                non_confused_digit_accuracy=0.0,
            ))
        rows.extend(run_ventral_push_phase2(seed=seed, n_examples=n_examples, k=k))

    narrative = _format_narrative(rows, phase)
    return VentralPushPanel(rows=rows, phase=phase, narrative=narrative)


def _format_narrative(rows: list[VentralPushRow], phase: str) -> str:
    lines = [
        f"Ventral push panel (phase={phase})",
        "=" * 48,
        "Target: 88%+ on recurrent ~80% baseline.",
        "",
    ]
    baseline_rows = [r for r in rows if "baseline" in r.name or r.name == "recurrent_connectome"]
    base = baseline_rows[0].mean_accuracy if baseline_rows else 0.0
    for r in rows:
        if r.name.startswith("---"):
            lines.append("")
            lines.append("Phase 2 — encoding + confused-pair curriculum:")
            continue
        delta = r.mean_accuracy - base
        delta_s = ""
        if r.name not in ("recurrent_connectome", "recurrent_baseline") and base > 0:
            delta_s = f"  ({delta:+.1%} vs baseline)"
        lines.append(
            f"  {r.name}: acc={r.mean_accuracy:.1%}{delta_s}  "
            f"confused={r.confused_digit_accuracy:.1%}  "
            f"other={r.non_confused_digit_accuracy:.1%}  {r.details}"
        )
    eval_rows = [r for r in rows if not r.name.startswith("---")]
    best = max(eval_rows, key=lambda r: r.mean_accuracy) if eval_rows else None
    if best:
        lines.extend([
            "",
            f"Best: {best.name} @ {best.mean_accuracy:.1%}",
            "Phase 2 notes: pair curriculum on fixed HIGH codes may saturate at ~80%;",
            "post-hoc learn_assembly (HIGH-only) currently regresses — integrate at train time.",
        ])
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ventral push toward 88%")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--phase", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()

    panel = run_ventral_push_panel(
        seed=args.seed, n_examples=args.n_examples, k=args.k, phase=args.phase,
    )
    print(panel.narrative)


if __name__ == "__main__":
    main()
