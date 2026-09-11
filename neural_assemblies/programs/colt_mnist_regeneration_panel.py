"""
Phase I-b — Generative completeness / attractor regeneration panel.

Measures whether HIGH assemblies are true attractors (H11), whether digit 3
survives structured absence (H12, Ember test), and LOW top-down regeneration (H13).

Run before investing in cross-modal binding or readout tricks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from neural_assemblies.assembly_calculus.ops import _snap, pattern_complete
from neural_assemblies.programs.colt_mnist_absence import (
    ABSENCE_PROTOCOLS,
    apply_absence_mask,
    digit3_absence_battery,
)
from neural_assemblies.programs.colt_mnist_forward_completion import (
    encode_and_predict,
    has_generative_head,
    measure_digit_forward_completion_battery,
    readout_prototypes,
)
from neural_assemblies.programs.colt_mnist_attractor import (
    consolidate_low_from_high,
    measure_recovery_on_bundle,
)
from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners
from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, LOW, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict, load_recurrent_bundle

# H11–H13 generative gates (supplement H6 binding gate).
REGENERATION_GATE = {
    "digit3_forward_overlap_min": 0.35,
    "digit3_structured_absence_acc_min": 0.55,
    "mean_low_regeneration_min": 0.15,
    "mean_forward_prototype_overlap_min": 0.30,
    "pattern_complete_recovery_min": 0.50,  # E2 fixed: overlap is already normalized [0,1]
}


@dataclass
class RegenerationPanelResult:
    """Structured generative audit for one bundle."""

    per_digit_forward_overlap: np.ndarray
    digit3_forward_by_protocol: dict[str, dict[str, float]]
    digit3_forward_overlap_mean: float
    per_digit_recovery: np.ndarray
    per_digit_cls_after_completion: np.ndarray
    digit3_absence_by_protocol: dict[str, float]
    digit3_absence_mean: float
    mean_high_recovery: float
    mean_low_regeneration: float | None
    per_digit_low_regeneration: np.ndarray | None
    regeneration_gate_open: bool
    regeneration_gate_reasons: list[str]
    baseline_label: str
    narrative: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in (
            "per_digit_recovery", "per_digit_cls_after_completion",
            "per_digit_low_regeneration", "per_digit_forward_overlap",
        ):
            val = d.get(key)
            if isinstance(val, np.ndarray):
                d[key] = val.tolist()
        return d


def _classify_after_completion(
    bundle,
    readout_brain,
    *,
    fraction: float = 0.5,
    rounds: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Per-digit connectome accuracy after HIGH pattern_complete."""
    brain = bundle.brain
    k = bundle.k
    per_digit = np.zeros(NUM_DIGITS)
    has_rec = float(np.sum(brain.connectomes[HIGH][HIGH].weights)) > 0.0
    for digit in range(NUM_DIGITS):
        hits = 0
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            set_kcap_winners(brain, HIGH, hv)
            if has_rec:
                for _ in range(3):
                    brain.project({}, {HIGH: [HIGH]})
            pattern_complete(
                brain, HIGH, fraction=fraction, rounds=rounds,
                seed=seed + digit * 100 + j,
                observation_mode="plastic",
            )
            hv2 = np.zeros_like(hv)
            hv2[np.asarray(_snap(brain, HIGH).winners, dtype=int)] = 1.0
            hits += connectome_predict(hv2, readout_brain, k=k) == digit
        per_digit[digit] = hits / bundle.n_examples
    return per_digit


def _structured_absence_accuracy(
    bundle,
    readout_brain,
    digit: int,
    protocol: str,
    *,
    use_forward_encode: bool = True,
    seed: int = 42,
) -> float:
    """Classify digit from structurally masked LOW via forward encoding (generative path)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for j in range(bundle.n_examples):
        pat = apply_absence_mask(bundle.examples[digit, j], protocol, rng=rng)
        pred, _, _ = encode_and_predict(
            bundle, pat,
            head="generative" if has_generative_head(bundle) else "auto",
        )
        hits += pred == digit
    return hits / bundle.n_examples


def _low_regeneration_per_digit(bundle) -> np.ndarray | None:
    """HIGH→LOW overlap vs reference, per digit (requires HIGH→LOW connectome)."""
    brain = bundle.brain
    if HIGH not in brain.connectomes or LOW not in brain.connectomes.get(HIGH, {}):
        return None
    if float(np.sum(brain.connectomes[HIGH][LOW].weights)) <= 0:
        return None
    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    out = np.zeros(NUM_DIGITS)
    try:
        for digit in range(NUM_DIGITS):
            vals = [
                consolidate_low_from_high(
                    brain, bundle.high_outputs[digit, j], bundle.examples[digit, j], bundle.k,
                )
                for j in range(min(5, bundle.n_examples))
            ]
            out[digit] = float(np.mean(vals))
    finally:
        brain.disable_plasticity = saved
    return out


def regeneration_gate_open(
    panel: RegenerationPanelResult,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if panel.digit3_forward_overlap_mean >= REGENERATION_GATE["digit3_forward_overlap_min"]:
        reasons.append("digit3_forward_overlap")
    if panel.digit3_absence_mean >= REGENERATION_GATE["digit3_structured_absence_acc_min"]:
        reasons.append("digit3_structured_absence")
    if float(panel.per_digit_forward_overlap.mean()) >= REGENERATION_GATE["mean_forward_prototype_overlap_min"]:
        reasons.append("mean_forward_overlap")
    if panel.mean_low_regeneration is not None:
        if panel.mean_low_regeneration >= REGENERATION_GATE["mean_low_regeneration_min"]:
            reasons.append("mean_low_regeneration")
    if panel.mean_high_recovery >= REGENERATION_GATE["pattern_complete_recovery_min"]:
        reasons.append("pattern_complete_recovery")
    return len(reasons) > 0, reasons


def run_regeneration_panel(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    bundle=None,
    readout_bundle=None,
    baseline_label: str = "recurrent",
    **kwargs,
) -> RegenerationPanelResult:
    """Generative completeness audit on a recurrent/attractor bundle."""
    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes

    if bundle is None:
        if baseline_label == "attractor":
            from neural_assemblies.programs.colt_mnist_tier_util import load_attractor_bundle
            bundle = load_attractor_bundle(seed=seed, n_examples=n_examples, k=k, **kwargs)
        else:
            bundle = load_recurrent_bundle(seed=seed, n_examples=n_examples, k=k, **kwargs)

    readout_brain = bundle.brain
    readout_protos = readout_prototypes(bundle)
    wire_class_from_prototypes(readout_brain, readout_protos, HIGH, CLASS, bundle.k)

    saved_plast = readout_brain.disable_plasticity
    readout_brain.disable_plasticity = True
    try:
        recovery = measure_recovery_on_bundle(bundle, seed=seed)
        cls_after = _classify_after_completion(bundle, readout_brain, seed=seed)

        forward_per_digit = np.zeros(NUM_DIGITS)
        for digit in range(NUM_DIGITS):
            battery = measure_digit_forward_completion_battery(
                bundle, digit, ("center_band",), seed=seed + digit,
            )
            forward_per_digit[digit] = battery["center_band"]["prototype_overlap"]

        d3_forward = measure_digit_forward_completion_battery(
            bundle, 3, digit3_absence_battery(), seed=seed,
        )
        d3_forward_mean = float(np.mean([v["prototype_overlap"] for v in d3_forward.values()]))

        d3_proto: dict[str, float] = {}
        for proto in digit3_absence_battery():
            d3_proto[proto] = _structured_absence_accuracy(
                bundle, readout_brain, 3, proto, use_forward_encode=True, seed=seed,
            )
        d3_mean = float(np.mean(list(d3_proto.values()))) if d3_proto else 0.0

        low_regen = _low_regeneration_per_digit(bundle)
        mean_low = float(np.mean(low_regen)) if low_regen is not None else None
    finally:
        readout_brain.disable_plasticity = saved_plast

    panel = RegenerationPanelResult(
        per_digit_forward_overlap=forward_per_digit,
        digit3_forward_by_protocol=d3_forward,
        digit3_forward_overlap_mean=d3_forward_mean,
        per_digit_recovery=recovery,
        per_digit_cls_after_completion=cls_after,
        digit3_absence_by_protocol=d3_proto,
        digit3_absence_mean=d3_mean,
        mean_high_recovery=float(recovery.mean()),
        mean_low_regeneration=mean_low,
        per_digit_low_regeneration=low_regen,
        regeneration_gate_open=False,
        regeneration_gate_reasons=[],
        baseline_label=baseline_label,
        narrative="",
        parameters={"seed": seed, "n_examples": n_examples, "k": k, "baseline": baseline_label, **kwargs},
    )
    open_gate, reasons = regeneration_gate_open(panel)
    panel.regeneration_gate_open = open_gate
    panel.regeneration_gate_reasons = reasons
    panel.narrative = _build_narrative(panel)
    return panel


def _build_narrative(panel: RegenerationPanelResult) -> str:
    lines = [
        "Generative completeness panel (Phase I-b)",
        "=" * 42,
        f"Bundle: {panel.baseline_label}",
        "",
        f"Mean forward prototype overlap (center_band): {float(panel.per_digit_forward_overlap.mean()):.1%}",
        f"Digit-3 forward overlap mean: {panel.digit3_forward_overlap_mean:.1%}  "
        f"(H11 gate >= {REGENERATION_GATE['digit3_forward_overlap_min']:.0%})",
        "",
        f"pattern_complete recovery@50%: {panel.mean_high_recovery:.1%}",
        f"Digit-3 pattern_complete: {panel.per_digit_recovery[3]:.1%}",
        "",
        "Per-digit forward overlap (center_band) / pattern_complete recovery:",
    ]
    for d in range(NUM_DIGITS):
        tag = "*" if d in {2, 3, 5, 8} else " "
        lines.append(
            f"  {tag} {d}: forward={panel.per_digit_forward_overlap[d]:.1%}  "
            f"pc_recovery={panel.per_digit_recovery[d]:.1%}  "
            f"cls_after_pc={panel.per_digit_cls_after_completion[d]:.1%}"
        )
    lines.extend(["", "Digit-3 structured absence classification (Ember test):"])
    for proto, acc in panel.digit3_absence_by_protocol.items():
        desc = ABSENCE_PROTOCOLS.get(proto, proto)
        fwd = panel.digit3_forward_by_protocol.get(proto, {})
        lines.append(
            f"  {proto}: cls={acc:.1%}  forward_ov={fwd.get('prototype_overlap', 0):.1%}  ({desc})"
        )
    lines.append(
        f"  mean: {panel.digit3_absence_mean:.1%}  "
        f"(H12 gate >= {REGENERATION_GATE['digit3_structured_absence_acc_min']:.0%})"
    )
    if panel.mean_low_regeneration is not None:
        lines.extend([
            "",
            f"Mean LOW regeneration overlap: {panel.mean_low_regeneration:.1%}  "
            f"(H13 gate >= {REGENERATION_GATE['mean_low_regeneration_min']:.0%})",
        ])
    else:
        lines.extend(["", "LOW regeneration: not available (no HIGH→LOW pathway)"])
    lines.extend([
        "",
        f"Regeneration gate: {'OPEN' if panel.regeneration_gate_open else 'CLOSED'}",
    ])
    if panel.regeneration_gate_reasons:
        lines.append(f"  reasons: {', '.join(panel.regeneration_gate_reasons)}")
    else:
        lines.append(
            "  need digit3_forward>=35%, digit3_absence>=55%, or mean_forward>=30%"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generative completeness panel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--baseline", choices=("recurrent", "attractor"), default="recurrent")
    parser.add_argument("--export", type=str)
    args = parser.parse_args()

    panel = run_regeneration_panel(
        seed=args.seed, n_examples=args.n_examples, baseline_label=args.baseline,
    )
    print(panel.narrative)
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(panel.to_dict(), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
