"""
Assembly Calculus ventral synthesis — empirical scorecard and audit.

Maps neurobiological success criteria to measurable quantities:

1. **Assembly stability** — HIGH→HIGH persistence, pattern_complete recovery
2. **Separability geometry** — confused-pair overlap vs chance k/n
3. **Generative encoding** — structured partial LOW → HIGH (forward path)
4. **Predictive readout** — connectome margin on confused pairs
5. **Top-down decode** — HIGH→LOW overlap (when pathway exists)

Run::

    python -m neural_assemblies.programs.colt_mnist_synthesis --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap, pattern_complete
from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners
from neural_assemblies.programs.colt_mnist_forward_completion import (
    encode_and_predict,
    has_generative_head,
    measure_digit_forward_completion_battery,
    readout_prototypes,
)
from neural_assemblies.programs.colt_mnist_absence import apply_bundle_absence
from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, LOW, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_tier_util import (
    CONFUSED_DIGITS,
    CONFUSED_PAIRS,
    connectome_predict,
)


@dataclass
class ACScorecard:
    """Neurobiologically grounded metrics (not leaderboard accuracy alone)."""

    discriminative_accuracy: float
    confused_digit_accuracy: float
    non_confused_digit_accuracy: float
    mean_confused_pair_overlap: float
    chance_overlap: float
    high_self_persistence: float
    pattern_complete_recovery: float
    forward_center_band_overlap: float
    digit3_absence_accuracy: float
    low_regeneration: float | None
    generative_head_available: bool
    assembly_stable: bool
    geometry_adequate: bool
    generative_viable: bool
    narrative: str
    details: dict[str, Any] = field(default_factory=dict)


def _confused_pair_overlap_mean(prototypes: np.ndarray, k: int) -> float:
    from neural_assemblies.assembly_calculus.assembly import Assembly

    mat = np.zeros((NUM_DIGITS, NUM_DIGITS))
    lex = {
        d: Assembly(HIGH, np.flatnonzero(prototypes[d] > 0).astype(np.uint32))
        for d in range(NUM_DIGITS)
    }
    for i in range(NUM_DIGITS):
        for j in range(NUM_DIGITS):
            mat[i, j] = overlap(lex[i], lex[j])
    vals = [float(mat[a, b]) for a, b in CONFUSED_PAIRS]
    return float(np.mean(vals)) if vals else 0.0


def measure_high_persistence(brain, high_vec: np.ndarray, *, rounds: int = 5) -> float:
    """HIGH→HIGH overlap after warmup (attractor stability)."""
    set_kcap_winners(brain, HIGH, high_vec)
    for _ in range(3):
        brain.project({}, {HIGH: [HIGH]})
    a = _snap(brain, HIGH)
    for _ in range(rounds):
        brain.project({}, {HIGH: [HIGH]})
    b = _snap(brain, HIGH)
    return float(overlap(a, b))


def audit_bundle(
    bundle,
    *,
    label: str = "bundle",
    seed: int = 42,
) -> ACScorecard:
    """Full AC scorecard on one VentralBundle."""
    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes
    from neural_assemblies.programs.colt_mnist_absence import digit3_absence_battery
    from neural_assemblies.programs.patch_registry import patch_absence_protocols, resolve_patch_graph
    from neural_assemblies.programs.colt_mnist_attractor import consolidate_low_from_high

    brain = bundle.brain
    k = bundle.k
    n = brain.areas[HIGH].n
    chance = k / n

    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, k)

    per_digit = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], brain, k=k) == digit
            for j in range(bundle.n_examples)
        )
        per_digit[digit] = hits / bundle.n_examples

    confused = float(np.mean([per_digit[d] for d in CONFUSED_DIGITS]))
    other = float(np.mean([per_digit[d] for d in range(NUM_DIGITS) if d not in CONFUSED_DIGITS]))
    pair_ov = _confused_pair_overlap_mean(bundle.prototypes, k)

    persist_vals = [
        measure_high_persistence(brain, bundle.high_outputs[d, 0])
        for d in range(NUM_DIGITS)
    ]
    mean_persist = float(np.mean(persist_vals))

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    has_rec = float(np.sum(brain.connectomes[HIGH][HIGH].weights)) > 0.0
    pc_vals: list[float] = []
    try:
        for digit in range(NUM_DIGITS):
            digit_recs: list[float] = []
            for j in range(min(5, bundle.n_examples)):
                set_kcap_winners(brain, HIGH, bundle.high_outputs[digit, j])
                if has_rec:
                    for _ in range(3):
                        brain.project({}, {HIGH: [HIGH]})
                _, rec = pattern_complete(
                    brain, HIGH, fraction=0.5, rounds=5,
                    seed=seed + digit * 100 + j, observation_mode="plastic",
                )
                digit_recs.append(float(rec))
            pc_vals.append(float(np.mean(digit_recs)))
    finally:
        brain.disable_plasticity = saved
    mean_pc = float(np.mean(pc_vals))

    saved_fwd = brain.disable_plasticity
    brain.disable_plasticity = True
    try:
        fwd_protocol = "center_patch" if resolve_patch_graph(bundle) else "center_band"
        try:
            fwd = measure_digit_forward_completion_battery(
                bundle, 3, (fwd_protocol,), seed=seed,
            )
            center_ov = float(fwd[fwd_protocol]["prototype_overlap"])
        except (ValueError, KeyError):
            fwd = measure_digit_forward_completion_battery(
                bundle, 3, ("center_band",), seed=seed,
            )
            center_ov = float(fwd["center_band"]["prototype_overlap"])

        gen_available = has_generative_head(bundle)
        absence_battery = patch_absence_protocols(bundle)
        if resolve_patch_graph(bundle) is None:
            absence_battery = digit3_absence_battery()
        d3_hits = d3_total = 0
        eval_digit = 3 if resolve_patch_graph(bundle) is None else 0
        for proto in absence_battery[:3]:
            for j in range(bundle.n_examples):
                pat = apply_bundle_absence(bundle, bundle.examples[eval_digit, j], proto)
                pred, _, _ = encode_and_predict(
                    bundle, pat, head="generative" if gen_available else "discriminative",
                )
                d3_total += 1
                d3_hits += pred == eval_digit
        d3_abs = d3_hits / max(d3_total, 1)
    finally:
        brain.disable_plasticity = saved_fwd

    low_regen: float | None = None
    if (
        HIGH in brain.connectomes
        and LOW in brain.connectomes.get(HIGH, {})
        and float(np.sum(brain.connectomes[HIGH][LOW].weights)) > 0
    ):
        protos = readout_prototypes(bundle)
        vals = [
            consolidate_low_from_high(
                brain, protos[d], bundle.examples[d, 0], k, rounds=5,
            )
            for d in range(NUM_DIGITS)
        ]
        low_regen = float(np.mean(vals))

    assembly_stable = mean_persist >= 0.85 and mean_pc >= 0.50
    geometry_adequate = pair_ov < 0.35 and confused + 0.05 < other
    generative_viable = d3_abs >= 0.50 or (center_ov >= 0.25 and d3_abs >= 0.40)

    lines = [
        f"AC Scorecard — {label}",
        "=" * 40,
        f"Discriminative accuracy: {float(per_digit.mean()):.1%}  "
        f"(confused {confused:.1%} / other {other:.1%})",
        f"Confused-pair overlap: {pair_ov:.3f}  (chance {chance:.3f})",
        f"HIGH self-persistence: {mean_persist:.1%}  |  pattern_complete: {mean_pc:.1%}",
        f"Forward center_band (d=3): {center_ov:.1%}  |  d3 absence cls: {d3_abs:.1%}",
    ]
    if low_regen is not None:
        lines.append(f"LOW regeneration: {low_regen:.1%}")
    lines.extend([
        "",
        f"Assembly stable: {'YES' if assembly_stable else 'no'}",
        f"Geometry adequate: {'YES' if geometry_adequate else 'no'}",
        f"Generative viable: {'YES' if generative_viable else 'no'}",
    ])

    return ACScorecard(
        discriminative_accuracy=float(per_digit.mean()),
        confused_digit_accuracy=confused,
        non_confused_digit_accuracy=other,
        mean_confused_pair_overlap=pair_ov,
        chance_overlap=chance,
        high_self_persistence=mean_persist,
        pattern_complete_recovery=mean_pc,
        forward_center_band_overlap=center_ov,
        digit3_absence_accuracy=d3_abs,
        low_regeneration=low_regen,
        generative_head_available=gen_available,
        assembly_stable=assembly_stable,
        geometry_adequate=geometry_adequate,
        generative_viable=generative_viable,
        narrative="\n".join(lines),
        details={
            "per_digit_accuracy": per_digit.tolist(),
            "per_digit_persistence": persist_vals,
            "per_digit_pc_recovery": pc_vals,
            "label": label,
        },
    )


def run_synthesis_audit(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
) -> dict[str, ACScorecard]:
    """Compare recurrent vs attractor bundles on AC scorecard."""
    from neural_assemblies.programs.colt_mnist_tier_util import (
        clear_ventral_bundle_cache,
        load_attractor_bundle,
        load_recurrent_bundle,
    )

    clear_ventral_bundle_cache()
    rec = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )
    attr = load_attractor_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=False,
    )
    return {
        "recurrent": audit_bundle(rec, label="recurrent", seed=seed),
        "attractor": audit_bundle(attr, label="attractor", seed=seed),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AC ventral synthesis scorecard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()

    cards = run_synthesis_audit(
        seed=args.seed, n_examples=args.n_examples, k=args.k,
    )
    for _name, card in cards.items():
        print(card.narrative)
        print()

