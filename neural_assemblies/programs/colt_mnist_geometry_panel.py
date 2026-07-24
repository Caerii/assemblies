"""
Phase I — Representational geometry panel.

Measures HIGH assembly geometry *before* training tricks or hub fusion:
pairwise prototype overlap, connectome margins, confused-pair concentration,
and occlusion recovery.  Used to gate cross-modal binding (H6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import pattern_complete
from neural_assemblies.programs.colt_mnist_advanced_util import read_class_connectome_scores
from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners
from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_tier_util import (
    CONFUSED_DIGITS,
    CONFUSED_PAIRS,
    connectome_predict,
    load_ventral_bundle,
)

# H6 gate: re-open cross-modal binding when any threshold is met.
BINDING_GATE = {
    "vision_semantic_accuracy_min": 0.70,
    "contrastive_margin_min": 0.05,
    "confused_digit_accuracy_min": 0.80,
}


@dataclass
class GeometryPanelResult:
    """Structured geometry audit for one ventral bundle."""

    prototype_overlap: np.ndarray
    pair_overlap_ranked: list[tuple[int, int, float]]
    per_digit_connectome_accuracy: np.ndarray
    confused_digit_accuracy: float
    non_confused_digit_accuracy: float
    mean_margin_correct: float
    mean_margin_error: float
    confused_pair_margins: dict[str, float]
    occlusion_fractions: list[float]
    occlusion_accuracies: list[float]
    binding_gate_open: bool
    binding_gate_reasons: list[str]
    narrative: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prototype_overlap"] = self.prototype_overlap.tolist()
        d["per_digit_connectome_accuracy"] = self.per_digit_connectome_accuracy.tolist()
        return d


def prototype_overlap_matrix(prototypes: np.ndarray, k: int) -> np.ndarray:
    """10x10 normalized prototype overlap (HIGH support)."""
    mat = np.zeros((NUM_DIGITS, NUM_DIGITS), dtype=np.float64)
    lex = {
        str(d): Assembly(HIGH, np.flatnonzero(prototypes[d] > 0).astype(np.uint32))
        for d in range(NUM_DIGITS)
    }
    for i in range(NUM_DIGITS):
        for j in range(NUM_DIGITS):
            mat[i, j] = overlap(lex[str(i)], lex[str(j)])
    return mat


def _pair_overlap_ranking(mat: np.ndarray) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for i in range(NUM_DIGITS):
        for j in range(i + 1, NUM_DIGITS):
            pairs.append((i, j, float(mat[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _margin_stats(bundle, brain) -> tuple[float, float, dict[str, float]]:
    margins_correct: list[float] = []
    margins_error: list[float] = []
    pair_margins: dict[str, list[float]] = {
        f"{a}_{b}": [] for a, b in CONFUSED_PAIRS
    }

    for digit in range(NUM_DIGITS):
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            scores = read_class_connectome_scores(
                hv, brain, HIGH, CLASS, bundle.k, NUM_DIGITS,
            )
            order = np.argsort(scores)[::-1]
            margin = float(scores[order[0]] - scores[order[1]])
            if order[0] == digit:
                margins_correct.append(margin)
            else:
                margins_error.append(margin)
            for a, b in CONFUSED_PAIRS:
                if digit in (a, b):
                    pair_margins[f"{a}_{b}"].append(margin)

    mc = float(np.mean(margins_correct)) if margins_correct else 0.0
    me = float(np.mean(margins_error)) if margins_error else 0.0
    pair_means = {k: float(np.mean(v)) if v else 0.0 for k, v in pair_margins.items()}
    return mc, me, pair_means


def _occlusion_curve(
    bundle,
    rec_bundle,
    *,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    completion_rounds: int = 5,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Connectome accuracy vs HIGH masking fraction (with completion when >0)."""
    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes

    brain = bundle.brain
    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, bundle.k)
    rec_brain = rec_bundle.brain
    accs: list[float] = []

    for frac in fractions:
        hits = 0
        total = NUM_DIGITS * bundle.n_examples
        for digit in range(NUM_DIGITS):
            for j in range(bundle.n_examples):
                hv = bundle.high_outputs[digit, j].copy()
                if frac > 0:
                    set_kcap_winners(rec_brain, HIGH, hv)
                    if float(np.sum(rec_brain.connectomes[HIGH][HIGH].weights)) > 0:
                        for _ in range(3):
                            rec_brain.project({}, {HIGH: [HIGH]})
                    _, _ = pattern_complete(
                        rec_brain, HIGH, fraction=1.0 - frac,
                        rounds=completion_rounds, seed=seed + digit * 100 + j,
                    )
                    winners = np.asarray(rec_brain.areas[HIGH].winners, dtype=int)
                    hv = np.zeros_like(hv)
                    if winners.size:
                        hv[winners] = 1.0
                pred = connectome_predict(hv, brain, k=bundle.k)
                hits += pred == digit
        accs.append(hits / total)
    return list(fractions), accs


def binding_gate_open(panel, *, vision_semantic_accuracy: float | None = None, contrastive_margin: float | None = None) -> bool:
    """Whether H6 gate permits cross-modal binding investment."""
    reasons: list[str] = []
    if panel.confused_digit_accuracy >= BINDING_GATE["confused_digit_accuracy_min"]:
        reasons.append("confused_digit_accuracy")
    if vision_semantic_accuracy is not None and vision_semantic_accuracy >= BINDING_GATE["vision_semantic_accuracy_min"]:
        reasons.append("vision_semantic_accuracy")
    if contrastive_margin is not None and contrastive_margin >= BINDING_GATE["contrastive_margin_min"]:
        reasons.append("contrastive_margin")
    return len(reasons) > 0


def run_geometry_panel(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    bundle=None,
    **kwargs,
) -> GeometryPanelResult:
    """Phase I geometry audit on a ventral (or recurrent) bundle."""
    if bundle is None:
        bundle = load_ventral_bundle(seed=seed, n_examples=n_examples, k=k, **kwargs)

    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes

    brain = bundle.brain
    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, bundle.k)

    mat = prototype_overlap_matrix(bundle.prototypes, bundle.k)
    ranked = _pair_overlap_ranking(mat)

    per_digit = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        )
        per_digit[digit] = hits / bundle.n_examples

    confused_acc = float(np.mean([per_digit[d] for d in CONFUSED_DIGITS]))
    non_confused_acc = float(
        np.mean([per_digit[d] for d in range(NUM_DIGITS) if d not in CONFUSED_DIGITS])
    )
    mc, me, pair_margins = _margin_stats(bundle, brain)

    from neural_assemblies.programs.colt_mnist_tier_util import load_recurrent_bundle

    rec_kw = {key: val for key, val in kwargs.items() if key != "use_cache"}
    rec_cache = kwargs.get("use_cache", True)
    rec_bundle = load_recurrent_bundle(
        seed=seed, n_examples=n_examples, k=k, use_cache=rec_cache, **rec_kw,
    )
    fracs, occ_accs = _occlusion_curve(bundle, rec_bundle, seed=seed)

    gate_reasons: list[str] = []
    if confused_acc >= BINDING_GATE["confused_digit_accuracy_min"]:
        gate_reasons.append("confused_digit_accuracy")
    binding_open = len(gate_reasons) > 0

    narrative = _build_narrative(
        mat, ranked, per_digit, confused_acc, non_confused_acc, mc, me, pair_margins,
        fracs, occ_accs, binding_open, gate_reasons,
    )

    return GeometryPanelResult(
        prototype_overlap=mat,
        pair_overlap_ranked=ranked[:15],
        per_digit_connectome_accuracy=per_digit,
        confused_digit_accuracy=confused_acc,
        non_confused_digit_accuracy=non_confused_acc,
        mean_margin_correct=mc,
        mean_margin_error=me,
        confused_pair_margins=pair_margins,
        occlusion_fractions=fracs,
        occlusion_accuracies=occ_accs,
        binding_gate_open=binding_open,
        binding_gate_reasons=gate_reasons,
        narrative=narrative,
        parameters={"seed": seed, "n_examples": n_examples, "k": k, **kwargs},
    )


def _build_narrative(
    mat, ranked, per_digit, confused_acc, non_confused_acc,
    mc, me, pair_margins, fracs, occ_accs, binding_open, gate_reasons,
) -> str:
    lines = [
        "Representational geometry panel (Phase I)",
        "=" * 40,
        "",
        "Highest prototype overlap pairs (H3 diagnostic):",
    ]
    for a, b, ov in ranked[:6]:
        tag = " CONFUSED" if (a, b) in CONFUSED_PAIRS or (b, a) in CONFUSED_PAIRS else ""
        lines.append(f"  ({a},{b}): overlap={ov:.3f}{tag}")
    lines.extend([
        "",
        f"Connectome accuracy — confused digits: {confused_acc:.1%}",
        f"Connectome accuracy — others: {non_confused_acc:.1%}",
        f"Margin correct={mc:.4f}  error={me:.4f}",
        "",
        "Per-digit connectome:",
    ])
    for d in range(NUM_DIGITS):
        tag = "*" if d in CONFUSED_DIGITS else " "
        lines.append(f"  {tag} {d}: {per_digit[d]:.1%}")
    lines.extend(["", "Occlusion curve (fraction masked -> accuracy):"])
    for f, a in zip(fracs, occ_accs):
        lines.append(f"  mask={f:.0%} -> {a:.1%}")
    lines.extend([
        "",
        f"H6 binding gate: {'OPEN' if binding_open else 'CLOSED'}",
    ])
    if gate_reasons:
        lines.append(f"  reasons: {', '.join(gate_reasons)}")
    else:
        lines.append(
            f"  need confused>={BINDING_GATE['confused_digit_accuracy_min']:.0%} "
            "before cross-modal binding investment"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase I geometry panel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--export", type=str)
    args = parser.parse_args()

    panel = run_geometry_panel(seed=args.seed, n_examples=args.n_examples)
    print(panel.narrative)
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(panel.to_dict(), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
