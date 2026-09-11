"""
Causal profiling for cross-domain vision–language fusion.

Measures per-route accuracy, ablations, confusion structure, readout margins,
and tie-break utility to identify information bottlenecks gating performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs.colt_mnist_advanced_util import read_class_connectome_scores
from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners
from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_lri_readout import connectome_lri_predict
from neural_assemblies.programs.colt_mnist_tier_a import predict_multi_prototype
from neural_assemblies.programs.colt_mnist_tier_util import CONFUSED_DIGITS, CONFUSED_PAIRS, connectome_predict
from neural_assemblies.programs.cross_domain_assemblies import (
    SEMANTIC,
    TrainedCrossDomainHub,
    _anchor_readout,
    _contrastive_matrix,
    _fuse_visual_prediction,
    _train_cross_domain_hub,
)


@dataclass
class RouteProfile:
    """Single readout route statistics."""

    name: str
    accuracy: float
    per_digit: np.ndarray
    confusion: np.ndarray
    mean_margin_correct: float
    mean_margin_error: float


@dataclass
class AblationProfile:
    """Accuracy when one fusion component is removed."""

    name: str
    accuracy: float
    delta_vs_fused: float


@dataclass
class TiebreakProfile:
    """When connectome and multi-prototype disagree, does semantic help?"""

    n_disagreements: int
    semantic_agrees_with_winner: int
    fusion_picks_semantic: int
    fusion_correct_when_semantic_picked: int
    fusion_correct_when_connectome_picked: int


@dataclass
class BottleneckFinding:
    """Ranked causal bottleneck hypothesis."""

    id: str
    severity: str  # critical | major | minor
    evidence: str
    recommendation: str


@dataclass
class CrossDomainProfile:
    """Full causal profile of cross-domain fusion."""

    fused_accuracy: float
    routes: list[RouteProfile]
    ablations: list[AblationProfile]
    tiebreak: TiebreakProfile
    confused_digit_accuracy: float
    non_confused_digit_accuracy: float
    contrastive_margin: float
    contrastive_diagonal: float
    semantic_anchor_overlap_true: float
    semantic_anchor_overlap_max: float
    bottlenecks: list[BottleneckFinding]
    narrative: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for r in d["routes"]:
            r["per_digit"] = r["per_digit"].tolist()
            r["confusion"] = r["confusion"].tolist()
        return d


def _confusion_matrix(
    predict_fn: Callable[[int, int, np.ndarray], int],
    bundle,
) -> tuple[np.ndarray, np.ndarray]:
    """Build confusion matrix and per-digit accuracy for predict(true_digit, j, hv)->pred."""
    conf = np.zeros((NUM_DIGITS, NUM_DIGITS), dtype=np.int64)
    per_digit = np.zeros(NUM_DIGITS, dtype=np.float64)
    for digit in range(NUM_DIGITS):
        hits = 0
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            pred = predict_fn(digit, j, hv)
            conf[digit, pred] += 1
            hits += pred == digit
        per_digit[digit] = hits / bundle.n_examples
    return conf, per_digit


def _margin_stats(
    score_fn: Callable[[np.ndarray], np.ndarray],
    bundle,
) -> tuple[float, float]:
    """Mean top1-top2 margin on correct vs incorrect trials."""
    margins_correct: list[float] = []
    margins_error: list[float] = []
    for digit in range(NUM_DIGITS):
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            scores = score_fn(hv)
            order = np.argsort(scores)[::-1]
            margin = float(scores[order[0]] - scores[order[1]]) if scores.size > 1 else 0.0
            if order[0] == digit:
                margins_correct.append(margin)
            else:
                margins_error.append(margin)
    mc = float(np.mean(margins_correct)) if margins_correct else 0.0
    me = float(np.mean(margins_error)) if margins_error else 0.0
    return mc, me


def _semantic_conn_scores(hv: np.ndarray, brain, k: int) -> np.ndarray:
    return read_class_connectome_scores(hv, brain, HIGH, SEMANTIC, k, NUM_DIGITS)


def _class_conn_scores(hv: np.ndarray, brain, k: int) -> np.ndarray:
    return read_class_connectome_scores(hv, brain, HIGH, CLASS, k, NUM_DIGITS)


def _anchor_score_vector(asm, anchors: dict, k: int) -> np.ndarray:
    return np.array([overlap(asm, anchors[str(d)]) for d in range(NUM_DIGITS)])


def profile_cross_domain_hub(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    hub: TrainedCrossDomainHub | None = None,
    **kwargs,
) -> CrossDomainProfile:
    """Train (or reuse) hub and run full causal profiling."""
    if hub is None:
        hub = _train_cross_domain_hub(seed=seed, n_examples=n_examples, k=k, **kwargs)

    brain = hub.brain
    bundle = hub.bundle
    seed = hub.seed
    n_lang = hub.n_lang

    def _conn(d, j, hv):
        return connectome_predict(hv, brain, k=bundle.k, src=HIGH, dst=CLASS)

    def _mp(d, j, hv):
        return predict_multi_prototype(hv, hub.mp_lex)

    def _sem_conn(d, j, hv):
        return int(np.argmax(_semantic_conn_scores(hv, brain, bundle.k)))

    def _sem_anchor(d, j, hv):
        set_kcap_winners(brain, HIGH, hv)
        brain.project({}, {HIGH: [SEMANTIC]})
        return _anchor_readout(_snap(brain, SEMANTIC), hub.semantic_lex, bundle.k)

    def _sem_slot(d, j, hv):
        return connectome_predict(hv, brain, k=bundle.k, src=HIGH, dst=SEMANTIC)

    def _conn_lri(d, j, hv):
        pred, _, _ = connectome_lri_predict(hv, brain, hub.mp_lex, k=bundle.k)
        return pred

    def _fused(d, j, hv):
        pred, _ = _fuse_visual_prediction(hv, brain, bundle, hub.semantic_lex, hub.mp_lex)
        return pred

    def _fused_no_sem_tie(d, j, hv):
        pred, _ = _fuse_visual_prediction(
            hv, brain, bundle, hub.semantic_lex, hub.mp_lex, use_semantic_tiebreak=False,
        )
        return pred

    def _fused_conn_only(d, j, hv):
        return _conn(d, j, hv)

    def _fused_mp_only(d, j, hv):
        return _mp(d, j, hv)

    route_defs = [
        ("fused", _fused),
        ("connectome_class", _conn),
        ("connectome_lri", _conn_lri),
        ("multi_prototype", _mp),
        ("semantic_connectome", _sem_conn),
        ("semantic_anchor_project", _sem_anchor),
        ("semantic_slot_readout", _sem_slot),
    ]
    routes: list[RouteProfile] = []

    for name, fn in route_defs:
        conf, per_digit = _confusion_matrix(fn, bundle)
        acc = float(per_digit.mean())
        if name == "connectome_class":
            mc, me = _margin_stats(lambda hv: _class_conn_scores(hv, brain, bundle.k), bundle)
        elif name == "semantic_connectome":
            mc, me = _margin_stats(lambda hv: _semantic_conn_scores(hv, brain, bundle.k), bundle)
        elif name == "semantic_anchor_project":
            def anchor_scores(hv):
                set_kcap_winners(brain, HIGH, hv)
                brain.project({}, {HIGH: [SEMANTIC]})
                return _anchor_score_vector(_snap(brain, SEMANTIC), hub.semantic_lex, bundle.k)
            mc, me = _margin_stats(anchor_scores, bundle)
        else:
            mc, me = 0.0, 0.0
        routes.append(RouteProfile(name, acc, per_digit, conf, mc, me))

    fused_acc = routes[0].accuracy
    ablations = [
        AblationProfile(n, float(a), float(a - fused_acc))
        for n, a in [
            ("connectome_only", _confusion_matrix(_fused_conn_only, bundle)[1].mean()),
            ("multi_proto_only", _confusion_matrix(_fused_mp_only, bundle)[1].mean()),
            ("no_semantic_tiebreak", _confusion_matrix(_fused_no_sem_tie, bundle)[1].mean()),
        ]
    ]

    # Tie-break analysis
    tb = TiebreakProfile(0, 0, 0, 0, 0)
    for digit in range(NUM_DIGITS):
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            conn = _conn(digit, j, hv)
            mp = _mp(digit, j, hv)
            if conn == mp:
                continue
            tb.n_disagreements += 1
            sem = _sem_anchor(digit, j, hv)
            if sem in (conn, mp):
                tb.semantic_agrees_with_winner += 1
            fused, _ = _fuse_visual_prediction(hv, brain, bundle, hub.semantic_lex, hub.mp_lex)
            if fused == sem:
                tb.fusion_picks_semantic += 1
                tb.fusion_correct_when_semantic_picked += fused == digit
            if fused == conn:
                tb.fusion_correct_when_connectome_picked += fused == digit

    # Confused vs non-confused digits
    fused_per_digit = routes[0].per_digit
    confused_acc = float(np.mean([fused_per_digit[d] for d in CONFUSED_DIGITS]))
    non_confused_acc = float(np.mean([fused_per_digit[d] for d in range(NUM_DIGITS) if d not in CONFUSED_DIGITS]))

    # Semantic anchor overlap diagnostic
    true_overlaps: list[float] = []
    max_overlaps: list[float] = []
    for digit in range(NUM_DIGITS):
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            set_kcap_winners(brain, HIGH, hv)
            brain.project({}, {HIGH: [SEMANTIC]})
            asm = _snap(brain, SEMANTIC)
            scores = _anchor_score_vector(asm, hub.semantic_lex, bundle.k)
            true_overlaps.append(scores[digit])
            max_overlaps.append(float(np.max(scores)))

    cm = _contrastive_matrix(brain, bundle, hub.semantic_lex, seed=seed, n_lang=n_lang)
    diag = float(np.trace(cm) / NUM_DIGITS)
    offdiag = float((cm.sum() - np.trace(cm)) / (NUM_DIGITS * (NUM_DIGITS - 1)))
    margin = diag - offdiag

    bottlenecks = _infer_bottlenecks(
        routes=routes,
        ablations=ablations,
        tiebreak=tb,
        confused_acc=confused_acc,
        non_confused_acc=non_confused_acc,
        contrastive_margin=margin,
        anchor_overlap_true=float(np.mean(true_overlaps)),
        anchor_overlap_max=float(np.mean(max_overlaps)),
    )
    narrative = _build_narrative(routes, ablations, tb, bottlenecks, confused_acc, non_confused_acc, margin)

    return CrossDomainProfile(
        fused_accuracy=fused_acc,
        routes=routes,
        ablations=ablations,
        tiebreak=tb,
        confused_digit_accuracy=confused_acc,
        non_confused_digit_accuracy=non_confused_acc,
        contrastive_margin=margin,
        contrastive_diagonal=diag,
        semantic_anchor_overlap_true=float(np.mean(true_overlaps)),
        semantic_anchor_overlap_max=float(np.mean(max_overlaps)),
        bottlenecks=bottlenecks,
        narrative=narrative,
        parameters=hub.parameters,
    )


def _infer_bottlenecks(
    *,
    routes: list[RouteProfile],
    ablations: list[AblationProfile],
    tiebreak: TiebreakProfile,
    confused_acc: float,
    non_confused_acc: float,
    contrastive_margin: float,
    anchor_overlap_true: float,
    anchor_overlap_max: float,
) -> list[BottleneckFinding]:
    """Rank causal bottlenecks from profile metrics."""
    by_name = {r.name: r for r in routes}
    findings: list[BottleneckFinding] = []

    conn = by_name["connectome_class"]
    sem = by_name["semantic_anchor_project"]
    fused = by_name["fused"]

    if conn.accuracy >= fused.accuracy - 0.005:
        findings.append(BottleneckFinding(
            id="B1_fusion_ceiling_at_connectome",
            severity="critical",
            evidence=(
                f"Fused {fused.accuracy:.1%} ~ connectome {conn.accuracy:.1%}; "
                "semantic hub does not raise visual ceiling."
            ),
            recommendation="Strengthen HIGH->SEMANTIC binding before fusion; wire semantic connectome.",
        ))

    if sem.accuracy < 0.30:
        findings.append(BottleneckFinding(
            id="B2_high_semantic_projection",
            severity="critical",
            evidence=(
                f"Vision->SEMANTIC anchor readout {sem.accuracy:.1%}; "
                f"true-anchor overlap {anchor_overlap_true:.4f} vs max {anchor_overlap_max:.4f}."
            ),
            recommendation="Wire HIGH->SEMANTIC from prototypes/anchors; confused-pair contrastive curriculum.",
        ))

    if confused_acc + 0.08 < non_confused_acc:
        findings.append(BottleneckFinding(
            id="B3_stroke_confusion_pairs",
            severity="major",
            evidence=(
                f"Confused digits {confused_acc:.1%} vs others {non_confused_acc:.1%} "
                f"(pairs {list(CONFUSED_PAIRS)})."
            ),
            recommendation="Target associate/separate curriculum on confused pairs; LRI cascade at readout.",
        ))

    mp = by_name["multi_prototype"]
    if mp.accuracy < conn.accuracy - 0.10:
        findings.append(BottleneckFinding(
            id="B4_weak_view_manifold",
            severity="major",
            evidence=f"Multi-prototype {mp.accuracy:.1%} << connectome {conn.accuracy:.1%}.",
            recommendation="Increase prototypes_per_digit or merge-halves part structure.",
        ))

    if contrastive_margin < 0.05:
        findings.append(BottleneckFinding(
            id="B5_flat_cross_modal_similarity",
            severity="major",
            evidence=f"Contrastive margin (diag-offdiag overlap) = {contrastive_margin:.4f}.",
            recommendation="More contrastive epochs; hard negatives on confused pairs; anchor reset.",
        ))

    no_sem = next(a for a in ablations if a.name == "no_semantic_tiebreak")
    if abs(no_sem.delta_vs_fused) < 0.005 and tiebreak.n_disagreements > 0:
        findings.append(BottleneckFinding(
            id="B6_tiebreak_inactive",
            severity="minor",
            evidence=(
                f"{tiebreak.n_disagreements} conn!=mp disagreements; "
                f"removing semantic tie-break delta={no_sem.delta_vs_fused:+.1%}."
            ),
            recommendation="Fusion logic is not the bottleneck; fix representation routes first.",
        ))

    if conn.mean_margin_error < conn.mean_margin_correct * 0.5 and conn.mean_margin_correct > 0:
        findings.append(BottleneckFinding(
            id="B7_low_error_margin",
            severity="major",
            evidence=(
                f"Connectome margin correct={conn.mean_margin_correct:.3f} "
                f"error={conn.mean_margin_error:.3f}."
            ),
            recommendation="Errors are low-confidence confusions; LRI cascade or hard-negative binding.",
        ))

    severity_order = {"critical": 0, "major": 1, "minor": 2}
    findings.sort(key=lambda f: severity_order.get(f.severity, 3))
    return findings


def _build_narrative(
    routes: list[RouteProfile],
    ablations: list[AblationProfile],
    tiebreak: TiebreakProfile,
    bottlenecks: list[BottleneckFinding],
    confused_acc: float,
    non_confused_acc: float,
    margin: float,
) -> str:
    lines = ["Cross-domain causal profile", "=" * 28, "", "Route accuracy:"]
    for r in sorted(routes, key=lambda x: x.accuracy, reverse=True):
        lines.append(f"  {r.name:28s} {r.accuracy:6.1%}")
    lines.extend(["", "Ablations (delta vs fused):"])
    for a in ablations:
        lines.append(f"  {a.name:28s} {a.accuracy:6.1%} ({a.delta_vs_fused:+.1%})")
    lines.extend([
        "",
        f"Confused digits: {confused_acc:.1%}  |  Others: {non_confused_acc:.1%}",
        f"Contrastive margin: {margin:.4f}",
        f"Tie-break disagreements (conn!=mp): {tiebreak.n_disagreements}",
        "",
        "Ranked bottlenecks:",
    ])
    for b in bottlenecks:
        lines.append(f"  [{b.severity.upper()}] {b.id}")
        lines.append(f"    {b.evidence}")
        lines.append(f"    -> {b.recommendation}")
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Profile cross-domain fusion bottlenecks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--export", type=str, help="Write JSON profile")
    args = parser.parse_args()

    profile = profile_cross_domain_hub(seed=args.seed, n_examples=args.n_examples)
    print(profile.narrative)
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
            f.write("\n")
        print(f"Wrote {args.export}")


if __name__ == "__main__":
    main()
