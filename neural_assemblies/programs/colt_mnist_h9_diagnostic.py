"""
H9 diagnostic — part-merge confused-pair overlap vs ventral baseline.

Hypothesis H9 (``colt_mnist_ventral_theory``): merge(TOP,BOT)→MID→HIGH should
*lower* ``separate``-style prototype overlap on stroke-confusable pairs before
global accuracy rises.

This module compares confused-pair HIGH geometry across streams and reports
whether merge improves separation (not routing fidelity to a ventral teacher).

Run::

    python -m neural_assemblies.programs.colt_mnist_h9_diagnostic --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from neural_assemblies.programs.colt_mnist_geometry_panel import prototype_overlap_matrix
from neural_assemblies.programs.colt_mnist_hierarchical_brain import NUM_DIGITS
from neural_assemblies.programs.colt_mnist_tier_util import CONFUSED_PAIRS


H9_KEY_PAIRS: tuple[tuple[int, int], ...] = ((2, 8), (3, 5), (4, 9), (1, 7))


@dataclass
class StreamGeometry:
    """Prototype overlap geometry for one encoding stream."""

    name: str
    mean_accuracy: float
    mean_confused_overlap: float
    chance_overlap: float
    pair_overlaps: dict[str, float]
    mean_non_confused_overlap: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class H9DiagnosticResult:
    streams: list[StreamGeometry]
    pair_deltas: dict[str, dict[str, float]]
    h9_verdict: Literal["supported", "partial", "falsified", "inconclusive"]
    narrative: str


def _pair_overlap_dict(prototypes: np.ndarray, k: int, pairs: tuple[tuple[int, int], ...]) -> dict[str, float]:
    mat = prototype_overlap_matrix(prototypes, k)
    return {f"{a}_{b}": float(mat[a, b]) for a, b in pairs}


def _mean_overlap(prototypes: np.ndarray, k: int, pairs: tuple[tuple[int, int], ...]) -> float:
    if not pairs:
        return 0.0
    vals = _pair_overlap_dict(prototypes, k, pairs)
    return float(np.mean(list(vals.values())))


def _non_confused_pairs() -> tuple[tuple[int, int], ...]:
    confused = set(CONFUSED_PAIRS) | {(b, a) for a, b in CONFUSED_PAIRS}
    return tuple(
        (i, j)
        for i in range(NUM_DIGITS)
        for j in range(i + 1, NUM_DIGITS)
        if (i, j) not in confused
    )


def stream_geometry(
    name: str,
    prototypes: np.ndarray,
    *,
    k: int,
    mean_accuracy: float,
    n_high: int = 2000,
    extra: dict[str, Any] | None = None,
) -> StreamGeometry:
    chance = k / n_high
    return StreamGeometry(
        name=name,
        mean_accuracy=mean_accuracy,
        mean_confused_overlap=_mean_overlap(prototypes, k, CONFUSED_PAIRS),
        chance_overlap=chance,
        pair_overlaps=_pair_overlap_dict(prototypes, k, H9_KEY_PAIRS),
        mean_non_confused_overlap=_mean_overlap(prototypes, k, _non_confused_pairs()),
        details=extra or {},
    )


def _h9_verdict(
    baseline: StreamGeometry,
    merge: StreamGeometry,
    *,
    min_delta: float = 0.01,
) -> Literal["supported", "partial", "falsified", "inconclusive"]:
    """Compare merge stream to recurrent ventral baseline on confused-pair overlap."""
    mean_delta = merge.mean_confused_overlap - baseline.mean_confused_overlap
    pair_improved = sum(
        1 for key in baseline.pair_overlaps
        if merge.pair_overlaps.get(key, 1.0) < baseline.pair_overlaps[key] - min_delta
    )
    pair_worse = sum(
        1 for key in baseline.pair_overlaps
        if merge.pair_overlaps.get(key, 0.0) > baseline.pair_overlaps[key] + min_delta
    )
    if mean_delta <= -min_delta and pair_improved >= 2:
        return "supported"
    if mean_delta >= min_delta:
        return "falsified"
    if pair_improved > 0 and pair_worse > 0:
        return "partial"
    if abs(mean_delta) < min_delta:
        return "inconclusive"
    return "partial"


def run_h9_diagnostic(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
) -> H9DiagnosticResult:
    from neural_assemblies.programs.colt_mnist_tier_a import run_merge_halves_mnist
    from neural_assemblies.programs.colt_mnist_tier_util import (
        clear_ventral_bundle_cache,
        connectome_predict,
        load_multiscale_spatial_bundle,
        load_recurrent_bundle,
        load_ventral_bundle,
    )
    from neural_assemblies.programs.patch_merge import run_grid_patch_merge_mnist

    clear_ventral_bundle_cache()
    kw = dict(seed=seed, n_examples=n_examples, k=k, use_cache=False)
    train_kw = {k: v for k, v in kw.items() if k != "use_cache"}

    recurrent = load_recurrent_bundle(**kw)
    ventral = load_ventral_bundle(**kw)
    multiscale = load_multiscale_spatial_bundle(**kw)

    def _acc(bundle) -> float:
        from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes
        from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH

        wire_class_from_prototypes(bundle.brain, bundle.prototypes, HIGH, CLASS, k)
        hits = total = 0
        for d in range(NUM_DIGITS):
            for j in range(bundle.n_examples):
                total += 1
                hits += connectome_predict(bundle.high_outputs[d, j], bundle.brain, k=k) == d
        return hits / max(total, 1)

    streams = [
        stream_geometry(
            "recurrent_ventral",
            recurrent.prototypes,
            k=k,
            mean_accuracy=_acc(recurrent),
            extra={"routing_fidelity": None},
        ),
        stream_geometry(
            "feedforward_ventral",
            ventral.prototypes,
            k=k,
            mean_accuracy=_acc(ventral),
        ),
        stream_geometry(
            "spatial_multiscale",
            multiscale.prototypes,
            k=k,
            mean_accuracy=_acc(multiscale),
        ),
    ]

    merge = run_merge_halves_mnist(bundle=ventral, **train_kw)
    merge_proto = merge.extra.get("prototypes")
    if merge_proto is not None:
        streams.append(stream_geometry(
            "merge_halves",
            merge_proto,
            k=k,
            mean_accuracy=merge.mean_accuracy,
            extra={"routing_fidelity": merge.extra.get("routing_fidelity")},
        ))

    grid = run_grid_patch_merge_mnist(bundle=ventral, grid=2, merge_mode="chain", **train_kw)
    grid_proto = grid.extra.get("prototypes")
    if grid_proto is not None:
        streams.append(stream_geometry(
            "grid_merge_2x2",
            grid_proto,
            k=k,
            mean_accuracy=grid.mean_accuracy,
            extra={"routing_fidelity": grid.extra.get("routing_fidelity")},
        ))

    baseline = streams[0]
    merge_stream = next((s for s in streams if s.name == "merge_halves"), None)
    verdict: Literal["supported", "partial", "falsified", "inconclusive"] = "inconclusive"
    pair_deltas: dict[str, dict[str, float]] = {}
    if merge_stream is not None:
        verdict = _h9_verdict(baseline, merge_stream)
        pair_deltas["merge_halves_vs_recurrent"] = {
            key: merge_stream.pair_overlaps.get(key, 0.0) - baseline.pair_overlaps.get(key, 0.0)
            for key in baseline.pair_overlaps
        }
        pair_deltas["merge_halves_vs_recurrent"]["mean_confused"] = (
            merge_stream.mean_confused_overlap - baseline.mean_confused_overlap
        )

    narrative = _format_narrative(streams, pair_deltas, verdict)
    return H9DiagnosticResult(
        streams=streams,
        pair_deltas=pair_deltas,
        h9_verdict=verdict,
        narrative=narrative,
    )


def _format_narrative(
    streams: list[StreamGeometry],
    pair_deltas: dict[str, dict[str, float]],
    verdict: str,
) -> str:
    lines = [
        "H9 diagnostic — confused-pair prototype overlap",
        "=" * 48,
        f"Verdict: {verdict.upper()}",
        "",
        "Streams (accuracy | mean confused overlap | vs chance):",
    ]
    for s in streams:
        rf = s.details.get("routing_fidelity")
        rf_s = f"  routing={rf:.1%}" if rf is not None else ""
        lines.append(
            f"  {s.name}: acc={s.mean_accuracy:.1%}  "
            f"confused_ov={s.mean_confused_overlap:.3f}  "
            f"non_confused_ov={s.mean_non_confused_overlap:.3f}  "
            f"chance={s.chance_overlap:.3f}{rf_s}"
        )
    lines.extend(["", "Key pair overlaps (2_8, 3_5, 4_9, 1_7):"])
    for s in streams:
        pairs = "  ".join(f"{k}={v:.3f}" for k, v in s.pair_overlaps.items())
        lines.append(f"  {s.name}: {pairs}")
    if pair_deltas:
        lines.extend(["", "Merge-halves delta vs recurrent (negative = H9 direction):"])
        for key, deltas in pair_deltas.items():
            parts = "  ".join(f"{k}={v:+.3f}" for k, v in deltas.items())
            lines.append(f"  {key}: {parts}")
    lines.extend([
        "",
        "Interpretation:",
        "  supported  — merge lowers confused overlap on multiple pairs",
        "  falsified  — merge overlap >= ventral (k-cap binding not helping geometry)",
        "  partial    — mixed per-pair effects",
        "  inconclusive — deltas below threshold",
    ])
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="H9 part-merge overlap diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()

    result = run_h9_diagnostic(
        seed=args.seed, n_examples=args.n_examples, k=args.k,
    )
    print(result.narrative)


if __name__ == "__main__":
    main()
