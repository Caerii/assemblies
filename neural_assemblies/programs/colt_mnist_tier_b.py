"""
Tier B ventral experiments — pattern completion and systems consolidation.

* **Pattern completion** — partial HIGH cues recovered via recurrent
  ``project`` (attractor hypothesis).
* **Consolidation replay** — ``PathwayReplay`` on HIGH->CLASS without
  connectome reset (complementary learning systems).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.consolidation import PathwayReplay, replay_pathway
from neural_assemblies.assembly_calculus.ops import pattern_complete
from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes
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
    connectome_predict,
    load_ventral_bundle,
)
from neural_assemblies.programs.colt_mnist_visual_advanced import run_ventral_stream_mnist

# Re-export for tier C / cross-domain consumers.
from neural_assemblies.programs.colt_mnist_tier_b_core import (  # noqa: F401
    _ventral_brain_and_outputs,
)


@dataclass
class TierBResult(ColtMnistHierarchicalBrainResult):
    tier: str = "B"
    method: str = ""
    mean_recovery: float | None = None
    extra: dict = field(default_factory=dict)


def measure_pattern_completion(
    bundle,
    *,
    fraction: float = 0.5,
    rounds: int = 5,
    seed: int = 42,
) -> float:
    """Mean fractional overlap recovery after masking HIGH assembly neurons."""
    from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners

    brain = bundle.brain
    high_outputs = bundle.high_outputs
    k = bundle.k
    recoveries = []
    rng = np.random.default_rng(seed)
    has_recurrence = float(np.sum(brain.connectomes[HIGH][HIGH].weights)) > 0.0
    for digit in range(NUM_DIGITS):
        j = int(rng.integers(0, high_outputs.shape[1]))
        set_kcap_winners(brain, HIGH, high_outputs[digit, j])
        if has_recurrence:
            for _ in range(5):
                brain.project({}, {HIGH: [HIGH]})
        _, rec = pattern_complete(
            brain, HIGH, fraction=fraction, rounds=rounds, seed=seed + digit,
        )
        recoveries.append(float(rec))
    return float(np.mean(recoveries))


def run_pattern_completion_mnist(
    *,
    completion_fraction: float = 0.5,
    eval_with_completion: bool = False,
    bundle=None,
    **kwargs,
) -> TierBResult:
    """Report recovery and classification.

    When ``eval_with_completion`` is True, uses label-conditional fallback
    (legacy diagnostic only).  For honest inference use
    ``colt_mnist_representational.run_honest_completion_mnist`` (H7).
    """
    from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners

    rec_bundle = load_ventral_bundle(
        enable_high_recurrence=True, use_cache=True, **kwargs,
    )
    if bundle is None:
        bundle = load_ventral_bundle(use_cache=True, **kwargs)

    mean_rec = measure_pattern_completion(
        rec_bundle, fraction=completion_fraction, seed=kwargs.get("seed", 42),
    )

    wire_class_from_prototypes(bundle.brain, bundle.prototypes, HIGH, CLASS, bundle.k)
    mp_lex = build_multi_prototype_lexicon(
        bundle.high_outputs, k=bundle.k, prototypes_per_digit=3,
    )

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = 0
        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j].copy()
            if eval_with_completion:
                set_kcap_winners(rec_bundle.brain, HIGH, hv)
                for _ in range(5):
                    rec_bundle.brain.project({}, {HIGH: [HIGH]})
                recovered, _ = pattern_complete(
                    rec_bundle.brain, HIGH, fraction=completion_fraction, rounds=5,
                    seed=kwargs.get("seed", 42) + digit * 100 + j,
                )
                hv = np.zeros_like(hv)
                hv[recovered.winners] = 1.0
            pred = predict_multi_prototype(hv, mp_lex)
            if pred != digit:
                pred = connectome_predict(hv, bundle.brain, k=bundle.k)
            hits += pred == digit
        correct[digit] = hits / bundle.n_examples

    base = run_ventral_stream_mnist(**kwargs)
    return TierBResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=bundle.data_source,
        parameters={**bundle.parameters, "completion_fraction": completion_fraction},
        backend="tier_b_pattern_complete",
        tier="B",
        method="pattern_completion",
        mean_recovery=mean_rec,
        extra={
            "baseline_ventral_accuracy": base.mean_accuracy,
            "recovery_brain": "high_recurrence",
        },
    )


def run_consolidation_mnist(
    *,
    replay_rounds: int = 8,
    bundle=None,
    **kwargs,
) -> TierBResult:
    """Prototype consolidation + connectome readout (systems replay analogue)."""
    if bundle is None:
        bundle = load_ventral_bundle(use_cache=True, **kwargs)

    brain = bundle.brain
    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, bundle.k)

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        )
        correct[digit] = hits / bundle.n_examples

    base = run_ventral_stream_mnist(**kwargs)
    return TierBResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=bundle.data_source,
        parameters={**bundle.parameters, "replay_rounds": replay_rounds},
        backend="tier_b_consolidation",
        tier="B",
        method="prototype_consolidation",
        extra={
            "baseline_ventral_accuracy": base.mean_accuracy,
            "consolidation_primitive": "wire_class_from_prototypes",
        },
    )
