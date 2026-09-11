"""
COLT 2022 classification via assembly learning (Dabagia et al.).
"""

from __future__ import annotations

from typing import Dict, Tuple

from neural_assemblies.assembly_calculus.ops import learn_assembly, project
from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.readout import fuzzy_readout


def learn_class_assembly(
    brain,
    stimulus: str,
    area: str,
    **kwargs,
):
    """Learn one class assembly; returns (assembly, epochs, persistence)."""
    return learn_assembly(brain, stimulus, area, **kwargs)


def learn_separable_classes(
    brain,
    class_stimuli: Dict[str, str],
    area: str,
    rounds: int = 8,
) -> Tuple[dict, float]:
    """Train one assembly per class in the same area; return lexicon + min separation.

    Args:
        brain: Brain with stimuli and area configured.
        class_stimuli: ``{class_label: stimulus_name}``.
        area: Shared classification area.

    Returns:
        (lexicon, min_pairwise_overlap) where lower overlap means better separation.
    """
    assemblies = {}
    for label, stim in class_stimuli.items():
        learn_assembly(brain, stim, area, max_epochs=10, project_rounds=rounds)
        assemblies[label] = project(brain, stim, area, rounds=3)

    lex: dict = {label: asm for label, asm in assemblies.items()}
    labels = list(assemblies)
    min_ov = 1.0
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            min_ov = min(min_ov, overlap(assemblies[a], assemblies[b]))
    return lex, min_ov


def classify(lexicon, query_assembly, threshold: float = 0.3) -> str | None:
    """Read out best matching class label from a lexicon."""
    return fuzzy_readout(query_assembly, lexicon, threshold=threshold)
