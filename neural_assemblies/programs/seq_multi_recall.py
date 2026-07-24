"""SEQ25 multi-item ordered recall protocol (SEQ25-E06)."""

from __future__ import annotations

from dataclasses import dataclass

from neural_assemblies.assembly_calculus.ops import overlap, ordered_recall, sequence_memorize


@dataclass
class SeqMultiRecallResult:
    first_item_overlap: float
    second_item_overlap: float
    recalled_length: int
    parameters: dict


def run_seq_multi_recall(
    *,
    seed: int = 42,
    n: int = 1000,
    k: int = 50,
    beta: float = 0.1,
    p: float = 0.05,
    rounds_per_step: int = 5,
    repetitions: int = 3,
    refractory_period: int = 3,
    inhibition_strength: float = 100.0,
    max_steps: int = 10,
) -> SeqMultiRecallResult:
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse")
    stimuli = [f"s{i}" for i in range(3)]
    for s in stimuli:
        brain.add_stimulus(s, k)
    brain.add_area("A", n, k, beta)

    memorized = sequence_memorize(
        brain, stimuli, "A",
        rounds_per_step=rounds_per_step,
        repetitions=repetitions,
    )
    brain.set_lri("A", refractory_period=refractory_period,
                  inhibition_strength=inhibition_strength)
    recalled = ordered_recall(
        brain, "A", stimuli[0], max_steps=max_steps,
        known_assemblies=list(memorized),
    )

    first_ov = overlap(recalled[0], memorized[0]) if recalled else 0.0
    second_ov = (
        overlap(recalled[1], memorized[1])
        if len(recalled) > 1 and len(memorized) > 1 else 0.0
    )
    params = {
        "seed": seed, "n": n, "k": k, "beta": beta, "p": p,
        "rounds_per_step": rounds_per_step, "repetitions": repetitions,
        "refractory_period": refractory_period,
        "inhibition_strength": inhibition_strength,
        "max_steps": max_steps,
    }
    return SeqMultiRecallResult(
        first_item_overlap=float(first_ov),
        second_item_overlap=float(second_ov),
        recalled_length=len(recalled),
        parameters=params,
    )
