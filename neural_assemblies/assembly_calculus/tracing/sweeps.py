"""Small deterministic sweeps used by teaching notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from neural_assemblies.assembly_calculus.assembly import chance_overlap, overlap
from neural_assemblies.assembly_calculus.ops import sequence_memorize
from neural_assemblies.core.brain import Brain

from .operations import ordered_recall_trace, project_trace


@dataclass(frozen=True)
class ProjectionSweepConfig:
    """One projection-stability parameter setting."""

    n: int
    k: int
    beta: float
    rounds: int
    seed: int = 42
    p: float = 0.05
    engine: str = "numpy_sparse"


@dataclass(frozen=True)
class RecallSweepConfig:
    """One sequence-recall parameter setting."""

    refractory_period: int
    inhibition_strength: float
    beta_boost: float | None = None
    phase_b_ratio: float | None = 0.5
    n: int = 4_000
    k: int = 50
    beta: float = 0.10
    rounds_per_step: int = 8
    repetitions: int = 2
    seed: int = 42
    p: float = 0.05
    engine: str = "numpy_sparse"
    sequence: tuple[str, ...] = ("item_0", "item_1", "item_2", "item_3")
    match_threshold: float = 0.3


def projection_sweep(configs: Sequence[ProjectionSweepConfig]) -> list[dict[str, object]]:
    """Run independent projection traces and return compact diagnostic rows."""
    rows: list[dict[str, object]] = []
    for config in configs:
        brain = Brain(p=config.p, save_winners=True, seed=config.seed, engine=config.engine)
        brain.add_stimulus("stim", config.k)
        brain.add_area("A", config.n, config.k, config.beta)
        trace = project_trace(brain, "stim", "A", rounds=config.rounds)
        summary = trace.summary()
        rows.append(
            {
                "n": config.n,
                "k": config.k,
                "density": config.k / config.n,
                "beta": config.beta,
                "rounds": config.rounds,
                "seed": config.seed,
                "chance_overlap": chance_overlap(config.k, config.n),
                "final_overlap_prev": summary["final_overlap_prev"],
                "stabilized_at": summary["stabilized_at"],
                "new_winners_final_round": summary["new_winners_final_round"],
            }
        )
    return rows


def lri_recall_sweep(configs: Sequence[RecallSweepConfig]) -> list[dict[str, object]]:
    """Run independent sequence-memory/LRI recalls and summarize trace quality."""
    rows: list[dict[str, object]] = []
    for config in configs:
        brain = Brain(p=config.p, save_winners=True, seed=config.seed, engine=config.engine)
        for stimulus in config.sequence:
            brain.add_stimulus(stimulus, config.k)
        brain.add_area("SEQ", config.n, config.k, config.beta)

        memorized = sequence_memorize(
            brain,
            list(config.sequence),
            "SEQ",
            rounds_per_step=config.rounds_per_step,
            repetitions=config.repetitions,
            phase_b_ratio=config.phase_b_ratio,
            beta_boost=config.beta_boost,
        )
        brain.set_lri("SEQ", config.refractory_period, config.inhibition_strength)
        trace = ordered_recall_trace(
            brain,
            "SEQ",
            config.sequence[0],
            max_steps=len(config.sequence) + 4,
            known_assemblies=list(memorized),
        )

        best_overlaps = [
            max(overlap(step.assembly, known) for known in memorized)
            for step in trace
        ]
        ordered_matches = 0
        for step, known in zip(trace, memorized):
            if overlap(step.assembly, known) >= config.match_threshold:
                ordered_matches += 1
            else:
                break

        rows.append(
            {
                "refractory_period": config.refractory_period,
                "inhibition_strength": config.inhibition_strength,
                "beta_boost": config.beta_boost,
                "phase_b_ratio": config.phase_b_ratio,
                "recall_steps": len(trace),
                "ordered_matches": ordered_matches,
                "mean_best_overlap": mean(best_overlaps),
                "final_best_overlap": best_overlaps[-1],
                "chance_overlap": chance_overlap(config.k, config.n),
                "seed": config.seed,
            }
        )
    return rows
