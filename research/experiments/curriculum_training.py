"""Shared curriculum execution for emergent parser experiments."""

from collections.abc import Sequence


def train_curriculum(parser, stages: Sequence[str]):
    """Train *parser* through the declared stages in order.

    The stage sequence remains a caller-owned protocol parameter; this helper
    only owns construction and execution of the trainer, so studies cannot
    drift in loop behavior while retaining distinct curricula.
    """
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )

    trainer = CurriculumTrainer(parser)
    for stage in stages:
        trainer.train_stage(stage)
    return parser
