"""
Developmental Stages
====================

Curriculum organized by developmental stage, similar to child language acquisition.

Stages:
1. First Words (12-18 months) - Single words, basic vocabulary
2. Two-Word Stage (18-24 months) - Word combinations
3. Telegraphic (24-30 months) - Simple sentences
4. Complex (30-36 months) - Questions, pronouns
5. Fluent (36+ months) - Full grammar, cognitive verbs

Each stage builds on previous stages.
"""

from typing import List
from dataclasses import dataclass

from ...params import GroundedSentence
from ..generators import SentenceGenerator


@dataclass
class Stage:
    """A developmental stage in the curriculum."""
    name: str
    description: str
    age_range: str
    patterns: List[str]  # Generator method names (without 'generate_')
    repetitions: int = 10
    
    def generate(self, generator: SentenceGenerator) -> List[GroundedSentence]:
        """Generate curriculum for this stage."""
        sentences = []
        
        for pattern in self.patterns:
            method_name = f"generate_{pattern}"
            if hasattr(generator, method_name):
                method = getattr(generator, method_name)
                sentences.extend(method(self.repetitions))
        
        return sentences


# === Stage Definitions ===

STAGE_1 = Stage(
    name="First Words",
    description="Single words and simple two-word combinations",
    age_range="12-18 months",
    patterns=['adjective_noun'],
    repetitions=20,
)

STAGE_2 = Stage(
    name="Two-Word Stage",
    description="Agent-action and action-object combinations",
    age_range="18-24 months",
    patterns=['intransitive', 'adjective_noun'],
    repetitions=15,
)

STAGE_3 = Stage(
    name="Telegraphic Speech",
    description="Simple SVO sentences",
    age_range="24-30 months",
    patterns=['intransitive', 'transitive', 'copular'],
    repetitions=12,
)

STAGE_4 = Stage(
    name="Complex Sentences",
    description="Questions and pronouns",
    age_range="30-36 months",
    patterns=[
        'intransitive', 'transitive', 'copular',
        'question_who', 'question_what', 'question_yesno',
        'first_person', 'second_person',
    ],
    repetitions=8,
)

STAGE_5 = Stage(
    name="Fluent Speech",
    description="Full grammar including cognitive verbs and self-reference",
    age_range="36+ months",
    patterns=[
        'intransitive', 'transitive', 'copular',
        'question_who', 'question_what', 'question_yesno', 'question_where',
        'first_person', 'second_person',
        'cognitive', 'self_query',
    ],
    repetitions=5,
)

# All stages in order
STAGES = [STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5]


def get_stage_curriculum(stage_index: int = 4, seed: int = 42) -> List[GroundedSentence]:
    """
    Get curriculum up to and including the specified stage.
    
    Args:
        stage_index: 0-4, which stage to train to
        seed: Random seed for reproducibility
    
    Returns:
        List of grounded sentences for training
    """
    generator = SentenceGenerator(seed=seed)
    sentences = []
    
    for i, stage in enumerate(STAGES):
        if i > stage_index:
            break
        sentences.extend(stage.generate(generator))
    
    return sentences


def get_full_curriculum(seed: int = 42) -> List[GroundedSentence]:
    """Get the complete curriculum (all stages)."""
    return get_stage_curriculum(stage_index=len(STAGES) - 1, seed=seed)


def get_stage_by_name(name: str) -> Stage:
    """Get a stage by its name."""
    for stage in STAGES:
        if stage.name.lower() == name.lower():
            return stage
    raise ValueError(f"Unknown stage: {name}")


__all__ = [
    'Stage',
    'STAGES',
    'STAGE_1', 'STAGE_2', 'STAGE_3', 'STAGE_4', 'STAGE_5',
    'get_stage_curriculum',
    'get_full_curriculum',
    'get_stage_by_name',
]
