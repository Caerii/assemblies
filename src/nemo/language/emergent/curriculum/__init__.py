"""
NEMO Curriculum Module
======================

Modular curriculum for emergent language learning.

Structure:
    curriculum/
    ├── __init__.py           # This file - main exports
    ├── vocabulary/           # Word definitions with grounding
    │   ├── types.py          # WordInfo dataclass
    │   ├── grounding.py      # Domain-to-grounding mapping
    │   └── loader.py         # Vocabulary loading
    ├── generators/           # Sentence pattern generators
    │   ├── base.py           # BaseGenerator class
    │   ├── declarative.py    # Statement generators
    │   ├── interrogative.py  # Question generators
    │   └── self_referential.py  # I/you generators
    ├── stages/               # Developmental stages
    │   └── __init__.py       # Stage definitions
    └── dialogue/             # Dialogue patterns
        └── __init__.py       # Q-A pair generators

Key Principles:
1. Categories EMERGE from grounding patterns
2. Vocabulary integrates with lexicon/data/
3. Stages mirror child language acquisition
4. Dialogue patterns teach Q-A mappings
"""

from typing import List, Optional

from .vocabulary import (
    WordInfo,
    get_vocabulary,
    get_word_info,
    create_grounding_from_domains,
)
from .generators import (
    SentenceGenerator,
    BaseGenerator,
)
from .stages import (
    Stage,
    STAGES,
    get_stage_curriculum,
    get_full_curriculum,
)
from .dialogue import (
    DialoguePair,
    DialoguePatternGenerator,
    get_dialogue_curriculum,
)
from ..params import GroundedSentence


def get_training_curriculum(
    include_dialogue: bool = True,
    stage: int = 4,
    seed: int = 42
) -> List[GroundedSentence]:
    """
    Get complete training curriculum.
    
    Args:
        include_dialogue: Whether to include dialogue patterns
        stage: Maximum developmental stage (0-4)
        seed: Random seed
    
    Returns:
        List of grounded sentences for training
    """
    sentences = []
    
    # Get stage-based curriculum
    sentences.extend(get_stage_curriculum(stage_index=stage, seed=seed))
    
    # Add dialogue patterns
    if include_dialogue:
        sentences.extend(get_dialogue_curriculum(seed=seed))
    
    return sentences


def get_curriculum_stats(sentences: List[GroundedSentence]) -> dict:
    """Get statistics about a curriculum."""
    stats = {
        'total_sentences': len(sentences),
        'declarative': 0,
        'interrogative': 0,
        'unique_words': set(),
        'avg_length': 0,
    }
    
    total_words = 0
    for s in sentences:
        if s.mood == 'declarative':
            stats['declarative'] += 1
        elif s.mood == 'interrogative':
            stats['interrogative'] += 1
        
        stats['unique_words'].update(s.words)
        total_words += len(s.words)
    
    stats['unique_words'] = len(stats['unique_words'])
    stats['avg_length'] = total_words / len(sentences) if sentences else 0
    
    return stats


__all__ = [
    # Vocabulary
    'WordInfo',
    'get_vocabulary',
    'get_word_info',
    'create_grounding_from_domains',
    
    # Generators
    'SentenceGenerator',
    'BaseGenerator',
    
    # Stages
    'Stage',
    'STAGES',
    'get_stage_curriculum',
    'get_full_curriculum',
    
    # Dialogue
    'DialoguePair',
    'DialoguePatternGenerator',
    'get_dialogue_curriculum',
    
    # Main API
    'get_training_curriculum',
    'get_curriculum_stats',
]
