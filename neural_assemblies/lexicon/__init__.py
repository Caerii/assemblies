"""
Massive Lexicon System for Assembly Calculus
=============================================

A comprehensive, organized lexicon with:
- 5000+ words across all categories
- Frequency statistics from real corpora
- Curriculum-based learning progression
- Semantic features and relationships

Structure:
- data/: Word lists organized by category
- curriculum/: Learning stages and progressions
- statistics/: Word frequency and co-occurrence data
"""

from .lexicon_manager import LexiconManager, Word, WordCategory
from .statistics import WordStatistics

__all__ = [
    'LexiconManager',
    'Word', 
    'WordCategory',
    'WordStatistics',
]

