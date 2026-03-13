"""
Word Statistics
===============

Frequency and co-occurrence statistics for vocabulary learning.
Based on corpus linguistics research.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math
import json
import os

from .lexicon_manager import WordCategory


@dataclass
class WordStats:
    """Statistics for a single word"""
    lemma: str
    frequency: float = 0.0          # Log10 frequency per million
    rank: int = 0                   # Frequency rank
    dispersion: float = 0.0         # How evenly distributed
    age_of_acquisition: float = 0.0 # Typical age learned
    concreteness: float = 0.0       # 1-5 scale
    imageability: float = 0.0       # 1-5 scale


class WordStatistics:
    """
    Manages word frequency and co-occurrence statistics.
    
    Based on:
    - SUBTLEX-US (film subtitle frequencies)
    - Age of Acquisition norms
    - Concreteness ratings
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        
        self.word_stats: Dict[str, WordStats] = {}
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.trigram_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.category_transitions: Dict[Tuple[WordCategory, WordCategory], int] = defaultdict(int)
        
        # Total counts for normalization
        self.total_words = 0
        self.total_bigrams = 0
        self.total_trigrams = 0
    
    def add_word_stats(self, stats: WordStats):
        """Add statistics for a word"""
        self.word_stats[stats.lemma] = stats
    
    def get_frequency(self, word: str) -> float:
        """Get word frequency (log scale)"""
        stats = self.word_stats.get(word.lower())
        return stats.frequency if stats else 0.0
    
    def get_aoa(self, word: str) -> float:
        """Get age of acquisition"""
        stats = self.word_stats.get(word.lower())
        return stats.age_of_acquisition if stats else 18.0  # Default to adult
    
    def record_bigram(self, word1: str, word2: str):
        """Record a word bigram"""
        self.bigram_counts[(word1.lower(), word2.lower())] += 1
        self.total_bigrams += 1
    
    def record_trigram(self, word1: str, word2: str, word3: str):
        """Record a word trigram"""
        self.trigram_counts[(word1.lower(), word2.lower(), word3.lower())] += 1
        self.total_trigrams += 1
    
    def record_category_transition(self, cat1: WordCategory, cat2: WordCategory):
        """Record a category transition"""
        self.category_transitions[(cat1, cat2)] += 1
    
    def get_bigram_probability(self, word1: str, word2: str) -> float:
        """Get P(word2 | word1)"""
        bigram_count = self.bigram_counts.get((word1.lower(), word2.lower()), 0)
        word1_count = sum(c for (w1, _), c in self.bigram_counts.items() if w1 == word1.lower())
        
        if word1_count == 0:
            return 0.0
        return bigram_count / word1_count
    
    def get_category_transition_probability(self, cat1: WordCategory, cat2: WordCategory) -> float:
        """Get P(cat2 | cat1)"""
        transition_count = self.category_transitions.get((cat1, cat2), 0)
        cat1_count = sum(c for (c1, _), c in self.category_transitions.items() if c1 == cat1)
        
        if cat1_count == 0:
            return 0.0
        return transition_count / cat1_count
    
    def get_most_likely_next(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most likely next words"""
        word = word.lower()
        candidates = [(w2, c) for (w1, w2), c in self.bigram_counts.items() if w1 == word]
        candidates.sort(key=lambda x: -x[1])
        
        total = sum(c for _, c in candidates)
        if total == 0:
            return []
        
        return [(w, c/total) for w, c in candidates[:top_k]]
    
    def get_category_distribution(self) -> Dict[WordCategory, float]:
        """Get distribution of categories"""
        total = sum(self.category_transitions.values())
        if total == 0:
            return {}
        
        cat_counts = defaultdict(int)
        for (cat1, cat2), count in self.category_transitions.items():
            cat_counts[cat1] += count
            cat_counts[cat2] += count
        
        return {cat: count / (2 * total) for cat, count in cat_counts.items()}
    
    def compute_entropy(self, word: str) -> float:
        """Compute entropy of next-word distribution"""
        probs = [p for _, p in self.get_most_likely_next(word, top_k=100)]
        if not probs:
            return 0.0
        
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def save(self, filename: str = 'statistics.json'):
        """Save statistics to file"""
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            'word_stats': {k: {
                'lemma': v.lemma,
                'frequency': v.frequency,
                'rank': v.rank,
                'dispersion': v.dispersion,
                'age_of_acquisition': v.age_of_acquisition,
                'concreteness': v.concreteness,
                'imageability': v.imageability,
            } for k, v in self.word_stats.items()},
            'bigrams': {f"{w1}|{w2}": c for (w1, w2), c in self.bigram_counts.items()},
            'category_transitions': {
                f"{c1.name}|{c2.name}": c 
                for (c1, c2), c in self.category_transitions.items()
            },
            'totals': {
                'words': self.total_words,
                'bigrams': self.total_bigrams,
                'trigrams': self.total_trigrams,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str = 'statistics.json'):
        """Load statistics from file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for lemma, stats in data.get('word_stats', {}).items():
            self.word_stats[lemma] = WordStats(**stats)
        
        for key, count in data.get('bigrams', {}).items():
            w1, w2 = key.split('|')
            self.bigram_counts[(w1, w2)] = count
        
        for key, count in data.get('category_transitions', {}).items():
            c1, c2 = key.split('|')
            self.category_transitions[(WordCategory[c1], WordCategory[c2])] = count
        
        totals = data.get('totals', {})
        self.total_words = totals.get('words', 0)
        self.total_bigrams = totals.get('bigrams', 0)
        self.total_trigrams = totals.get('trigrams', 0)

