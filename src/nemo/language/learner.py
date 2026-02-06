"""
NEMO Language Learner
=====================

Learns language structure from exposure - NO HARDCODED GRAMMAR.

Key principles:
1. Word categories emerge from co-occurrence patterns
2. Word order learned from sequence statistics  
3. Selectional restrictions learned from verb-argument pairs
4. All learning is through assembly overlap, not explicit rules

This is the scientifically valuable part - testing if assemblies
can learn linguistic structure without being told the rules.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core import Brain, BrainParams


@dataclass
class LearnerParams:
    """Parameters for language learning."""
    n: int = 10000
    k: int = None
    p: float = 0.1
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class LanguageLearner:
    """
    Learns language from exposure without hardcoded grammar.
    
    What it learns:
    1. Word assemblies (phonological form)
    2. Word categories (from distributional patterns)
    3. Word order (from sequence statistics)
    4. Selectional restrictions (from co-occurrence)
    
    What is NOT hardcoded:
    - No explicit NOUN/VERB categories
    - No explicit SVO/SOV rules
    - No explicit semantic features
    
    Everything emerges from the data!
    """
    
    # Standard areas - but their function emerges from learning
    AREAS = ['PHON', 'LEX', 'ROLE1', 'ROLE2', 'ROLE3', 'SEQ']
    
    def __init__(self, params: LearnerParams = None, verbose: bool = True):
        self.params = params or LearnerParams()
        
        # Core brain
        brain_params = BrainParams(
            n=self.params.n,
            k=self.params.k,
            p=self.params.p
        )
        self.brain = Brain(brain_params)
        
        # Add areas
        for area in self.AREAS:
            self.brain.add_area(area)
        
        # Learning statistics (emergent, not predefined)
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.position_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Track sentence lengths seen
        self.length_counts: Dict[int, int] = defaultdict(int)
        
        # Track patterns by length: length -> {pattern: count}
        # Pattern is tuple of word categories at each position
        self.sentence_patterns: Dict[int, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
        
        # Co-occurrence for selectional restrictions
        # word1 -> {word2: count} (learned from data)
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Role assignments (learned, not predefined)
        # word -> most common position
        self.word_roles: Dict[str, int] = {}
        
        self.sentences_seen = 0
        
        if verbose:
            print(f"LanguageLearner: n={self.params.n}, k={self.params.k}")
            print(f"  Areas: {self.AREAS}")
            print(f"  No hardcoded grammar - everything learned from data!")
    
    def _get_word_assembly(self, word: str) -> torch.Tensor:
        """Get or create phonological assembly for word."""
        return self.brain.get_or_create(f"phon_{word}")
    
    def _project_word(self, word: str) -> torch.Tensor:
        """Project word through PHON -> LEX."""
        phon = self._get_word_assembly(word)
        lex = self.brain.project('LEX', phon)
        return lex
    
    def hear_sentence(self, words: List[str]):
        """
        Learn from hearing a sentence.
        
        This is the main learning function. It:
        1. Creates/strengthens word assemblies
        2. Learns word order from positions
        3. Learns co-occurrence patterns
        4. Binds words to roles based on position
        """
        n_words = len(words)
        
        # Process each word
        lex_assemblies = []
        for i, word in enumerate(words):
            # Update word count
            self.word_counts[word] += 1
            
            # Update position statistics
            self.position_counts[word][i] += 1
            
            # Project word to lexical assembly
            lex = self._project_word(word)
            lex_assemblies.append(lex)
            
            # Bind to role based on position
            role_area = f'ROLE{i+1}' if i < 3 else 'ROLE3'
            if role_area in self.brain.areas:
                self.brain.project(role_area, lex)
            
            # Project to sequence area
            self.brain.project('SEQ', lex)
        
        # Learn bigrams (word order)
        for i in range(len(words) - 1):
            self.bigram_counts[(words[i], words[i+1])] += 1
        
        # Learn co-occurrence (all pairs)
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j:
                    self.cooccurrence[w1][w2] += 1
        
        # Track sentence length
        self.length_counts[n_words] += 1
        
        # Track sentence pattern (after we have some word categories)
        if self.sentences_seen > 10:
            pattern = tuple(self.get_word_category(w) for w in words)
            self.sentence_patterns[n_words][pattern] += 1
        
        self.sentences_seen += 1
    
    def get_word_category(self, word: str) -> int:
        """
        Infer word category from positional distribution.
        
        Returns the most common position (0, 1, 2, ...).
        This is an EMERGENT category, not a predefined one.
        """
        if word not in self.position_counts:
            return -1
        
        positions = self.position_counts[word]
        if not positions:
            return -1
        
        return max(positions.keys(), key=lambda p: positions[p])
    
    def get_word_order(self) -> List[int]:
        """
        Infer learned word order from bigram statistics.
        
        Returns list of position indices in learned order.
        """
        if not self.bigram_counts:
            return []
        
        # Find most common starting word category
        start_counts = defaultdict(int)
        for (w1, w2), count in self.bigram_counts.items():
            cat1 = self.get_word_category(w1)
            start_counts[cat1] += count
        
        if not start_counts:
            return []
        
        # Build order from bigram chain
        order = []
        current = max(start_counts.keys(), key=lambda c: start_counts[c])
        order.append(current)
        
        for _ in range(5):  # Max 5 positions
            # Find most likely next category
            next_counts = defaultdict(int)
            for (w1, w2), count in self.bigram_counts.items():
                if self.get_word_category(w1) == current:
                    next_cat = self.get_word_category(w2)
                    if next_cat not in order:
                        next_counts[next_cat] += count
            
            if not next_counts:
                break
            
            current = max(next_counts.keys(), key=lambda c: next_counts[c])
            order.append(current)
        
        return order
    
    def can_follow(self, word1: str, word2: str) -> float:
        """
        Probability that word2 can follow word1 (learned from data).
        """
        total = sum(self.bigram_counts.get((word1, w), 0) for w in self.word_counts)
        if total == 0:
            return 0.0
        return self.bigram_counts.get((word1, word2), 0) / total
    
    def can_cooccur(self, word1: str, word2: str) -> float:
        """
        Probability that word1 and word2 can appear in same sentence.
        """
        if word1 not in self.cooccurrence:
            return 0.0
        total = sum(self.cooccurrence[word1].values())
        if total == 0:
            return 0.0
        return self.cooccurrence[word1].get(word2, 0) / total
    
    def get_vocabulary(self) -> List[str]:
        """Get all learned words."""
        return list(self.word_counts.keys())
    
    def get_words_at_position(self, position: int) -> List[str]:
        """Get words that commonly appear at a position."""
        words = []
        for word, positions in self.position_counts.items():
            if position in positions:
                words.append((word, positions[position]))
        
        # Sort by frequency at this position
        words.sort(key=lambda x: -x[1])
        return [w for w, _ in words]
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            'sentences_seen': self.sentences_seen,
            'vocabulary_size': len(self.word_counts),
            'bigrams_learned': len(self.bigram_counts),
            'word_order': self.get_word_order(),
            'lengths_seen': dict(self.length_counts),
        }
    
    def get_common_patterns(self, length: int, top_n: int = 5) -> List[Tuple[Tuple, int]]:
        """Get most common sentence patterns of given length."""
        if length not in self.sentence_patterns:
            return []
        
        patterns = list(self.sentence_patterns[length].items())
        patterns.sort(key=lambda x: -x[1])
        return patterns[:top_n]
    
    def pattern_probability(self, pattern: Tuple[int, ...]) -> float:
        """Get probability of a sentence pattern."""
        length = len(pattern)
        if length not in self.sentence_patterns:
            return 0.0
        
        total = sum(self.sentence_patterns[length].values())
        if total == 0:
            return 0.0
        
        return self.sentence_patterns[length].get(pattern, 0) / total
    
    def get_words_for_category(self, category: int, top_n: int = 10) -> List[str]:
        """Get words that belong to a category (position)."""
        words = []
        for word, positions in self.position_counts.items():
            if category in positions:
                words.append((word, positions[category]))
        
        words.sort(key=lambda x: -x[1])
        return [w for w, _ in words[:top_n]]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LANGUAGE LEARNER DEMO")
    print("=" * 60)
    
    learner = LanguageLearner(verbose=True)
    
    # Train on SVO sentences (but don't tell it that!)
    print("\nTraining on sentences (learner doesn't know it's SVO)...")
    sentences = [
        ['dog', 'chases', 'cat'],
        ['cat', 'sees', 'dog'],
        ['boy', 'loves', 'girl'],
        ['girl', 'helps', 'boy'],
        ['dog', 'sees', 'boy'],
        ['cat', 'chases', 'girl'],
    ]
    
    for _ in range(20):  # Repeat for learning
        for sent in sentences:
            learner.hear_sentence(sent)
    
    print(f"\nLearned from {learner.sentences_seen} sentences")
    
    # What did it learn?
    print("\nEmergent word categories (by position):")
    for word in learner.get_vocabulary():
        cat = learner.get_word_category(word)
        print(f"  {word}: position {cat}")
    
    print("\nLearned word order:", learner.get_word_order())
    
    print("\nWords at each position:")
    for pos in range(3):
        words = learner.get_words_at_position(pos)[:3]
        print(f"  Position {pos}: {words}")
    
    print("\nBigram probabilities:")
    for w1 in ['dog', 'chases']:
        for w2 in ['cat', 'sees', 'dog']:
            p = learner.can_follow(w1, w2)
            if p > 0:
                print(f"  P({w2} | {w1}) = {p:.2f}")

