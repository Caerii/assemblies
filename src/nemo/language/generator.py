"""
NEMO Sentence Generator
=======================

Generates sentences from learned patterns - NO HARDCODED GRAMMAR.

Key principles:
1. Uses learned word order (not predefined SVO/SOV)
2. Uses learned co-occurrence (not predefined features)
3. Samples from learned distributions
4. All constraints emerge from training data

This tests if the learned representations support generation.
"""

import numpy as np
from typing import List, Tuple, Optional

from .learner import LanguageLearner


class SentenceGenerator:
    """
    Generates sentences from a trained LanguageLearner.
    
    Generation strategy:
    1. Sample first word from position-0 distribution
    2. Sample next word from bigram distribution
    3. Continue until sentence complete
    
    No hardcoded grammar - everything from learned statistics.
    """
    
    def __init__(self, learner: LanguageLearner):
        self.learner = learner
    
    def _sample_from_distribution(self, items: List[Tuple[str, float]]) -> Optional[str]:
        """Sample an item from a weighted distribution."""
        if not items:
            return None
        
        words, weights = zip(*items)
        weights = np.array(weights, dtype=np.float64)
        
        if weights.sum() == 0:
            return np.random.choice(words)
        
        weights = weights / weights.sum()
        return np.random.choice(words, p=weights)
    
    def generate_word_at_position(self, position: int, 
                                   exclude: List[str] = None) -> Optional[str]:
        """
        Generate a word for a given position.
        
        Uses learned positional distribution.
        """
        exclude = exclude or []
        
        # Get words that appear at this position
        candidates = []
        for word, positions in self.learner.position_counts.items():
            if word not in exclude and position in positions:
                candidates.append((word, positions[position]))
        
        return self._sample_from_distribution(candidates)
    
    def generate_next_word(self, prev_word: str, 
                           exclude: List[str] = None) -> Optional[str]:
        """
        Generate next word given previous word.
        
        Uses learned bigram distribution.
        """
        exclude = exclude or []
        
        # Get words that can follow prev_word
        candidates = []
        for (w1, w2), count in self.learner.bigram_counts.items():
            if w1 == prev_word and w2 not in exclude:
                candidates.append((w2, count))
        
        return self._sample_from_distribution(candidates)
    
    def generate_sentence(self, length: int = 3) -> List[str]:
        """
        Generate a sentence of given length.
        
        Strategy:
        1. Sample a learned pattern for this length
        2. Fill each position with a word of the right category
        3. Use bigrams to prefer coherent sequences
        """
        # Get common patterns for this length
        patterns = self.learner.get_common_patterns(length, top_n=10)
        
        if patterns:
            # Sample a pattern weighted by frequency
            pattern_list, counts = zip(*patterns)
            probs = np.array(counts, dtype=float)
            probs /= probs.sum()
            pattern = pattern_list[np.random.choice(len(pattern_list), p=probs)]
            
            return self._generate_from_pattern(pattern)
        
        # Fall back to sequential generation
        return self._generate_sequential(length)
    
    def _generate_from_pattern(self, pattern: Tuple[int, ...]) -> List[str]:
        """Generate sentence following a learned pattern."""
        sentence = []
        
        for i, category in enumerate(pattern):
            # Get words for this category
            candidates = self.learner.get_words_for_category(category)
            
            if not candidates:
                # Fall back to position-based
                word = self.generate_word_at_position(i, exclude=sentence)
            else:
                # Filter out already used words
                candidates = [w for w in candidates if w not in sentence]
                
                if candidates and sentence:
                    # Prefer words that follow the previous word
                    prev = sentence[-1]
                    scored = []
                    for w in candidates:
                        bigram_score = self.learner.can_follow(prev, w)
                        scored.append((w, bigram_score + 0.1))  # Small base prob
                    
                    words, scores = zip(*scored)
                    probs = np.array(scores)
                    probs /= probs.sum()
                    word = np.random.choice(words, p=probs)
                elif candidates:
                    word = np.random.choice(candidates)
                else:
                    word = self.generate_word_at_position(i, exclude=sentence)
            
            if word:
                sentence.append(word)
        
        return sentence
    
    def _generate_sequential(self, length: int) -> List[str]:
        """Generate sentence sequentially (fallback)."""
        sentence = []
        
        # First word
        first = self.generate_word_at_position(0)
        if first is None:
            return []
        sentence.append(first)
        
        # Subsequent words
        for i in range(1, length):
            # Try bigram first
            next_word = self.generate_next_word(sentence[-1], exclude=sentence)
            
            # Fall back to positional
            if next_word is None:
                next_word = self.generate_word_at_position(i, exclude=sentence)
            
            if next_word is None:
                break
            
            sentence.append(next_word)
        
        return sentence
    
    def generate_with_constraint(self, must_include: str, 
                                  length: int = 3) -> List[str]:
        """
        Generate a sentence that must include a specific word.
        
        Uses learned position of that word.
        """
        # Find most likely position for this word
        if must_include not in self.learner.position_counts:
            return []
        
        positions = self.learner.position_counts[must_include]
        best_pos = max(positions.keys(), key=lambda p: positions[p])
        
        # Generate around this constraint
        sentence = [None] * length
        sentence[best_pos] = must_include
        
        # Fill other positions
        for i in range(length):
            if sentence[i] is not None:
                continue
            
            # Use bigram if possible
            if i > 0 and sentence[i-1] is not None:
                word = self.generate_next_word(
                    sentence[i-1], 
                    exclude=[w for w in sentence if w is not None]
                )
            else:
                word = self.generate_word_at_position(
                    i, 
                    exclude=[w for w in sentence if w is not None]
                )
            
            sentence[i] = word
        
        return [w for w in sentence if w is not None]
    
    def score_sentence(self, words: List[str]) -> float:
        """
        Score a sentence based on learned patterns.
        
        Higher score = more likely given training data.
        """
        if len(words) < 2:
            return 0.0
        
        score = 0.0
        
        # Positional score
        for i, word in enumerate(words):
            if word in self.learner.position_counts:
                pos_counts = self.learner.position_counts[word]
                total = sum(pos_counts.values())
                if total > 0 and i in pos_counts:
                    score += pos_counts[i] / total
        
        # Bigram score
        for i in range(len(words) - 1):
            p = self.learner.can_follow(words[i], words[i+1])
            score += p
        
        return score / len(words)
    
    def rank_sentences(self, sentences: List[List[str]]) -> List[Tuple[List[str], float]]:
        """Rank sentences by their scores."""
        scored = [(sent, self.score_sentence(sent)) for sent in sentences]
        scored.sort(key=lambda x: -x[1])
        return scored


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from .learner import LanguageLearner
    
    print("=" * 60)
    print("SENTENCE GENERATOR DEMO")
    print("=" * 60)
    
    # Train learner
    learner = LanguageLearner(verbose=False)
    
    sentences = [
        ['dog', 'chases', 'cat'],
        ['cat', 'sees', 'dog'],
        ['boy', 'loves', 'girl'],
        ['girl', 'helps', 'boy'],
        ['dog', 'sees', 'boy'],
        ['cat', 'chases', 'girl'],
        ['boy', 'chases', 'dog'],
        ['girl', 'sees', 'cat'],
    ]
    
    print("Training on sentences...")
    for _ in range(30):
        for sent in sentences:
            learner.hear_sentence(sent)
    
    print(f"Trained on {learner.sentences_seen} sentences")
    print(f"Vocabulary: {learner.get_vocabulary()}")
    
    # Generate
    generator = SentenceGenerator(learner)
    
    print("\nGenerating sentences (no hardcoded grammar!):")
    for i in range(10):
        sent = generator.generate_sentence(length=3)
        score = generator.score_sentence(sent)
        print(f"  {i+1}. {' '.join(sent)} (score: {score:.2f})")
    
    print("\nGenerating with constraint (must include 'dog'):")
    for i in range(5):
        sent = generator.generate_with_constraint('dog', length=3)
        print(f"  {i+1}. {' '.join(sent)}")
    
    print("\nScoring novel sentences:")
    test_sentences = [
        ['dog', 'chases', 'cat'],   # Seen
        ['cat', 'chases', 'dog'],   # Novel but valid
        ['chases', 'dog', 'cat'],   # Wrong order
    ]
    
    ranked = generator.rank_sentences(test_sentences)
    for sent, score in ranked:
        print(f"  {' '.join(sent)}: {score:.2f}")

