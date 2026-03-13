"""
NEMO Curriculum Learning
========================

Structured curriculum for learning increasingly complex language.

Key Principles:
1. Start simple, build complexity gradually (like children)
2. Each stage builds on previous learning
3. Structure emerges from patterns, not explicit rules

Curriculum Stages:
1. Single words (naming)
2. Two-word combinations (agent-action, action-object)
3. Simple sentences (SVO)
4. Modified sentences (adjectives, adverbs)
5. Complex sentences (conjunctions, relative clauses)
6. Questions and negation

Scientific Value:
- Models child language acquisition
- Tests if assemblies can learn hierarchical structure
- Measures when each structure is acquired
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

from .learner import LanguageLearner


class StructureType(Enum):
    """Types of linguistic structure to learn."""
    WORD = auto()           # Single word
    PAIR = auto()           # Two-word combination
    TRIPLE = auto()         # Three-word sentence
    MODIFIED = auto()        # With adjective/adverb
    COMPOUND = auto()        # With conjunction
    EMBEDDED = auto()        # With relative clause
    QUESTION = auto()        # Question form
    NEGATION = auto()        # Negated sentence


@dataclass
class LearningMilestone:
    """Tracks when a structure type is acquired."""
    structure: StructureType
    sentences_needed: int = 0
    accuracy: float = 0.0
    acquired: bool = False


@dataclass
class Curriculum:
    """
    A curriculum for language learning.
    
    Provides sentences of increasing complexity.
    """
    
    # Vocabulary pools
    nouns: List[str] = field(default_factory=lambda: [
        'dog', 'cat', 'bird', 'fish', 'boy', 'girl', 'man', 'woman',
        'ball', 'book', 'tree', 'house', 'car', 'food', 'water'
    ])
    
    verbs: List[str] = field(default_factory=lambda: [
        'sees', 'chases', 'eats', 'loves', 'helps', 'finds',
        'wants', 'likes', 'has', 'gives', 'takes', 'makes'
    ])
    
    adjectives: List[str] = field(default_factory=lambda: [
        'big', 'small', 'red', 'blue', 'happy', 'sad', 'fast', 'slow',
        'old', 'young', 'good', 'bad', 'hot', 'cold'
    ])
    
    adverbs: List[str] = field(default_factory=lambda: [
        'quickly', 'slowly', 'loudly', 'quietly', 'happily', 'sadly'
    ])
    
    conjunctions: List[str] = field(default_factory=lambda: [
        'and', 'but', 'or', 'because', 'when', 'while'
    ])
    
    relativizers: List[str] = field(default_factory=lambda: [
        'that', 'which', 'who'
    ])
    
    def random_noun(self) -> str:
        return np.random.choice(self.nouns)
    
    def random_verb(self) -> str:
        return np.random.choice(self.verbs)
    
    def random_adj(self) -> str:
        return np.random.choice(self.adjectives)
    
    def random_adv(self) -> str:
        return np.random.choice(self.adverbs)
    
    # =========================================================================
    # SENTENCE GENERATORS (increasing complexity)
    # =========================================================================
    
    def generate_word(self) -> List[str]:
        """Stage 1: Single word (naming)."""
        return [self.random_noun()]
    
    def generate_pair(self) -> List[str]:
        """Stage 2: Two-word combination."""
        patterns = [
            lambda: [self.random_noun(), self.random_verb()],  # dog runs
            lambda: [self.random_verb(), self.random_noun()],  # see dog
            lambda: [self.random_adj(), self.random_noun()],   # big dog
        ]
        return np.random.choice(patterns)()
    
    def generate_triple(self) -> List[str]:
        """Stage 3: Simple SVO sentence."""
        subj = self.random_noun()
        verb = self.random_verb()
        obj = self.random_noun()
        while obj == subj:
            obj = self.random_noun()
        return [subj, verb, obj]
    
    def generate_modified(self) -> List[str]:
        """Stage 4: Sentence with modifier."""
        patterns = [
            # Adjective + Noun + Verb + Noun
            lambda: [self.random_adj(), self.random_noun(), 
                    self.random_verb(), self.random_noun()],
            # Noun + Verb + Adjective + Noun
            lambda: [self.random_noun(), self.random_verb(),
                    self.random_adj(), self.random_noun()],
            # Noun + Adverb + Verb + Noun
            lambda: [self.random_noun(), self.random_adv(),
                    self.random_verb(), self.random_noun()],
        ]
        sent = np.random.choice(patterns)()
        # Ensure no repeated nouns
        while len(set(w for w in sent if w in self.nouns)) < 2:
            sent = np.random.choice(patterns)()
        return sent
    
    def generate_compound(self) -> List[str]:
        """Stage 5: Compound sentence with conjunction."""
        # S V O conj S V O
        s1, v1, o1 = self.generate_triple()
        conj = np.random.choice(self.conjunctions[:3])  # and, but, or
        s2, v2, o2 = self.generate_triple()
        return [s1, v1, o1, conj, s2, v2, o2]
    
    def generate_embedded(self) -> List[str]:
        """Stage 6: Sentence with relative clause."""
        # The N that V O V O
        # "the dog that chases cats sees birds"
        subj = self.random_noun()
        rel = np.random.choice(self.relativizers)
        v1 = self.random_verb()
        o1 = self.random_noun()
        v2 = self.random_verb()
        o2 = self.random_noun()
        
        return ['the', subj, rel, v1, o1, v2, o2]
    
    def generate_question(self) -> List[str]:
        """Stage 7: Question form."""
        patterns = [
            # Does S V O?
            lambda: ['does', self.random_noun(), self.random_verb(), 
                    self.random_noun()],
            # What does S V?
            lambda: ['what', 'does', self.random_noun(), self.random_verb()],
            # Who V O?
            lambda: ['who', self.random_verb(), self.random_noun()],
        ]
        return np.random.choice(patterns)()
    
    def generate_negation(self) -> List[str]:
        """Stage 8: Negated sentence."""
        # S does not V O
        return [self.random_noun(), 'does', 'not', 
                self.random_verb(), self.random_noun()]
    
    def generate(self, structure: StructureType) -> List[str]:
        """Generate a sentence of given structure type."""
        generators = {
            StructureType.WORD: self.generate_word,
            StructureType.PAIR: self.generate_pair,
            StructureType.TRIPLE: self.generate_triple,
            StructureType.MODIFIED: self.generate_modified,
            StructureType.COMPOUND: self.generate_compound,
            StructureType.EMBEDDED: self.generate_embedded,
            StructureType.QUESTION: self.generate_question,
            StructureType.NEGATION: self.generate_negation,
        }
        return generators[structure]()


class CurriculumLearner:
    """
    Learns language through a structured curriculum.
    
    Tracks acquisition of each structure type.
    """
    
    def __init__(self, learner: LanguageLearner = None, verbose: bool = True):
        self.learner = learner or LanguageLearner(verbose=False)
        self.curriculum = Curriculum()
        self.verbose = verbose
        
        # Track milestones
        self.milestones: Dict[StructureType, LearningMilestone] = {
            st: LearningMilestone(structure=st)
            for st in StructureType
        }
        
        # Track what structures we've trained on
        self.structure_counts: Dict[StructureType, int] = defaultdict(int)
        
        # Current stage
        self.current_stage = 0
        self.stages = list(StructureType)
    
    def train_stage(self, structure: StructureType, n_sentences: int = 50):
        """Train on a specific structure type."""
        if self.verbose:
            print(f"Training on {structure.name}...")
        
        for _ in range(n_sentences):
            sent = self.curriculum.generate(structure)
            self.learner.hear_sentence(sent)
            self.structure_counts[structure] += 1
        
        # Update milestone
        milestone = self.milestones[structure]
        milestone.sentences_needed = self.structure_counts[structure]
    
    def train_curriculum(self, sentences_per_stage: int = 100):
        """Train through entire curriculum."""
        for stage in self.stages:
            self.train_stage(stage, sentences_per_stage)
            
            if self.verbose:
                stats = self.learner.get_stats()
                print(f"  Vocab: {stats['vocabulary_size']}, "
                      f"Bigrams: {stats['bigrams_learned']}")
    
    def test_structure(self, structure: StructureType, n_tests: int = 20) -> float:
        """
        Test if a structure has been learned.
        
        Returns accuracy (0-1).
        """
        from .generator import SentenceGenerator
        generator = SentenceGenerator(self.learner)
        
        correct = 0
        for _ in range(n_tests):
            # Generate a sentence
            target = self.curriculum.generate(structure)
            length = len(target)
            
            # See if we can generate something similar
            generated = generator.generate_sentence(length)
            
            # Score based on structure match
            if len(generated) == length:
                # Check if positions match expected categories
                match = True
                for i, (t, g) in enumerate(zip(target, generated)):
                    t_cat = self.learner.get_word_category(t)
                    g_cat = self.learner.get_word_category(g)
                    if t_cat != g_cat and t_cat != -1 and g_cat != -1:
                        match = False
                        break
                if match:
                    correct += 1
        
        accuracy = correct / n_tests
        self.milestones[structure].accuracy = accuracy
        self.milestones[structure].acquired = accuracy > 0.7
        
        return accuracy
    
    def get_acquisition_report(self) -> str:
        """Generate report on structure acquisition."""
        lines = ["Structure Acquisition Report", "=" * 40]
        
        for st in self.stages:
            m = self.milestones[st]
            status = "✓" if m.acquired else "○"
            lines.append(
                f"{status} {st.name:12} | "
                f"trained: {self.structure_counts[st]:4} | "
                f"accuracy: {m.accuracy:.0%}"
            )
        
        return "\n".join(lines)


# =============================================================================
# DYNAMIC STRUCTURE LEARNING
# =============================================================================

class StructureDetector:
    """
    Detects emerging structure in learned patterns.
    
    Key insight: Structure is detected through statistical regularities,
    not explicit rules.
    """
    
    def __init__(self, learner: LanguageLearner):
        self.learner = learner
    
    def detect_word_classes(self) -> Dict[int, List[str]]:
        """
        Detect word classes from positional distribution.
        
        Words that appear in same positions form a class.
        """
        classes = defaultdict(list)
        
        for word in self.learner.get_vocabulary():
            primary_pos = self.learner.get_word_category(word)
            if primary_pos >= 0:
                classes[primary_pos].append(word)
        
        return dict(classes)
    
    def detect_phrase_structure(self) -> List[Tuple[str, str, float]]:
        """
        Detect phrase structure from bigram patterns.
        
        Returns list of (word1, word2, strength) for strong associations.
        """
        phrases = []
        
        for (w1, w2), count in self.learner.bigram_counts.items():
            # Normalize by word frequencies
            w1_count = self.learner.word_counts.get(w1, 1)
            w2_count = self.learner.word_counts.get(w2, 1)
            
            # Pointwise mutual information (simplified)
            total = self.learner.sentences_seen
            if total > 0:
                p_joint = count / total
                p_w1 = w1_count / total
                p_w2 = w2_count / total
                
                if p_w1 > 0 and p_w2 > 0:
                    pmi = np.log2(p_joint / (p_w1 * p_w2) + 1e-10)
                    if pmi > 1.0:  # Strong association
                        phrases.append((w1, w2, pmi))
        
        phrases.sort(key=lambda x: -x[2])
        return phrases
    
    def detect_sentence_patterns(self) -> List[Tuple[List[int], int]]:
        """
        Detect common sentence patterns.
        
        Returns patterns as sequences of position categories.
        """
        patterns = defaultdict(int)
        
        # Reconstruct patterns from position statistics
        for word, positions in self.learner.position_counts.items():
            for pos, count in positions.items():
                # This word appears at this position
                pass
        
        # For now, just return the learned word order
        order = self.learner.get_word_order()
        return [(order, self.learner.sentences_seen)]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CURRICULUM LEARNING DEMO")
    print("=" * 60)
    
    # Create curriculum learner
    cl = CurriculumLearner(verbose=True)
    
    # Train through curriculum
    print("\nTraining through curriculum stages...")
    print("-" * 40)
    
    for stage in [StructureType.WORD, StructureType.PAIR, 
                  StructureType.TRIPLE, StructureType.MODIFIED]:
        cl.train_stage(stage, n_sentences=50)
    
    # Detect structure
    print("\n" + "=" * 60)
    print("DETECTED STRUCTURE")
    print("=" * 60)
    
    detector = StructureDetector(cl.learner)
    
    print("\nWord Classes (by position):")
    classes = detector.detect_word_classes()
    for pos, words in sorted(classes.items()):
        print(f"  Position {pos}: {words[:5]}...")
    
    print("\nStrong Phrases (by PMI):")
    phrases = detector.detect_phrase_structure()[:10]
    for w1, w2, pmi in phrases:
        print(f"  {w1} {w2}: {pmi:.2f}")
    
    # Test generation
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    from .generator import SentenceGenerator
    generator = SentenceGenerator(cl.learner)
    
    print("\nGenerated sentences:")
    for i in range(5):
        sent = generator.generate_sentence(4)
        print(f"  {' '.join(sent)}")
    
    # Acquisition report
    print("\n" + "=" * 60)
    print("ACQUISITION STATUS")
    print("=" * 60)
    
    for stage in [StructureType.TRIPLE, StructureType.MODIFIED]:
        acc = cl.test_structure(stage)
        print(f"  {stage.name}: {acc:.0%}")

