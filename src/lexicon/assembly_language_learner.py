"""
Assembly Language Learner
=========================

Integrates the lexicon and curriculum with the assembly brain
to test how the system learns language.

Uses explicit areas with full connectivity to enable proper Hebbian learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

# Import brain
from brain import Brain, Area

# Import lexicon
from src.lexicon.build_lexicon import build_lexicon
from src.lexicon.curriculum.grounded_training import (
    GroundedCorpus, GroundedUtterance, GroundedContext, SpeechAct,
    create_stage1_corpus
)


class AssemblyLanguageLearner:
    """
    A language learner based on neural assemblies.
    
    Uses EXPLICIT areas with full weight matrices for proper Hebbian learning.
    
    Architecture:
    - LEX: Lexical area (word forms)
    - VISUAL: Visual grounding area
    - CORE: Shared grammatical category area
    """
    
    def __init__(self, 
                 n: int = 1000,       # Neurons per area (smaller for explicit)
                 k: int = 50,         # Active neurons
                 p: float = 0.1,      # Connection probability (higher for explicit)
                 beta: float = 0.1,   # Plasticity
                 verbose: bool = False):
        
        self.n = n
        self.k = k
        self.p = p
        self.beta = beta
        self.verbose = verbose
        
        # Create brain
        self.brain = Brain(p=p, save_winners=True)
        
        # Use explicit areas for proper Hebbian learning
        self.brain.add_explicit_area('LEX', n, k, beta)
        self.brain.add_explicit_area('VISUAL', n, k, beta)
        self.brain.add_explicit_area('CORE', n, k, beta)
        
        # Track learned words
        self.word_assemblies: Dict[str, np.ndarray] = {}  # word -> assembly neurons
        self.visual_assemblies: Dict[str, np.ndarray] = {}  # visual concept -> assembly
        self.word_exposures: Dict[str, int] = {}
        
        # Track learning
        self.learning_history: List[Dict] = []
    
    def _get_or_create_stimulus(self, area_name: str, concept: str) -> np.ndarray:
        """Get or create a stimulus pattern for a concept"""
        # Use hash of concept to generate consistent random pattern
        np.random.seed(hash(concept) % (2**32))
        stimulus = np.random.choice(self.n, self.k, replace=False)
        np.random.seed()  # Reset seed
        return stimulus
    
    def learn_word(self, word: str, pos: str, grounding: List[str], n_rounds: int = 5):
        """
        Learn a single word with grounding using Hebbian association.
        
        The key insight: we activate VISUAL and LEX simultaneously,
        then project both to CORE. This creates associations:
        - VISUAL -> CORE
        - LEX -> CORE
        - CORE -> VISUAL (for retrieval)
        - CORE -> LEX (for retrieval)
        """
        if self.verbose:
            print(f"Learning word '{word}' ({pos}) with grounding {grounding}")
        
        # Get/create word assembly in LEX
        word_stimulus = self._get_or_create_stimulus('LEX', word)
        
        # Get/create grounding assemblies in VISUAL
        grounding_stimuli = [self._get_or_create_stimulus('VISUAL', g) for g in grounding]
        combined_grounding = np.concatenate(grounding_stimuli)[:self.k]
        
        # Learning: Associate word with grounding through CORE
        for _ in range(n_rounds):
            # 1. Activate both VISUAL and LEX simultaneously
            self.brain.areas['VISUAL'].winners = combined_grounding
            self.brain.areas['VISUAL'].fix_assembly()
            
            self.brain.areas['LEX'].winners = word_stimulus
            self.brain.areas['LEX'].fix_assembly()
            
            # 2. Project both to CORE (creates shared representation)
            self.brain.project({}, {'VISUAL': ['CORE'], 'LEX': ['CORE']}, 0)
            
            # 3. Project back from CORE to both areas (strengthens association)
            self.brain.project({}, {'CORE': ['VISUAL', 'LEX']}, 0)
        
        # Store learned word
        self.word_assemblies[word] = word_stimulus
        for g in grounding:
            self.visual_assemblies[g] = self._get_or_create_stimulus('VISUAL', g)
        
        self.word_exposures[word] = self.word_exposures.get(word, 0) + n_rounds
        
        if self.verbose:
            print(f"  Word '{word}' learned with {n_rounds} rounds")
    
    def test_retrieval(self, grounding: List[str], expected_word: str) -> Tuple[bool, float]:
        """
        Test if the system can retrieve a word from its grounding.
        
        Retrieval path: VISUAL -> CORE -> LEX
        
        Args:
            grounding: Visual/conceptual grounding concepts
            expected_word: The word we expect to retrieve
            
        Returns:
            (success, confidence)
        """
        if expected_word not in self.word_assemblies:
            return False, 0.0
        
        # Activate grounding
        grounding_stimuli = [self._get_or_create_stimulus('VISUAL', g) for g in grounding]
        combined_grounding = np.concatenate(grounding_stimuli)[:self.k]
        self.brain.areas['VISUAL'].winners = combined_grounding
        self.brain.areas['VISUAL'].fix_assembly()
        
        # Project VISUAL -> CORE
        try:
            self.brain.project({}, {'VISUAL': ['CORE']}, 0)
        except (ValueError, IndexError):
            return False, 0.0
        
        # Project CORE -> LEX
        try:
            self.brain.project({}, {'CORE': ['LEX']}, 0)
        except (ValueError, IndexError):
            return False, 0.0
        
        # Check overlap with expected word assembly
        expected_assembly = self.word_assemblies[expected_word]
        current_winners = self.brain.areas['LEX'].winners
        
        if hasattr(current_winners, '__len__') and len(current_winners) > 0:
            overlap = len(set(expected_assembly) & set(current_winners))
            confidence = overlap / self.k
            success = confidence > 0.5  # More than 50% overlap
            return success, confidence
        
        return False, 0.0
    
    def learn_from_utterance(self, utterance: GroundedUtterance):
        """Learn from a grounded utterance"""
        words = utterance.words
        pos_tags = utterance.pos_tags
        context = utterance.context
        
        # Learn each content word with its grounding
        for word, pos in zip(words, pos_tags):
            # Find relevant grounding for this word
            grounding = []
            for obj in context.visual_objects:
                if word.upper() in obj or obj in word.upper():
                    grounding.append(obj)
            
            if not grounding:
                grounding = context.visual_objects[:1] if context.visual_objects else [word.upper()]
            
            self.learn_word(word, pos, grounding, n_rounds=1)
    
    def train_on_corpus(self, corpus: GroundedCorpus, n_epochs: int = 1):
        """Train on a grounded corpus"""
        for epoch in range(n_epochs):
            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            for utterance in corpus.examples:
                self.learn_from_utterance(utterance)
        
        if self.verbose:
            print(f"\nTraining complete. Learned {len(self.word_assemblies)} words.")
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'words_learned': len(self.word_assemblies),
            'visual_concepts': len(self.visual_assemblies),
            'total_exposures': sum(self.word_exposures.values()),
            'avg_exposures_per_word': (
                sum(self.word_exposures.values()) / max(len(self.word_exposures), 1)
            ),
            'word_exposures': dict(self.word_exposures),
        }


def test_word_learning_curve():
    """Test how many exposures are needed to learn a word"""
    print("=" * 60)
    print("WORD LEARNING CURVE EXPERIMENT")
    print("=" * 60)
    
    results = []
    
    for n_exposures in [1, 2, 3, 5, 7, 10, 15, 20]:
        # Create fresh learner (smaller for explicit areas)
        learner = AssemblyLanguageLearner(n=1000, k=50, p=0.1, verbose=False)
        
        # Learn word with n exposures
        word = 'dog'
        grounding = ['DOG', 'ANIMAL']
        learner.learn_word(word, 'NOUN', grounding, n_rounds=n_exposures)
        
        # Test retrieval
        success, confidence = learner.test_retrieval(grounding, word)
        
        results.append({
            'exposures': n_exposures,
            'success': success,
            'confidence': confidence,
        })
        
        status = 'OK' if success else 'FAIL'
        print(f"  {n_exposures} exposures: {status} (confidence: {confidence:.2f})")
    
    print("\nLearning curve summary:")
    for r in results:
        bar = '#' * int(r['confidence'] * 20)
        print(f"  {r['exposures']:2d} exposures: {bar} {r['confidence']:.2f}")
    
    return results


def test_vocabulary_learning():
    """Test learning a vocabulary from Stage 1 corpus"""
    print("\n" + "=" * 60)
    print("VOCABULARY LEARNING FROM CORPUS")
    print("=" * 60)
    
    # Create learner (smaller for explicit areas)
    learner = AssemblyLanguageLearner(n=1000, k=50, p=0.1, verbose=False)
    
    # Create Stage 1 corpus
    corpus = create_stage1_corpus()
    
    print(f"\nCorpus has {len(corpus.examples)} examples")
    print(f"Unique words: {len(corpus.word_exposures)}")
    
    # Train
    print("\nTraining...")
    learner.train_on_corpus(corpus, n_epochs=1)
    
    # Get stats
    stats = learner.get_learning_stats()
    print(f"\nLearning stats:")
    print(f"  Words learned: {stats['words_learned']}")
    print(f"  Avg exposures per word: {stats['avg_exposures_per_word']:.1f}")
    
    # Test retrieval for some words
    print("\nRetrieval tests:")
    test_words = [
        ('dog', ['DOG']),
        ('cat', ['CAT']),
        ('ball', ['BALL']),
        ('milk', ['MILK', 'DRINK']),
    ]
    
    for word, grounding in test_words:
        success, confidence = learner.test_retrieval(grounding, word)
        status = 'OK' if success else 'FAIL'
        print(f"  '{word}' from {grounding}: {status} ({confidence:.2f})")
    
    return learner


if __name__ == '__main__':
    # Run experiments
    test_word_learning_curve()
    test_vocabulary_learning()

