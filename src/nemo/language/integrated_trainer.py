"""
Integrated NEMO Trainer
=======================

Version: 1.0.0
Date: 2025-12-01

Integrates the NEMO learner with:
- Rich lexicon (nouns, verbs, adjectives, etc.)
- Curriculum stages (child language acquisition)
- Grounded training data

This allows training on realistic language data following
the developmental trajectory of human language acquisition.
"""

import sys
import os

# Add parent directories to path for imports
_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import time
from typing import List, Set
from dataclasses import dataclass

# Import NEMO learner
from src.nemo.language.nemo_learner import (
    NemoLanguageLearner, NemoParams,
    GroundedContext
)

# Import lexicon
from lexicon.build_lexicon import build_lexicon

# Import curriculum
from lexicon.curriculum.stage1_first_words import STAGE1_CORPUS
from lexicon.curriculum.stage2_vocabulary_spurt import STAGE2_CORPUS
from lexicon.curriculum.stage3_two_word import STAGE3_CORPUS
from lexicon.curriculum.stage4_sentences import STAGE4_CORPUS


@dataclass
class TrainingStats:
    """Statistics for training progress"""
    stage: int = 0
    sentences_seen: int = 0
    words_learned: int = 0
    noun_accuracy: float = 0.0
    verb_accuracy: float = 0.0
    word_order_correct: bool = False
    
    def __str__(self):
        return (f"Stage {self.stage}: {self.sentences_seen} sentences, "
                f"{self.words_learned} words, "
                f"N:{self.noun_accuracy:.0%} V:{self.verb_accuracy:.0%}")


class IntegratedNemoTrainer:
    """
    Trains NEMO on realistic language data with curriculum.
    
    Features:
    - Loads vocabulary from rich lexicon
    - Registers words with proper grounding (visual for nouns, motor for verbs)
    - Trains through curriculum stages
    - Tracks learning progress
    """
    
    def __init__(self, params: NemoParams = None, verbose: bool = True):
        self.params = params or NemoParams(n=10000)  # k = sqrt(n) = 100
        self.verbose = verbose
        
        # Create NEMO learner
        self.learner = NemoLanguageLearner(self.params, verbose=verbose)
        
        # Load lexicon
        if verbose:
            print("\nLoading lexicon...")
        self.lexicon = build_lexicon()
        if verbose:
            stats = self.lexicon.get_stats()
            print(f"  Total words: {stats['total_words']}")
        
        # Track registered words
        self.registered_nouns: Set[str] = set()
        self.registered_verbs: Set[str] = set()
        self.registered_adjectives: Set[str] = set()
        self.registered_other: Set[str] = set()
        
        # Training history
        self.history: List[TrainingStats] = []
        
        # Current stage
        self.current_stage = 0
    
    def register_vocabulary_by_aoa(self, max_aoa: float):
        """
        Register vocabulary up to a certain age of acquisition.
        
        This simulates the gradual vocabulary growth in children.
        """
        if self.verbose:
            print(f"\nRegistering vocabulary with AoA <= {max_aoa}...")
        
        # Get words by AoA
        words = self.lexicon.get_by_aoa(max_aoa)
        
        nouns_added = 0
        verbs_added = 0
        adj_added = 0
        other_added = 0
        
        for word in words:
            lemma = word.lemma
            # Use category name for comparison (avoids enum import issues)
            cat_name = word.category.name
            
            if cat_name == 'NOUN':
                if lemma not in self.registered_nouns:
                    # Get visual grounding from semantic domains
                    visual = [d.name for d in word.semantic_domains]
                    if not visual:
                        visual = [lemma.upper()]
                    self.learner.register_noun(lemma, visual)
                    self.registered_nouns.add(lemma)
                    nouns_added += 1
                    
            elif cat_name == 'VERB':
                if lemma not in self.registered_verbs:
                    # Get motor grounding from semantic domains
                    motor = [d.name for d in word.semantic_domains]
                    if not motor:
                        motor = [lemma.upper()]
                    self.learner.register_verb(lemma, motor)
                    self.registered_verbs.add(lemma)
                    verbs_added += 1
                    
            elif cat_name == 'ADJECTIVE':
                if lemma not in self.registered_adjectives:
                    self.registered_adjectives.add(lemma)
                    adj_added += 1
                    
            else:
                if lemma not in self.registered_other:
                    self.registered_other.add(lemma)
                    other_added += 1
        
        if self.verbose:
            print(f"  Nouns: +{nouns_added} (total: {len(self.registered_nouns)})")
            print(f"  Verbs: +{verbs_added} (total: {len(self.registered_verbs)})")
            print(f"  Adjectives: +{adj_added} (total: {len(self.registered_adjectives)})")
            print(f"  Other: +{other_added} (total: {len(self.registered_other)})")
    
    def _create_context_from_sentence(self, words: List[str]) -> GroundedContext:
        """Create grounded context from sentence words"""
        visual = []
        motor = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if it's a registered noun
            if word_lower in self.registered_nouns:
                visual.append(word_lower.upper())
            
            # Check if it's a registered verb
            elif word_lower in self.registered_verbs:
                motor.append(word_lower.upper())
        
        return GroundedContext(visual=visual, motor=motor)
    
    def _infer_roles(self, words: List[str]) -> List[str]:
        """Infer syntactic roles from word positions and categories"""
        roles = []
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Simple heuristics based on position and category
            if word_lower in self.registered_verbs:
                roles.append('VERB')
            elif word_lower in self.registered_nouns:
                # First noun is usually subject, later nouns are objects
                if 'SUBJ' not in roles:
                    roles.append('SUBJ')
                else:
                    roles.append('OBJ')
            else:
                # Skip function words for now
                roles.append('OTHER')
        
        return roles
    
    def train_on_corpus(self, corpus: List[str], n_epochs: int = 1):
        """Train on a list of sentences"""
        sentences_trained = 0
        
        for epoch in range(n_epochs):
            for sentence in corpus:
                # Tokenize
                words = sentence.lower().split()
                
                # Filter to known words
                known_words = [w for w in words 
                              if w in self.registered_nouns or w in self.registered_verbs]
                
                if len(known_words) < 2:
                    continue
                
                # Check we have at least one noun and one verb
                has_noun = any(w in self.registered_nouns for w in known_words)
                has_verb = any(w in self.registered_verbs for w in known_words)
                
                if not (has_noun and has_verb):
                    continue
                
                # Create context and roles
                context = self._create_context_from_sentence(known_words)
                roles = self._infer_roles(known_words)
                
                # Filter out 'OTHER' roles
                filtered_words = []
                filtered_roles = []
                for w, r in zip(known_words, roles):
                    if r != 'OTHER':
                        filtered_words.append(w)
                        filtered_roles.append(r)
                
                if len(filtered_words) >= 2:
                    self.learner.present_grounded_sentence(
                        filtered_words, context, filtered_roles, learn=True)
                    sentences_trained += 1
        
        return sentences_trained
    
    def train_stage(self, stage: int, n_epochs: int = 1) -> TrainingStats:
        """Train on a specific curriculum stage"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STAGE {stage}")
            print(f"{'='*60}")
        
        # Register vocabulary for this stage
        aoa_by_stage = {1: 2.0, 2: 2.5, 3: 3.0, 4: 4.0}
        max_aoa = aoa_by_stage.get(stage, 5.0)
        self.register_vocabulary_by_aoa(max_aoa)
        
        # Get corpus for this stage
        corpus_by_stage = {
            1: STAGE1_CORPUS,
            2: STAGE2_CORPUS,
            3: STAGE3_CORPUS,
            4: STAGE4_CORPUS,
        }
        corpus = corpus_by_stage.get(stage, STAGE1_CORPUS)
        
        # Train
        if self.verbose:
            print(f"\nTraining on {len(corpus)} sentences (x{n_epochs} epochs)...")
        
        start = time.perf_counter()
        trained = self.train_on_corpus(corpus, n_epochs=n_epochs)
        elapsed = time.perf_counter() - start
        
        if self.verbose:
            print(f"Training complete: {trained} sentences in {elapsed:.2f}s")
        
        # Evaluate
        stats = self._evaluate()
        stats.stage = stage
        self.history.append(stats)
        self.current_stage = stage
        
        return stats
    
    def _evaluate(self) -> TrainingStats:
        """Evaluate current learning"""
        stats = TrainingStats()
        stats.sentences_seen = self.learner.sentences_seen
        stats.words_learned = len(self.learner.brain.words_learned)
        
        # Test noun classification
        noun_correct = 0
        noun_total = 0
        for noun in list(self.registered_nouns)[:20]:  # Test subset
            pred, _, _ = self.learner.classify_word(noun)
            if pred == 'NOUN':
                noun_correct += 1
            noun_total += 1
        
        stats.noun_accuracy = noun_correct / max(noun_total, 1)
        
        # Test verb classification
        verb_correct = 0
        verb_total = 0
        for verb in list(self.registered_verbs)[:20]:  # Test subset
            pred, _, _ = self.learner.classify_word(verb)
            if pred == 'VERB':
                verb_correct += 1
            verb_total += 1
        
        stats.verb_accuracy = verb_correct / max(verb_total, 1)
        
        # Test word order
        word_order = self.learner.get_learned_word_order()
        stats.word_order_correct = word_order == ['SUBJ', 'VERB', 'OBJ'] or word_order == ['SUBJ', 'VERB']
        
        if self.verbose:
            print("\nEvaluation:")
            print(f"  Sentences seen: {stats.sentences_seen}")
            print(f"  Words learned: {stats.words_learned}")
            print(f"  Noun accuracy: {stats.noun_accuracy:.1%}")
            print(f"  Verb accuracy: {stats.verb_accuracy:.1%}")
            print(f"  Word order: {word_order} ({'✓' if stats.word_order_correct else '✗'})")
        
        return stats
    
    def train_full_curriculum(self, epochs_per_stage: int = 3) -> List[TrainingStats]:
        """Train through all curriculum stages"""
        if self.verbose:
            print("\n" + "="*70)
            print("FULL CURRICULUM TRAINING")
            print("="*70)
        
        results = []
        for stage in [1, 2, 3, 4]:
            stats = self.train_stage(stage, n_epochs=epochs_per_stage)
            results.append(stats)
        
        if self.verbose:
            print("\n" + "="*70)
            print("CURRICULUM COMPLETE")
            print("="*70)
            print("\nProgress through stages:")
            for stats in results:
                print(f"  {stats}")
        
        return results
    
    def generate_sentence(self, length: int = 3, animate_subject: bool = True) -> List[str]:
        """
        Generate a sentence using learned patterns.
        
        Args:
            length: Number of words
            animate_subject: If True, prefer animate nouns as subjects
        """
        word_order = self.learner.get_learned_word_order()
        
        # Get animate nouns (people, animals)
        animate_nouns = []
        inanimate_nouns = []
        for lemma in self.registered_nouns:
            word = self.lexicon.get_word(lemma)
            if word:
                domains = [d.name for d in word.semantic_domains]
                if 'ANIMAL' in domains or 'PERSON' in domains:
                    animate_nouns.append(lemma)
                else:
                    inanimate_nouns.append(lemma)
        
        # Get action verbs
        action_verbs = []
        for lemma in self.registered_verbs:
            word = self.lexicon.get_word(lemma)
            if word:
                domains = [d.name for d in word.semantic_domains]
                if 'MOTION' in domains or 'PERCEPTION' in domains or 'CONSUMPTION' in domains:
                    action_verbs.append(lemma)
                else:
                    action_verbs.append(lemma)  # Include all for now
        
        sentence = []
        for role in word_order[:length]:
            if role == 'SUBJ':
                # Prefer animate subjects
                if animate_subject and animate_nouns:
                    word = np.random.choice(animate_nouns)
                elif self.registered_nouns:
                    word = np.random.choice(list(self.registered_nouns))
                else:
                    continue
                sentence.append(word)
                
            elif role == 'VERB':
                if action_verbs:
                    word = np.random.choice(action_verbs)
                elif self.registered_verbs:
                    word = np.random.choice(list(self.registered_verbs))
                else:
                    continue
                sentence.append(word)
                
            elif role == 'OBJ':
                # Prefer inanimate objects
                available = [n for n in inanimate_nouns if n not in sentence]
                if not available:
                    available = [n for n in self.registered_nouns if n not in sentence]
                if available:
                    word = np.random.choice(available)
                    sentence.append(word)
        
        return sentence


def demo():
    """Demo the integrated trainer"""
    print("="*70)
    print("INTEGRATED NEMO TRAINER DEMO")
    print("="*70)
    
    # Create trainer
    params = NemoParams(n=10000)  # Smaller for faster demo
    trainer = IntegratedNemoTrainer(params, verbose=True)
    
    # Train through curriculum (more epochs for better learning)
    results = trainer.train_full_curriculum(epochs_per_stage=5)
    
    # Generate some sentences
    print("\n" + "="*70)
    print("GENERATED SENTENCES")
    print("="*70)
    
    for i in range(5):
        sentence = trainer.generate_sentence(3)
        print(f"  {i+1}. {' '.join(sentence)}")
    
    # Show final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"  Registered nouns: {len(trainer.registered_nouns)}")
    print(f"  Registered verbs: {len(trainer.registered_verbs)}")
    print(f"  Sentences seen: {trainer.learner.sentences_seen}")
    print(f"  Word order learned: {trainer.learner.get_learned_word_order()}")
    
    # Show learning curve
    print("\n" + "="*70)
    print("LEARNING CURVE")
    print("="*70)
    print(f"{'Stage':>6} {'Sentences':>10} {'Words':>8} {'Noun Acc':>10} {'Verb Acc':>10}")
    print("-"*50)
    for stats in results:
        print(f"{stats.stage:>6} {stats.sentences_seen:>10} {stats.words_learned:>8} "
              f"{stats.noun_accuracy:>10.1%} {stats.verb_accuracy:>10.1%}")


if __name__ == "__main__":
    demo()

