#!/usr/bin/env python3
"""
Assembly Calculus Text Generator
================================

A prototype text generator using neural assemblies.

Architecture:
- Each word/token = one assembly (k neurons)
- Sequence learning via cross-area projection
- Generation via pattern completion

This is a research prototype to explore if Assembly Calculus
can perform meaningful text generation.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

try:
    import brain as b
except ImportError:
    print("Warning: brain.py not found, using mock implementation")
    b = None


@dataclass
class AssemblyConfig:
    """Configuration for assembly-based text generation"""
    n_neurons: int = 10000      # Neurons per area
    k_active: int = 100         # Active neurons per assembly (sqrt(n))
    n_areas: int = 3            # Lexicon, Sequence, Context
    beta: float = 0.1           # Learning rate
    p_connect: float = 0.1      # Connection probability
    seed: int = 42


class AssemblyTokenizer:
    """Maps tokens to/from neural assemblies"""
    
    def __init__(self, config: AssemblyConfig):
        self.config = config
        self.token_to_pattern: Dict[str, np.ndarray] = {}
        self.pattern_cache: Dict[str, Set[int]] = {}
        self.rng = np.random.default_rng(config.seed)
        
    def encode(self, token: str) -> np.ndarray:
        """Convert token to stimulus pattern (k active neurons)"""
        if token not in self.token_to_pattern:
            # Deterministic pattern from token hash
            token_seed = hash(token) % (2**31)
            token_rng = np.random.default_rng(token_seed)
            pattern = token_rng.choice(
                self.config.n_neurons, 
                self.config.k_active, 
                replace=False
            ).astype(np.uint32)
            self.token_to_pattern[token] = pattern
            self.pattern_cache[token] = set(pattern)
        return self.token_to_pattern[token]
    
    def decode(self, assembly: np.ndarray) -> Tuple[str, float]:
        """Find closest token to given assembly, return (token, confidence)"""
        if len(self.token_to_pattern) == 0:
            return ("<UNK>", 0.0)
        
        assembly_set = set(assembly)
        best_token = "<UNK>"
        best_overlap = 0
        
        for token, pattern_set in self.pattern_cache.items():
            overlap = len(assembly_set & pattern_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_token = token
        
        confidence = best_overlap / self.config.k_active
        return (best_token, confidence)
    
    def get_vocabulary(self) -> List[str]:
        """Return all known tokens"""
        return list(self.token_to_pattern.keys())


class AssemblyBrain:
    """Wrapper around brain.py for text generation"""
    
    def __init__(self, config: AssemblyConfig):
        self.config = config
        
        if b is None:
            raise RuntimeError("brain.py not available")
        
        # Create brain with explicit areas for pattern completion
        self.brain = b.Brain(p=config.p_connect, seed=config.seed)
        
        # Add areas - explicit areas support pattern completion
        # Lexicon: where word assemblies live
        self.brain.add_explicit_area("LEX", config.n_neurons, config.k_active, config.beta)
        
        # Note: brain.py creates fibers implicitly during projection
        # No need to call add_fiber explicitly
        
        # Track assemblies
        self.word_assemblies: Dict[str, np.ndarray] = {}
        
    def create_word_assembly(self, word: str, stimulus: np.ndarray) -> np.ndarray:
        """Create or retrieve assembly for a word - use deterministic stimulus pattern"""
        if word in self.word_assemblies:
            return self.word_assemblies[word]
        
        # Use the stimulus pattern directly as the assembly
        # This ensures tokenizer.decode will work correctly
        assembly = stimulus.copy()
        self.word_assemblies[word] = assembly
        
        return assembly
    
    def associate_words(self, word1: str, word2: str):
        """Learn that word1 is followed by word2 using recurrent connections"""
        if word1 not in self.word_assemblies or word2 not in self.word_assemblies:
            return
        
        # Get the LEX->LEX weight matrix
        # connectomes[source][target] gives the weight matrix
        weights = self.brain.connectomes["LEX"]["LEX"]
        
        w1_neurons = self.word_assemblies[word1]
        w2_neurons = self.word_assemblies[word2]
        
        # Direct Hebbian update: strengthen word1 -> word2 connections
        # weights[i, j] = connection from neuron i to neuron j
        for i in w1_neurons:
            for j in w2_neurons:
                weights[i, j] += self.config.beta
    
    def predict_next(self, current_word: str) -> np.ndarray:
        """Given current word, predict next word's assembly via learned weights"""
        if current_word not in self.word_assemblies:
            return np.array([], dtype=np.uint32)
        
        # Get the LEX->LEX weight matrix
        weights = self.brain.connectomes["LEX"]["LEX"]
        
        # Get current word's neurons
        current_neurons = self.word_assemblies[current_word]
        
        # Compute activation for all neurons based on connections from current word
        # activation[j] = sum over i in current_neurons of weights[i, j]
        activations = np.zeros(self.config.n_neurons, dtype=np.float32)
        for i in current_neurons:
            activations += weights[i, :]
        
        # Select top-k neurons as the predicted next assembly
        top_k_indices = np.argpartition(activations, -self.config.k_active)[-self.config.k_active:]
        
        return top_k_indices.astype(np.uint32)


class AssemblyTextGenerator:
    """Main text generation class"""
    
    def __init__(self, config: Optional[AssemblyConfig] = None):
        self.config = config or AssemblyConfig()
        self.tokenizer = AssemblyTokenizer(self.config)
        self.brain = AssemblyBrain(self.config)
        self.trained = False
        
    def train(self, corpus: List[str], verbose: bool = True):
        """Train on a corpus of sentences"""
        if verbose:
            print(f"Training on {len(corpus)} sentences...")
        
        # First pass: create all word assemblies
        all_words = set()
        for sentence in corpus:
            words = sentence.lower().split()
            all_words.update(words)
        
        if verbose:
            print(f"Vocabulary size: {len(all_words)}")
        
        for word in all_words:
            stimulus = self.tokenizer.encode(word)
            self.brain.create_word_assembly(word, stimulus)
        
        if verbose:
            print(f"Created {len(all_words)} word assemblies")
        
        # Second pass: learn transitions
        n_transitions = 0
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                self.brain.associate_words(words[i], words[i+1])
                n_transitions += 1
        
        if verbose:
            print(f"Learned {n_transitions} transitions")
        
        self.trained = True
    
    def generate(self, prompt: str, max_tokens: int = 20, verbose: bool = False,
                 no_repeat_ngram: int = 2) -> str:
        """Generate text continuation from prompt
        
        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            verbose: Print step-by-step predictions
            no_repeat_ngram: Prevent repeating n-grams of this size
        """
        if not self.trained:
            raise RuntimeError("Model not trained!")
        
        words = prompt.lower().split()
        generated = list(words)
        
        if len(words) == 0:
            return ""
        
        current_word = words[-1]
        recent_words = list(words[-no_repeat_ngram:]) if len(words) >= no_repeat_ngram else list(words)
        
        for i in range(max_tokens):
            # Predict next assembly
            next_assembly = self.brain.predict_next(current_word)
            
            if len(next_assembly) == 0:
                if verbose:
                    print(f"  Step {i}: No prediction for '{current_word}'")
                break
            
            # Decode to word
            next_word, confidence = self.tokenizer.decode(next_assembly)
            
            if verbose:
                print(f"  Step {i}: '{current_word}' -> '{next_word}' (conf={confidence:.2f})")
            
            if next_word == "<UNK>" or confidence < 0.1:
                break
            
            # Check for n-gram repetition
            if no_repeat_ngram > 0:
                test_ngram = recent_words[-(no_repeat_ngram-1):] + [next_word] if len(recent_words) >= no_repeat_ngram-1 else [next_word]
                full_text = " ".join(generated)
                test_text = " ".join(test_ngram)
                if test_text in full_text:
                    if verbose:
                        print(f"  Step {i}: Stopping due to n-gram repetition")
                    break
            
            generated.append(next_word)
            recent_words.append(next_word)
            if len(recent_words) > no_repeat_ngram:
                recent_words.pop(0)
            current_word = next_word
        
        return " ".join(generated)


def demo():
    """Demo the text generator"""
    print("=" * 60)
    print("ASSEMBLY CALCULUS TEXT GENERATOR DEMO")
    print("=" * 60)
    print()
    
    # Simple training corpus
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat ran to the door",
        "the dog sat on the floor",
        "a cat is a pet",
        "a dog is a pet",
        "the pet sat on the mat",
        "the cat and the dog",
    ]
    
    print("Training corpus:")
    for s in corpus:
        print(f"  - {s}")
    print()
    
    # Create and train
    config = AssemblyConfig(
        n_neurons=5000,
        k_active=70,  # ~sqrt(5000)
        beta=0.2
    )
    
    generator = AssemblyTextGenerator(config)
    generator.train(corpus, verbose=True)
    print()
    
    # Generate
    prompts = ["the cat", "the dog", "a pet", "the"]
    
    print("Generation:")
    for prompt in prompts:
        result = generator.generate(prompt, max_tokens=10, verbose=True)
        print(f"  RESULT: '{prompt}' -> '{result}'")
    
    print()
    print("Done!")


if __name__ == "__main__":
    demo()

