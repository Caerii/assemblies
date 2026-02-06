"""
NEMO Corrected Implementation

Based on the notebook code, fixing:
1. Weight update formula (prevents explosion)
2. Noise during inference
3. Proper inhibit/reset
4. Multiple recurrence rounds
5. Sparse matrices for efficiency
"""

import numpy as np
from scipy.sparse import csr_array, lil_array
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def k_cap(inputs: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k neurons"""
    return np.argpartition(inputs, -k)[-k:]


def random_sparse_array(n_rows: int, n_cols: int, density: float) -> csr_array:
    """Create sparse random weight matrix"""
    # Use lil_array for efficient construction
    W = lil_array((n_rows, n_cols), dtype=np.float32)
    
    # Random sparse connections
    n_connections = int(n_rows * n_cols * density)
    rows = np.random.randint(0, n_rows, n_connections)
    cols = np.random.randint(0, n_cols, n_connections)
    
    for r, c in zip(rows, cols):
        W[r, c] = 1.0
    
    return W.tocsr()


class NemoArea:
    """
    A single brain area with recurrent connections.
    
    Based on RecurrentArea from the notebook.
    """
    
    def __init__(self, n_inputs: int, n_neurons: int, cap_size: int, 
                 density: float = 0.1, plasticity: float = 0.1):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.k = cap_size
        self.density = density
        self.beta = plasticity
        
        # Input weights (sparse)
        self.W_inp = random_sparse_array(n_inputs, n_neurons, density)
        
        # Recurrent weights (sparse)
        self.W_rec = random_sparse_array(n_neurons, n_neurons, density)
        
        # Current activation (indices of firing neurons)
        self.activated: Optional[np.ndarray] = None
        
        # Weight update tracking
        self.weight_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Parameters for weight formula
        self.lam = 24
        self.beta_weight = 1000
    
    def get_weight(self, n_rounds: int) -> float:
        """
        Weight formula that converges instead of exploding.
        From the notebook code.
        """
        if n_rounds == 0:
            return 1.0
        w = 1 + np.log(self.beta_weight * self.lam) / self.lam
        for _ in range(n_rounds - 1):
            w += self.beta_weight * np.exp(-self.lam * (w - 1))
        return w
    
    def inhibit(self):
        """Reset/clear the area"""
        self.activated = None
    
    def forward(self, input_indices: np.ndarray, update: bool = True, 
                noise: float = 0.0) -> np.ndarray:
        """
        Process input and update activation.
        
        Args:
            input_indices: Indices of active input neurons
            update: Whether to apply Hebbian learning
            noise: Amount of noise to add (for generation)
        """
        # Compute input from external input
        total_input = np.zeros(self.n_neurons, dtype=np.float32)
        
        # Sum input weights from active inputs
        for idx in input_indices:
            if idx < self.n_inputs:
                total_input += self.W_inp[idx].toarray().flatten()
        
        # Add recurrent input if we have previous activation
        if self.activated is not None:
            for idx in self.activated:
                total_input += self.W_rec[idx].toarray().flatten()
        
        # Add noise (calibrated as in notebook)
        if noise > 0:
            randomness = noise * np.sqrt(self.k * self.density) * np.random.randn(self.n_neurons)
            total_input += randomness
        
        # k-cap: select top-k neurons
        new_activated = k_cap(total_input, self.k)
        
        # Hebbian update using the weight formula
        if update and self.activated is not None:
            for src in self.activated:
                for dst in new_activated:
                    key = (src, dst)
                    self.weight_counts[key] += 1
                    new_weight = self.get_weight(self.weight_counts[key])
                    # Update recurrent weight
                    self.W_rec[src, dst] = new_weight
        
        self.activated = new_activated
        return new_activated
    
    def read(self, dense: bool = False) -> np.ndarray:
        """Read current activation"""
        if self.activated is None:
            if dense:
                return np.zeros(self.n_neurons)
            return np.array([])
        
        if dense:
            vec = np.zeros(self.n_neurons)
            vec[self.activated] = 1.0
            return vec
        return self.activated


class NemoLanguageSystem:
    """
    Language learning system using corrected NEMO.
    
    Architecture:
    - Phon: Phonological input
    - Visual: Visual input (for nouns)
    - Motor: Motor input (for verbs)
    - Lex1: Lexical area for nouns
    - Lex2: Lexical area for verbs
    """
    
    def __init__(self, n_neurons: int = 10000, cap_size: int = 50, 
                 density: float = 0.1, plasticity: float = 0.1):
        self.n = n_neurons
        self.k = cap_size
        
        # Input areas (just store assemblies, no learning)
        self.phon_assemblies: Dict[str, np.ndarray] = {}
        self.visual_assemblies: Dict[str, np.ndarray] = {}
        self.motor_assemblies: Dict[str, np.ndarray] = {}
        
        # Lexical areas
        self.lex1 = NemoArea(n_neurons * 2, n_neurons, cap_size, density, plasticity)
        self.lex2 = NemoArea(n_neurons * 2, n_neurons, cap_size, density, plasticity)
        
        # Strong connections (higher density for Lex1-Visual, Lex2-Motor)
        # This is implicit in how we route inputs
        
        self.sentences_seen = 0
    
    def create_assembly(self, word: str, area: str) -> np.ndarray:
        """Create random assembly for a word"""
        indices = np.random.choice(self.n, self.k, replace=False)
        
        if area == 'Phon':
            self.phon_assemblies[word] = indices
        elif area == 'Visual':
            self.visual_assemblies[word] = indices
        elif area == 'Motor':
            self.motor_assemblies[word] = indices
        
        return indices
    
    def present_grounded_sentence(self, noun: str, verb: str, n_rounds: int = 5):
        """
        Present a grounded sentence.
        
        Key insight: Visual/Motor fire throughout, words presented sequentially.
        """
        # Ensure assemblies exist
        if noun not in self.phon_assemblies:
            self.create_assembly(noun, 'Phon')
        if noun not in self.visual_assemblies:
            self.create_assembly(noun, 'Visual')
        if verb not in self.phon_assemblies:
            self.create_assembly(verb, 'Phon')
        if verb not in self.motor_assemblies:
            self.create_assembly(verb, 'Motor')
        
        # Get assemblies
        phon_noun = self.phon_assemblies[noun]
        phon_verb = self.phon_assemblies[verb]
        visual = self.visual_assemblies[noun]
        motor = self.motor_assemblies[verb]
        
        # Present noun
        self.lex1.inhibit()
        self.lex2.inhibit()
        
        # Combine Phon[noun] + Visual (for Lex1)
        # Combine Phon[noun] + Motor (for Lex2) - but Motor is weaker for nouns
        noun_input_lex1 = np.concatenate([phon_noun, visual + self.n])
        noun_input_lex2 = np.concatenate([phon_noun, motor + self.n])
        
        for _ in range(n_rounds):
            self.lex1.forward(noun_input_lex1, update=True)
            self.lex2.forward(noun_input_lex2, update=True)
        
        # Present verb
        # Combine Phon[verb] + Visual (weaker for verbs)
        # Combine Phon[verb] + Motor (for Lex2)
        verb_input_lex1 = np.concatenate([phon_verb, visual + self.n])
        verb_input_lex2 = np.concatenate([phon_verb, motor + self.n])
        
        for _ in range(n_rounds):
            self.lex1.forward(verb_input_lex1, update=True)
            self.lex2.forward(verb_input_lex2, update=True)
        
        self.sentences_seen += 1
    
    def measure_stability(self, word: str, area: str, n_rounds: int = 10) -> float:
        """
        Measure assembly stability.
        
        Stable = same neurons keep firing after recurrence.
        """
        if word not in self.phon_assemblies:
            return 0.0
        
        phon = self.phon_assemblies[word]
        lex = self.lex1 if area == 'Lex1' else self.lex2
        
        # Get semantic grounding
        if word in self.visual_assemblies:
            sem = self.visual_assemblies[word]
        elif word in self.motor_assemblies:
            sem = self.motor_assemblies[word]
        else:
            sem = np.array([])
        
        # Combine input
        if len(sem) > 0:
            combined_input = np.concatenate([phon, sem + self.n])
        else:
            combined_input = phon
        
        # Fire and measure stability
        lex.inhibit()
        
        first_activation = None
        last_activation = None
        
        for i in range(n_rounds):
            lex.forward(combined_input, update=False, noise=0)
            
            if i == 0:
                first_activation = set(lex.activated)
            last_activation = set(lex.activated)
        
        if first_activation is None or last_activation is None:
            return 0.0
        
        overlap = len(first_activation & last_activation) / self.k
        return overlap
    
    def classify_word(self, word: str) -> str:
        """Classify word as NOUN or VERB based on stability"""
        s1 = self.measure_stability(word, 'Lex1')
        s2 = self.measure_stability(word, 'Lex2')
        return 'NOUN' if s1 > s2 else 'VERB'


def run_experiment():
    print("=" * 70)
    print("NEMO CORRECTED IMPLEMENTATION")
    print("=" * 70)
    
    # Create system
    system = NemoLanguageSystem(n_neurons=5000, cap_size=50, density=0.1)
    
    # Define words
    nouns = ['dog', 'cat', 'bird', 'ball', 'baby']
    verbs = ['runs', 'walks', 'sleeps', 'eats', 'plays']
    
    # Create assemblies
    for word in nouns:
        system.create_assembly(word, 'Phon')
        system.create_assembly(word, 'Visual')
    for word in verbs:
        system.create_assembly(word, 'Phon')
        system.create_assembly(word, 'Motor')
    
    # Test before training
    print("\nBEFORE TRAINING:")
    print(f"{'Word':>10} {'Lex1':>8} {'Lex2':>8} {'Pred':>8}")
    print("-" * 40)
    
    for word in nouns[:3] + verbs[:3]:
        s1 = system.measure_stability(word, 'Lex1')
        s2 = system.measure_stability(word, 'Lex2')
        pred = 'NOUN' if s1 > s2 else 'VERB'
        print(f"{word:>10} {s1:>8.2f} {s2:>8.2f} {pred:>8}")
    
    # Train
    print("\nTRAINING...")
    n_sentences = 100
    for i in range(n_sentences):
        noun = np.random.choice(nouns)
        verb = np.random.choice(verbs)
        system.present_grounded_sentence(noun, verb)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_sentences} sentences")
    
    # Test after training
    print("\nAFTER TRAINING:")
    print(f"{'Word':>10} {'Type':>6} {'Lex1':>8} {'Lex2':>8} {'Pred':>8} {'OK':>4}")
    print("-" * 50)
    
    correct = 0
    for word in nouns + verbs:
        is_noun = word in nouns
        s1 = system.measure_stability(word, 'Lex1')
        s2 = system.measure_stability(word, 'Lex2')
        pred = 'NOUN' if s1 > s2 else 'VERB'
        
        true_type = 'NOUN' if is_noun else 'VERB'
        ok = '✓' if pred == true_type else '✗'
        if pred == true_type:
            correct += 1
        
        print(f"{word:>10} {true_type:>6} {s1:>8.2f} {s2:>8.2f} {pred:>8} {ok:>4}")
    
    print(f"\nAccuracy: {correct}/{len(nouns)+len(verbs)} = {correct/(len(nouns)+len(verbs)):.1%}")
    
    # Check weight statistics
    print("\nWEIGHT STATISTICS:")
    print(f"  Lex1 recurrent weights updated: {len(system.lex1.weight_counts)}")
    print(f"  Lex2 recurrent weights updated: {len(system.lex2.weight_counts)}")
    
    if system.lex1.weight_counts:
        max_count = max(system.lex1.weight_counts.values())
        max_weight = system.lex1.get_weight(max_count)
        print(f"  Max update count: {max_count}, corresponding weight: {max_weight:.2f}")


if __name__ == "__main__":
    run_experiment()

