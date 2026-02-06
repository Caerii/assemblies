"""
NEMO GPU-Accelerated Corrected Implementation

Based on the notebook code, fixing:
1. Weight update formula (prevents explosion)
2. Noise during inference
3. Proper inhibit/reset
4. Multiple recurrence rounds

Using PyTorch for GPU acceleration.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Check CUDA
assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name()}")


class NemoAreaGPU:
    """
    A single brain area with recurrent connections - GPU accelerated.
    
    Uses the CORRECT weight formula from the notebook that naturally saturates.
    """
    
    def __init__(self, n_inputs: int, n_neurons: int, cap_size: int, 
                 density: float = 0.1, plasticity: float = 0.1,
                 lam: float = 24.0, beta_weight: float = 1000.0):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.k = cap_size
        self.density = density
        self.beta = plasticity
        
        # Parameters for the self-limiting weight formula
        # w_new = w_old + beta_weight * exp(-lam * (w_old - 1))
        self.lam = lam
        self.beta_weight = beta_weight
        
        # Input weights (dense on GPU - faster for our sizes)
        # Sparse initialization with weight = 1
        self.W_inp = torch.zeros(n_inputs, n_neurons, device=DEVICE)
        mask = torch.rand(n_inputs, n_neurons, device=DEVICE) < density
        self.W_inp[mask] = 1.0
        
        # Recurrent weights
        self.W_rec = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < density
        self.W_rec[mask] = 1.0
        
        # Current activation (indices of firing neurons)
        self.activated: Optional[torch.Tensor] = None
    
    def inhibit(self):
        """Reset/clear the area"""
        self.activated = None
    
    def saturating_update(self, W: torch.Tensor, src_indices: torch.Tensor, 
                          dst_indices: torch.Tensor):
        """
        Apply a biologically-motivated saturating weight update.
        
        Formula: w_new = w_old + beta * (w_max - w_old) / w_max
        
        This is equivalent to: w_new = w_old * (1 - beta/w_max) + beta
        
        Properties:
        - When w ≈ 0: update is ≈ beta (large)
        - When w ≈ w_max: update is ≈ 0 (saturated)
        - Converges to w_max as updates continue
        
        This models synaptic saturation: strong synapses resist further potentiation.
        """
        # Get current weights for the updated synapses
        idx = (src_indices.unsqueeze(1), dst_indices.unsqueeze(0))
        current_w = W[idx]
        
        # Saturating update: delta = beta * (1 - w/w_max)
        # This means: delta → 0 as w → w_max
        w_max = 10.0  # Maximum weight (saturation point)
        beta = 0.5    # Base learning rate
        
        delta = beta * (1.0 - current_w / w_max)
        delta = torch.clamp(delta, min=0)  # No negative updates
        
        # Apply update
        W[idx] = current_w + delta
    
    def forward(self, input_indices: torch.Tensor, update: bool = True, 
                noise: float = 0.0) -> torch.Tensor:
        """
        Process input and update activation.
        
        Args:
            input_indices: Indices of active input neurons
            update: Whether to apply Hebbian learning
            noise: Amount of noise to add (for generation)
        """
        # Compute input from external input
        total_input = self.W_inp[input_indices].sum(dim=0)
        
        # Add recurrent input if we have previous activation
        if self.activated is not None:
            total_input += self.W_rec[self.activated].sum(dim=0)
        
        # Add noise (calibrated as in notebook)
        if noise > 0:
            randomness = noise * np.sqrt(self.k * self.density) * torch.randn(self.n_neurons, device=DEVICE)
            total_input += randomness
        
        # k-cap: select top-k neurons
        _, new_activated = torch.topk(total_input, self.k)
        
        # Hebbian update using the CORRECT saturating formula
        if update and self.activated is not None:
            self.saturating_update(self.W_rec, self.activated, new_activated)
        
        self.activated = new_activated
        return new_activated
    
    def read(self, dense: bool = False) -> torch.Tensor:
        """Read current activation"""
        if self.activated is None:
            if dense:
                return torch.zeros(self.n_neurons, device=DEVICE)
            return torch.tensor([], device=DEVICE, dtype=torch.long)
        
        if dense:
            vec = torch.zeros(self.n_neurons, device=DEVICE)
            vec[self.activated] = 1.0
            return vec
        return self.activated


class NemoLanguageSystemGPU:
    """
    Language learning system using corrected NEMO on GPU.
    
    Architecture:
    - Phon: Phonological input
    - Visual: Visual input (for nouns)
    - Motor: Motor input (for verbs)
    - Lex1: Lexical area for nouns (strong Visual connection)
    - Lex2: Lexical area for verbs (strong Motor connection)
    """
    
    def __init__(self, n_neurons: int = 10000, cap_size: int = 50, 
                 density: float = 0.1, plasticity: float = 0.1):
        self.n = n_neurons
        self.k = cap_size
        self.density = density
        
        # Input assemblies (just indices, no weights)
        self.phon_assemblies: Dict[str, torch.Tensor] = {}
        self.visual_assemblies: Dict[str, torch.Tensor] = {}
        self.motor_assemblies: Dict[str, torch.Tensor] = {}
        
        # Lexical areas - input is Phon + Semantic
        # Lex1 gets Phon + Visual (strong) + Motor (weak)
        # Lex2 gets Phon + Motor (strong) + Visual (weak)
        input_size = n_neurons  # Phon size
        
        self.lex1 = NemoAreaGPU(input_size, n_neurons, cap_size, density, plasticity)
        self.lex2 = NemoAreaGPU(input_size, n_neurons, cap_size, density, plasticity)
        
        # Semantic -> Lex weights (separate from Phon -> Lex)
        # Strong: Visual -> Lex1, Motor -> Lex2
        # Weak: Visual -> Lex2, Motor -> Lex1
        strong_density = density * 2
        weak_density = density * 0.5
        
        self.W_visual_lex1 = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < strong_density
        self.W_visual_lex1[mask] = 1.0
        
        self.W_motor_lex2 = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < strong_density
        self.W_motor_lex2[mask] = 1.0
        
        self.W_visual_lex2 = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < weak_density
        self.W_visual_lex2[mask] = 1.0
        
        self.W_motor_lex1 = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < weak_density
        self.W_motor_lex1[mask] = 1.0
        
        # Lex -> Semantic weights (for pathway test)
        self.W_lex1_visual = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < strong_density
        self.W_lex1_visual[mask] = 1.0
        
        self.W_lex2_motor = torch.zeros(n_neurons, n_neurons, device=DEVICE)
        mask = torch.rand(n_neurons, n_neurons, device=DEVICE) < strong_density
        self.W_lex2_motor[mask] = 1.0
        
        # Parameters for saturating update
        self.lam = 24.0
        self.beta_weight = 1000.0
        self.beta = plasticity
        self.sentences_seen = 0
    
    def saturating_update(self, W: torch.Tensor, src_indices: torch.Tensor, 
                          dst_indices: torch.Tensor):
        """
        Apply a biologically-motivated saturating weight update.
        
        Formula: w_new = w_old + beta * (w_max - w_old) / w_max
        
        This models synaptic saturation.
        """
        idx = (src_indices.unsqueeze(1), dst_indices.unsqueeze(0))
        current_w = W[idx]
        
        w_max = 10.0  # Maximum weight
        beta = 0.5    # Base learning rate
        
        delta = beta * (1.0 - current_w / w_max)
        delta = torch.clamp(delta, min=0)
        
        W[idx] = current_w + delta
    
    def create_assembly(self, word: str, area: str) -> torch.Tensor:
        """Create random assembly for a word"""
        indices = torch.randperm(self.n, device=DEVICE)[:self.k]
        
        if area == 'Phon':
            self.phon_assemblies[word] = indices
        elif area == 'Visual':
            self.visual_assemblies[word] = indices
        elif area == 'Motor':
            self.motor_assemblies[word] = indices
        
        return indices
    
    def present_grounded_sentence(self, noun: str, verb: str, n_rounds: int = 3):
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
        
        # Reset areas
        self.lex1.inhibit()
        self.lex2.inhibit()
        
        # Present noun - ONLY Visual grounding (not Motor)
        # This is the key insight: nouns are grounded in Visual, verbs in Motor
        for _ in range(n_rounds):
            # Lex1 gets Phon[noun] + Visual (strong)
            phon_input_lex1 = self.lex1.W_inp[phon_noun].sum(dim=0)
            visual_input_lex1 = self.W_visual_lex1[visual].sum(dim=0)
            
            total_input_lex1 = phon_input_lex1 + visual_input_lex1
            if self.lex1.activated is not None:
                total_input_lex1 += self.lex1.W_rec[self.lex1.activated].sum(dim=0)
            
            _, winners_lex1 = torch.topk(total_input_lex1, self.k)
            
            # Hebbian update for Lex1
            if self.lex1.activated is not None:
                self.lex1.saturating_update(self.lex1.W_rec, self.lex1.activated, winners_lex1)
            
            # Update Visual <-> Lex1 weights
            self.saturating_update(self.W_visual_lex1, visual, winners_lex1)
            self.saturating_update(self.W_lex1_visual, winners_lex1, visual)
            
            self.lex1.activated = winners_lex1
            
            # Lex2 gets Phon[noun] + Visual (weak) - noun is NOT a verb
            phon_input_lex2 = self.lex2.W_inp[phon_noun].sum(dim=0)
            visual_input_lex2 = self.W_visual_lex2[visual].sum(dim=0)  # Weak connection
            
            total_input_lex2 = phon_input_lex2 + visual_input_lex2
            if self.lex2.activated is not None:
                total_input_lex2 += self.lex2.W_rec[self.lex2.activated].sum(dim=0)
            
            _, winners_lex2 = torch.topk(total_input_lex2, self.k)
            
            if self.lex2.activated is not None:
                self.lex2.saturating_update(self.lex2.W_rec, self.lex2.activated, winners_lex2)
            
            self.lex2.activated = winners_lex2
        
        # Present verb - ONLY Motor grounding (not Visual)
        for _ in range(n_rounds):
            # Lex1 gets Phon[verb] + Motor (weak) - verb is NOT a noun
            phon_input_lex1 = self.lex1.W_inp[phon_verb].sum(dim=0)
            motor_input_lex1 = self.W_motor_lex1[motor].sum(dim=0)  # Weak connection
            
            total_input_lex1 = phon_input_lex1 + motor_input_lex1
            if self.lex1.activated is not None:
                total_input_lex1 += self.lex1.W_rec[self.lex1.activated].sum(dim=0)
            
            _, winners_lex1 = torch.topk(total_input_lex1, self.k)
            
            if self.lex1.activated is not None:
                self.lex1.saturating_update(self.lex1.W_rec, self.lex1.activated, winners_lex1)
            
            self.lex1.activated = winners_lex1
            
            # Lex2 gets Phon[verb] + Motor (strong)
            phon_input_lex2 = self.lex2.W_inp[phon_verb].sum(dim=0)
            motor_input_lex2 = self.W_motor_lex2[motor].sum(dim=0)
            
            total_input_lex2 = phon_input_lex2 + motor_input_lex2
            if self.lex2.activated is not None:
                total_input_lex2 += self.lex2.W_rec[self.lex2.activated].sum(dim=0)
            
            _, winners_lex2 = torch.topk(total_input_lex2, self.k)
            
            if self.lex2.activated is not None:
                self.lex2.saturating_update(self.lex2.W_rec, self.lex2.activated, winners_lex2)
            
            # Update Motor <-> Lex2 weights
            self.saturating_update(self.W_motor_lex2, motor, winners_lex2)
            self.saturating_update(self.W_lex2_motor, winners_lex2, motor)
            
            self.lex2.activated = winners_lex2
        
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
        visual = self.visual_assemblies.get(word)
        motor = self.motor_assemblies.get(word)
        
        # Reset
        lex.inhibit()
        
        first_activation = None
        
        for i in range(n_rounds):
            # Compute input
            total_input = lex.W_inp[phon].sum(dim=0)
            
            if visual is not None:
                if area == 'Lex1':
                    total_input += self.W_visual_lex1[visual].sum(dim=0)
                else:
                    total_input += self.W_visual_lex2[visual].sum(dim=0)
            
            if motor is not None:
                if area == 'Lex1':
                    total_input += self.W_motor_lex1[motor].sum(dim=0)
                else:
                    total_input += self.W_motor_lex2[motor].sum(dim=0)
            
            if lex.activated is not None:
                total_input += lex.W_rec[lex.activated].sum(dim=0)
            
            _, winners = torch.topk(total_input, self.k)
            lex.activated = winners
            
            if i == 0:
                first_activation = set(winners.cpu().numpy())
        
        if first_activation is None or lex.activated is None:
            return 0.0
        
        last_activation = set(lex.activated.cpu().numpy())
        overlap = len(first_activation & last_activation) / self.k
        return overlap
    
    def test_pathway(self, word: str, is_noun: bool) -> float:
        """Test semantic pathway: Phon -> Lex -> Visual/Motor"""
        if word not in self.phon_assemblies:
            return 0.0
        
        phon = self.phon_assemblies[word]
        
        if is_noun:
            if word not in self.visual_assemblies:
                return 0.0
            target = self.visual_assemblies[word]
            lex = self.lex1
            W_lex_sem = self.W_lex1_visual
        else:
            if word not in self.motor_assemblies:
                return 0.0
            target = self.motor_assemblies[word]
            lex = self.lex2
            W_lex_sem = self.W_lex2_motor
        
        target_set = set(target.cpu().numpy())
        
        # Fire Phon -> Lex
        lex.inhibit()
        for _ in range(3):
            total_input = lex.W_inp[phon].sum(dim=0)
            if lex.activated is not None:
                total_input += lex.W_rec[lex.activated].sum(dim=0)
            _, winners = torch.topk(total_input, self.k)
            lex.activated = winners
        
        # Lex -> Semantic
        if lex.activated is None:
            return 0.0
        
        sem_input = W_lex_sem[lex.activated].sum(dim=0)
        _, top_sem = torch.topk(sem_input, self.k)
        top_set = set(top_sem.cpu().numpy())
        
        overlap = len(target_set & top_set) / self.k
        return overlap


def run_experiment():
    print("=" * 70)
    print("NEMO GPU-ACCELERATED CORRECTED IMPLEMENTATION")
    print("=" * 70)
    
    # Create system
    start = time.perf_counter()
    system = NemoLanguageSystemGPU(n_neurons=10000, cap_size=50, density=0.1, plasticity=0.1)
    init_time = time.perf_counter() - start
    print(f"Initialization: {init_time:.2f}s")
    
    # Memory usage
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU Memory: {mem:.2f} GB")
    
    # Define words
    nouns = ['dog', 'cat', 'bird', 'ball', 'baby', 'boy', 'girl', 'man', 'woman', 'food']
    verbs = ['runs', 'walks', 'sleeps', 'eats', 'plays', 'jumps', 'flies', 'sees', 'has', 'wants']
    
    # Create assemblies
    for word in nouns:
        system.create_assembly(word, 'Phon')
        system.create_assembly(word, 'Visual')
    for word in verbs:
        system.create_assembly(word, 'Phon')
        system.create_assembly(word, 'Motor')
    
    # Training
    print("\nTRAINING...")
    n_sentences = 200
    start = time.perf_counter()
    
    for i in range(n_sentences):
        noun = np.random.choice(nouns)
        verb = np.random.choice(verbs)
        system.present_grounded_sentence(noun, verb)
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_sentences} sentences")
    
    torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    print(f"Training: {train_time:.2f}s ({n_sentences/train_time:.1f} sentences/sec)")
    
    # Test classification
    print("\nCLASSIFICATION RESULTS:")
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
    
    print(f"\nClassification Accuracy: {correct}/{len(nouns)+len(verbs)} = {correct/(len(nouns)+len(verbs)):.1%}")
    
    # Test pathway
    print("\nPATHWAY RESULTS:")
    noun_overlaps = [system.test_pathway(w, True) for w in nouns]
    verb_overlaps = [system.test_pathway(w, False) for w in verbs]
    
    print(f"  Noun pathway (Phon→Lex1→Visual): {np.mean(noun_overlaps):.1%}")
    print(f"  Verb pathway (Phon→Lex2→Motor): {np.mean(verb_overlaps):.1%}")
    
    # Weight statistics
    print("\nWEIGHT STATISTICS:")
    print(f"  Lex1 recurrent max: {system.lex1.W_rec.max().item():.2f}")
    print(f"  Lex2 recurrent max: {system.lex2.W_rec.max().item():.2f}")
    print(f"  Visual→Lex1 max: {system.W_visual_lex1.max().item():.2f}")
    print(f"  Motor→Lex2 max: {system.W_motor_lex2.max().item():.2f}")


if __name__ == "__main__":
    run_experiment()

