"""
NEMO Diagnostics

Investigate why:
1. Verb accuracy is lower than noun accuracy
2. Semantic pathway doesn't work
3. What parameters affect learning

From paper Figure 3:
- n = 10^5, p = 0.05, β = 0.06, k_Lex = 50, k_Context = 20, k_other = 100, τ = 2
"""

import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

assert torch.cuda.is_available()
DEVICE = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name()}")


@dataclass  
class PaperParams:
    """Parameters from the paper Figure 3"""
    n: int = 100000          # 10^5 neurons per area
    p: float = 0.05          # Connection probability (paper says 0.05, not 0.01!)
    beta: float = 0.06       # Plasticity (paper says 0.06)
    k_lex: int = 50          # k for Lex areas
    k_context: int = 20      # k for context areas
    k_other: int = 100       # k for other areas
    tau: int = 2             # Firing steps per word


class SimplifiedNemoBrain:
    """
    Simplified NEMO brain focusing on the word learning experiment.
    
    Only the essential areas:
    - Phon (input)
    - Visual (input, for nouns)
    - Motor (input, for verbs)
    - Lex1 (learns nouns)
    - Lex2 (learns verbs)
    """
    
    def __init__(self, n: int = 10000, p: float = 0.05, beta: float = 0.06, 
                 k: int = 50, tau: int = 2, verbose: bool = True):
        self.n = n
        self.p = p
        self.beta = beta
        self.k = k
        self.tau = tau
        
        # Weight matrices
        # Paper: "Four fibers... have increased parameters β and p"
        # Strong: Phon↔Lex1, Phon↔Lex2, Lex1↔Visual, Lex2↔Motor
        
        self.W = {}
        
        # Strong fibers (higher p and β)
        strong_p = p * 2  # Double connection probability
        self.strong_fibers = [
            ('Phon', 'Lex1'), ('Lex1', 'Phon'),
            ('Phon', 'Lex2'), ('Lex2', 'Phon'),
            ('Visual', 'Lex1'), ('Lex1', 'Visual'),
            ('Motor', 'Lex2'), ('Lex2', 'Motor'),
        ]
        
        # Weak fibers (cross-connections)
        self.weak_fibers = [
            ('Visual', 'Lex2'), ('Lex2', 'Visual'),  # Visual to verb area (weak)
            ('Motor', 'Lex1'), ('Lex1', 'Motor'),    # Motor to noun area (weak)
        ]
        
        # Recurrent connections
        self.recurrent_fibers = [
            ('Lex1', 'Lex1'),
            ('Lex2', 'Lex2'),
        ]
        
        # Initialize
        for fiber in self.strong_fibers:
            self.W[fiber] = self._init_weights(strong_p)
        for fiber in self.weak_fibers:
            self.W[fiber] = self._init_weights(p)
        for fiber in self.recurrent_fibers:
            self.W[fiber] = self._init_weights(p)
        
        # Activations
        self.act = {}
        
        # Input assemblies
        self.input_assemblies = {'Phon': {}, 'Visual': {}, 'Motor': {}}
        
        if verbose:
            print(f"SimplifiedNemoBrain: n={n}, p={p}, β={beta}, k={k}")
            mem = len(self.W) * n * n * 4 / 1e9
            print(f"  Fibers: {len(self.W)}, Memory: {mem:.2f} GB")
    
    def _init_weights(self, p: float) -> torch.Tensor:
        """Sparse random initialization with weight=1"""
        W = torch.zeros(self.n, self.n, device=DEVICE)
        mask = torch.rand(self.n, self.n, device=DEVICE) < p
        W[mask] = 1.0
        return W
    
    def create_assembly(self, area: str, name: str) -> torch.Tensor:
        """Create random assembly"""
        if name not in self.input_assemblies[area]:
            indices = torch.randperm(self.n, device=DEVICE)[:self.k]
            self.input_assemblies[area][name] = indices
        return self.input_assemblies[area][name]
    
    def activate(self, area: str, name: str):
        """Activate an assembly"""
        indices = self.input_assemblies[area][name]
        act = torch.zeros(self.n, device=DEVICE)
        act[indices] = 1.0
        self.act[area] = act
    
    def fire_step(self, active_areas: List[str], plastic_areas: List[str]):
        """One firing step with k-cap and Hebbian learning"""
        new_act = {}
        
        for dst in active_areas:
            if dst in ['Phon', 'Visual', 'Motor']:
                # Input areas don't change
                if dst in self.act:
                    new_act[dst] = self.act[dst]
                continue
            
            # Sum inputs from all connected areas
            total_input = torch.zeros(self.n, device=DEVICE)
            
            for src in self.act:
                fiber = (src, dst)
                if fiber not in self.W:
                    continue
                total_input += self.W[fiber] @ self.act[src]
            
            if total_input.sum() == 0:
                continue
            
            # k-cap: top-k winners
            _, winners = torch.topk(total_input, self.k)
            
            act = torch.zeros(self.n, device=DEVICE)
            act[winners] = 1.0
            new_act[dst] = act
            
            # Hebbian learning
            if dst in plastic_areas:
                for src in self.act:
                    fiber = (src, dst)
                    if fiber not in self.W:
                        continue
                    
                    src_active = (self.act[src] > 0).nonzero(as_tuple=True)[0]
                    if len(src_active) == 0:
                        continue
                    
                    # Multiplicative update: w *= (1 + β)
                    beta = self.beta * 2 if fiber in self.strong_fibers else self.beta
                    self.W[fiber][winners.unsqueeze(1), src_active.unsqueeze(0)] *= (1 + beta)
        
        # Update activations
        for area, act in new_act.items():
            self.act[area] = act
    
    def test_pathway(self, word: str, is_noun: bool) -> float:
        """
        Test semantic pathway: Phon → Lex → Visual/Motor
        
        This tests if hearing a word activates its meaning.
        """
        target_area = 'Visual' if is_noun else 'Motor'
        lex_area = 'Lex1' if is_noun else 'Lex2'
        
        # Get target assembly
        target_indices = self.input_assemblies[target_area][word]
        target_set = set(target_indices.cpu().numpy())
        
        # Step 1: Phon → Lex (activate word's lexical representation)
        self.clear()
        self.activate('Phon', word)
        
        for _ in range(3):
            self.fire_step(['Phon', lex_area], plastic_areas=[])
        
        # Step 2: Lex → Target
        # Compute what Visual/Motor neurons would be activated by Lex
        if lex_area not in self.act:
            return 0.0
        
        fiber = (lex_area, target_area)
        if fiber not in self.W:
            return 0.0
        
        input_to_target = self.W[fiber] @ self.act[lex_area]
        _, top_indices = torch.topk(input_to_target, self.k)
        top_set = set(top_indices.cpu().numpy())
        
        overlap = len(target_set & top_set) / self.k
        return overlap
    
    def clear(self):
        self.act.clear()
    
    def measure_stability(self, area: str, word: str, n_steps: int = 10) -> float:
        """
        Measure assembly stability.
        
        From paper: Fire Phon[word] into area, then continue firing
        recurrently. Stable = same neurons keep firing.
        """
        self.clear()
        self.activate('Phon', word)
        
        history = []
        
        for step in range(n_steps):
            # Keep Phon active
            self.activate('Phon', word)
            
            # Fire
            self.fire_step(['Phon', area], plastic_areas=[])
            
            if area in self.act:
                winners = (self.act[area] > 0).nonzero(as_tuple=True)[0]
                history.append(set(winners.cpu().numpy()))
        
        if len(history) < 2:
            return 0.0
        
        # Overlap between first and last
        first = history[0]
        last = history[-1]
        overlap = len(first & last) / self.k
        return overlap


def run_diagnostics():
    print("=" * 70)
    print("NEMO DIAGNOSTICS")
    print("=" * 70)
    
    # Test with paper-like parameters (scaled down for speed)
    brain = SimplifiedNemoBrain(n=10000, p=0.05, beta=0.06, k=50, tau=2)
    
    # Create words
    nouns = ['dog', 'cat', 'bird', 'ball', 'baby']
    verbs = ['runs', 'walks', 'sleeps', 'eats', 'plays']
    
    for word in nouns:
        brain.create_assembly('Phon', word)
        brain.create_assembly('Visual', word)
    
    for word in verbs:
        brain.create_assembly('Phon', word)
        brain.create_assembly('Motor', word)
    
    # Test 1: Before training - stability should be random/low
    print("\n" + "=" * 70)
    print("TEST 1: Stability BEFORE training")
    print("=" * 70)
    
    print(f"\n{'Word':>10} {'Type':>6} {'Lex1':>8} {'Lex2':>8}")
    print("-" * 40)
    
    for word in nouns[:3] + verbs[:3]:
        is_noun = word in nouns
        s1 = brain.measure_stability('Lex1', word)
        s2 = brain.measure_stability('Lex2', word)
        word_type = 'NOUN' if is_noun else 'VERB'
        print(f"{word:>10} {word_type:>6} {s1:>8.2f} {s2:>8.2f}")
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING: 100 grounded sentences")
    print("=" * 70)
    
    n_sentences = 100
    for i in range(n_sentences):
        noun = np.random.choice(nouns)
        verb = np.random.choice(verbs)
        
        brain.clear()
        
        # Present grounded sentence
        for word in [noun, verb]:
            # Grounding fires throughout
            brain.activate('Visual', noun)
            brain.activate('Motor', verb)
            brain.activate('Phon', word)
            
            # Fire for tau steps
            # KEY FIX: Also update Lex→Visual and Lex→Motor weights!
            # The paper says there are bidirectional fibers.
            # We do this by making Visual/Motor "plastic" for the reverse direction.
            for _ in range(brain.tau):
                # Forward: Phon/Visual/Motor → Lex
                brain.fire_step(['Phon', 'Visual', 'Motor', 'Lex1', 'Lex2'],
                               plastic_areas=['Lex1', 'Lex2'])
                
                # Also strengthen Lex → Visual/Motor (for the pathway)
                # This happens when Lex fires after Visual/Motor
                # We need to manually update these weights
                for lex_area, sem_area in [('Lex1', 'Visual'), ('Lex2', 'Motor')]:
                    if lex_area in brain.act and sem_area in brain.act:
                        lex_active = (brain.act[lex_area] > 0).nonzero(as_tuple=True)[0]
                        sem_active = (brain.act[sem_area] > 0).nonzero(as_tuple=True)[0]
                        
                        if len(lex_active) > 0 and len(sem_active) > 0:
                            fiber = (lex_area, sem_area)
                            if fiber in brain.W:
                                beta = brain.beta * 2  # Strong fiber
                                brain.W[fiber][sem_active.unsqueeze(1), lex_active.unsqueeze(0)] *= (1 + beta)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_sentences} sentences")
    
    # Test 2: After training - nouns should be stable in Lex1, verbs in Lex2
    print("\n" + "=" * 70)
    print("TEST 2: Stability AFTER training")
    print("=" * 70)
    
    print(f"\n{'Word':>10} {'Type':>6} {'Lex1':>8} {'Lex2':>8} {'Pred':>8} {'OK':>4}")
    print("-" * 55)
    
    correct = 0
    total = 0
    
    for word in nouns + verbs:
        is_noun = word in nouns
        s1 = brain.measure_stability('Lex1', word)
        s2 = brain.measure_stability('Lex2', word)
        
        predicted_noun = s1 > s2
        correct_pred = predicted_noun == is_noun
        if correct_pred:
            correct += 1
        total += 1
        
        word_type = 'NOUN' if is_noun else 'VERB'
        pred_type = 'NOUN' if predicted_noun else 'VERB'
        ok = '✓' if correct_pred else '✗'
        
        print(f"{word:>10} {word_type:>6} {s1:>8.2f} {s2:>8.2f} {pred_type:>8} {ok:>4}")
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")
    
    # Test 3: Semantic pathway
    print("\n" + "=" * 70)
    print("TEST 3: Semantic Pathway (Phon → Lex → Visual/Motor)")
    print("=" * 70)
    
    print("\nMethod 1: Simultaneous firing (wrong)")
    for word in nouns[:2] + verbs[:2]:
        is_noun = word in nouns
        target_area = 'Visual' if is_noun else 'Motor'
        lex_area = 'Lex1' if is_noun else 'Lex2'
        
        target_indices = brain.input_assemblies[target_area][word]
        target_set = set(target_indices.cpu().numpy())
        
        brain.clear()
        brain.activate('Phon', word)
        
        for _ in range(5):
            brain.fire_step(['Phon', lex_area, target_area], plastic_areas=[])
        
        if target_area in brain.act:
            current = (brain.act[target_area] > 0).nonzero(as_tuple=True)[0]
            current_set = set(current.cpu().numpy())
            overlap = len(target_set & current_set) / brain.k
        else:
            overlap = 0.0
        
        print(f"  {word} ({target_area}): overlap = {overlap:.2f}")
    
    print("\nMethod 2: Sequential (Phon→Lex, then Lex→Target)")
    for word in nouns[:2] + verbs[:2]:
        is_noun = word in nouns
        overlap = brain.test_pathway(word, is_noun)
        target_area = 'Visual' if is_noun else 'Motor'
        print(f"  {word} ({target_area}): overlap = {overlap:.2f}")
    
    # Test 4: Weight analysis
    print("\n" + "=" * 70)
    print("TEST 4: Weight Analysis")
    print("=" * 70)
    
    for fiber in [('Phon', 'Lex1'), ('Phon', 'Lex2'), 
                  ('Visual', 'Lex1'), ('Motor', 'Lex2'),
                  ('Lex1', 'Visual'), ('Lex2', 'Motor'),  # These are key for pathway!
                  ('Visual', 'Lex2'), ('Motor', 'Lex1')]:
        W = brain.W[fiber]
        mean_w = W[W > 0].mean().item() if (W > 0).any() else 0
        max_w = W.max().item()
        nnz = (W > 0).sum().item()
        print(f"  {fiber[0]:>8} → {fiber[1]:<8}: mean={mean_w:.2f}, max={max_w:.2f}, nnz={nnz}")
    
    # Test 5: Debug pathway step by step
    print("\n" + "=" * 70)
    print("TEST 5: Pathway Debug (step by step)")
    print("=" * 70)
    
    word = 'dog'
    print(f"\nTesting pathway for '{word}':")
    
    # Get target
    target_indices = brain.input_assemblies['Visual'][word]
    target_set = set(target_indices.cpu().numpy())
    print(f"  Target Visual assembly: {sorted(list(target_set))[:10]}...")
    
    # Step 1: Activate Phon
    brain.clear()
    brain.activate('Phon', word)
    phon_indices = brain.input_assemblies['Phon'][word]
    print(f"  Phon assembly: {sorted(list(phon_indices.cpu().numpy()))[:10]}...")
    
    # Step 2: Fire Phon → Lex1
    brain.fire_step(['Phon', 'Lex1'], plastic_areas=[])
    if 'Lex1' in brain.act:
        lex1_indices = (brain.act['Lex1'] > 0).nonzero(as_tuple=True)[0]
        print(f"  Lex1 after Phon→Lex1: {sorted(list(lex1_indices.cpu().numpy()))[:10]}...")
    else:
        print("  Lex1: NO ACTIVATION")
    
    # Step 3: Fire Lex1 → Visual
    brain.fire_step(['Lex1', 'Visual'], plastic_areas=[])
    if 'Visual' in brain.act:
        vis_indices = (brain.act['Visual'] > 0).nonzero(as_tuple=True)[0]
        vis_set = set(vis_indices.cpu().numpy())
        overlap = len(target_set & vis_set) / brain.k
        print(f"  Visual after Lex1→Visual: {sorted(list(vis_set))[:10]}...")
        print(f"  Overlap with target: {overlap:.2f}")
    else:
        print("  Visual: NO ACTIVATION")
    
    # Check Lex1 → Visual weights for this word's Lex1 assembly
    if 'Lex1' in brain.act:
        lex1_active = (brain.act['Lex1'] > 0).nonzero(as_tuple=True)[0]
        W_lex_vis = brain.W[('Lex1', 'Visual')]
        
        # What Visual neurons get input from this Lex1 assembly?
        input_to_visual = W_lex_vis @ brain.act['Lex1']
        top_visual = torch.topk(input_to_visual, brain.k)[1]
        top_set = set(top_visual.cpu().numpy())
        
        print(f"\n  Lex1→Visual analysis:")
        print(f"    Top Visual neurons from Lex1: {sorted(list(top_set))[:10]}...")
        print(f"    Overlap with target: {len(target_set & top_set) / brain.k:.2f}")


def run_learning_curve():
    """Test how pathway accuracy improves with more training"""
    print("\n" + "=" * 70)
    print("LEARNING CURVE: Pathway accuracy vs training")
    print("=" * 70)
    
    nouns = ['dog', 'cat', 'bird', 'ball', 'baby']
    verbs = ['runs', 'walks', 'sleeps', 'eats', 'plays']
    
    checkpoints = [10, 20, 50, 100, 200]
    
    print(f"\n{'Sentences':>12} {'Noun Path':>12} {'Verb Path':>12} {'N/V Class':>12}")
    print("-" * 52)
    
    for n_sent in checkpoints:
        brain = SimplifiedNemoBrain(n=10000, p=0.05, beta=0.06, k=50, verbose=False)
        
        # Create assemblies
        for word in nouns:
            brain.create_assembly('Phon', word)
            brain.create_assembly('Visual', word)
        for word in verbs:
            brain.create_assembly('Phon', word)
            brain.create_assembly('Motor', word)
        
        # Train
        for _ in range(n_sent):
            noun = np.random.choice(nouns)
            verb = np.random.choice(verbs)
            
            brain.clear()
            for word in [noun, verb]:
                brain.activate('Visual', noun)
                brain.activate('Motor', verb)
                brain.activate('Phon', word)
                
                for _ in range(brain.tau):
                    brain.fire_step(['Phon', 'Visual', 'Motor', 'Lex1', 'Lex2'],
                                   plastic_areas=['Lex1', 'Lex2'])
                    
                    for lex_area, sem_area in [('Lex1', 'Visual'), ('Lex2', 'Motor')]:
                        if lex_area in brain.act and sem_area in brain.act:
                            lex_active = (brain.act[lex_area] > 0).nonzero(as_tuple=True)[0]
                            sem_active = (brain.act[sem_area] > 0).nonzero(as_tuple=True)[0]
                            if len(lex_active) > 0 and len(sem_active) > 0:
                                fiber = (lex_area, sem_area)
                                if fiber in brain.W:
                                    brain.W[fiber][sem_active.unsqueeze(1), lex_active.unsqueeze(0)] *= 1.12
        
        # Test pathway
        noun_overlaps = [brain.test_pathway(w, True) for w in nouns]
        verb_overlaps = [brain.test_pathway(w, False) for w in verbs]
        
        # Test classification
        correct = 0
        for word in nouns + verbs:
            is_noun = word in nouns
            s1 = brain.measure_stability('Lex1', word)
            s2 = brain.measure_stability('Lex2', word)
            if (s1 > s2) == is_noun:
                correct += 1
        
        print(f"{n_sent:>12} {np.mean(noun_overlaps):>12.1%} {np.mean(verb_overlaps):>12.1%} {correct/10:>12.1%}")


if __name__ == "__main__":
    run_diagnostics()
    run_learning_curve()

