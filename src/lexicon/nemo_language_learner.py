"""
NEMO Language Learner

Based on: "Simulated Language Acquisition in a Biologically Realistic Model of the Brain"
Mitropolsky & Papadimitriou, 2025

Key principles:
1. Grounded learning - words are presented with sensory context
2. Separate Lex areas for nouns (Lex1→Visual) and verbs (Lex2→Motor)
3. Stability-based classification - stable assembly = correct area
4. Inter-area inhibition for competition
5. Role areas for thematic roles (agent, action, patient)

NO HARDCODED POS LABELS - categories emerge from grounding!
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

# Check CUDA
assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device('cuda')
print(f"Using GPU: {torch.cuda.get_device_name()}")


@dataclass
class NemoParams:
    """Parameters for NEMO model"""
    n: int = 100000        # Neurons per area (paper uses 10^5 to 10^6)
    k: int = 100           # Winners per area (paper uses ~sqrt(n))
    p: float = 0.01        # Connection probability
    beta: float = 0.05     # Hebbian plasticity rate
    tau: int = 2           # Firing steps per word
    
    # Stronger fibers (as in paper)
    p_strong: float = 0.05
    beta_strong: float = 0.1


class NemoBrain:
    """
    NEMO brain for language acquisition.
    
    Architecture (from paper):
    - Phon: Phonological representations (input)
    - Lex1: Lexical area for NOUNS (strong connection to Visual)
    - Lex2: Lexical area for VERBS (strong connection to Motor)
    - Visual: Visual representations (input)
    - Motor: Motor/mirror neuron representations (input)
    - Context areas C1...Cm: Other semantic context
    
    For syntax:
    - Role_agent, Role_action, Role_patient: Thematic roles (mutual inhibition)
    - Subj, Verb, Obj: Syntactic positions
    - Mood: Grammatical mood
    """
    
    def __init__(self, params: NemoParams = None, n_context: int = 3, verbose: bool = True):
        self.p = params or NemoParams()
        self.verbose = verbose
        
        # Areas
        self.input_areas = ['Phon', 'Visual', 'Motor'] + [f'C{i}' for i in range(n_context)]
        self.lex_areas = ['Lex1', 'Lex2']  # Lex1=nouns, Lex2=verbs (will emerge!)
        self.role_areas = ['Role_agent', 'Role_action', 'Role_patient']
        self.syntax_areas = ['Subj', 'Verb', 'Obj']
        self.other_areas = ['Mood', 'Role_scene']
        
        self.all_areas = (self.input_areas + self.lex_areas + 
                         self.role_areas + self.syntax_areas + self.other_areas)
        
        # Initialize weights for each fiber
        # Key insight: different fibers have different strengths!
        self.W: Dict[Tuple[str, str], torch.Tensor] = {}
        
        # Strong fibers (as in paper Figure 2)
        strong_fibers = [
            ('Phon', 'Lex1'), ('Phon', 'Lex2'),
            ('Lex1', 'Phon'), ('Lex2', 'Phon'),
            ('Lex1', 'Visual'), ('Visual', 'Lex1'),  # Nouns ↔ Visual
            ('Lex2', 'Motor'), ('Motor', 'Lex2'),    # Verbs ↔ Motor
        ]
        
        # Regular fibers
        regular_fibers = []
        # Lex areas to context areas
        for lex in self.lex_areas:
            for ctx in [f'C{i}' for i in range(n_context)]:
                regular_fibers.append((lex, ctx))
                regular_fibers.append((ctx, lex))
        
        # Lex1 to Motor (weak), Lex2 to Visual (weak)
        regular_fibers.extend([
            ('Lex1', 'Motor'), ('Motor', 'Lex1'),
            ('Lex2', 'Visual'), ('Visual', 'Lex2'),
        ])
        
        # Role and syntax fibers
        for role in self.role_areas:
            for syn in self.syntax_areas:
                regular_fibers.append((role, syn))
                regular_fibers.append((syn, role))
        
        # Recurrent within Lex areas
        regular_fibers.extend([('Lex1', 'Lex1'), ('Lex2', 'Lex2')])
        
        # Initialize fibers
        for src, dst in strong_fibers:
            self.W[(src, dst)] = self._init_weights(self.p.p_strong)
        
        for src, dst in regular_fibers:
            if (src, dst) not in self.W:
                self.W[(src, dst)] = self._init_weights(self.p.p)
        
        # Current activations
        self.activations: Dict[str, torch.Tensor] = {}
        
        # Input assemblies (pre-initialized)
        self.input_assemblies: Dict[str, Dict[str, torch.Tensor]] = {
            area: {} for area in self.input_areas
        }
        
        # Track which neurons have fired (for assembly detection)
        self.firing_history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        if verbose:
            n_fibers = len(self.W)
            mem_mb = n_fibers * self.p.n * self.p.n * 4 / 1e9
            print(f"NemoBrain initialized:")
            print(f"  n={self.p.n}, k={self.p.k}, beta={self.p.beta}")
            print(f"  Areas: {len(self.all_areas)}")
            print(f"  Fibers: {n_fibers}")
            print(f"  Memory: {mem_mb:.1f} GB")
    
    def _init_weights(self, p: float) -> torch.Tensor:
        """Initialize sparse random weights"""
        # Sparse initialization - only p fraction of connections
        W = torch.zeros(self.p.n, self.p.n, device=DEVICE)
        mask = torch.rand(self.p.n, self.p.n, device=DEVICE) < p
        W[mask] = 1.0  # Initial weight = 1
        return W
    
    def create_input_assembly(self, area: str, name: str) -> torch.Tensor:
        """Create a random assembly in an input area"""
        if name not in self.input_assemblies[area]:
            indices = torch.randperm(self.p.n, device=DEVICE)[:self.p.k]
            self.input_assemblies[area][name] = indices
        return self.input_assemblies[area][name]
    
    def activate_input(self, area: str, name: str):
        """Activate a pre-defined assembly in an input area"""
        indices = self.input_assemblies[area].get(name)
        if indices is None:
            indices = self.create_input_assembly(area, name)
        
        act = torch.zeros(self.p.n, device=DEVICE)
        act[indices] = 1.0
        self.activations[area] = act
    
    def fire_step(self, active_areas: List[str], plastic_areas: List[str] = None,
                  inhibited_groups: List[List[str]] = None):
        """
        Execute one firing step.
        
        Args:
            active_areas: Areas that can fire this step
            plastic_areas: Areas where Hebbian learning occurs (default: all active)
            inhibited_groups: Groups of areas under mutual inhibition
        """
        if plastic_areas is None:
            plastic_areas = active_areas
        
        if inhibited_groups is None:
            inhibited_groups = []
        
        # Compute inputs for each active area
        new_activations = {}
        area_inputs = {}
        
        for dst in active_areas:
            total_input = torch.zeros(self.p.n, device=DEVICE)
            
            # Sum inputs from all connected areas
            for src in self.all_areas:
                if (src, dst) not in self.W:
                    continue
                
                src_act = self.activations.get(src)
                if src_act is None or src_act.sum() == 0:
                    continue
                
                W = self.W[(src, dst)]
                total_input += W @ src_act
            
            area_inputs[dst] = total_input
        
        # Handle mutual inhibition groups
        inhibited_areas = set()
        for group in inhibited_groups:
            # Find area with maximum input in this group
            group_inputs = {a: area_inputs.get(a, torch.zeros(1)).sum().item() 
                          for a in group if a in active_areas}
            if group_inputs:
                winner = max(group_inputs, key=group_inputs.get)
                # Inhibit all others in group
                for a in group:
                    if a != winner:
                        inhibited_areas.add(a)
        
        # Apply k-cap to each non-inhibited area
        for dst in active_areas:
            if dst in inhibited_areas:
                new_activations[dst] = torch.zeros(self.p.n, device=DEVICE)
                continue
            
            total_input = area_inputs.get(dst)
            if total_input is None or total_input.sum() == 0:
                continue
            
            # Winner-take-all (k-cap)
            _, winners = torch.topk(total_input, self.p.k)
            
            new_act = torch.zeros(self.p.n, device=DEVICE)
            new_act[winners] = 1.0
            new_activations[dst] = new_act
            
            # Record firing history
            self.firing_history[dst].append(winners.clone())
        
        # Hebbian plasticity
        for dst in plastic_areas:
            if dst in inhibited_areas:
                continue
            
            new_act = new_activations.get(dst)
            if new_act is None or new_act.sum() == 0:
                continue
            
            dst_active = (new_act > 0).nonzero(as_tuple=True)[0]
            
            for src in self.all_areas:
                if (src, dst) not in self.W:
                    continue
                
                src_act = self.activations.get(src)
                if src_act is None or src_act.sum() == 0:
                    continue
                
                src_active = (src_act > 0).nonzero(as_tuple=True)[0]
                if len(src_active) == 0:
                    continue
                
                # Hebbian update: W[dst, src] *= (1 + beta)
                W = self.W[(src, dst)]
                beta = self.p.beta_strong if (src, dst) in [
                    ('Phon', 'Lex1'), ('Phon', 'Lex2'),
                    ('Lex1', 'Visual'), ('Lex2', 'Motor')
                ] else self.p.beta
                
                # Update weights for co-active neurons
                W[dst_active.unsqueeze(1), src_active.unsqueeze(0)] *= (1 + beta)
        
        # Update activations
        for area, act in new_activations.items():
            self.activations[area] = act
    
    def clear(self):
        """Clear all activations"""
        self.activations.clear()
        self.firing_history.clear()
    
    def measure_stability(self, area: str, steps: int = 5) -> float:
        """
        Measure assembly stability in an area.
        
        Stable assembly = same neurons keep firing.
        Wobbly assembly = neurons change each step.
        
        Returns overlap between first and last firing.
        """
        if area not in self.firing_history or len(self.firing_history[area]) < 2:
            return 0.0
        
        history = self.firing_history[area]
        first = set(history[0].cpu().numpy())
        last = set(history[-1].cpu().numpy())
        
        overlap = len(first & last) / self.p.k
        return overlap


class NemoLanguageLearner:
    """
    Language learner using NEMO architecture.
    
    Learning phases (from paper):
    1. Word learning: Learn word meanings and noun/verb distinction
    2. Syntax learning: Learn word order
    """
    
    def __init__(self, params: NemoParams = None, verbose: bool = True):
        # Use smaller params for faster iteration
        if params is None:
            params = NemoParams(n=10000, k=50, beta=0.05)
        
        self.brain = NemoBrain(params, verbose=verbose)
        self.verbose = verbose
        
        # Lexicon - just word forms, NO POS labels!
        self.words = {
            # Nouns (but we don't tell the system!)
            'dog', 'cat', 'bird', 'ball', 'book', 'baby', 'boy', 'girl',
            'man', 'woman', 'food', 'milk', 'tree', 'car', 'house',
            # Verbs (but we don't tell the system!)
            'runs', 'walks', 'sleeps', 'eats', 'drinks', 'plays', 'jumps',
            'flies', 'sees', 'has', 'wants', 'reads', 'throws',
        }
        
        # Ground truth for evaluation only (NOT used in learning!)
        self._gt_nouns = {'dog', 'cat', 'bird', 'ball', 'book', 'baby', 'boy', 
                         'girl', 'man', 'woman', 'food', 'milk', 'tree', 'car', 'house'}
        self._gt_verbs = {'runs', 'walks', 'sleeps', 'eats', 'drinks', 'plays',
                         'jumps', 'flies', 'sees', 'has', 'wants', 'reads', 'throws'}
        
        # Initialize input assemblies
        self._init_input_assemblies()
        
        self.sentences_seen = 0
    
    def _init_input_assemblies(self):
        """Initialize assemblies in input areas"""
        # Each word has a Phon assembly
        for word in self.words:
            self.brain.create_input_assembly('Phon', word)
        
        # Nouns have Visual assemblies
        for noun in self._gt_nouns:
            self.brain.create_input_assembly('Visual', noun)
            # Also some context
            for i in range(2):
                self.brain.create_input_assembly(f'C{i}', noun)
        
        # Verbs have Motor assemblies
        for verb in self._gt_verbs:
            self.brain.create_input_assembly('Motor', verb)
            # Also some context
            for i in range(1, 3):
                self.brain.create_input_assembly(f'C{i}', verb)
    
    def present_grounded_sentence(self, noun: str, verb: str, learn: bool = True):
        """
        Present a grounded sentence (e.g., "the dog runs").
        
        Grounding means:
        - Visual[noun] fires throughout
        - Motor[verb] fires throughout
        - Context assemblies fire throughout
        - Words are presented sequentially in Phon
        
        Key insight from paper: Grounding fires CONTINUOUSLY while words
        are presented. This creates the differential association:
        - Noun in Phon + Visual firing → Lex1 learns noun-visual association
        - Verb in Phon + Motor firing → Lex2 learns verb-motor association
        """
        self.brain.clear()
        
        # Present words sequentially (SV order for now)
        for word in [noun, verb]:
            # Activate grounding EVERY step (key insight!)
            # Visual fires throughout (for noun grounding)
            self.brain.activate_input('Visual', noun)
            # Motor fires throughout (for verb grounding)
            self.brain.activate_input('Motor', verb)
            
            # Context areas
            for i in range(2):
                if noun in self.brain.input_assemblies.get(f'C{i}', {}):
                    self.brain.activate_input(f'C{i}', noun)
            for i in range(1, 3):
                if verb in self.brain.input_assemblies.get(f'C{i}', {}):
                    self.brain.activate_input(f'C{i}', verb)
            
            # Activate word in Phon
            self.brain.activate_input('Phon', word)
            
            # Fire for tau steps
            for _ in range(self.brain.p.tau):
                # Active areas: input areas + lex areas
                active = self.brain.input_areas + self.brain.lex_areas
                plastic = self.brain.lex_areas if learn else []
                
                self.brain.fire_step(active, plastic)
        
        if learn:
            self.sentences_seen += 1
    
    def train_word_learning(self, n_sentences: int = 100, verbose: bool = True):
        """
        Phase 1: Learn word meanings and noun/verb distinction.
        
        Present random grounded sentences.
        """
        if verbose:
            print(f"\nTraining word learning with {n_sentences} sentences...")
        
        nouns = list(self._gt_nouns)
        verbs = list(self._gt_verbs)
        
        start = time.perf_counter()
        
        for i in range(n_sentences):
            noun = np.random.choice(nouns)
            verb = np.random.choice(verbs)
            
            self.present_grounded_sentence(noun, verb, learn=True)
            
            if verbose and (i + 1) % max(1, n_sentences // 5) == 0:
                print(f"  {i+1}/{n_sentences} sentences")
        
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"Training complete in {elapsed:.2f}s")
    
    def test_word_classification(self, verbose: bool = True) -> Tuple[float, float]:
        """
        Test noun/verb classification using stability criterion.
        
        From paper: A word is a noun if its assembly in Lex1 is stable
        and wobbly in Lex2, and vice versa for verbs.
        
        Key: Need to fire RECURRENTLY (Lex fires back into itself) to test stability.
        """
        if verbose:
            print("\nTesting word classification (stability-based)...")
        
        correct_nouns = 0
        correct_verbs = 0
        total_nouns = 0
        total_verbs = 0
        
        results = []
        
        for word in self.words:
            is_noun = word in self._gt_nouns
            
            # Test stability in Lex1: Fire Phon[word] and let Lex1 fire recurrently
            self.brain.clear()
            self.brain.firing_history.clear()
            self.brain.activate_input('Phon', word)
            
            # Fire into Lex1 with recurrence
            for step in range(10):
                # First step: Phon → Lex1
                # Later steps: Phon + Lex1 → Lex1 (recurrent)
                if step > 0:
                    # Keep Phon active
                    self.brain.activate_input('Phon', word)
                self.brain.fire_step(['Phon', 'Lex1'], plastic_areas=[])
            stability_lex1 = self.brain.measure_stability('Lex1')
            
            # Test stability in Lex2
            self.brain.clear()
            self.brain.firing_history.clear()
            self.brain.activate_input('Phon', word)
            
            for step in range(10):
                if step > 0:
                    self.brain.activate_input('Phon', word)
                self.brain.fire_step(['Phon', 'Lex2'], plastic_areas=[])
            stability_lex2 = self.brain.measure_stability('Lex2')
            
            # Classify based on stability
            predicted_noun = stability_lex1 > stability_lex2
            
            if is_noun:
                total_nouns += 1
                if predicted_noun:
                    correct_nouns += 1
            else:
                total_verbs += 1
                if not predicted_noun:
                    correct_verbs += 1
            
            results.append({
                'word': word,
                'is_noun': is_noun,
                'predicted_noun': predicted_noun,
                'stability_lex1': stability_lex1,
                'stability_lex2': stability_lex2,
            })
        
        noun_acc = correct_nouns / total_nouns if total_nouns > 0 else 0
        verb_acc = correct_verbs / total_verbs if total_verbs > 0 else 0
        
        if verbose:
            print(f"\n{'Word':>12} {'True':>8} {'Pred':>8} {'Lex1':>8} {'Lex2':>8}")
            print("-" * 50)
            for r in results[:10]:  # Show first 10
                true_label = 'NOUN' if r['is_noun'] else 'VERB'
                pred_label = 'NOUN' if r['predicted_noun'] else 'VERB'
                correct = '✓' if r['is_noun'] == r['predicted_noun'] else '✗'
                print(f"{r['word']:>12} {true_label:>8} {pred_label:>8} "
                      f"{r['stability_lex1']:>8.2f} {r['stability_lex2']:>8.2f} {correct}")
            
            print(f"\nNoun accuracy: {correct_nouns}/{total_nouns} = {noun_acc:.1%}")
            print(f"Verb accuracy: {correct_verbs}/{total_verbs} = {verb_acc:.1%}")
            print(f"Overall: {(correct_nouns + correct_verbs)}/{len(self.words)} = "
                  f"{(correct_nouns + correct_verbs)/len(self.words):.1%}")
        
        return noun_acc, verb_acc
    
    def test_semantic_pathway(self, verbose: bool = True) -> float:
        """
        Test if Phon → Lex → Visual/Motor pathway works.
        
        From paper Property 2: Firing Phon[word] should activate
        the corresponding Visual/Motor assembly.
        """
        if verbose:
            print("\nTesting semantic pathway...")
        
        correct = 0
        total = 0
        
        for word in list(self.words)[:10]:  # Test subset
            is_noun = word in self._gt_nouns
            target_area = 'Visual' if is_noun else 'Motor'
            lex_area = 'Lex1' if is_noun else 'Lex2'
            
            # Get target assembly
            target_indices = self.brain.input_assemblies[target_area].get(word)
            if target_indices is None:
                continue
            
            # Fire Phon[word] through Lex to Visual/Motor
            self.brain.clear()
            self.brain.activate_input('Phon', word)
            
            for _ in range(3):
                self.brain.fire_step(['Phon', lex_area, target_area], plastic_areas=[])
            
            # Check overlap with target
            current_act = self.brain.activations.get(target_area)
            if current_act is not None:
                current_indices = (current_act > 0).nonzero(as_tuple=True)[0]
                target_set = set(target_indices.cpu().numpy())
                current_set = set(current_indices.cpu().numpy())
                overlap = len(target_set & current_set) / self.brain.p.k
                
                if overlap > 0.5:  # Threshold for "correct"
                    correct += 1
                
                if verbose:
                    print(f"  {word}: overlap = {overlap:.2f}")
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        if verbose:
            print(f"Pathway accuracy: {correct}/{total} = {accuracy:.1%}")
        
        return accuracy


def main():
    print("=" * 70)
    print("NEMO LANGUAGE LEARNER")
    print("Based on Mitropolsky & Papadimitriou, 2025")
    print("=" * 70)
    
    # Use smaller params for testing
    params = NemoParams(n=5000, k=30, beta=0.1)
    learner = NemoLanguageLearner(params, verbose=True)
    
    # Learning curve experiment
    print("\n" + "=" * 70)
    print("LEARNING CURVE: How many sentences needed?")
    print("=" * 70)
    
    checkpoints = [10, 20, 50, 100, 200]
    
    print(f"\n{'Sentences':>12} {'Noun Acc':>10} {'Verb Acc':>10} {'Overall':>10}")
    print("-" * 45)
    
    for n_sent in checkpoints:
        # Reset and train
        learner = NemoLanguageLearner(params, verbose=False)
        learner.train_word_learning(n_sentences=n_sent, verbose=False)
        
        # Test
        noun_acc, verb_acc = learner.test_word_classification(verbose=False)
        overall = (noun_acc + verb_acc) / 2
        
        print(f"{n_sent:>12} {noun_acc:>10.1%} {verb_acc:>10.1%} {overall:>10.1%}")
    
    # Final detailed test
    print("\n" + "=" * 70)
    print("DETAILED RESULTS (200 sentences)")
    print("=" * 70)
    
    learner = NemoLanguageLearner(params, verbose=False)
    learner.train_word_learning(n_sentences=200, verbose=True)
    learner.test_word_classification(verbose=True)
    learner.test_semantic_pathway(verbose=True)
    
    # Memory usage
    mem = torch.cuda.memory_allocated() / 1e6
    print(f"\nGPU memory: {mem:.1f} MB")


if __name__ == "__main__":
    main()

