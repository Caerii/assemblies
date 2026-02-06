"""
NEMO Efficient Implementation

Uses IMPLICIT sparse representation:
- Initial connections computed on-demand (not stored)
- Only learned weights are stored
- Scales to n=100,000+ with minimal memory

Memory: O(vocabulary * k^2) instead of O(n^2)
"""

import torch
import numpy as np
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name()}")


@dataclass
class NemoParams:
    n: int = 100000          # Neurons per area (paper scale!)
    k: int = 50              # Winners (cap size)
    p: float = 0.05          # Connection probability
    beta: float = 0.06       # Plasticity
    w_max: float = 10.0      # Weight saturation
    tau: int = 2             # Firing steps per word


class ImplicitSparseMatrix:
    """
    Sparse matrix with implicit random initialization.
    
    - Initial connections are computed on-demand using a hash function
    - Only learned weight modifications are stored
    - Memory: O(number of updates) instead of O(n^2)
    """
    
    def __init__(self, n_src: int, n_dst: int, p: float, seed: int = 42):
        self.n_src = n_src
        self.n_dst = n_dst
        self.p = p
        self.seed = seed
        
        # Only store weight MODIFICATIONS (delta from initial weight of 1.0)
        # Key: (src_idx, dst_idx), Value: weight delta
        self.weight_deltas: Dict[Tuple[int, int], float] = {}
        
        # Cache for frequently accessed rows (optional optimization)
        self._row_cache: Dict[int, torch.Tensor] = {}
        self._cache_size = 1000
    
    def _has_connection(self, src: int, dst: int) -> bool:
        """Check if initial random connection exists using hash"""
        # Deterministic pseudo-random based on indices and seed
        h = hash((src, dst, self.seed)) & 0xFFFFFFFF
        threshold = int(self.p * 0xFFFFFFFF)
        return h < threshold
    
    def _get_row_connections(self, src: int) -> torch.Tensor:
        """Get all destination indices that src connects to"""
        # Check cache first
        if src in self._row_cache:
            return self._row_cache[src]
        
        # Compute connections for this row
        # Use vectorized random for efficiency
        rng = np.random.RandomState(self.seed + src)
        mask = rng.random(self.n_dst) < self.p
        connections = torch.tensor(np.where(mask)[0], device=DEVICE, dtype=torch.long)
        
        # Cache if room
        if len(self._row_cache) < self._cache_size:
            self._row_cache[src] = connections
        
        return connections
    
    def precompute_cache(self, indices: torch.Tensor, verbose: bool = False):
        """Precompute row cache for given indices"""
        for i, idx in enumerate(indices.cpu().numpy()):
            if verbose and i % 10 == 0:
                print(f"    Caching row {i+1}/{len(indices)}...", end='\r')
            self._get_row_connections(int(idx))
    
    def get_weight(self, src: int, dst: int) -> float:
        """Get weight for a specific connection"""
        if not self._has_connection(src, dst):
            return 0.0
        
        # Base weight is 1.0, add any learned delta
        delta = self.weight_deltas.get((src, dst), 0.0)
        return 1.0 + delta
    
    def update_weights(self, src_indices: torch.Tensor, dst_indices: torch.Tensor,
                       beta: float, w_max: float):
        """
        Saturating weight update for co-active neurons.
        Only updates connections that exist.
        """
        src_list = src_indices.cpu().numpy()
        dst_list = dst_indices.cpu().numpy()
        
        for src in src_list:
            for dst in dst_list:
                if self._has_connection(int(src), int(dst)):
                    key = (int(src), int(dst))
                    current_delta = self.weight_deltas.get(key, 0.0)
                    current_w = 1.0 + current_delta
                    
                    # Saturating update
                    update = beta * (1.0 - current_w / w_max)
                    if update > 0:
                        self.weight_deltas[key] = current_delta + update
    
    def sum_inputs(self, active_indices: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Sum input weights from active source neurons.
        Returns: tensor of shape (n_dst,) with total input to each dst neuron.
        """
        result = torch.zeros(self.n_dst, device=DEVICE)
        
        indices_list = active_indices.cpu().numpy()
        for i, src in enumerate(indices_list):
            if verbose and i % 10 == 0:
                print(f"      Processing input {i+1}/{len(indices_list)}...", end='\r')
            
            src = int(src)
            connections = self._get_row_connections(src)
            
            if len(connections) == 0:
                continue
            
            # Get weights for these connections
            weights = torch.ones(len(connections), device=DEVICE)
            for j, dst in enumerate(connections.cpu().numpy()):
                delta = self.weight_deltas.get((src, int(dst)), 0.0)
                weights[j] = 1.0 + delta
            
            # Add to result
            result[connections] += weights
        
        if verbose:
            print()  # Clear the line
        
        return result
    
    def memory_usage(self) -> int:
        """Return approximate memory usage in bytes"""
        # Each delta entry: 2 ints (8 bytes) + 1 float (8 bytes) = 16 bytes
        return len(self.weight_deltas) * 16


class NemoEfficientBrain:
    """
    Efficient NEMO brain using implicit sparse matrices.
    
    Can scale to n=100,000+ with minimal memory.
    """
    
    def __init__(self, params: NemoParams = None, verbose: bool = True):
        self.p = params or NemoParams()
        self.verbose = verbose
        n = self.p.n
        
        # Assemblies
        self.phon_assemblies: Dict[str, torch.Tensor] = {}
        self.visual_assemblies: Dict[str, torch.Tensor] = {}
        self.motor_assemblies: Dict[str, torch.Tensor] = {}
        
        # Activations
        self.lex1_activated: Optional[torch.Tensor] = None
        self.lex2_activated: Optional[torch.Tensor] = None
        
        # Weight matrices using implicit sparse representation
        # Use different seeds for each matrix
        self.W_phon_lex1 = ImplicitSparseMatrix(n, n, self.p.p * 2, seed=1)
        self.W_phon_lex2 = ImplicitSparseMatrix(n, n, self.p.p * 2, seed=2)
        self.W_visual_lex1 = ImplicitSparseMatrix(n, n, self.p.p * 2, seed=3)
        self.W_motor_lex2 = ImplicitSparseMatrix(n, n, self.p.p * 2, seed=4)
        self.W_lex1_rec = ImplicitSparseMatrix(n, n, self.p.p, seed=5)
        self.W_lex2_rec = ImplicitSparseMatrix(n, n, self.p.p, seed=6)
        self.W_lex1_visual = ImplicitSparseMatrix(n, n, self.p.p * 2, seed=7)
        self.W_lex2_motor = ImplicitSparseMatrix(n, n, self.p.p * 2, seed=8)
        
        self.sentences_seen = 0
        
        if verbose:
            print(f"NemoEfficientBrain initialized: n={n:,}")
            print(f"  Using IMPLICIT sparse matrices")
            print(f"  Initial memory: ~0 MB (connections computed on-demand)")
    
    def create_assembly(self, area: str, name: str) -> torch.Tensor:
        """Create random assembly"""
        indices = torch.randperm(self.p.n, device=DEVICE)[:self.p.k]
        
        if area == 'Phon':
            self.phon_assemblies[name] = indices
        elif area == 'Visual':
            self.visual_assemblies[name] = indices
        elif area == 'Motor':
            self.motor_assemblies[name] = indices
        
        return indices
    
    def inhibit_all(self):
        self.lex1_activated = None
        self.lex2_activated = None
    
    def present_word(self, word: str, is_noun: bool, learn: bool = True):
        """Present a word with grounding"""
        if word not in self.phon_assemblies:
            self.create_assembly('Phon', word)
        
        phon = self.phon_assemblies[word]
        
        if is_noun:
            if word not in self.visual_assemblies:
                self.create_assembly('Visual', word)
            visual = self.visual_assemblies[word]
            
            # Compute input
            total_input = self.W_phon_lex1.sum_inputs(phon)
            total_input += self.W_visual_lex1.sum_inputs(visual)
            if self.lex1_activated is not None:
                total_input += self.W_lex1_rec.sum_inputs(self.lex1_activated)
            
            _, winners = torch.topk(total_input, self.p.k)
            
            if learn:
                if self.lex1_activated is not None:
                    self.W_lex1_rec.update_weights(self.lex1_activated, winners,
                                                   self.p.beta, self.p.w_max)
                self.W_visual_lex1.update_weights(visual, winners, self.p.beta, self.p.w_max)
                self.W_lex1_visual.update_weights(winners, visual, self.p.beta, self.p.w_max)
            
            self.lex1_activated = winners
        else:
            if word not in self.motor_assemblies:
                self.create_assembly('Motor', word)
            motor = self.motor_assemblies[word]
            
            total_input = self.W_phon_lex2.sum_inputs(phon)
            total_input += self.W_motor_lex2.sum_inputs(motor)
            if self.lex2_activated is not None:
                total_input += self.W_lex2_rec.sum_inputs(self.lex2_activated)
            
            _, winners = torch.topk(total_input, self.p.k)
            
            if learn:
                if self.lex2_activated is not None:
                    self.W_lex2_rec.update_weights(self.lex2_activated, winners,
                                                   self.p.beta, self.p.w_max)
                self.W_motor_lex2.update_weights(motor, winners, self.p.beta, self.p.w_max)
                self.W_lex2_motor.update_weights(winners, motor, self.p.beta, self.p.w_max)
            
            self.lex2_activated = winners
    
    def train(self, subject: str, verb: str, obj: str = None):
        """Train on a sentence"""
        self.inhibit_all()
        
        for _ in range(self.p.tau):
            self.present_word(subject, is_noun=True)
        
        self.lex1_activated = None
        self.lex2_activated = None
        
        for _ in range(self.p.tau):
            self.present_word(verb, is_noun=False)
        
        if obj:
            self.lex1_activated = None
            self.lex2_activated = None
            for _ in range(self.p.tau):
                self.present_word(obj, is_noun=True)
        
        self.sentences_seen += 1
    
    def measure_stability(self, word: str, area: str, n_rounds: int = 10) -> float:
        """Measure assembly stability"""
        if word not in self.phon_assemblies:
            return 0.0
        
        phon = self.phon_assemblies[word]
        visual = self.visual_assemblies.get(word)
        motor = self.motor_assemblies.get(word)
        
        first_activation = None
        activated = None
        
        for i in range(n_rounds):
            if area == 'Lex1':
                total_input = self.W_phon_lex1.sum_inputs(phon)
                if visual is not None:
                    total_input += self.W_visual_lex1.sum_inputs(visual)
                if activated is not None:
                    total_input += self.W_lex1_rec.sum_inputs(activated)
                _, activated = torch.topk(total_input, self.p.k)
            else:
                total_input = self.W_phon_lex2.sum_inputs(phon)
                if motor is not None:
                    total_input += self.W_motor_lex2.sum_inputs(motor)
                if activated is not None:
                    total_input += self.W_lex2_rec.sum_inputs(activated)
                _, activated = torch.topk(total_input, self.p.k)
            
            if i == 0:
                first_activation = set(activated.cpu().numpy())
        
        if first_activation is None or activated is None:
            return 0.0
        
        last_activation = set(activated.cpu().numpy())
        return len(first_activation & last_activation) / self.p.k
    
    def memory_usage(self) -> dict:
        """Get memory usage breakdown"""
        matrices = [
            ('W_phon_lex1', self.W_phon_lex1),
            ('W_phon_lex2', self.W_phon_lex2),
            ('W_visual_lex1', self.W_visual_lex1),
            ('W_motor_lex2', self.W_motor_lex2),
            ('W_lex1_rec', self.W_lex1_rec),
            ('W_lex2_rec', self.W_lex2_rec),
            ('W_lex1_visual', self.W_lex1_visual),
            ('W_lex2_motor', self.W_lex2_motor),
        ]
        
        usage = {}
        total = 0
        for name, mat in matrices:
            mem = mat.memory_usage()
            usage[name] = mem
            total += mem
        
        usage['total'] = total
        return usage


def run_experiment():
    print("=" * 70)
    print("NEMO EFFICIENT IMPLEMENTATION")
    print("Using implicit sparse matrices for O(vocab * k^2) memory")
    print("=" * 70)
    
    # Start with smaller n to test, then scale up
    # The issue is that computing random connections for n=100k is slow
    print("\nNote: Using smaller n for initial test (implicit sparse is slow for large n)")
    print("      The real solution is to use GPU-accelerated sparse ops")
    
    params = NemoParams(n=10000, k=50, p=0.05, beta=0.1)  # Start smaller
    print(f"\nInitializing brain with n={params.n:,}...")
    brain = NemoEfficientBrain(params, verbose=True)
    
    nouns = ['dog', 'cat', 'boy', 'girl', 'ball', 'food']
    verbs = ['runs', 'sleeps', 'eats', 'sees']
    
    print(f"\n{'='*70}")
    print("TRAINING")
    print("=" * 70)
    
    n_sentences = 20
    print(f"\nTraining on {n_sentences} sentences with n={params.n:,}...")
    print("(Each sentence takes time because we compute random connections on-demand)")
    
    start = time.perf_counter()
    for i in range(n_sentences):
        sentence_start = time.perf_counter()
        noun = np.random.choice(nouns)
        verb = np.random.choice(verbs)
        print(f"\n  Sentence {i+1}/{n_sentences}: '{noun} {verb}'")
        brain.train(noun, verb)
        sentence_time = time.perf_counter() - sentence_start
        
        mem = brain.memory_usage()
        print(f"    Time: {sentence_time:.2f}s, Learned weights: {mem['total'] / 1e3:.1f} KB")
    
    train_time = time.perf_counter() - start
    print(f"\nTotal training time: {train_time:.2f}s ({n_sentences/train_time:.1f} sentences/sec)")
    
    # Memory usage
    print(f"\n{'='*70}")
    print("MEMORY USAGE")
    print("=" * 70)
    
    mem = brain.memory_usage()
    print(f"\nLearned weights by matrix:")
    for name, usage in mem.items():
        if name != 'total':
            print(f"  {name}: {usage / 1e3:.1f} KB")
    print(f"\nTotal learned weights: {mem['total'] / 1e6:.2f} MB")
    print(f"GPU memory (PyTorch): {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    # Compare to dense
    dense_mem = 8 * params.n * params.n * 4  # 8 matrices
    print(f"\nComparison:")
    print(f"  Dense would use: {dense_mem / 1e9:.1f} GB")
    print(f"  We use: {mem['total'] / 1e6:.2f} MB")
    print(f"  Savings: {dense_mem / max(mem['total'], 1):.0f}x")
    
    # Test classification
    print(f"\n{'='*70}")
    print("CLASSIFICATION TEST")
    print("=" * 70)
    
    correct = 0
    for word in nouns + verbs:
        is_noun = word in nouns
        s1 = brain.measure_stability(word, 'Lex1')
        s2 = brain.measure_stability(word, 'Lex2')
        pred_noun = s1 > s2
        if pred_noun == is_noun:
            correct += 1
        print(f"  {word}: Lex1={s1:.2f}, Lex2={s2:.2f} -> {'NOUN' if pred_noun else 'VERB'} {'✓' if pred_noun == is_noun else '✗'}")
    
    print(f"\nAccuracy: {correct}/{len(nouns)+len(verbs)} = {correct/(len(nouns)+len(verbs)):.1%}")


if __name__ == "__main__":
    run_experiment()

