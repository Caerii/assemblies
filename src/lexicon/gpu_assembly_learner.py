"""
GPU-Accelerated Assembly Language Learner
==========================================

Uses custom CUDA kernels for fast learning.
"""

import numpy as np
import ctypes
from ctypes import c_uint32, c_float, c_void_p, POINTER
from typing import Dict, List, Tuple
import time

# Try to import PyTorch for GPU acceleration
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


class GPUAssemblyLearner:
    """
    GPU-accelerated language learner using PyTorch CUDA tensors.
    
    Much faster than the CPU version for explicit areas.
    """
    
    def __init__(self, 
                 n: int = 10000,      # Neurons per area
                 k: int = 100,        # Active neurons
                 p: float = 0.05,     # Connection probability
                 beta: float = 0.1,   # Plasticity
                 verbose: bool = False):
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch with CUDA not available!")
        
        self.n = n
        self.k = k
        self.p = p
        self.beta = beta
        self.verbose = verbose
        self.device = torch.device('cuda')
        
        # Initialize weight matrices on GPU
        # LEX <-> CORE, VISUAL <-> CORE
        self.W_lex_to_core = self._init_weights(n, n)
        self.W_core_to_lex = self._init_weights(n, n)
        self.W_visual_to_core = self._init_weights(n, n)
        self.W_core_to_visual = self._init_weights(n, n)
        
        # Recurrent connections
        self.W_core_to_core = self._init_weights(n, n)
        
        # Activations buffer
        self.activations = torch.zeros(n, device=self.device, dtype=torch.float32)
        
        # Track learned words
        self.word_assemblies: Dict[str, torch.Tensor] = {}
        self.visual_assemblies: Dict[str, torch.Tensor] = {}
        self.word_exposures: Dict[str, int] = {}
        
        if verbose:
            print(f"GPU Assembly Learner initialized on {torch.cuda.get_device_name()}")
            print(f"  n={n}, k={k}, p={p}, beta={beta}")
            mem_mb = (5 * n * n * 4) / (1024 * 1024)  # 5 weight matrices
            print(f"  GPU memory for weights: {mem_mb:.1f} MB")
    
    def _init_weights(self, n_in: int, n_out: int) -> torch.Tensor:
        """Initialize sparse random weights on GPU"""
        # Sparse initialization: only p fraction of connections
        W = torch.zeros(n_out, n_in, device=self.device, dtype=torch.float32)
        mask = torch.rand(n_out, n_in, device=self.device) < self.p
        W[mask] = torch.randn(mask.sum(), device=self.device) * 0.1
        return W
    
    def _get_or_create_stimulus(self, concept: str) -> torch.Tensor:
        """Get or create a stimulus pattern for a concept"""
        np.random.seed(hash(concept) % (2**32))
        indices = np.random.choice(self.n, self.k, replace=False)
        np.random.seed()
        return torch.tensor(indices, device=self.device, dtype=torch.long)
    
    def _accumulate_and_topk(self, W: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated weight accumulation and top-k selection.
        
        This is the key operation: sum weights from active neurons, select top-k.
        """
        # Sum weights from active neurons (matrix-vector multiply)
        # W is (n_out, n_in), active is (k,) indices
        activations = W[:, active].sum(dim=1)  # (n_out,)
        
        # Top-k selection
        _, top_k_indices = torch.topk(activations, self.k)
        
        return top_k_indices
    
    def _hebbian_update(self, W: torch.Tensor, pre: torch.Tensor, post: torch.Tensor):
        """
        GPU-accelerated Hebbian weight update.
        
        W[post, pre] += beta (outer product update)
        """
        # Create index grids
        post_grid = post.unsqueeze(1).expand(-1, len(pre))  # (k, k)
        pre_grid = pre.unsqueeze(0).expand(len(post), -1)   # (k, k)
        
        # Update weights
        W[post_grid.flatten(), pre_grid.flatten()] += self.beta
    
    def learn_word(self, word: str, pos: str, grounding: List[str], n_rounds: int = 5):
        """
        Learn a word with grounding using GPU-accelerated Hebbian learning.
        """
        if self.verbose:
            print(f"Learning word '{word}' ({pos}) with grounding {grounding}")
        
        # Get/create stimuli
        word_stim = self._get_or_create_stimulus(word)
        grounding_stim = self._get_or_create_stimulus(grounding[0] if grounding else word.upper())
        
        for _ in range(n_rounds):
            # 1. Project VISUAL and LEX to CORE
            # Accumulate from both sources
            visual_contrib = self.W_visual_to_core[:, grounding_stim].sum(dim=1)
            lex_contrib = self.W_lex_to_core[:, word_stim].sum(dim=1)
            core_activations = visual_contrib + lex_contrib
            
            # Select top-k for CORE
            _, core_active = torch.topk(core_activations, self.k)
            
            # 2. Hebbian update: strengthen VISUAL->CORE and LEX->CORE
            self._hebbian_update(self.W_visual_to_core, grounding_stim, core_active)
            self._hebbian_update(self.W_lex_to_core, word_stim, core_active)
            
            # 3. Project CORE back to VISUAL and LEX (for retrieval)
            self._hebbian_update(self.W_core_to_visual, core_active, grounding_stim)
            self._hebbian_update(self.W_core_to_lex, core_active, word_stim)
        
        # Store learned word
        self.word_assemblies[word] = word_stim
        self.visual_assemblies[grounding[0] if grounding else word.upper()] = grounding_stim
        self.word_exposures[word] = self.word_exposures.get(word, 0) + n_rounds
    
    def test_retrieval(self, grounding: List[str], expected_word: str) -> Tuple[bool, float]:
        """
        Test word retrieval from grounding.
        
        Path: VISUAL -> CORE -> LEX
        """
        if expected_word not in self.word_assemblies:
            return False, 0.0
        
        # Get grounding stimulus
        grounding_stim = self._get_or_create_stimulus(grounding[0] if grounding else expected_word.upper())
        
        # VISUAL -> CORE
        core_activations = self.W_visual_to_core[:, grounding_stim].sum(dim=1)
        _, core_active = torch.topk(core_activations, self.k)
        
        # CORE -> LEX
        lex_activations = self.W_core_to_lex[:, core_active].sum(dim=1)
        _, lex_active = torch.topk(lex_activations, self.k)
        
        # Check overlap with expected word assembly
        expected_assembly = self.word_assemblies[expected_word]
        
        # Convert to sets for overlap calculation
        lex_set = set(lex_active.cpu().numpy())
        expected_set = set(expected_assembly.cpu().numpy())
        
        overlap = len(lex_set & expected_set)
        confidence = overlap / self.k
        success = confidence > 0.5
        
        return success, confidence
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'words_learned': len(self.word_assemblies),
            'visual_concepts': len(self.visual_assemblies),
            'total_exposures': sum(self.word_exposures.values()),
            'avg_exposures_per_word': (
                sum(self.word_exposures.values()) / max(len(self.word_exposures), 1)
            ),
        }


def benchmark_cpu_vs_gpu():
    """Compare CPU vs GPU performance"""
    print("=" * 60)
    print("CPU vs GPU BENCHMARK")
    print("=" * 60)
    
    if not HAS_TORCH:
        print("PyTorch CUDA not available, skipping GPU benchmark")
        return
    
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
    
    # Test configurations - start small
    configs = [
        (1000, 50),
        (5000, 100),
    ]
    
    print(f"\n{'n':>8} {'k':>6} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 50)
    
    for n, k in configs:
        print(f"\n[Testing n={n}, k={k}]")
        
        # GPU timing FIRST (faster, less debug output)
        print("  Creating GPU learner...")
        try:
            gpu_learner = GPUAssemblyLearner(n=n, k=k, verbose=False)
            print("  GPU learner created, warming up...")
            
            # Warm up
            gpu_learner.learn_word('warmup', 'NOUN', ['WARMUP'], n_rounds=1)
            torch.cuda.synchronize()
            print("  Warmup done, timing GPU...")
            
            start = time.perf_counter()
            gpu_learner.learn_word('dog', 'NOUN', ['DOG'], n_rounds=5)
            torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - start) * 1000
            print(f"  GPU time: {gpu_time:.2f} ms")
        except Exception as e:
            gpu_time = float('inf')
            print(f"  GPU error: {e}")
        
        # CPU timing (slower, lots of debug output from brain.py)
        print("  Creating CPU learner (may have debug output)...")
        try:
            # Suppress debug by redirecting
            import io
            import contextlib
            
            # Create learner with suppressed output
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    from src.lexicon.assembly_language_learner import AssemblyLanguageLearner
                    cpu_learner = AssemblyLanguageLearner(n=n, k=k, verbose=False)
            
            print("  CPU learner created, timing CPU...")
            
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    start = time.perf_counter()
                    cpu_learner.learn_word('dog', 'NOUN', ['DOG'], n_rounds=5)
                    cpu_time = (time.perf_counter() - start) * 1000
            
            print(f"  CPU time: {cpu_time:.2f} ms")
        except Exception as e:
            cpu_time = float('inf')
            print(f"  CPU error: {e}")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n{n:>8} {k:>6} {cpu_time:>12.2f} {gpu_time:>12.2f} {speedup:>10.1f}x")


def test_gpu_learning_curve():
    """Test learning curve with GPU acceleration"""
    print("\n" + "=" * 60)
    print("GPU WORD LEARNING CURVE")
    print("=" * 60)
    
    if not HAS_TORCH:
        print("PyTorch CUDA not available")
        return
    
    results = []
    
    for n_exposures in [1, 2, 3, 5, 7, 10, 15, 20]:
        learner = GPUAssemblyLearner(n=10000, k=100, verbose=False)
        
        word = 'dog'
        grounding = ['DOG', 'ANIMAL']
        learner.learn_word(word, 'NOUN', grounding, n_rounds=n_exposures)
        
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


if __name__ == '__main__':
    benchmark_cpu_vs_gpu()
    test_gpu_learning_curve()

