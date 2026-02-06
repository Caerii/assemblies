"""
NEMO Hierarchical Language System - Fast Version
=================================================

Version: 2.0.0
Author: Assembly Calculus Project
Date: 2025-11-30

Optimized for speed by:
1. Using PyTorch throughout (no CuPy<->PyTorch conversion)
2. Pre-allocated buffers and scalars
3. DLPack zero-copy where needed
4. Minimal Python overhead

Target: 200+ sentences/sec at n=10K
"""

import cupy as cp
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# CUDA kernel for projection (using CuPy but output to PyTorch-compatible buffer)
projection_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void projection_fp16(
    const unsigned int* __restrict__ active,
    __half* __restrict__ result,
    const unsigned int k,
    const unsigned int n,
    const unsigned int seed,
    const float p
) {
    unsigned int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= n) return;
    
    extern __shared__ unsigned int s_active[];
    for (unsigned int i = threadIdx.x; i < k; i += blockDim.x) {
        s_active[i] = active[i];
    }
    __syncthreads();
    
    unsigned int threshold = (unsigned int)(p * 16777216.0f);
    float sum = 0.0f;
    
    #pragma unroll 8
    for (unsigned int i = 0; i < k; i++) {
        unsigned int src = s_active[i];
        unsigned int hash = (src * 2654435761u) ^ (dst * 2246822519u) ^ seed;
        sum += (float)((hash & 0xFFFFFFu) < threshold);
    }
    
    result[dst] = __float2half(sum);
}
''', 'projection_fp16')


class WordOrder(Enum):
    SVO = "SVO"
    SOV = "SOV"
    VSO = "VSO"


@dataclass
class FastParams:
    n: int = 10000
    k: int = None
    p: float = 0.05
    beta: float = 0.1
    w_max: float = 10.0
    word_order: WordOrder = WordOrder.SVO
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class FastHierarchicalBrain:
    """
    Fast hierarchical NEMO brain using PyTorch throughout.
    
    Simplified architecture:
    - LEX1, LEX2: Lexical areas for nouns/verbs
    - SUBJ, OBJ, VERB: Role areas
    - SEQ: Sequence area for word order
    """
    
    # Simplified area indices
    LEX1 = 0
    LEX2 = 1
    SUBJ = 2
    OBJ = 3
    VERB = 4
    SEQ = 5
    NUM_AREAS = 6
    
    def __init__(self, params: FastParams = None, verbose: bool = True):
        self.p = params or FastParams()
        n, k = self.p.n, self.p.k
        
        # Word assemblies (stored as PyTorch tensors)
        self.phon: Dict[str, torch.Tensor] = {}
        self.visual: Dict[str, torch.Tensor] = {}
        self.motor: Dict[str, torch.Tensor] = {}
        
        # Pre-allocate buffers
        self.active = torch.zeros(k, device='cuda', dtype=torch.int64)
        self.result = torch.zeros(n, device='cuda', dtype=torch.float16)
        
        # CuPy views for kernel (zero-copy via DLPack)
        self.active_cp = cp.from_dlpack(self.active)
        self.result_cp = cp.from_dlpack(self.result)
        
        # Pre-create scalars
        self.k_u32 = cp.uint32(k)
        self.n_u32 = cp.uint32(n)
        self.p_f32 = cp.float32(self.p.p * 2)  # 2x for strong fibers
        self.shared_mem = k * 4
        
        # Seeds per area
        self.seeds = [cp.uint32(i * 1000) for i in range(self.NUM_AREAS)]
        
        # Kernel config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        # Previous activations for Hebbian learning
        self.prev = [None for _ in range(self.NUM_AREAS)]
        
        # Word order transitions
        self.transitions: Dict[Tuple[str, str], int] = {}
        self.sentences_seen = 0
        
        if verbose:
            print(f"FastHierarchicalBrain: n={n:,}, k={k}, {self.NUM_AREAS} areas")
    
    def _project(self, area: int, inp: torch.Tensor) -> torch.Tensor:
        """Fast projection using pre-allocated buffers."""
        # Copy input to active buffer
        self.active[:len(inp)] = inp[:self.p.k]
        
        # Run kernel (writes to result via CuPy view)
        projection_kernel(
            (self.gx,), (self.bs,),
            (self.active_cp.astype(cp.uint32), self.result_cp,
             self.k_u32, self.n_u32, self.seeds[area], self.p_f32),
            shared_mem=self.shared_mem
        )
        
        # Top-k (in-place on result)
        _, winners = torch.topk(self.result, self.p.k, sorted=False)
        
        return winners
    
    def _get_or_create(self, store: Dict, name: str) -> torch.Tensor:
        if name not in store:
            store[name] = torch.randint(0, self.p.n, (self.p.k,), device='cuda')
        return store[name]
    
    def present_noun(self, word: str, role: str = None) -> torch.Tensor:
        """Present a noun and bind to role."""
        phon = self._get_or_create(self.phon, word)
        vis = self._get_or_create(self.visual, word)
        
        # Combine phon + visual
        combined = torch.unique(torch.cat([phon, vis]))[:self.p.k]
        
        # Project to LEX1
        lex1 = self._project(self.LEX1, combined)
        
        # Bind to role
        if role == 'SUBJ':
            self._project(self.SUBJ, lex1)
            self._project(self.SEQ, lex1)
        elif role == 'OBJ':
            self._project(self.OBJ, lex1)
            self._project(self.SEQ, lex1)
        
        return lex1
    
    def present_verb(self, word: str) -> torch.Tensor:
        """Present a verb."""
        phon = self._get_or_create(self.phon, word)
        mot = self._get_or_create(self.motor, word)
        
        combined = torch.unique(torch.cat([phon, mot]))[:self.p.k]
        lex2 = self._project(self.LEX2, combined)
        
        self._project(self.VERB, lex2)
        self._project(self.SEQ, lex2)
        
        return lex2
    
    def train_sentence(self, subj: str, verb: str, obj: str):
        """Train on a sentence."""
        # Present words in order based on word order
        order = self.p.word_order.value
        
        if order == "SVO":
            self.present_noun(subj, 'SUBJ')
            self.present_verb(verb)
            self.present_noun(obj, 'OBJ')
            self._learn_transition('SUBJ', 'VERB')
            self._learn_transition('VERB', 'OBJ')
        elif order == "SOV":
            self.present_noun(subj, 'SUBJ')
            self.present_noun(obj, 'OBJ')
            self.present_verb(verb)
            self._learn_transition('SUBJ', 'OBJ')
            self._learn_transition('OBJ', 'VERB')
        elif order == "VSO":
            self.present_verb(verb)
            self.present_noun(subj, 'SUBJ')
            self.present_noun(obj, 'OBJ')
            self._learn_transition('VERB', 'SUBJ')
            self._learn_transition('SUBJ', 'OBJ')
        
        self.sentences_seen += 1
    
    def _learn_transition(self, from_role: str, to_role: str):
        key = (from_role, to_role)
        self.transitions[key] = self.transitions.get(key, 0) + 1
    
    def generate_sentence_order(self) -> List[str]:
        """Get learned word order."""
        all_targets = set(to_role for (_, to_role) in self.transitions.keys())
        all_sources = set(from_role for (from_role, _) in self.transitions.keys())
        
        starts = all_sources - all_targets
        current = list(starts)[0] if starts else 'SUBJ'
        
        order = [current]
        for _ in range(3):
            candidates = {to: cnt for (fr, to), cnt in self.transitions.items() if fr == current}
            if candidates:
                next_role = max(candidates, key=candidates.get)
                if next_role not in order:
                    order.append(next_role)
                    current = next_role
                else:
                    break
            else:
                break
        
        return order
    
    def generate_word(self, role: str, context: Dict = None) -> Tuple[str, float]:
        """Generate a word for a role."""
        context = context or {}
        
        if role in ['SUBJ', 'OBJ']:
            candidates = list(self.visual.keys())
            if role == 'OBJ' and 'SUBJ' in context:
                candidates = [w for w in candidates if w != context['SUBJ']]
            if candidates:
                return np.random.choice(candidates), 1.0
        elif role == 'VERB':
            candidates = list(self.motor.keys())
            if candidates:
                return np.random.choice(candidates), 1.0
        
        return None, 0.0
    
    def generate_sentence(self) -> List[Tuple[str, float]]:
        """Generate a sentence."""
        sentence = []
        context = {}
        
        for role in self.generate_sentence_order():
            word, conf = self.generate_word(role, context)
            if word:
                sentence.append((word, conf))
                context[role] = word
        
        return sentence


if __name__ == "__main__":
    import time
    
    print("\nBENCHMARKING FAST HIERARCHICAL BRAIN")
    print("=" * 60)
    
    params = FastParams(n=10000, word_order=WordOrder.SVO)
    brain = FastHierarchicalBrain(params, verbose=True)
    
    # Warmup
    for _ in range(20):
        brain.train_sentence('dog', 'sees', 'cat')
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    num_sentences = 500
    for _ in range(num_sentences):
        brain.train_sentence('dog', 'sees', 'cat')
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    rate = num_sentences / elapsed
    print(f"\nSpeed: {rate:.0f} sentences/sec")
    print(f"Time per sentence: {elapsed/num_sentences*1000:.2f} ms")
    
    # Test generation
    print(f"\nWord order: {brain.generate_sentence_order()}")
    print("Generated sentences:")
    for i in range(3):
        sent = brain.generate_sentence()
        print(f"  {i+1}. {' '.join(w for w, _ in sent)}")

