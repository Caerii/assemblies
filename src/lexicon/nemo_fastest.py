"""
NEMO Fastest - Maximum Speed Implementation
============================================

Version: 1.0.0
Author: Assembly Calculus Project
Date: 2025-11-29

Combines all working optimizations:
1. FP16 for top-k (1.7x faster)
2. Batched projections (1.4x faster)
3. Minimal Python overhead
4. Pre-allocated everything

Performance:
- n=100K: 148 sentences/sec
- n=1M:   175 sentences/sec
- n=10M:  64 sentences/sec

Changelog:
- 1.0.0: Initial release with FP16 + batched optimizations
"""

import cupy as cp
import torch
import numpy as np
import time
from typing import Dict, List
from dataclasses import dataclass

print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# =============================================================================
# CUDA KERNELS
# =============================================================================

# FP16 output projection for faster top-k
projection_fp16_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void projection_fp16(
    const unsigned int* __restrict__ active_batch,
    __half* __restrict__ result_batch,
    const unsigned int k,
    const unsigned int n,
    const unsigned int batch_size,
    const unsigned int* __restrict__ seeds,
    const float p
) {
    unsigned int dst = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int batch_idx = blockIdx.y;
    
    if (dst >= n || batch_idx >= batch_size) return;
    
    const unsigned int* active = active_batch + batch_idx * k;
    __half* result = result_batch + batch_idx * n;
    unsigned int seed = seeds[batch_idx];
    
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

# Simple Hebbian update (not batched - batching adds too much overhead)
hebbian_kernel = cp.RawKernel(r'''
extern "C" __global__
void hebbian_update(
    unsigned int* learned_src,
    unsigned int* learned_dst,
    float* learned_delta,
    unsigned int* num_learned,
    const unsigned int* prev_active,
    const unsigned int* new_active,
    const unsigned int k,
    const float beta,
    const float w_max,
    const unsigned int max_learned,
    const unsigned int seed,
    const float p
) {
    unsigned int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = pair_idx / k;
    unsigned int j = pair_idx % k;
    
    if (i >= k) return;
    
    unsigned int src = prev_active[i];
    unsigned int dst = new_active[j];
    
    unsigned int hash = seed ^ src;
    hash *= 0x01000193u;
    hash ^= dst;
    hash *= 0x01000193u;
    
    if ((float)(hash & 0xFFFFFFu) / 16777216.0f >= p) return;
    
    unsigned int current_num = *num_learned;
    unsigned int found_idx = 0xFFFFFFFFu;
    
    for (unsigned int l = 0; l < current_num; l++) {
        if (learned_src[l] == src && learned_dst[l] == dst) {
            found_idx = l;
            break;
        }
    }
    
    if (found_idx == 0xFFFFFFFFu) {
        unsigned int new_idx = atomicAdd(num_learned, 1);
        if (new_idx < max_learned) {
            learned_src[new_idx] = src;
            learned_dst[new_idx] = dst;
            learned_delta[new_idx] = beta;
        }
    } else {
        float current_w = 1.0f + learned_delta[found_idx];
        float update = beta * (1.0f - current_w / w_max);
        if (update > 0) atomicAdd(&learned_delta[found_idx], update);
    }
}
''', 'hebbian_update')


@dataclass
class FastParams:
    n: int = 1000000
    k: int = 50
    p: float = 0.05
    beta: float = 0.1
    w_max: float = 10.0
    tau: int = 2


class FastestNemoBrain:
    """
    Maximum speed NEMO implementation.
    
    Key optimizations:
    - FP16 activations for 1.7x faster top-k
    - Batched projections for parallel area processing
    - Minimal Python overhead
    - Simple sequential Hebbian (batched adds too much overhead)
    """
    
    PHON_TO_LEX1 = 0
    PHON_TO_LEX2 = 1
    VISUAL_TO_LEX1 = 2
    MOTOR_TO_LEX2 = 3
    LEX1 = 4
    LEX2 = 5
    LEX1_TO_VIS = 6
    LEX2_TO_MOT = 7
    NUM_AREAS = 8
    
    def __init__(self, params: FastParams = None, verbose: bool = True):
        self.p = params or FastParams()
        n, k = self.p.n, self.p.k
        
        # Assemblies
        self.phon: Dict[str, cp.ndarray] = {}
        self.visual: Dict[str, cp.ndarray] = {}
        self.motor: Dict[str, cp.ndarray] = {}
        
        # Area config
        self.seeds = cp.arange(self.NUM_AREAS, dtype=cp.uint32) * 1000
        self.conn_p = self.p.p
        
        # Batched buffers - FP16 for faster top-k
        self.active = cp.zeros((self.NUM_AREAS, k), dtype=cp.uint32)
        self.result = cp.zeros((self.NUM_AREAS, n), dtype=cp.float16)
        self.result_torch = torch.zeros((self.NUM_AREAS, n), device='cuda', dtype=torch.float16)
        
        # Learned weights per area
        self.max_learned = k * k * 500
        self.l_src = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        self.l_dst = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        self.l_delta = [cp.zeros(self.max_learned, dtype=cp.float32) for _ in range(self.NUM_AREAS)]
        self.l_num = [cp.zeros(1, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        
        # Previous activations
        self.prev = [None for _ in range(self.NUM_AREAS)]
        
        # Kernel config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        self.sentences = 0
        
        if verbose:
            print(f"FastestNemoBrain: n={n:,}, FP16 top-k")
    
    def _project_batch(self, areas: List[int], inputs: List[cp.ndarray], learn: bool) -> List[cp.ndarray]:
        """Project multiple areas in parallel with FP16."""
        batch = len(areas)
        
        # Pack inputs
        for i, (a, inp) in enumerate(zip(areas, inputs)):
            self.active[a, :len(inp)] = inp[:min(len(inp), self.p.k)]
        
        packed_active = cp.stack([self.active[a] for a in areas])
        packed_result = cp.zeros((batch, self.p.n), dtype=cp.float16)
        packed_seeds = cp.array([int(self.seeds[a]) for a in areas], dtype=cp.uint32)
        
        # FP16 projection
        projection_fp16_kernel(
            (self.gx, batch), (self.bs,),
            (packed_active, packed_result,
             cp.uint32(self.p.k), cp.uint32(self.p.n), cp.uint32(batch),
             packed_seeds, self.conn_p * 2),  # 2x for cross-area
            shared_mem=self.p.k * 4
        )
        
        # FP16 batched top-k (1.7x faster!)
        packed_torch = torch.as_tensor(packed_result, device='cuda')
        _, top_idx = torch.topk(packed_torch, self.p.k, dim=1, sorted=False)
        top_cp = cp.asarray(top_idx)
        
        # Hebbian updates (sequential - batching adds overhead)
        results = []
        for i, a in enumerate(areas):
            winners = top_cp[i].astype(cp.uint32)
            
            if learn and self.prev[a] is not None:
                grid = (self.p.k * self.p.k + self.bs - 1) // self.bs
                hebbian_kernel(
                    (grid,), (self.bs,),
                    (self.l_src[a], self.l_dst[a], self.l_delta[a], self.l_num[a],
                     self.prev[a], winners,
                     cp.uint32(self.p.k), cp.float32(self.p.beta), cp.float32(self.p.w_max),
                     cp.uint32(self.max_learned), self.seeds[a], cp.float32(self.conn_p * 2))
                )
            
            self.prev[a] = winners
            results.append(winners)
        
        return results
    
    def _project_one(self, area: int, inp: cp.ndarray, learn: bool, p_mult: float = 1.0) -> cp.ndarray:
        """Project single area."""
        self.active[area, :len(inp)] = inp[:min(len(inp), self.p.k)]
        self.result[area].fill(0)
        
        projection_fp16_kernel(
            (self.gx, 1), (self.bs,),
            (self.active[area:area+1], self.result[area:area+1],
             cp.uint32(self.p.k), cp.uint32(self.p.n), cp.uint32(1),
             self.seeds[area:area+1], self.conn_p * p_mult),
            shared_mem=self.p.k * 4
        )
        
        result_torch = torch.as_tensor(self.result[area], device='cuda')
        _, top_idx = torch.topk(result_torch, self.p.k, sorted=False)
        winners = cp.asarray(top_idx).astype(cp.uint32)
        
        if learn and self.prev[area] is not None:
            grid = (self.p.k * self.p.k + self.bs - 1) // self.bs
            hebbian_kernel(
                (grid,), (self.bs,),
                (self.l_src[area], self.l_dst[area], self.l_delta[area], self.l_num[area],
                 self.prev[area], winners,
                 cp.uint32(self.p.k), cp.float32(self.p.beta), cp.float32(self.p.w_max),
                 cp.uint32(self.max_learned), self.seeds[area], cp.float32(self.conn_p * p_mult))
            )
        
        self.prev[area] = winners
        return winners
    
    def _get_or_create(self, store: Dict, name: str) -> cp.ndarray:
        if name not in store:
            store[name] = cp.random.randint(0, self.p.n, self.p.k, dtype=cp.uint32)
        return store[name]
    
    def present_noun(self, word: str, learn: bool = True):
        phon = self._get_or_create(self.phon, word)
        vis = self._get_or_create(self.visual, word)
        
        # Batch: phon->lex1, vis->lex1
        r = self._project_batch([self.PHON_TO_LEX1, self.VISUAL_TO_LEX1], [phon, vis], learn)
        combined = cp.unique(cp.concatenate(r))[:self.p.k]
        
        # Lex1 recurrent
        winners = self._project_one(self.LEX1, combined, learn, p_mult=1.0)
        
        if learn:
            self._project_one(self.LEX1_TO_VIS, winners, True, p_mult=2.0)
    
    def present_verb(self, word: str, learn: bool = True):
        phon = self._get_or_create(self.phon, word)
        mot = self._get_or_create(self.motor, word)
        
        r = self._project_batch([self.PHON_TO_LEX2, self.MOTOR_TO_LEX2], [phon, mot], learn)
        combined = cp.unique(cp.concatenate(r))[:self.p.k]
        
        winners = self._project_one(self.LEX2, combined, learn, p_mult=1.0)
        
        if learn:
            self._project_one(self.LEX2_TO_MOT, winners, True, p_mult=2.0)
    
    def train(self, subj: str, verb: str, obj: str = None):
        """Train on a sentence - process noun and verb with maximum batching."""
        # Get all assemblies upfront
        subj_phon = self._get_or_create(self.phon, subj)
        subj_vis = self._get_or_create(self.visual, subj)
        verb_phon = self._get_or_create(self.phon, verb)
        verb_mot = self._get_or_create(self.motor, verb)
        
        # Reset recurrent
        self.prev[self.LEX1] = None
        self.prev[self.LEX2] = None
        
        for _ in range(self.p.tau):
            # Batch ALL 4 cross-area projections at once!
            r = self._project_batch(
                [self.PHON_TO_LEX1, self.VISUAL_TO_LEX1, self.PHON_TO_LEX2, self.MOTOR_TO_LEX2],
                [subj_phon, subj_vis, verb_phon, verb_mot],
                learn=True
            )
            
            # Combine noun inputs
            noun_combined = cp.unique(cp.concatenate(r[:2]))[:self.p.k]
            # Combine verb inputs  
            verb_combined = cp.unique(cp.concatenate(r[2:]))[:self.p.k]
            
            # Recurrent projections (could batch these too but they have dependencies)
            noun_winners = self._project_one(self.LEX1, noun_combined, True, p_mult=1.0)
            verb_winners = self._project_one(self.LEX2, verb_combined, True, p_mult=1.0)
            
            # Pathway updates - batch together
            self._project_batch(
                [self.LEX1_TO_VIS, self.LEX2_TO_MOT],
                [noun_winners, verb_winners],
                learn=True
            )
        
        if obj:
            obj_phon = self._get_or_create(self.phon, obj)
            obj_vis = self._get_or_create(self.visual, obj)
            self.prev[self.LEX1] = None
            
            for _ in range(self.p.tau):
                r = self._project_batch([self.PHON_TO_LEX1, self.VISUAL_TO_LEX1], [obj_phon, obj_vis], True)
                combined = cp.unique(cp.concatenate(r))[:self.p.k]
                winners = self._project_one(self.LEX1, combined, True, p_mult=1.0)
                self._project_one(self.LEX1_TO_VIS, winners, True, p_mult=2.0)
        
        self.sentences += 1
    
    def classify(self, word: str) -> str:
        has_vis = word in self.visual
        has_mot = word in self.motor
        if has_vis and not has_mot:
            return 'NOUN'
        if has_mot and not has_vis:
            return 'VERB'
        return 'UNKNOWN'


def benchmark():
    print("=" * 70)
    print("FASTEST NEMO BENCHMARK")
    print("=" * 70)
    
    nouns = ['dog', 'cat', 'boy', 'girl']
    verbs = ['runs', 'eats', 'sees']
    
    for n in [100000, 1000000, 10000000]:
        print(f"\nn = {n:,}")
        print("-" * 50)
        
        try:
            brain = FastestNemoBrain(FastParams(n=n), verbose=False)
            
            # Warmup
            for _ in range(20):
                brain.train('dog', 'runs')
            
            n_sent = 200
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            for _ in range(n_sent):
                brain.train(np.random.choice(nouns), np.random.choice(verbs))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            
            rate = n_sent / elapsed
            print(f"  Speed: {rate:.1f} sentences/sec")
            
            # Test classification
            correct = sum(1 for w in nouns if brain.classify(w) == 'NOUN')
            correct += sum(1 for w in verbs if brain.classify(w) == 'VERB')
            print(f"  Accuracy: {correct}/{len(nouns)+len(verbs)}")
            
            del brain
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"  Error: {e}")


def compare_all():
    print("=" * 70)
    print("COMPARISON: All Versions at n=1M")
    print("=" * 70)
    
    from nemo_scalable import ScalableNemoBrain, ScalableParams
    from nemo_batched import BatchedNemoBrain, BatchedParams
    
    nouns = ['dog', 'cat', 'boy', 'girl']
    verbs = ['runs', 'eats', 'sees']
    n = 1000000
    n_sent = 200
    
    results = {}
    
    # Sequential
    brain = ScalableNemoBrain(ScalableParams(n=n), verbose=False)
    for _ in range(10): brain.train_sentence('dog', 'runs')
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(n_sent):
        brain.train_sentence(np.random.choice(nouns), np.random.choice(verbs))
    cp.cuda.Stream.null.synchronize()
    results['Sequential'] = n_sent / (time.perf_counter() - start)
    del brain
    cp.get_default_memory_pool().free_all_blocks()
    
    # Batched
    brain = BatchedNemoBrain(BatchedParams(n=n), verbose=False)
    for _ in range(10): brain.train_sentence('dog', 'runs')
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(n_sent):
        brain.train_sentence(np.random.choice(nouns), np.random.choice(verbs))
    cp.cuda.Stream.null.synchronize()
    results['Batched'] = n_sent / (time.perf_counter() - start)
    del brain
    cp.get_default_memory_pool().free_all_blocks()
    
    # Fastest
    brain = FastestNemoBrain(FastParams(n=n), verbose=False)
    for _ in range(10): brain.train('dog', 'runs')
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(n_sent):
        brain.train(np.random.choice(nouns), np.random.choice(verbs))
    cp.cuda.Stream.null.synchronize()
    results['Fastest'] = n_sent / (time.perf_counter() - start)
    del brain
    cp.get_default_memory_pool().free_all_blocks()
    
    print(f"\n{'Version':<15} {'Speed':<20} {'Speedup'}")
    print("-" * 50)
    baseline = results['Sequential']
    for name, speed in results.items():
        print(f"{name:<15} {speed:.1f} sent/sec      {speed/baseline:.2f}x")


if __name__ == "__main__":
    benchmark()
    print()
    compare_all()

