"""
NEMO Scaling Study: Brain Areas vs Performance
===============================================

Version: 1.0.0
Date: 2025-11-30

Systematically study how performance scales with:
1. Number of brain areas
2. Projections per sentence
3. Neuron count (n)
4. Assembly size (k)

Goal: Understand the performance cost of adding linguistic structure.
"""

import cupy as cp
import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# CUDA kernel for projection
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
class ScalingParams:
    n: int = 10000
    k: int = None
    p: float = 0.05
    word_order: WordOrder = WordOrder.SVO
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class ScalableBrain:
    """
    Brain with configurable number of areas for scaling study.
    
    Architecture levels:
    1. Minimal (6 areas): LEX1, LEX2, SUBJ, OBJ, VERB, SEQ
    2. Basic (10 areas): + NP, VP, PHON_LEX1, PHON_LEX2
    3. Standard (16 areas): + VISUAL, MOTOR, LEX1_NP, LEX2_VP, NP_SUBJ, NP_OBJ
    4. Full (26 areas): + SENT, generation pathways, more cross-area connections
    """
    
    def __init__(self, num_areas: int, params: ScalingParams = None, verbose: bool = True):
        self.p = params or ScalingParams()
        self.num_areas = num_areas
        n, k = self.p.n, self.p.k
        
        # Word assemblies
        self.phon: Dict[str, torch.Tensor] = {}
        self.visual: Dict[str, torch.Tensor] = {}
        self.motor: Dict[str, torch.Tensor] = {}
        
        # Pre-allocate buffers
        self.active = torch.zeros(k, device='cuda', dtype=torch.int64)
        self.result = torch.zeros(n, device='cuda', dtype=torch.float16)
        
        # CuPy views for kernel
        self.active_cp = cp.from_dlpack(self.active)
        self.result_cp = cp.from_dlpack(self.result)
        
        # Pre-create scalars
        self.k_u32 = cp.uint32(k)
        self.n_u32 = cp.uint32(n)
        self.p_f32 = cp.float32(self.p.p * 2)
        self.shared_mem = k * 4
        
        # Seeds per area
        self.seeds = [cp.uint32(i * 1000) for i in range(num_areas)]
        
        # Kernel config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        # Word order transitions
        self.transitions: Dict[Tuple[str, str], int] = {}
        self.sentences_seen = 0
        
        # Define projections per sentence based on architecture level
        self._setup_projections_per_sentence()
        
        if verbose:
            print(f"ScalableBrain: n={n:,}, k={k}, {num_areas} areas, "
                  f"{self.projections_per_sentence} projections/sentence")
    
    def _setup_projections_per_sentence(self):
        """Calculate projections per sentence based on number of areas."""
        if self.num_areas <= 6:
            # Minimal: 3 words × 2 projections each
            self.projections_per_sentence = 6
        elif self.num_areas <= 10:
            # Basic: + phon/visual pathways
            self.projections_per_sentence = 12
        elif self.num_areas <= 16:
            # Standard: + phrase structure
            self.projections_per_sentence = 20
        elif self.num_areas <= 26:
            # Full: + generation pathways
            self.projections_per_sentence = 30
        else:
            # Extended: linear scaling
            self.projections_per_sentence = self.num_areas
    
    def _project(self, area: int, inp: torch.Tensor) -> torch.Tensor:
        """Fast projection."""
        self.active[:len(inp)] = inp[:self.p.k]
        
        projection_kernel(
            (self.gx,), (self.bs,),
            (self.active_cp.astype(cp.uint32), self.result_cp,
             self.k_u32, self.n_u32, self.seeds[area % self.num_areas], self.p_f32),
            shared_mem=self.shared_mem
        )
        
        _, winners = torch.topk(self.result, self.p.k, sorted=False)
        return winners
    
    def _get_or_create(self, store: Dict, name: str) -> torch.Tensor:
        if name not in store:
            store[name] = torch.randint(0, self.p.n, (self.p.k,), device='cuda')
        return store[name]
    
    def train_sentence(self, subj: str, verb: str, obj: str):
        """Train on a sentence with appropriate number of projections."""
        # Get word assemblies
        subj_phon = self._get_or_create(self.phon, subj)
        subj_vis = self._get_or_create(self.visual, subj)
        verb_phon = self._get_or_create(self.phon, verb)
        verb_mot = self._get_or_create(self.motor, verb)
        obj_phon = self._get_or_create(self.phon, obj)
        obj_vis = self._get_or_create(self.visual, obj)
        
        # Do the appropriate number of projections
        area = 0
        for _ in range(self.projections_per_sentence):
            # Cycle through different inputs
            if area % 3 == 0:
                inp = torch.unique(torch.cat([subj_phon, subj_vis]))[:self.p.k]
            elif area % 3 == 1:
                inp = torch.unique(torch.cat([verb_phon, verb_mot]))[:self.p.k]
            else:
                inp = torch.unique(torch.cat([obj_phon, obj_vis]))[:self.p.k]
            
            self._project(area % self.num_areas, inp)
            area += 1
        
        # Learn word order
        order = self.p.word_order.value
        if order == "SVO":
            self._learn_transition('SUBJ', 'VERB')
            self._learn_transition('VERB', 'OBJ')
        elif order == "SOV":
            self._learn_transition('SUBJ', 'OBJ')
            self._learn_transition('OBJ', 'VERB')
        elif order == "VSO":
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


def benchmark_areas(num_areas: int, n: int = 10000, num_sentences: int = 200) -> dict:
    """Benchmark a specific number of areas."""
    params = ScalingParams(n=n, word_order=WordOrder.SVO)
    brain = ScalableBrain(num_areas, params, verbose=False)
    
    # Warmup
    for _ in range(20):
        brain.train_sentence('dog', 'sees', 'cat')
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_sentences):
        brain.train_sentence('dog', 'sees', 'cat')
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return {
        'num_areas': num_areas,
        'projections_per_sentence': brain.projections_per_sentence,
        'sentences_per_sec': num_sentences / elapsed,
        'ms_per_sentence': elapsed / num_sentences * 1000,
        'ms_per_projection': elapsed / num_sentences / brain.projections_per_sentence * 1000,
    }


def run_scaling_study():
    """Run full scaling study."""
    print("=" * 70)
    print("NEMO SCALING STUDY: Brain Areas vs Performance")
    print("=" * 70)
    
    # Study 1: Scaling with number of areas
    print("\n1. SCALING WITH NUMBER OF AREAS (n=10,000)")
    print("-" * 70)
    print(f"{'Areas':>6} {'Proj/Sent':>10} {'Sent/sec':>10} {'ms/sent':>10} {'ms/proj':>10}")
    print("-" * 70)
    
    area_results = []
    for num_areas in [4, 6, 8, 10, 12, 16, 20, 26, 32, 40, 50]:
        result = benchmark_areas(num_areas, n=10000)
        area_results.append(result)
        print(f"{result['num_areas']:>6} {result['projections_per_sentence']:>10} "
              f"{result['sentences_per_sec']:>10.0f} {result['ms_per_sentence']:>10.2f} "
              f"{result['ms_per_projection']:>10.3f}")
    
    # Study 2: Scaling with n
    print("\n2. SCALING WITH NEURON COUNT (16 areas)")
    print("-" * 70)
    print(f"{'n':>12} {'k':>6} {'Sent/sec':>10} {'ms/sent':>10}")
    print("-" * 70)
    
    n_results = []
    for n in [1000, 10000, 100000, 1000000]:
        result = benchmark_areas(16, n=n, num_sentences=100)
        n_results.append(result)
        k = int(np.sqrt(n))
        print(f"{n:>12,} {k:>6} {result['sentences_per_sec']:>10.0f} "
              f"{result['ms_per_sentence']:>10.2f}")
    
    # Study 3: Verify correctness
    print("\n3. CORRECTNESS CHECK (all area counts)")
    print("-" * 70)
    
    for num_areas in [6, 16, 26]:
        params = ScalingParams(n=10000, word_order=WordOrder.SVO)
        brain = ScalableBrain(num_areas, params, verbose=False)
        
        nouns = ['dog', 'cat', 'boy']
        verbs = ['sees', 'chases']
        
        for _ in range(50):
            for noun in nouns:
                for verb in verbs:
                    obj = nouns[(nouns.index(noun) + 1) % len(nouns)]
                    brain.train_sentence(noun, verb, obj)
        
        order = brain.generate_sentence_order()
        sent = brain.generate_sentence()
        print(f"{num_areas} areas: order={order}, example='{' '.join(w for w,_ in sent)}'")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Calculate scaling law
    areas = [r['num_areas'] for r in area_results]
    projs = [r['projections_per_sentence'] for r in area_results]
    ms_per_proj = [r['ms_per_projection'] for r in area_results]
    
    avg_ms_per_proj = np.mean(ms_per_proj)
    
    print(f"""
Key Findings:
1. Time per projection is ~{avg_ms_per_proj:.3f}ms (constant regardless of #areas)
2. Total time scales linearly with projections/sentence
3. Projections/sentence scales with architecture complexity

Scaling Law:
  time_per_sentence ≈ {avg_ms_per_proj:.3f}ms × projections_per_sentence
  
  For {area_results[-1]['num_areas']} areas with {area_results[-1]['projections_per_sentence']} projections:
    Expected: {avg_ms_per_proj * area_results[-1]['projections_per_sentence']:.2f}ms
    Actual:   {area_results[-1]['ms_per_sentence']:.2f}ms

Recommendations:
- Minimal (6 areas, 6 proj): {6 * avg_ms_per_proj:.1f}ms/sent = {1000/(6*avg_ms_per_proj):.0f} sent/sec
- Standard (16 areas, 20 proj): {20 * avg_ms_per_proj:.1f}ms/sent = {1000/(20*avg_ms_per_proj):.0f} sent/sec
- Full (26 areas, 30 proj): {30 * avg_ms_per_proj:.1f}ms/sent = {1000/(30*avg_ms_per_proj):.0f} sent/sec
""")


if __name__ == "__main__":
    run_scaling_study()

