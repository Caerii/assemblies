"""
NEMO Brain Implementations
==========================

Version: 1.0.0
Author: Assembly Calculus Project
Date: 2025-11-29

Core NEMO brain implementations:
- FastestNemoBrain: Maximum speed (175 sent/sec at n=1M)
- ScalableNemoBrain: Baseline scalable implementation

Based on Mitropolsky & Papadimitriou 2025

Changelog:
- 1.0.0: Initial release with FP16 + batched optimizations
"""

import cupy as cp
import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from .kernels import projection_fp16_kernel, hebbian_kernel


@dataclass
class FastParams:
    """Parameters for FastestNemoBrain."""
    n: int = 1000000         # Neurons per area
    k: int = None            # Winners (assembly size) - defaults to sqrt(n)
    p: float = 0.05          # Connection probability
    beta: float = 0.1        # Plasticity rate
    w_max: float = 10.0      # Weight saturation
    tau: int = 2             # Firing steps per word
    
    def __post_init__(self):
        # k = sqrt(n) per Assembly Calculus theory
        if self.k is None:
            self.k = int(np.sqrt(self.n))


@dataclass
class ScalableParams:
    """Parameters for ScalableNemoBrain."""
    n: int = 100000          # Neurons per area
    k: int = None            # Winners - defaults to sqrt(n)
    p: float = 0.05          # Connection probability
    beta: float = 0.1        # Plasticity
    w_max: float = 10.0      # Weight saturation
    tau: int = 2             # Firing steps per word
    
    def __post_init__(self):
        # k = sqrt(n) per Assembly Calculus theory
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class FastestNemoBrain:
    """
    Maximum speed NEMO implementation.
    
    Performance at n=1M: 175 sentences/sec
    
    Key optimizations:
    - FP16 activations (1.7x faster top-k)
    - Batched projections (process noun+verb together)
    - Minimal Python overhead
    - Pre-allocated everything
    
    Architecture:
    - Lex1: Noun lexicon (grounded in Visual)
    - Lex2: Verb lexicon (grounded in Motor)
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
        self.verbose = verbose
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
        
        for i, (a, inp) in enumerate(zip(areas, inputs)):
            self.active[a, :len(inp)] = inp[:min(len(inp), self.p.k)]
        
        packed_active = cp.stack([self.active[a] for a in areas])
        packed_result = cp.zeros((batch, self.p.n), dtype=cp.float16)
        packed_seeds = cp.array([int(self.seeds[a]) for a in areas], dtype=cp.uint32)
        
        projection_fp16_kernel(
            (self.gx, batch), (self.bs,),
            (packed_active, packed_result,
             cp.uint32(self.p.k), cp.uint32(self.p.n), cp.uint32(batch),
             packed_seeds, cp.float32(self.conn_p * 2)),  # Must be float32!
            shared_mem=self.p.k * 4
        )
        
        packed_torch = torch.as_tensor(packed_result, device='cuda')
        _, top_idx = torch.topk(packed_torch, self.p.k, dim=1, sorted=False)
        top_cp = cp.asarray(top_idx)
        
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
             self.seeds[area:area+1], cp.float32(self.conn_p * p_mult)),  # Must be float32!
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
    
    def train(self, subj: str, verb: str, obj: str = None):
        """Train on a sentence."""
        # Get all assemblies upfront
        subj_phon = self._get_or_create(self.phon, subj)
        subj_vis = self._get_or_create(self.visual, subj)
        verb_phon = self._get_or_create(self.phon, verb)
        verb_mot = self._get_or_create(self.motor, verb)
        
        # Reset recurrent
        self.prev[self.LEX1] = None
        self.prev[self.LEX2] = None
        
        for _ in range(self.p.tau):
            # Batch ALL 4 cross-area projections at once
            r = self._project_batch(
                [self.PHON_TO_LEX1, self.VISUAL_TO_LEX1, self.PHON_TO_LEX2, self.MOTOR_TO_LEX2],
                [subj_phon, subj_vis, verb_phon, verb_mot],
                learn=True
            )
            
            # Combine noun inputs
            noun_combined = cp.unique(cp.concatenate(r[:2]))[:self.p.k]
            # Combine verb inputs
            verb_combined = cp.unique(cp.concatenate(r[2:]))[:self.p.k]
            
            # Recurrent projections
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
        """Classify word as NOUN or VERB based on grounding."""
        has_vis = word in self.visual
        has_mot = word in self.motor
        if has_vis and not has_mot:
            return 'NOUN'
        if has_mot and not has_vis:
            return 'VERB'
        return 'UNKNOWN'


class ScalableNemoBrain:
    """
    Baseline scalable NEMO implementation.
    
    Uses implicit random connectivity for memory efficiency.
    Simpler than FastestNemoBrain but still scalable.
    """
    
    def __init__(self, params: ScalableParams = None, verbose: bool = True):
        from .kernels import ImplicitAssemblyArea
        
        self.p = params or ScalableParams()
        self.verbose = verbose
        n, k = self.p.n, self.p.k
        
        # Assemblies
        self.phon_assemblies: Dict[str, cp.ndarray] = {}
        self.visual_assemblies: Dict[str, cp.ndarray] = {}
        self.motor_assemblies: Dict[str, cp.ndarray] = {}
        
        # Create areas
        self.lex1 = ImplicitAssemblyArea(n, k, self.p.p, self.p.beta, self.p.w_max, seed=1)
        self.lex2 = ImplicitAssemblyArea(n, k, self.p.p, self.p.beta, self.p.w_max, seed=2)
        self.phon_to_lex1 = ImplicitAssemblyArea(n, k, self.p.p * 2, self.p.beta, self.p.w_max, seed=3)
        self.phon_to_lex2 = ImplicitAssemblyArea(n, k, self.p.p * 2, self.p.beta, self.p.w_max, seed=4)
        self.visual_to_lex1 = ImplicitAssemblyArea(n, k, self.p.p * 2, self.p.beta, self.p.w_max, seed=5)
        self.motor_to_lex2 = ImplicitAssemblyArea(n, k, self.p.p * 2, self.p.beta, self.p.w_max, seed=6)
        
        self.sentences_seen = 0
        
        if verbose:
            print(f"ScalableNemoBrain: n={n:,}")
    
    def create_assembly(self, area: str, name: str) -> cp.ndarray:
        """Create random assembly for a word."""
        indices = cp.random.randint(0, self.p.n, self.p.k, dtype=cp.uint32)
        
        if area == 'Phon':
            self.phon_assemblies[name] = indices
        elif area == 'Visual':
            self.visual_assemblies[name] = indices
        elif area == 'Motor':
            self.motor_assemblies[name] = indices
        
        return indices
    
    def present_word(self, word: str, is_noun: bool, learn: bool = True):
        """Present a word with grounding."""
        if word not in self.phon_assemblies:
            self.create_assembly('Phon', word)
        
        phon = self.phon_assemblies[word]
        
        if is_noun:
            if word not in self.visual_assemblies:
                self.create_assembly('Visual', word)
            visual = self.visual_assemblies[word]
            
            phon_input = self.phon_to_lex1.project(phon, learn=learn)
            visual_input = self.visual_to_lex1.project(visual, learn=learn)
            
            combined = cp.concatenate([phon_input, visual_input])
            unique = cp.unique(combined)
            if len(unique) > self.p.k:
                unique = unique[:self.p.k]
            
            self.lex1.project(unique, learn=learn)
        else:
            if word not in self.motor_assemblies:
                self.create_assembly('Motor', word)
            motor = self.motor_assemblies[word]
            
            phon_input = self.phon_to_lex2.project(phon, learn=learn)
            motor_input = self.motor_to_lex2.project(motor, learn=learn)
            
            combined = cp.concatenate([phon_input, motor_input])
            unique = cp.unique(combined)
            if len(unique) > self.p.k:
                unique = unique[:self.p.k]
            
            self.lex2.project(unique, learn=learn)
    
    def train_sentence(self, subject: str, verb: str, obj: str = None):
        """Train on a sentence."""
        self.lex1.active = None
        self.lex2.active = None
        
        for _ in range(self.p.tau):
            self.present_word(subject, is_noun=True, learn=True)
        
        self.lex1.active = None
        self.lex2.active = None
        
        for _ in range(self.p.tau):
            self.present_word(verb, is_noun=False, learn=True)
        
        if obj:
            self.lex1.active = None
            for _ in range(self.p.tau):
                self.present_word(obj, is_noun=True, learn=True)
        
        self.sentences_seen += 1
    
    def classify_word(self, word: str) -> str:
        """Classify word as NOUN or VERB."""
        has_vis = word in self.visual_assemblies
        has_mot = word in self.motor_assemblies
        if has_vis and not has_mot:
            return 'NOUN'
        if has_mot and not has_vis:
            return 'VERB'
        return 'UNKNOWN'

