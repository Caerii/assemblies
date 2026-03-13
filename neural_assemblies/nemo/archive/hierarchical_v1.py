"""
NEMO Hierarchical Language System
==================================

Version: 1.1.0
Author: Assembly Calculus Project
Date: 2025-11-30

Hierarchical syntactic structure using Assembly Calculus:
- Lexical areas: Lex1 (nouns), Lex2 (verbs)
- Phrase areas: NP, VP, Sent
- Role areas: SUBJ, OBJ, VERB
- Sequence area: SEQ (word order learning)
- Generation pathways: LEX1_TO_PHON, LEX2_TO_PHON, etc.

Features:
- Word order learning (SVO, SOV, VSO)
- Sentence generation from learned vocabulary
- GPU-accelerated with custom CUDA kernels
- k = sqrt(n) assembly size (biologically realistic)

Based on:
- Mitropolsky & Papadimitriou 2025
- Papadimitriou et al. 2020

Changelog:
- 1.1.0: Added sentence generation capability
- 1.0.0: Initial hierarchical implementation
"""

import cupy as cp
import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .kernels import projection_fp16_kernel, hebbian_kernel


class WordOrder(Enum):
    """Supported word orders."""
    SVO = "SVO"  # Subject-Verb-Object (English)
    SOV = "SOV"  # Subject-Object-Verb (Japanese, Korean)
    VSO = "VSO"  # Verb-Subject-Object (Arabic, Irish)


@dataclass
class HierarchicalParams:
    """Parameters for HierarchicalNemoBrain."""
    n: int = 100000          # Neurons per area
    k: int = None            # Winners - defaults to sqrt(n)
    p: float = 0.05          # Connection probability
    beta: float = 0.1        # Plasticity
    w_max: float = 10.0      # Weight saturation
    tau: int = 2             # Firing steps per word
    word_order: WordOrder = WordOrder.SVO
    
    def __post_init__(self):
        # k = sqrt(n) per Assembly Calculus theory
        if self.k is None:
            import numpy as np
            self.k = int(np.sqrt(self.n))


class HierarchicalNemoBrain:
    """
    Hierarchical NEMO brain with syntactic structure.
    
    26 brain areas for full language processing:
    - Lexical: Lex1, Lex2
    - Phrase: NP, VP, Sent
    - Role: SUBJ, OBJ, VERB
    - Sequence: SEQ
    - Cross-area projections
    """
    
    class Area(Enum):
        # Comprehension pathways
        PHON_TO_LEX1 = 0
        PHON_TO_LEX2 = 1
        VISUAL_TO_LEX1 = 2
        MOTOR_TO_LEX2 = 3
        LEX1 = 4
        LEX2 = 5
        NP = 6
        VP = 7
        SENT = 8
        SUBJ = 9
        OBJ = 10
        VERB_ROLE = 11
        SEQ = 12
        LEX1_TO_NP = 13
        LEX2_TO_VP = 14
        NP_TO_SENT = 15
        VP_TO_SENT = 16
        NP_TO_SUBJ = 17
        NP_TO_OBJ = 18
        LEX2_TO_VERB = 19
        SUBJ_TO_SEQ = 20
        OBJ_TO_SEQ = 21
        VERB_TO_SEQ = 22
        SEQ_TO_SUBJ = 23
        SEQ_TO_OBJ = 24
        SEQ_TO_VERB = 25
        # Generation pathways (reverse of comprehension)
        LEX1_TO_PHON = 26  # For generating nouns
        LEX2_TO_PHON = 27  # For generating verbs
        SUBJ_TO_LEX1 = 28  # Role -> Lex for generation
        OBJ_TO_LEX1 = 29
        VERB_TO_LEX2 = 30
    
    NUM_AREAS = 31
    
    def __init__(self, params: HierarchicalParams = None, verbose: bool = True):
        self.p = params or HierarchicalParams()
        self.verbose = verbose
        n, k = self.p.n, self.p.k
        
        # Assemblies
        self.phon: Dict[str, cp.ndarray] = {}
        self.visual: Dict[str, cp.ndarray] = {}
        self.motor: Dict[str, cp.ndarray] = {}
        
        # Area seeds
        self.seeds = cp.arange(self.NUM_AREAS, dtype=cp.uint32) * 1000
        
        # Batched buffers
        self.active = cp.zeros((self.NUM_AREAS, k), dtype=cp.uint32)
        self.result = cp.zeros((self.NUM_AREAS, n), dtype=cp.float16)
        
        # Learned weights per area
        self.max_learned = k * k * 500
        self.l_src = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        self.l_dst = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        self.l_delta = [cp.zeros(self.max_learned, dtype=cp.float32) for _ in range(self.NUM_AREAS)]
        self.l_num = [cp.zeros(1, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        
        # Previous and current activations
        self.prev = [None for _ in range(self.NUM_AREAS)]
        self.current = [None for _ in range(self.NUM_AREAS)]
        
        # Word order transitions
        self.transitions: Dict[Tuple[str, str], int] = {}
        
        # Kernel config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        self.sentences_seen = 0
        
        if verbose:
            print(f"HierarchicalNemoBrain: n={n:,}, {self.NUM_AREAS} areas")
            print(f"  Word order: {self.p.word_order.value}")
    
    def _project_batch(self, areas: List[int], inputs: List[cp.ndarray],
                       learn: bool, p_mult: float = 2.0) -> List[cp.ndarray]:
        """Project multiple areas in parallel."""
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
             packed_seeds, cp.float32(self.p.p * p_mult)),  # Must be float32!
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
                     cp.uint32(self.max_learned), self.seeds[a], cp.float32(self.p.p * p_mult))
                )
            
            self.prev[a] = winners
            self.current[a] = winners
            results.append(winners)
        
        return results
    
    def _project_one(self, area: int, inp: cp.ndarray, learn: bool, p_mult: float = 2.0) -> cp.ndarray:
        """Project single area - optimized to avoid batch overhead."""
        # Direct kernel call without batch wrapping
        self.active[area, :len(inp)] = inp[:min(len(inp), self.p.k)]
        self.result[area].fill(0)
        
        projection_fp16_kernel(
            (self.gx, 1), (self.bs,),
            (self.active[area:area+1], self.result[area:area+1],
             cp.uint32(self.p.k), cp.uint32(self.p.n), cp.uint32(1),
             self.seeds[area:area+1], cp.float32(self.p.p * p_mult)),
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
                 cp.uint32(self.max_learned), self.seeds[area], cp.float32(self.p.p * p_mult))
            )
        
        self.prev[area] = winners
        self.current[area] = winners
        return winners
    
    def _get_or_create(self, store: Dict, name: str) -> cp.ndarray:
        if name not in store:
            store[name] = cp.random.randint(0, self.p.n, self.p.k, dtype=cp.uint32)
        return store[name]
    
    def _reset_phrase_areas(self):
        """Reset phrase and sentence areas."""
        for a in [self.Area.NP.value, self.Area.VP.value, self.Area.SENT.value,
                  self.Area.SUBJ.value, self.Area.OBJ.value, self.Area.VERB_ROLE.value]:
            self.prev[a] = None
            self.current[a] = None
    
    def present_noun(self, word: str, role: str = None, learn: bool = True) -> cp.ndarray:
        """Present a noun with optional role binding."""
        phon = self._get_or_create(self.phon, word)
        vis = self._get_or_create(self.visual, word)
        
        # Comprehension: Phon + Visual -> Lex1
        r = self._project_batch(
            [self.Area.PHON_TO_LEX1.value, self.Area.VISUAL_TO_LEX1.value],
            [phon, vis], learn
        )
        combined = cp.unique(cp.concatenate(r))[:self.p.k]
        lex1_assembly = self._project_one(self.Area.LEX1.value, combined, learn)
        
        np_assembly = self._project_one(self.Area.LEX1_TO_NP.value, lex1_assembly, learn)
        self._project_one(self.Area.NP.value, np_assembly, learn)
        
        if role == 'SUBJ':
            self._project_one(self.Area.NP_TO_SUBJ.value, np_assembly, learn)
            subj = self._project_one(self.Area.SUBJ.value, np_assembly, learn)
            self._project_one(self.Area.SUBJ_TO_SEQ.value, subj, learn)
            self._project_one(self.Area.SEQ.value, subj, learn)
        elif role == 'OBJ':
            self._project_one(self.Area.NP_TO_OBJ.value, np_assembly, learn)
            obj = self._project_one(self.Area.OBJ.value, np_assembly, learn)
            self._project_one(self.Area.OBJ_TO_SEQ.value, obj, learn)
            self._project_one(self.Area.SEQ.value, obj, learn)
        
        return lex1_assembly
    
    def present_verb(self, word: str, learn: bool = True) -> cp.ndarray:
        """Present a verb."""
        phon = self._get_or_create(self.phon, word)
        mot = self._get_or_create(self.motor, word)
        
        # Comprehension: Phon + Motor -> Lex2
        r = self._project_batch(
            [self.Area.PHON_TO_LEX2.value, self.Area.MOTOR_TO_LEX2.value],
            [phon, mot], learn
        )
        combined = cp.unique(cp.concatenate(r))[:self.p.k]
        lex2_assembly = self._project_one(self.Area.LEX2.value, combined, learn)
        
        vp_assembly = self._project_one(self.Area.LEX2_TO_VP.value, lex2_assembly, learn)
        self._project_one(self.Area.VP.value, vp_assembly, learn)
        
        self._project_one(self.Area.LEX2_TO_VERB.value, lex2_assembly, learn)
        verb_role = self._project_one(self.Area.VERB_ROLE.value, lex2_assembly, learn)
        
        self._project_one(self.Area.VERB_TO_SEQ.value, verb_role, learn)
        self._project_one(self.Area.SEQ.value, verb_role, learn)
        
        return lex2_assembly
    
    def train_sentence(self, subject: str, verb: str, obj: str = None):
        """Train on a sentence with word order learning."""
        self._reset_phrase_areas()
        
        # Determine word order
        if self.p.word_order == WordOrder.SVO:
            order = [('SUBJ', subject), ('VERB', verb)]
            if obj:
                order.append(('OBJ', obj))
        elif self.p.word_order == WordOrder.SOV:
            order = [('SUBJ', subject)]
            if obj:
                order.append(('OBJ', obj))
            order.append(('VERB', verb))
        elif self.p.word_order == WordOrder.VSO:
            order = [('VERB', verb), ('SUBJ', subject)]
            if obj:
                order.append(('OBJ', obj))
        
        # Learn transitions
        for i in range(len(order) - 1):
            trans = (order[i][0], order[i+1][0])
            self.transitions[trans] = self.transitions.get(trans, 0) + 1
        
        # Present words
        for _ in range(self.p.tau):
            for role, word in order:
                if role == 'SUBJ':
                    self.present_noun(word, role='SUBJ', learn=True)
                elif role == 'OBJ':
                    self.present_noun(word, role='OBJ', learn=True)
                elif role == 'VERB':
                    self.present_verb(word, learn=True)
        
        # Build sentence representation
        if self.current[self.Area.NP.value] is not None and self.current[self.Area.VP.value] is not None:
            np_contrib = self._project_one(self.Area.NP_TO_SENT.value,
                                           self.current[self.Area.NP.value], True)
            vp_contrib = self._project_one(self.Area.VP_TO_SENT.value,
                                           self.current[self.Area.VP.value], True)
            combined = cp.unique(cp.concatenate([np_contrib, vp_contrib]))[:self.p.k]
            self._project_one(self.Area.SENT.value, combined, True)
        
        self.sentences_seen += 1
    
    def predict_next_role(self, current_role: str) -> str:
        """Predict next role based on learned transitions."""
        candidates = {}
        for (from_role, to_role), count in self.transitions.items():
            if from_role == current_role:
                candidates[to_role] = count
        
        if not candidates:
            return None
        return max(candidates, key=candidates.get)
    
    def generate_sentence_order(self) -> List[str]:
        """Generate word order from learned transitions."""
        all_targets = set(to_role for (_, to_role) in self.transitions.keys())
        all_sources = set(from_role for (from_role, _) in self.transitions.keys())
        
        starts = all_sources - all_targets
        current = list(starts)[0] if starts else 'SUBJ'
        
        order = [current]
        for _ in range(3):
            next_role = self.predict_next_role(current)
            if next_role and next_role not in order:
                order.append(next_role)
                current = next_role
            else:
                break
        
        return order
    
    def classify_word(self, word: str) -> str:
        """Classify word as NOUN or VERB."""
        has_vis = word in self.visual
        has_mot = word in self.motor
        if has_vis and not has_mot:
            return 'NOUN'
        if has_mot and not has_vis:
            return 'VERB'
        return 'UNKNOWN'
    
    # =========================================================================
    # GENERATION METHODS
    # =========================================================================
    
    def _compute_overlap(self, assembly1: cp.ndarray, assembly2: cp.ndarray) -> float:
        """Compute overlap between two assemblies (Jaccard-like)."""
        set1 = set(assembly1.get())
        set2 = set(assembly2.get())
        intersection = len(set1 & set2)
        return intersection / self.p.k
    
    def _get_word_lex_assembly(self, word: str, word_type: str) -> cp.ndarray:
        """
        Get the Lex assembly for a word by running it through comprehension.
        
        This is cached after first computation.
        """
        cache_key = f"{word}_{word_type}"
        if not hasattr(self, '_lex_cache'):
            self._lex_cache = {}
        
        if cache_key not in self._lex_cache:
            phon = self.phon.get(word)
            if phon is None:
                return None
            
            if word_type == 'noun':
                vis = self.visual.get(word)
                if vis is None:
                    return None
                r = self._project_batch(
                    [self.Area.PHON_TO_LEX1.value, self.Area.VISUAL_TO_LEX1.value],
                    [phon, vis], learn=False
                )
                combined = cp.unique(cp.concatenate(r))[:self.p.k]
                lex_assembly = self._project_one(self.Area.LEX1.value, combined, learn=False)
            else:
                mot = self.motor.get(word)
                if mot is None:
                    return None
                r = self._project_batch(
                    [self.Area.PHON_TO_LEX2.value, self.Area.MOTOR_TO_LEX2.value],
                    [phon, mot], learn=False
                )
                combined = cp.unique(cp.concatenate(r))[:self.p.k]
                lex_assembly = self._project_one(self.Area.LEX2.value, combined, learn=False)
            
            self._lex_cache[cache_key] = lex_assembly
        
        return self._lex_cache[cache_key]
    
    def _match_lex_to_word(self, lex_pattern: cp.ndarray, word_type: str) -> Tuple[str, float]:
        """
        Match a Lex pattern to the closest known word using Lex overlap.
        
        This is the key insight: instead of trying to reverse-project to Phon,
        we compare the generated Lex pattern to the Lex patterns of known words.
        
        Args:
            lex_pattern: The generated Lex assembly
            word_type: 'noun' or 'verb' to filter candidates
            
        Returns:
            (word, overlap_score) - best matching word and its score
        """
        best_word = None
        best_overlap = 0.0
        
        # Get candidate words based on type
        if word_type == 'noun':
            candidates = list(self.visual.keys())
        else:
            candidates = list(self.motor.keys())
        
        for word in candidates:
            word_lex = self._get_word_lex_assembly(word, word_type)
            if word_lex is not None:
                overlap = self._compute_overlap(lex_pattern, word_lex)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_word = word
        
        return best_word, best_overlap
    
    def generate_word(self, role: str, context: Dict[str, str] = None) -> Tuple[str, float]:
        """
        Generate a word for a given syntactic role.
        
        Strategy: Sample from learned vocabulary, weighted by how well each word
        fits the current context. Without context, sample uniformly.
        
        Args:
            role: 'SUBJ', 'OBJ', or 'VERB'
            context: Optional dict of already-generated words (e.g., {'SUBJ': 'dog'})
            
        Returns:
            (word, confidence) - generated word and match confidence
        """
        context = context or {}
        
        if role in ['SUBJ', 'OBJ']:
            # Get all learned nouns
            candidates = list(self.visual.keys())
            if not candidates:
                return None, 0.0
            
            # Avoid repeating the same noun as subject and object
            if role == 'OBJ' and 'SUBJ' in context:
                candidates = [w for w in candidates if w != context['SUBJ']]
            
            if not candidates:
                return None, 0.0
            
            # For now, sample uniformly from candidates
            # Future: weight by context compatibility
            word = np.random.choice(candidates)
            return word, 1.0
        
        elif role == 'VERB':
            # Get all learned verbs
            candidates = list(self.motor.keys())
            if not candidates:
                return None, 0.0
            
            # Sample uniformly
            word = np.random.choice(candidates)
            return word, 1.0
        
        return None, 0.0
    
    def generate_sentence(self) -> List[Tuple[str, float]]:
        """
        Generate a complete sentence using learned word order.
        
        Returns:
            List of (word, confidence) tuples
        """
        sentence = []
        order = self.generate_sentence_order()
        context = {}
        
        for role in order:
            word, confidence = self.generate_word(role, context)
            if word:
                sentence.append((word, confidence))
                context[role] = word
        
        return sentence
    
    def generate_sentence_str(self) -> str:
        """Generate a sentence as a string."""
        sentence = self.generate_sentence()
        return ' '.join(word for word, _ in sentence if word)

