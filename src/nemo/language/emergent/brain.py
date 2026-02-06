"""
Emergent NEMO Brain
===================

Version: 2.2.0
Date: 2025-11-30

The core brain class where word categories EMERGE from grounding patterns.
No pre-labeled categories - everything learned from experience.

Key insight: We store LEARNED ASSEMBLIES for each word in each area.
During generation, we use these stored assemblies and learned weights
to find compatible combinations - not fresh random projections.

NEW in 2.2.0: CUDA backend support for ~8x faster training!
Uses hash-based implicit connectivity with no weight matrix storage.
"""

import cupy as cp
import torch
import numpy as np
from typing import Dict, Optional, Set, Tuple, List
from collections import defaultdict

from src.nemo.core.kernel import projection_fp16_kernel, hebbian_kernel
from .areas import Area, NUM_AREAS, MUTUAL_INHIBITION_GROUPS
from .params import EmergentParams, GroundingModality

# Try to import CUDA backend
try:
    from .cuda_backend import CUDAProjector, check_cuda_available
    CUDA_BACKEND_AVAILABLE = check_cuda_available()
except ImportError:
    CUDA_BACKEND_AVAILABLE = False
    CUDAProjector = None

__all__ = ['EmergentNemoBrain']


class EmergentNemoBrain:
    """
    Brain where word categories EMERGE from grounding patterns.
    
    No pre-labeled categories - everything learned from:
    1. Which modalities are active when word is heard
    2. How consistently a word co-occurs with each modality
    3. Assembly stability in different areas
    
    Features:
    - 37 neurobiologically plausible areas
    - Mutual inhibition for competing roles
    - Core areas that emerge from grounding
    - Phrase structure areas for composition
    - Error detection via assembly stability
    """
    
    def __init__(self, params: EmergentParams = None, verbose: bool = True, 
                 use_cuda_backend: bool = True):
        self.p = params or EmergentParams()
        self.verbose = verbose
        n, k = self.p.n, self.p.k
        
        # =====================================================================
        # CUDA BACKEND - ~8x faster training with hash-based connectivity
        # =====================================================================
        self.use_cuda_backend = use_cuda_backend and CUDA_BACKEND_AVAILABLE
        self.cuda_projectors: Dict[Area, 'CUDAProjector'] = {}
        
        if self.use_cuda_backend:
            # Create a CUDA projector for each area
            for area in Area:
                area_seed = area.value * 1000
                self.cuda_projectors[area] = CUDAProjector(
                    n=n, k=k, p=self.p.p, beta=self.p.beta, 
                    w_max=self.p.w_max, seed=area_seed
                )
        
        # Input assemblies for each modality (created on demand)
        self.assemblies: Dict[Area, Dict[str, cp.ndarray]] = {
            Area.PHON: {},
            Area.VISUAL: {},
            Area.MOTOR: {},
            Area.PROPERTY: {},
            Area.SPATIAL: {},
            Area.TEMPORAL: {},
            Area.SOCIAL: {},
            Area.EMOTION: {},
            Area.MOOD: {},
            Area.TENSE: {},
            Area.POLARITY: {},
        }
        
        # Area seeds for implicit connectivity (used by CuPy backend)
        self.seeds = cp.arange(NUM_AREAS, dtype=cp.uint32) * 1000
        
        # Learned weights per area (used by CuPy backend)
        self.max_learned = self.p.max_learned
        self.l_src = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(NUM_AREAS)]
        self.l_dst = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(NUM_AREAS)]
        self.l_delta = [cp.zeros(self.max_learned, dtype=cp.float32) for _ in range(NUM_AREAS)]
        self.l_num = [cp.zeros(1, dtype=cp.uint32) for _ in range(NUM_AREAS)]
        
        # Current and previous activations
        self.current: Dict[Area, Optional[cp.ndarray]] = {a: None for a in Area}
        self.prev: Dict[Area, Optional[cp.ndarray]] = {a: None for a in Area}
        
        # Inhibition state (for mutual inhibition)
        self.inhibited: Set[Area] = set()
        
        # Word statistics (for emergent categorization)
        self.word_grounding_counts: Dict[str, Dict[GroundingModality, int]] = defaultdict(
            lambda: defaultdict(int))
        self.word_exposure_count: Dict[str, int] = defaultdict(int)
        
        # =====================================================================
        # LEARNED ASSEMBLIES - The key to NEMO-correct generation
        # =====================================================================
        # Store the LEARNED assembly for each word in each area.
        # This is what the brain "knows" about each word.
        # Format: learned_assemblies[area][word] = assembly (cp.ndarray)
        self.learned_assemblies: Dict[Area, Dict[str, cp.ndarray]] = defaultdict(dict)
        
        # Track how many times each word has been presented to each area
        # (for averaging/stabilizing assemblies)
        self.assembly_exposure_count: Dict[Area, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        
        # Kernel config (for CuPy backend)
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        if verbose:
            backend_str = "CUDA (8x faster)" if self.use_cuda_backend else "CuPy"
            print(f"EmergentNemoBrain initialized:")
            print(f"  n={n:,}, k={k}")
            print(f"  Backend: {backend_str}")
            print(f"  Areas: {NUM_AREAS}")
            print(f"    Input: 8, Lexical: 2, Core: 8")
            print(f"    Thematic: 6, Phrase: 5, Syntactic: 3")
            print(f"    Control: 4, Error: 1")
            print(f"  Mutual inhibition groups: {len(MUTUAL_INHIBITION_GROUPS)}")
            print(f"  NO pre-labeled categories - all emergent!")
    
    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================
    
    def _get_or_create(self, area: Area, name: str) -> cp.ndarray:
        """Get or create assembly for a concept in an input area"""
        if area not in self.assemblies:
            self.assemblies[area] = {}
        store = self.assemblies[area]
        if name not in store:
            store[name] = cp.random.randint(0, self.p.n, self.p.k, dtype=cp.uint32)
        return store[name]
    
    def _apply_mutual_inhibition(self, active_area: Area):
        """Apply mutual inhibition - if one area in a group is active, inhibit others"""
        for group in MUTUAL_INHIBITION_GROUPS:
            if active_area in group:
                for other in group:
                    if other != active_area:
                        self.inhibited.add(other)
    
    def _is_inhibited(self, area: Area) -> bool:
        """Check if area is currently inhibited"""
        return area in self.inhibited
    
    def _project(self, area: Area, inp: cp.ndarray, learn: bool = True) -> Optional[cp.ndarray]:
        """Project input to area, respecting inhibition"""
        if self._is_inhibited(area):
            return None
        
        n, k = self.p.n, self.p.k
        area_idx = area.value
        
        # Use CUDA backend if available (8x faster)
        if self.use_cuda_backend and area in self.cuda_projectors:
            return self._project_cuda(area, inp, learn)
        
        # Fall back to CuPy implementation
        return self._project_cupy(area, inp, learn)
    
    def _project_cuda(self, area: Area, inp: cp.ndarray, learn: bool = True) -> Optional[cp.ndarray]:
        """Project using CUDA backend (fast path)"""
        k = self.p.k
        area_seed = area.value * 1000
        
        # Convert input to numpy for CUDA backend
        active = np.zeros(k, dtype=np.uint32)
        inp_np = inp.get() if hasattr(inp, 'get') else np.asarray(inp)
        active[:min(len(inp_np), k)] = inp_np[:k]
        
        # Project using CUDA kernel
        projector = self.cuda_projectors[area]
        winners_np = projector.project(active, learn=learn, area_seed=area_seed)
        
        # Convert back to CuPy
        winners = cp.asarray(winners_np, dtype=cp.uint32)
        
        self.prev[area] = winners
        self.current[area] = winners
        
        # Apply mutual inhibition
        self._apply_mutual_inhibition(area)
        
        return winners
    
    def _project_cupy(self, area: Area, inp: cp.ndarray, learn: bool = True) -> Optional[cp.ndarray]:
        """Project using CuPy implementation (fallback)"""
        n, k = self.p.n, self.p.k
        area_idx = area.value
        
        active = cp.zeros(k, dtype=cp.uint32)
        active[:len(inp)] = inp[:k]
        
        result = cp.zeros(n, dtype=cp.float16)
        
        projection_fp16_kernel(
            (self.gx, 1), (self.bs,),
            (active.reshape(1, -1), result.reshape(1, -1),
             cp.uint32(k), cp.uint32(n), cp.uint32(1),
             self.seeds[area_idx:area_idx+1], cp.float32(self.p.p)),
            shared_mem=k * 4
        )
        
        result_torch = torch.as_tensor(result, device='cuda')
        _, winners_idx = torch.topk(result_torch, k, sorted=False)
        winners = cp.asarray(winners_idx).astype(cp.uint32)
        
        if learn and self.prev[area] is not None:
            grid = (k * k + self.bs - 1) // self.bs
            hebbian_kernel(
                (grid,), (self.bs,),
                (self.l_src[area_idx], self.l_dst[area_idx],
                 self.l_delta[area_idx], self.l_num[area_idx],
                 self.prev[area], winners,
                 cp.uint32(k), cp.float32(self.p.beta), cp.float32(self.p.w_max),
                 cp.uint32(self.max_learned), self.seeds[area_idx], cp.float32(self.p.p))
            )
        
        self.prev[area] = winners
        self.current[area] = winners
        
        # Apply mutual inhibition
        self._apply_mutual_inhibition(area)
        
        return winners
    
    def _clear_area(self, area: Area):
        """Clear area state"""
        self.current[area] = None
        self.prev[area] = None
    
    def clear_all(self):
        """Clear all areas and inhibition state"""
        for area in Area:
            self._clear_area(area)
        self.inhibited.clear()
    
    # =========================================================================
    # LEARNED ASSEMBLY STORAGE AND RETRIEVAL
    # =========================================================================
    
    def store_learned_assembly(self, area: Area, word: str, assembly: cp.ndarray):
        """
        Store the learned assembly for a word in an area.
        
        This is called after training to remember what assembly
        a word activates in each area.
        
        If the word already has an assembly, we update it using
        exponential moving average to stabilize over time.
        """
        if word in self.learned_assemblies[area]:
            # Update existing assembly with moving average
            old_assembly = self.learned_assemblies[area][word]
            count = self.assembly_exposure_count[area][word]
            
            # Combine old and new (favor neurons that appear in both)
            old_set = set(old_assembly.get().tolist())
            new_set = set(assembly.get().tolist())
            
            # Keep neurons that appear in both, plus some from each
            intersection = old_set & new_set
            old_only = old_set - new_set
            new_only = new_set - old_set
            
            # Weighted combination: favor intersection, then old, then new
            combined = list(intersection)
            
            # Add from old (proportional to exposure count)
            old_weight = min(count / (count + 1), 0.7)
            n_from_old = int(len(old_only) * old_weight)
            combined.extend(list(old_only)[:n_from_old])
            
            # Fill rest from new
            remaining = self.p.k - len(combined)
            combined.extend(list(new_only)[:remaining])
            
            # If still not enough, add from old
            if len(combined) < self.p.k:
                remaining = self.p.k - len(combined)
                combined.extend(list(old_only)[n_from_old:n_from_old + remaining])
            
            # Truncate to k
            combined = combined[:self.p.k]
            
            self.learned_assemblies[area][word] = cp.array(combined, dtype=cp.uint32)
        else:
            # First time - just store
            self.learned_assemblies[area][word] = assembly.copy()
        
        self.assembly_exposure_count[area][word] += 1
    
    def get_learned_assembly(self, area: Area, word: str) -> Optional[cp.ndarray]:
        """
        Get the learned assembly for a word in an area.
        
        Returns None if the word hasn't been learned in this area.
        """
        return self.learned_assemblies[area].get(word)
    
    def has_learned_assembly(self, area: Area, word: str) -> bool:
        """Check if a word has a learned assembly in an area."""
        return word in self.learned_assemblies[area]
    
    def get_assembly_overlap(self, assembly1: cp.ndarray, assembly2: cp.ndarray) -> float:
        """
        Compute overlap between two assemblies.
        
        This is the NEMO way to measure similarity/compatibility.
        High overlap = similar/compatible
        Low overlap = different/incompatible
        """
        if assembly1 is None or assembly2 is None:
            return 0.0
        
        set1 = set(assembly1.get().tolist())
        set2 = set(assembly2.get().tolist())
        
        intersection = len(set1 & set2)
        return intersection / self.p.k
    
    def find_best_matching_word(self, area: Area, target_assembly: cp.ndarray,
                                 word_list: List[str] = None) -> Tuple[Optional[str], float]:
        """
        Find the word whose learned assembly best matches a target assembly.
        
        This is NEMO-style retrieval: project to an area, then find which
        learned word assembly has highest overlap.
        
        Args:
            area: The area to search in
            target_assembly: The assembly to match against
            word_list: Optional list of words to consider (if None, check all)
        
        Returns: (best_word, overlap_score)
        """
        if target_assembly is None:
            return None, 0.0
        
        best_word = None
        best_overlap = 0.0
        
        words_to_check = word_list if word_list else list(self.learned_assemblies[area].keys())
        
        for word in words_to_check:
            word_assembly = self.get_learned_assembly(area, word)
            if word_assembly is not None:
                overlap = self.get_assembly_overlap(target_assembly, word_assembly)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_word = word
        
        return best_word, best_overlap
    
    def get_compatible_words(self, area: Area, target_assembly: cp.ndarray,
                              word_list: List[str] = None, 
                              min_overlap: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find all words whose learned assemblies have sufficient overlap with target.
        
        Returns list of (word, overlap) tuples, sorted by overlap descending.
        """
        if target_assembly is None:
            return []
        
        compatible = []
        words_to_check = word_list if word_list else list(self.learned_assemblies[area].keys())
        
        for word in words_to_check:
            word_assembly = self.get_learned_assembly(area, word)
            if word_assembly is not None:
                overlap = self.get_assembly_overlap(target_assembly, word_assembly)
                if overlap >= min_overlap:
                    compatible.append((word, overlap))
        
        # Sort by overlap descending
        compatible.sort(key=lambda x: x[1], reverse=True)
        return compatible
    
    # =========================================================================
    # STABILITY MEASUREMENT
    # =========================================================================
    
    def measure_stability(self, area: Area, rounds: int = 3) -> float:
        """
        Measure assembly stability in an area.
        
        Stable assembly = valid parse/category
        Wobbly assembly = error/wrong category
        """
        if self.current[area] is None:
            return 0.0
        
        initial = set(self.current[area].get().tolist())
        
        # Recurrent projection
        for _ in range(rounds):
            self._project(area, self.current[area], learn=False)
        
        if self.current[area] is None:
            return 0.0
        
        final = set(self.current[area].get().tolist())
        
        # Calculate overlap
        intersection = len(initial & final)
        return intersection / self.p.k
    
    def get_learned_strength(self, area: Area, inp: cp.ndarray) -> float:
        """Get strength of learned connections for input in an area"""
        area_idx = area.value
        num_learned = int(self.l_num[area_idx].get()[0])
        if num_learned == 0:
            return 0.0
        
        # Project and check activation
        self._clear_area(area)
        self._project(area, inp, learn=False)
        
        winners = self.current[area]
        if winners is None:
            return 0.0
        
        dst_arr = self.l_dst[area_idx][:num_learned].get()
        delta_arr = self.l_delta[area_idx][:num_learned].get()
        
        winners_set = set(winners.get().tolist())
        strength = sum(1.0 + delta for dst, delta in zip(dst_arr, delta_arr) if dst in winners_set)
        
        return strength
    
    # =========================================================================
    # PHRASE COMPOSITION (NEMO merge operations)
    # =========================================================================
    
    def merge_to_area(self, target_area: Area, source_assembly: cp.ndarray, 
                      learn: bool = True) -> Optional[cp.ndarray]:
        """
        Merge a source assembly into a target phrase area.
        
        This is the KEY NEMO operation for phrase building:
        - If target_area already has an assembly, the new input MERGES with it
        - The resulting assembly represents the combined phrase
        
        Example:
            merge_to_area(NP, "the")  → NP has assembly for "the"
            merge_to_area(NP, "big")  → NP has merged assembly for "the big"
            merge_to_area(NP, "dog")  → NP has merged assembly for "the big dog"
        """
        if self._is_inhibited(target_area):
            return None
        
        result = self._project(target_area, source_assembly, learn=learn)
        return result
    
    def bind_phrase_to_role(self, phrase_assembly: cp.ndarray, role: Area,
                            learn: bool = True) -> Optional[cp.ndarray]:
        """
        Bind a phrase to a syntactic role (SUBJ, OBJ, IOBJ).
        
        Mutual inhibition ensures only one role can be active.
        """
        if role not in [Area.SUBJ, Area.OBJ, Area.IOBJ]:
            raise ValueError(f"Role must be SUBJ, OBJ, or IOBJ, got {role}")
        
        if self._is_inhibited(role):
            return None
        
        result = self._project(role, phrase_assembly, learn=learn)
        return result
    
    def link_to_predicate(self, role_assembly: cp.ndarray, predicate_area: Area = Area.VP,
                          learn: bool = True) -> Optional[cp.ndarray]:
        """Link a role (SUBJ/OBJ) to the predicate (VP)."""
        if self._is_inhibited(predicate_area):
            return None
        
        result = self._project(predicate_area, role_assembly, learn=learn)
        return result
    
    def disinhibit_role(self, role: Area):
        """Explicitly disinhibit a role area (used after verb for object)."""
        if role in self.inhibited:
            self.inhibited.remove(role)
    
    def get_phrase_stability(self, phrase_area: Area, rounds: int = 3) -> float:
        """Measure phrase stability (high = well-formed, low = malformed)."""
        return self.measure_stability(phrase_area, rounds)
    
    def project_backwards(self, from_area: Area, to_area: Area) -> Optional[cp.ndarray]:
        """Project backwards for generation (SENT → VP → NP → LEX)."""
        if self.current[from_area] is None:
            return None
        
        result = self._project(to_area, self.current[from_area], learn=False)
        return result
    
    # =========================================================================
    # NEMO-STYLE COMPATIBILITY TESTING
    # =========================================================================
    # In NEMO, compatibility is tested by DOING the operation and checking
    # if the result is STABLE. We don't query - we test.
    
    def test_merge_stability(self, target_area: Area, candidate: cp.ndarray,
                             stability_rounds: int = 3) -> float:
        """
        Test if merging a candidate into an area produces a stable result.
        
        NEMO way: Don't ask "is this compatible?" - instead:
        1. Save current state
        2. Try the merge (without learning)
        3. Measure stability
        4. Restore state
        
        High stability (>0.5) = compatible
        Low stability (<0.3) = incompatible (wobbly)
        
        Returns: stability score (0.0 to 1.0)
        """
        # Save current state
        saved_current = self.current[target_area]
        saved_prev = self.prev[target_area]
        
        # Try the merge without learning
        self._project(target_area, candidate, learn=False)
        
        # Measure stability through recurrence
        stability = self.measure_stability(target_area, rounds=stability_rounds)
        
        # Restore state
        self.current[target_area] = saved_current
        self.prev[target_area] = saved_prev
        
        return stability
    
    def settle_to_pattern(self, area: Area, input_assembly: cp.ndarray,
                          max_rounds: int = 5) -> cp.ndarray:
        """
        Let an area settle to a stable pattern given input.
        
        NEMO way: Don't enumerate options - project and let the brain settle.
        The area will naturally activate the most compatible assembly.
        
        Returns: The settled assembly
        """
        # Project input to area
        self._project(area, input_assembly, learn=False)
        
        # Let it settle through recurrence
        for _ in range(max_rounds):
            if self.current[area] is not None:
                prev_assembly = self.current[area].copy()
                self._project(area, self.current[area], learn=False)
                
                # Check if settled (no change)
                if self.current[area] is not None:
                    overlap = len(set(prev_assembly.get().tolist()) & 
                                 set(self.current[area].get().tolist())) / self.p.k
                    if overlap > 0.9:
                        break
        
        return self.current[area]

