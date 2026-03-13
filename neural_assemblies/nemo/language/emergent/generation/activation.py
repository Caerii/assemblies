"""
Activation Spreading
====================

Spreads activation through learned connections until convergence.

Key Principles:
1. Activation spreads through LEARNED weights (Hebbian connections)
2. k-winners provides competition (only top-k neurons fire)
3. Pattern settles when it stops changing (attractor dynamics)
4. No explicit pathways - connectivity is implicit in learned weights

This is the core of emergent generation:
- Seed with input activation
- Let it spread through what was learned
- The settled pattern IS the response
"""

from typing import Dict, List, Set, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..brain import EmergentNemoBrain

from ..areas import Area


class ActivationSpreader:
    """
    Spreads activation through learned connections.
    
    Uses the brain's existing projection mechanism, which already
    incorporates learned weights (Hebbian updates).
    
    The key insight: we don't need to define pathways explicitly.
    The learned weights determine where activation flows.
    """
    
    def __init__(self, brain: 'EmergentNemoBrain'):
        self.brain = brain
        
        # Areas that participate in spreading
        # (not all areas - some are input-only)
        self.spreading_areas = [
            Area.NOUN_CORE,
            Area.VERB_CORE,
            Area.ADJ_CORE,
            Area.ADV_CORE,
            Area.PRON_CORE,
            Area.PREP_CORE,
            Area.VP,
            Area.NP,
            Area.LEX_CONTENT,
            Area.LEX_FUNCTION,
        ]
        
        # Track activation history for convergence detection
        self.history: Dict[Area, List[Set[int]]] = {}
    
    def spread(self, seeds: Dict[Area, cp.ndarray],
               max_rounds: int = 10) -> Dict[Area, cp.ndarray]:
        """
        Spread activation from seeds until convergence.
        
        Args:
            seeds: Initial activations {area: assembly}
            max_rounds: Maximum iterations before stopping
        
        Returns:
            Final activation pattern {area: assembly}
        """
        # Clear history
        self.history = {area: [] for area in self.spreading_areas}
        
        # Initialize with seeds
        self.brain.clear_all()
        for area, assembly in seeds.items():
            self.brain.current[area] = assembly.copy()
            self._record_state(area)
        
        # Spread until convergence or max rounds
        for round_num in range(max_rounds):
            changed = self._spread_one_round()
            
            if not changed:
                # Converged - pattern stopped changing
                break
        
        # Return final state
        return self._get_current_state()
    
    def _spread_one_round(self) -> bool:
        """
        Perform one round of activation spreading.
        
        Returns True if any area changed, False if converged.
        """
        changed = False
        
        # Snapshot current state
        prev_state = self._get_current_state()
        
        # For each active area, project to related areas
        for source_area in self.spreading_areas:
            if self.brain.current[source_area] is None:
                continue
            
            source_assembly = self.brain.current[source_area]
            
            # Project to areas that this source connects to
            target_areas = self._get_target_areas(source_area)
            
            for target_area in target_areas:
                if target_area in self.brain.inhibited:
                    continue
                
                # Project without learning
                self.brain._project(target_area, source_assembly, learn=False)
        
        # Check if anything changed
        curr_state = self._get_current_state()
        
        for area in self.spreading_areas:
            prev = prev_state.get(area)
            curr = curr_state.get(area)
            
            if prev is None and curr is None:
                continue
            if prev is None or curr is None:
                changed = True
                break
            
            # Check overlap
            prev_set = set(prev.get().tolist())
            curr_set = set(curr.get().tolist())
            
            overlap = len(prev_set & curr_set) / max(len(prev_set), 1)
            if overlap < 0.95:  # Allow tiny fluctuations
                changed = True
                break
        
        # Record new state
        for area in self.spreading_areas:
            self._record_state(area)
        
        return changed
    
    def _get_target_areas(self, source_area: Area) -> List[Area]:
        """
        Get areas that a source area projects to.
        
        This defines the connectivity structure.
        Based on NEMO architecture and linguistic hierarchy.
        """
        # Define connectivity based on area type
        connectivity = {
            # Core areas project to phrase areas
            Area.NOUN_CORE: [Area.NP, Area.VP],
            Area.VERB_CORE: [Area.VP],
            Area.ADJ_CORE: [Area.NP],
            Area.ADV_CORE: [Area.VP],
            Area.PRON_CORE: [Area.NP, Area.VP],
            Area.PREP_CORE: [Area.NP],
            
            # Phrase areas project to each other and back to core
            Area.NP: [Area.VP, Area.NOUN_CORE, Area.ADJ_CORE],
            Area.VP: [Area.NP, Area.VERB_CORE, Area.NOUN_CORE],
            
            # Lexical areas
            Area.LEX_CONTENT: [Area.NOUN_CORE, Area.VERB_CORE, Area.ADJ_CORE],
            Area.LEX_FUNCTION: [Area.NP, Area.VP],
        }
        
        return connectivity.get(source_area, [])
    
    def _get_current_state(self) -> Dict[Area, cp.ndarray]:
        """Get current activation state."""
        return {
            area: self.brain.current[area].copy() 
            for area in self.spreading_areas
            if self.brain.current[area] is not None
        }
    
    def _record_state(self, area: Area):
        """Record current state for history."""
        if self.brain.current[area] is not None:
            state = set(self.brain.current[area].get().tolist())
            self.history[area].append(state)
    
    # =========================================================================
    # TARGETED SPREADING (for specific queries)
    # =========================================================================
    
    def spread_from_verb(self, verb: str, 
                         max_rounds: int = 5) -> Dict[Area, cp.ndarray]:
        """
        Spread activation from a verb to find related subjects/objects.
        
        Used for "who verbs?" and "what does X verb?" questions.
        """
        # Get verb's assembly
        verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
        
        if verb_assembly is None:
            return {}
        
        # Seed with verb
        seeds = {Area.VERB_CORE: verb_assembly}
        
        return self.spread(seeds, max_rounds)
    
    def spread_from_noun(self, noun: str,
                         max_rounds: int = 5) -> Dict[Area, cp.ndarray]:
        """
        Spread activation from a noun to find related verbs/properties.
        
        Used for "what does X do?" questions.
        """
        # Get noun's assembly
        noun_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, noun)
        
        if noun_assembly is None:
            # Try pronoun
            noun_assembly = self.brain.get_learned_assembly(Area.PRON_CORE, noun)
        
        if noun_assembly is None:
            return {}
        
        # Seed with noun
        seeds = {Area.NOUN_CORE: noun_assembly}
        
        return self.spread(seeds, max_rounds)
    
    def spread_from_vp(self, vp_key: str,
                       max_rounds: int = 5) -> Dict[Area, cp.ndarray]:
        """
        Spread activation from a VP assembly.
        
        Used to "unpack" a proposition into its components.
        """
        vp_assembly = self.brain.get_learned_assembly(Area.VP, vp_key)
        
        if vp_assembly is None:
            return {}
        
        seeds = {Area.VP: vp_assembly}
        
        return self.spread(seeds, max_rounds)
    
    # =========================================================================
    # BIASED SPREADING (for question answering)
    # =========================================================================
    
    def spread_with_bias(self, seeds: Dict[Area, cp.ndarray],
                         bias_area: Area,
                         max_rounds: int = 5) -> Dict[Area, cp.ndarray]:
        """
        Spread activation with a bias toward a specific area.
        
        Used for question answering:
        - "who" questions bias toward NOUN_CORE/PRON_CORE
        - "what" questions bias toward NOUN_CORE
        - "where" questions bias toward PREP_CORE
        
        The bias is implemented by giving the target area
        extra activation rounds.
        """
        # First, do normal spreading
        result = self.spread(seeds, max_rounds)
        
        # Then, do extra rounds focusing on bias area
        if bias_area in result:
            for _ in range(2):
                self.brain._project(bias_area, result[bias_area], learn=False)
        
        return self._get_current_state()


__all__ = ['ActivationSpreader']


