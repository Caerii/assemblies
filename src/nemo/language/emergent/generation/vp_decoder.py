"""
VP Assembly Decoder
===================

Decodes VP assemblies into word sequences.

Key Insight: VP assemblies encode full propositions (subject-verb-object).
The key (e.g., "dog_chases_cat") tells us the words and their order.
The assembly itself can be used for compatibility checking.

This is the bridge between neural representations and language output.
"""

from typing import List, Optional, Tuple, Dict, Set, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner
    from ..brain import EmergentNemoBrain

from ..areas import Area


class VPDecoder:
    """
    Decodes VP assemblies to word sequences.
    
    Two decoding modes:
    1. Key-based: Parse VP key string (fast, exact)
    2. Assembly-based: Find words through overlap (slower, emergent)
    
    The assembly-based mode is more NEMO-like but requires
    that we've stored word assemblies in core areas.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
    
    # =========================================================================
    # KEY-BASED DECODING (Fast path)
    # =========================================================================
    
    def decode_vp_key(self, vp_key: str) -> Dict[str, str]:
        """
        Decode a VP key to its component words.
        
        Args:
            vp_key: VP key like "dog_chases_cat" or "dog_runs"
        
        Returns:
            Dict with 'subject', 'verb', and optionally 'object'
        """
        parts = vp_key.split('_')
        
        result = {'subject': None, 'verb': None, 'object': None}
        
        if len(parts) >= 2:
            result['subject'] = parts[0]
            result['verb'] = parts[1]
        
        if len(parts) >= 3:
            result['object'] = parts[2]
        
        return result
    
    def decode_vp_key_to_sentence(self, vp_key: str, 
                                   add_determiner: bool = True) -> List[str]:
        """
        Decode VP key to a word sequence.
        
        Args:
            vp_key: VP key like "dog_chases_cat"
            add_determiner: Whether to add "the" before nouns
        
        Returns:
            List of words in order
        """
        parts = self.decode_vp_key(vp_key)
        words = []
        
        if parts['subject']:
            if add_determiner and self._needs_determiner(parts['subject']):
                words.append('the')
            words.append(parts['subject'])
        
        if parts['verb']:
            words.append(parts['verb'])
        
        if parts['object']:
            if add_determiner and self._needs_determiner(parts['object']):
                words.append('the')
            words.append(parts['object'])
        
        return words
    
    def _needs_determiner(self, word: str) -> bool:
        """Check if word needs a determiner (nouns do, pronouns don't)."""
        cat, _ = self.learner.get_emergent_category(word)
        return cat == 'NOUN'
    
    # =========================================================================
    # ASSEMBLY-BASED DECODING (Emergent path)
    # =========================================================================
    
    def decode_vp_assembly(self, vp_assembly: cp.ndarray,
                           min_overlap: float = 0.1) -> Dict[str, List[Tuple[str, float]]]:
        """
        Decode a VP assembly by finding overlapping word assemblies.
        
        This is the NEMO way: the VP assembly should overlap with
        the assemblies of its component words.
        
        Args:
            vp_assembly: The VP assembly to decode
            min_overlap: Minimum overlap to consider a match
        
        Returns:
            Dict with 'subjects', 'verbs', 'objects' - each a list of (word, overlap)
        """
        result = {
            'subjects': [],
            'verbs': [],
            'objects': [],
        }
        
        # Find overlapping noun assemblies (potential subjects/objects)
        nouns = self._find_overlapping_words(
            vp_assembly, Area.NOUN_CORE, min_overlap
        )
        
        # Find overlapping verb assemblies
        verbs = self._find_overlapping_words(
            vp_assembly, Area.VERB_CORE, min_overlap
        )
        
        # Find overlapping pronoun assemblies (can be subjects)
        pronouns = self._find_overlapping_words(
            vp_assembly, Area.PRON_CORE, min_overlap
        )
        
        # Nouns and pronouns can be subjects or objects
        # For now, put all in both lists - caller can disambiguate
        result['subjects'] = nouns + pronouns
        result['verbs'] = verbs
        result['objects'] = nouns  # Objects are typically nouns
        
        return result
    
    def _find_overlapping_words(self, target_assembly: cp.ndarray,
                                 area: Area,
                                 min_overlap: float) -> List[Tuple[str, float]]:
        """Find words whose assemblies overlap with target."""
        matches = []
        
        for word, word_assembly in self.brain.learned_assemblies[area].items():
            overlap = self.brain.get_assembly_overlap(target_assembly, word_assembly)
            if overlap >= min_overlap:
                matches.append((word, overlap))
        
        # Sort by overlap descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    # =========================================================================
    # VP QUERY METHODS
    # =========================================================================
    
    def find_vp_by_verb(self, verb: str) -> List[str]:
        """Find all VP keys containing this verb."""
        matches = []
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            if len(parts) >= 2 and parts[1] == verb:
                matches.append(key)
        return matches
    
    def find_vp_by_subject(self, subject: str) -> List[str]:
        """Find all VP keys with this subject."""
        matches = []
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            if len(parts) >= 1 and parts[0] == subject:
                matches.append(key)
        return matches
    
    def find_vp_by_object(self, obj: str) -> List[str]:
        """Find all VP keys with this object."""
        matches = []
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            if len(parts) >= 3 and parts[2] == obj:
                matches.append(key)
        return matches
    
    def find_vp_by_pattern(self, subject: str = None, 
                           verb: str = None,
                           obj: str = None) -> List[str]:
        """Find VP keys matching a pattern (None = wildcard)."""
        matches = []
        
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            
            if len(parts) < 2:
                continue
            
            # Check subject
            if subject is not None and parts[0] != subject:
                continue
            
            # Check verb
            if verb is not None and parts[1] != verb:
                continue
            
            # Check object
            if obj is not None:
                if len(parts) < 3 or parts[2] != obj:
                    continue
            
            matches.append(key)
        
        return matches
    
    # =========================================================================
    # ASSEMBLY-BASED VP MATCHING
    # =========================================================================
    
    def find_compatible_vps(self, seed_assembly: cp.ndarray,
                            area: Area,
                            min_overlap: float = 0.05) -> List[Tuple[str, float]]:
        """
        Find VP assemblies compatible with a seed assembly.
        
        This is the NEMO way to answer questions:
        - Activate the known part (e.g., verb assembly)
        - Find VP assemblies that overlap with it
        - Those VPs are "compatible" answers
        
        Args:
            seed_assembly: Assembly to match against
            area: Area the seed came from (for context)
            min_overlap: Minimum overlap threshold
        
        Returns:
            List of (vp_key, overlap) sorted by overlap
        """
        matches = []
        
        for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
            overlap = self.brain.get_assembly_overlap(seed_assembly, vp_assembly)
            if overlap >= min_overlap:
                matches.append((vp_key, overlap))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def find_vps_containing_word(self, word: str) -> List[Tuple[str, float]]:
        """
        Find VP assemblies that contain this word's assembly.
        
        Uses assembly overlap, not string matching.
        More NEMO-like than string-based search.
        """
        # Get word's assembly from its core area
        word_assembly = None
        
        for area in [Area.NOUN_CORE, Area.VERB_CORE, Area.PRON_CORE, Area.ADJ_CORE]:
            assembly = self.brain.get_learned_assembly(area, word)
            if assembly is not None:
                word_assembly = assembly
                break
        
        if word_assembly is None:
            return []
        
        return self.find_compatible_vps(word_assembly, Area.VP, min_overlap=0.05)


__all__ = ['VPDecoder']


