"""
Neural Decoder
==============

Decodes neural assemblies to words through REVERSE PROJECTION.

This is the key to truly emergent generation:
1. We have an active assembly in a core area (e.g., NOUN_CORE)
2. We project BACKWARDS to PHON
3. The PHON pattern that emerges overlaps with learned word assemblies
4. The word with highest overlap IS the decoded word

No string lookups. No templates. Pure neural activation.
"""

from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..brain import EmergentNemoBrain

from ..areas import Area


class NeuralDecoder:
    """
    Decodes neural assemblies to words through reverse projection.
    
    The NEMO way: words are PHON assemblies. To decode a concept,
    we project back to PHON and see which word assembly activates.
    """
    
    def __init__(self, brain: 'EmergentNemoBrain'):
        self.brain = brain
        
        # Cache of word -> PHON assembly for fast lookup
        # This is populated during learning
        self._phon_cache: Dict[str, cp.ndarray] = {}
    
    def cache_phon_assemblies(self):
        """
        Cache all known PHON assemblies for decoding.
        
        Call this after training to prepare for generation.
        """
        self._phon_cache = {}
        
        if Area.PHON in self.brain.assemblies:
            for word, assembly in self.brain.assemblies[Area.PHON].items():
                self._phon_cache[word] = assembly.copy()
    
    def decode_to_word(self, source_area: Area, 
                       source_assembly: cp.ndarray) -> Tuple[Optional[str], float]:
        """
        Decode an assembly to a word through assembly overlap.
        
        Process:
        1. Compare source_assembly to all learned word assemblies in source_area
        2. The word with highest overlap is the decoded word
        
        This is emergent: the word emerges from which learned assembly
        the current activation pattern most resembles.
        
        Note: We tried reverse projection to PHON, but it doesn't work well
        because PHON assemblies are arbitrary (not phonologically structured).
        Direct overlap comparison in the source area is more reliable.
        
        Args:
            source_area: Area the assembly is from
            source_assembly: The assembly to decode
        
        Returns:
            (word, confidence) or (None, 0.0) if no match
        """
        if source_assembly is None:
            return None, 0.0
        
        # Find word whose learned assembly in this area has highest overlap
        return self.brain.find_best_matching_word(source_area, source_assembly)
    
    def decode_to_words(self, source_area: Area,
                        source_assembly: cp.ndarray,
                        top_k: int = 5,
                        min_overlap: float = 0.05) -> List[Tuple[str, float]]:
        """
        Decode an assembly to multiple candidate words.
        
        Returns top-k words ranked by overlap with learned assemblies.
        """
        if source_assembly is None:
            return []
        
        # Get all words with sufficient overlap
        compatible = self.brain.get_compatible_words(
            source_area, source_assembly, 
            min_overlap=min_overlap
        )
        
        return compatible[:top_k]
    
    def _project_to_phon(self, source_area: Area, 
                         source_assembly: cp.ndarray) -> Optional[cp.ndarray]:
        """
        Project from a core area back to PHON.
        
        This is the reverse of encoding. We use the same projection
        mechanism, but in the opposite direction.
        
        The learned weights determine which PHON neurons activate.
        """
        # Clear PHON first
        self.brain._clear_area(Area.PHON)
        
        # Project source assembly to PHON
        # This uses the implicit connectivity + learned weights
        result = self.brain._project(Area.PHON, source_assembly, learn=False)
        
        return result
    
    def _find_best_phon_match(self, phon_activation: cp.ndarray) -> Tuple[Optional[str], float]:
        """Find the word whose PHON assembly best matches the activation."""
        best_word = None
        best_overlap = 0.0
        
        activation_set = set(phon_activation.get().tolist())
        
        for word, phon_assembly in self._phon_cache.items():
            phon_set = set(phon_assembly.get().tolist())
            overlap = len(activation_set & phon_set) / self.brain.p.k
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = word
        
        return best_word, best_overlap
    
    def _find_matching_phon(self, phon_activation: cp.ndarray,
                            top_k: int,
                            min_overlap: float) -> List[Tuple[str, float]]:
        """Find top-k words matching the PHON activation."""
        matches = []
        
        activation_set = set(phon_activation.get().tolist())
        
        for word, phon_assembly in self._phon_cache.items():
            phon_set = set(phon_assembly.get().tolist())
            overlap = len(activation_set & phon_set) / self.brain.p.k
            
            if overlap >= min_overlap:
                matches.append((word, overlap))
        
        # Sort by overlap descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    # =========================================================================
    # AREA-SPECIFIC DECODING
    # =========================================================================
    
    def decode_noun(self, noun_assembly: cp.ndarray) -> Tuple[Optional[str], float]:
        """Decode a NOUN_CORE assembly to a word."""
        return self.decode_to_word(Area.NOUN_CORE, noun_assembly)
    
    def decode_verb(self, verb_assembly: cp.ndarray) -> Tuple[Optional[str], float]:
        """Decode a VERB_CORE assembly to a word."""
        return self.decode_to_word(Area.VERB_CORE, verb_assembly)
    
    def decode_nouns(self, noun_assembly: cp.ndarray, 
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """Decode a NOUN_CORE assembly to multiple candidate nouns."""
        return self.decode_to_words(Area.NOUN_CORE, noun_assembly, top_k)
    
    def decode_verbs(self, verb_assembly: cp.ndarray,
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """Decode a VERB_CORE assembly to multiple candidate verbs."""
        return self.decode_to_words(Area.VERB_CORE, verb_assembly, top_k)


class EmergentRetriever:
    """
    Retrieves knowledge through activation spreading, not database lookup.
    
    The NEMO way to answer "who runs":
    1. Activate "runs" in VERB_CORE
    2. Let activation spread to VP (via learned weights)
    3. VP activation spreads back to NOUN_CORE
    4. The NOUN_CORE pattern that emerges represents the subjects
    5. Decode NOUN_CORE to words
    
    No VP key lookup. No string matching. Pure neural retrieval.
    """
    
    def __init__(self, brain: 'EmergentNemoBrain'):
        self.brain = brain
        self.decoder = NeuralDecoder(brain)
    
    def prepare(self):
        """Prepare for retrieval by caching PHON assemblies."""
        self.decoder.cache_phon_assemblies()
    
    def retrieve_subjects_for_verb(self, verb: str, 
                                   top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find subjects that perform a verb through VP assembly overlap.
        
        Process:
        1. Get verb's VERB_CORE assembly
        2. Find VP assemblies that overlap with verb assembly
        3. For each matching VP, find overlapping NOUN_CORE assemblies
        4. The nouns that overlap with matching VPs are the subjects
        
        This is emergent: we use assembly overlap, not string matching.
        """
        # Get verb assembly
        verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
        if verb_assembly is None:
            return []
        
        # Find VP assemblies that contain this verb
        # (high overlap with verb assembly)
        matching_vps = []
        for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
            overlap = self.brain.get_assembly_overlap(verb_assembly, vp_assembly)
            if overlap > 0.05:  # Threshold for "contains verb"
                matching_vps.append((vp_key, vp_assembly, overlap))
        
        if not matching_vps:
            return []
        
        # For each matching VP, find which nouns overlap with it
        subject_scores = {}
        
        for vp_key, vp_assembly, vp_overlap in matching_vps:
            # Find nouns that overlap with this VP
            for noun, noun_assembly in self.brain.learned_assemblies[Area.NOUN_CORE].items():
                noun_overlap = self.brain.get_assembly_overlap(noun_assembly, vp_assembly)
                if noun_overlap > 0.05:
                    # Weight by both VP match and noun match
                    score = vp_overlap * noun_overlap
                    if noun not in subject_scores or score > subject_scores[noun]:
                        subject_scores[noun] = score
        
        # Sort by score
        subjects = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
        
        return subjects[:top_k]
    
    def retrieve_verbs_for_subject(self, subject: str,
                                   top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find verbs that a subject performs through VP assembly overlap.
        
        Process:
        1. Get subject's NOUN_CORE assembly
        2. Find VP assemblies that overlap with subject assembly
        3. For each matching VP, find overlapping VERB_CORE assemblies
        4. The verbs that overlap with matching VPs are the actions
        """
        # Get subject assembly (try NOUN_CORE first, then PRON_CORE)
        subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
        if subj_assembly is None:
            subj_assembly = self.brain.get_learned_assembly(Area.PRON_CORE, subject)
        if subj_assembly is None:
            return []
        
        # Find VP assemblies that contain this subject
        matching_vps = []
        for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
            overlap = self.brain.get_assembly_overlap(subj_assembly, vp_assembly)
            if overlap > 0.05:
                matching_vps.append((vp_key, vp_assembly, overlap))
        
        if not matching_vps:
            return []
        
        # For each matching VP, find which verbs overlap with it
        verb_scores = {}
        
        for vp_key, vp_assembly, vp_overlap in matching_vps:
            for verb, verb_assembly in self.brain.learned_assemblies[Area.VERB_CORE].items():
                verb_overlap = self.brain.get_assembly_overlap(verb_assembly, vp_assembly)
                if verb_overlap > 0.05:
                    score = vp_overlap * verb_overlap
                    if verb not in verb_scores or score > verb_scores[verb]:
                        verb_scores[verb] = score
        
        # Sort by score
        verbs = sorted(verb_scores.items(), key=lambda x: x[1], reverse=True)
        
        return verbs[:top_k]
    
    def retrieve_objects_for_subject_verb(self, subject: str, verb: str,
                                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find objects for a subject-verb pair through neural activation.
        
        Process:
        1. Activate subject and verb
        2. Project both to VP (creates merged pattern)
        3. Project VP to NOUN_CORE (for objects)
        4. Decode, filtering out the subject
        """
        subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
        verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
        
        if subj_assembly is None or verb_assembly is None:
            return []
        
        # Clear areas
        self.brain.clear_all()
        
        # Project subject to VP
        self.brain._project(Area.VP, subj_assembly, learn=False)
        
        # Project verb to VP (merges with subject pattern)
        self.brain._project(Area.VP, verb_assembly, learn=False)
        
        if self.brain.current[Area.VP] is None:
            return []
        
        # Project VP to NOUN_CORE (for objects)
        self.brain._project(Area.NOUN_CORE, self.brain.current[Area.VP], learn=False)
        
        if self.brain.current[Area.NOUN_CORE] is None:
            return []
        
        # Decode NOUN_CORE to words, filtering out subject
        candidates = self.decoder.decode_nouns(self.brain.current[Area.NOUN_CORE], top_k + 1)
        
        return [(w, s) for w, s in candidates if w != subject][:top_k]
    
    def check_pattern_exists(self, subject: str, verb: str, 
                             obj: str = None) -> Tuple[bool, float]:
        """
        Check if a pattern exists through neural activation.
        
        Instead of looking up a VP key, we:
        1. Activate subject, verb, (object)
        2. Project to VP
        3. Check VP stability - high stability = learned pattern
        
        Returns: (exists, confidence)
        """
        subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
        verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
        
        if subj_assembly is None or verb_assembly is None:
            return False, 0.0
        
        # Clear areas
        self.brain.clear_all()
        
        # Project subject to VP
        self.brain._project(Area.VP, subj_assembly, learn=False)
        
        # Project verb to VP
        self.brain._project(Area.VP, verb_assembly, learn=False)
        
        # If object specified, project it too
        if obj:
            obj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, obj)
            if obj_assembly is not None:
                self.brain._project(Area.VP, obj_assembly, learn=False)
        
        if self.brain.current[Area.VP] is None:
            return False, 0.0
        
        # Measure stability - high stability means this is a learned pattern
        stability = self.brain.measure_stability(Area.VP, rounds=3)
        
        # Threshold for "exists"
        exists = stability > 0.3
        
        return exists, stability


__all__ = ['NeuralDecoder', 'EmergentRetriever']

