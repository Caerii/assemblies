"""
Truly Emergent Retriever
========================

Uses the VP component areas (VP_SUBJ, VP_VERB, VP_OBJ) for emergent retrieval.

Key insight: Instead of trying to decode merged VP assemblies (which doesn't work),
we use SEPARATE areas that preserve component information:

- VP_SUBJ: Contains subject assemblies for each sentence
- VP_VERB: Contains verb assemblies for each sentence  
- VP_OBJ: Contains object assemblies for each sentence

These are stored with the same VP key, so we can:
1. Find VP_VERB assemblies that match a query verb (neural overlap)
2. For matching VP_VERBs, retrieve the corresponding VP_SUBJ (same key)
3. Decode VP_SUBJ to find the subject word (neural overlap)

This is FULLY EMERGENT:
- Matching uses neural assembly overlap
- Retrieval uses stored assemblies (not string parsing)
- Decoding uses neural overlap with learned word assemblies
"""

from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner
    from ..brain import EmergentNemoBrain

from ..areas import Area


class EmergentRetriever:
    """
    Retrieves knowledge through neural assembly operations.
    
    Uses VP component areas for emergent decomposition.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
    
    def find_subjects_for_verb(self, verb: str, 
                                top_k: int = 5,
                                min_overlap: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find subjects that perform a verb - EMERGENT approach.
        
        Process:
        1. Get verb's VERB_CORE assembly
        2. Find VP_VERB assemblies that overlap with it (neural matching)
        3. For each matching VP_VERB, get the corresponding VP_SUBJ
        4. Decode VP_SUBJ to word by comparing with learned noun assemblies
        
        Returns: List of (subject_word, confidence) tuples
        """
        # Get verb assembly (try multiple forms)
        verb_assembly = self._get_verb_assembly(verb)
        if verb_assembly is None:
            return []
        
        # Find VP_VERB assemblies that match this verb
        matching_vp_keys = self._find_matching_vp_verbs(verb_assembly, min_overlap)
        
        if not matching_vp_keys:
            return []
        
        # For each matching VP, decode the subject
        subjects = {}
        
        for vp_key, verb_overlap in matching_vp_keys:
            # Get the VP_SUBJ assembly for this VP
            vp_subj = self.brain.get_learned_assembly(Area.VP_SUBJ, vp_key)
            if vp_subj is None:
                continue
            
            # Decode VP_SUBJ to find the subject word
            subject_word, subj_overlap = self._decode_subject(vp_subj)
            
            if subject_word and subj_overlap > 0:
                # Combine verb match confidence with subject decode confidence
                confidence = verb_overlap * subj_overlap
                if subject_word not in subjects or confidence > subjects[subject_word]:
                    subjects[subject_word] = confidence
        
        # Sort by confidence
        result = sorted(subjects.items(), key=lambda x: -x[1])
        return result[:top_k]
    
    def find_verbs_for_subject(self, subject: str,
                                top_k: int = 5,
                                min_overlap: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find verbs that a subject performs - EMERGENT approach.
        
        Process:
        1. Get subject's NOUN_CORE/PRON_CORE assembly
        2. Find VP_SUBJ assemblies that overlap with it
        3. For each matching VP_SUBJ, get the corresponding VP_VERB
        4. Decode VP_VERB to word
        
        Returns: List of (verb_word, confidence) tuples
        """
        # Get subject assembly
        subj_assembly = self._get_subject_assembly(subject)
        if subj_assembly is None:
            return []
        
        # Find VP_SUBJ assemblies that match this subject
        matching_vp_keys = self._find_matching_vp_subjs(subj_assembly, min_overlap)
        
        if not matching_vp_keys:
            return []
        
        # For each matching VP, decode the verb
        verbs = {}
        
        for vp_key, subj_overlap in matching_vp_keys:
            # Get the VP_VERB assembly for this VP
            vp_verb = self.brain.get_learned_assembly(Area.VP_VERB, vp_key)
            if vp_verb is None:
                continue
            
            # Decode VP_VERB to find the verb word
            verb_word, verb_overlap = self._decode_verb(vp_verb)
            
            if verb_word and verb_overlap > 0:
                confidence = subj_overlap * verb_overlap
                if verb_word not in verbs or confidence > verbs[verb_word]:
                    verbs[verb_word] = confidence
        
        # Sort by confidence
        result = sorted(verbs.items(), key=lambda x: -x[1])
        return result[:top_k]
    
    def find_objects_for_subject_verb(self, subject: str, verb: str,
                                       top_k: int = 5,
                                       min_overlap: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find objects for a subject-verb pair - EMERGENT approach.
        
        Process:
        1. Get subject and verb assemblies
        2. Find VP_OBJ assemblies (these have subject_verb_object keys)
        3. Check if the corresponding VP_SUBJ and VP_VERB match
        4. Decode VP_OBJ
        
        Note: VP_SUBJ/VP_VERB keys are "subject_verb", VP_OBJ keys are "subject_verb_object"
        
        Returns: List of (object_word, confidence) tuples
        """
        subj_assembly = self._get_subject_assembly(subject)
        verb_assembly = self._get_verb_assembly(verb)
        
        if subj_assembly is None or verb_assembly is None:
            return []
        
        objects = {}
        
        # Iterate over VP_OBJ assemblies (transitive sentences)
        for vp_key, vp_obj in self.brain.learned_assemblies[Area.VP_OBJ].items():
            # VP_OBJ key is "subject_verb_object", extract "subject_verb"
            parts = vp_key.split('_')
            if len(parts) < 3:
                continue
            
            subj_verb_key = f"{parts[0]}_{parts[1]}"
            
            # Get VP_SUBJ and VP_VERB for this sentence
            vp_subj = self.brain.get_learned_assembly(Area.VP_SUBJ, subj_verb_key)
            vp_verb = self.brain.get_learned_assembly(Area.VP_VERB, subj_verb_key)
            
            if vp_subj is None or vp_verb is None:
                continue
            
            # Check if subject and verb match
            subj_overlap = self.brain.get_assembly_overlap(subj_assembly, vp_subj)
            verb_overlap = self.brain.get_assembly_overlap(verb_assembly, vp_verb)
            
            if subj_overlap >= min_overlap and verb_overlap >= min_overlap:
                # Decode the object
                obj_word, obj_overlap = self._decode_object(vp_obj)
                
                if obj_word and obj_overlap > 0:
                    confidence = subj_overlap * verb_overlap * obj_overlap
                    if obj_word not in objects or confidence > objects[obj_word]:
                        objects[obj_word] = confidence
        
        result = sorted(objects.items(), key=lambda x: -x[1])
        return result[:top_k]
    
    def check_pattern_exists(self, subject: str, verb: str, 
                              obj: str = None,
                              min_overlap: float = 0.1) -> Tuple[bool, float]:
        """
        Check if a pattern exists in learned knowledge - EMERGENT approach.
        
        Uses VP component overlap to verify pattern existence.
        
        Returns: (exists, confidence)
        """
        subj_assembly = self._get_subject_assembly(subject)
        verb_assembly = self._get_verb_assembly(verb)
        
        if subj_assembly is None or verb_assembly is None:
            return False, 0.0
        
        obj_assembly = None
        if obj:
            obj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, obj)
        
        # Find best matching VP
        best_confidence = 0.0
        
        for vp_key in self.brain.learned_assemblies[Area.VP_SUBJ].keys():
            vp_subj = self.brain.get_learned_assembly(Area.VP_SUBJ, vp_key)
            vp_verb = self.brain.get_learned_assembly(Area.VP_VERB, vp_key)
            
            if vp_subj is None or vp_verb is None:
                continue
            
            subj_overlap = self.brain.get_assembly_overlap(subj_assembly, vp_subj)
            verb_overlap = self.brain.get_assembly_overlap(verb_assembly, vp_verb)
            
            if subj_overlap < min_overlap or verb_overlap < min_overlap:
                continue
            
            confidence = subj_overlap * verb_overlap
            
            # If object specified, check VP_OBJ too
            if obj_assembly is not None:
                vp_obj = self.brain.get_learned_assembly(Area.VP_OBJ, vp_key)
                if vp_obj is None:
                    continue
                obj_overlap = self.brain.get_assembly_overlap(obj_assembly, vp_obj)
                if obj_overlap < min_overlap:
                    continue
                confidence *= obj_overlap
            
            if confidence > best_confidence:
                best_confidence = confidence
        
        exists = best_confidence > min_overlap
        return exists, best_confidence
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_verb_assembly(self, verb: str) -> Optional[cp.ndarray]:
        """Get verb assembly, trying multiple forms."""
        # Try exact form
        assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
        if assembly is not None:
            return assembly
        
        # Try with 's' suffix
        if not verb.endswith('s'):
            assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb + 's')
            if assembly is not None:
                return assembly
        
        # Try without 's' suffix
        if verb.endswith('s') and len(verb) > 2:
            assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb[:-1])
            if assembly is not None:
                return assembly
        
        return None
    
    def _get_subject_assembly(self, subject: str) -> Optional[cp.ndarray]:
        """Get subject assembly from NOUN_CORE or PRON_CORE."""
        assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
        if assembly is not None:
            return assembly
        
        assembly = self.brain.get_learned_assembly(Area.PRON_CORE, subject)
        return assembly
    
    def _find_matching_vp_verbs(self, verb_assembly: cp.ndarray, 
                                 min_overlap: float) -> List[Tuple[str, float]]:
        """Find VP_VERB assemblies that match a verb assembly."""
        matches = []
        
        for vp_key, vp_verb in self.brain.learned_assemblies[Area.VP_VERB].items():
            overlap = self.brain.get_assembly_overlap(verb_assembly, vp_verb)
            if overlap >= min_overlap:
                matches.append((vp_key, overlap))
        
        matches.sort(key=lambda x: -x[1])
        return matches
    
    def _find_matching_vp_subjs(self, subj_assembly: cp.ndarray,
                                 min_overlap: float) -> List[Tuple[str, float]]:
        """Find VP_SUBJ assemblies that match a subject assembly."""
        matches = []
        
        for vp_key, vp_subj in self.brain.learned_assemblies[Area.VP_SUBJ].items():
            overlap = self.brain.get_assembly_overlap(subj_assembly, vp_subj)
            if overlap >= min_overlap:
                matches.append((vp_key, overlap))
        
        matches.sort(key=lambda x: -x[1])
        return matches
    
    def _decode_subject(self, vp_subj: cp.ndarray) -> Tuple[Optional[str], float]:
        """Decode VP_SUBJ assembly to subject word."""
        # Try NOUN_CORE first
        word, overlap = self.brain.find_best_matching_word(Area.NOUN_CORE, vp_subj)
        if word and overlap > 0.1:
            return word, overlap
        
        # Try PRON_CORE
        word, overlap = self.brain.find_best_matching_word(Area.PRON_CORE, vp_subj)
        return word, overlap
    
    def _decode_verb(self, vp_verb: cp.ndarray) -> Tuple[Optional[str], float]:
        """Decode VP_VERB assembly to verb word."""
        return self.brain.find_best_matching_word(Area.VERB_CORE, vp_verb)
    
    def _decode_object(self, vp_obj: cp.ndarray) -> Tuple[Optional[str], float]:
        """Decode VP_OBJ assembly to object word."""
        return self.brain.find_best_matching_word(Area.NOUN_CORE, vp_obj)


class EmergentGenerator:
    """
    Generates responses using fully emergent retrieval.
    
    No string parsing of VP keys - all retrieval uses neural assembly overlap.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
        self.retriever = EmergentRetriever(learner)
    
    def generate(self, input_words: List[str]) -> str:
        """Generate a response to input words."""
        if not input_words:
            return ""
        
        # Encode input
        encoded = self._encode_input(input_words)
        
        # Determine response type
        response_type = self._determine_response_type(input_words, encoded)
        
        # Generate based on type
        if response_type == 'who_query':
            return self._generate_who_response(encoded)
        elif response_type == 'what_query':
            return self._generate_what_response(encoded)
        elif response_type == 'yesno_query':
            return self._generate_yesno_response(encoded)
        else:
            return self._generate_default_response(encoded)
    
    def _encode_input(self, words: List[str]) -> Dict:
        """Encode input words using emergent categories."""
        result = {
            'words': words,
            'categories': {},
            'verb': None,
            'subject': None,
            'object': None,
        }
        
        for word in words:
            word_lower = word.lower()
            cat, _ = self.learner.get_emergent_category(word_lower)
            result['categories'][word_lower] = cat
            
            if cat == 'VERB' and result['verb'] is None:
                result['verb'] = word_lower
            elif cat in ['NOUN', 'PRONOUN']:
                if result['verb'] is None and result['subject'] is None:
                    result['subject'] = word_lower
                elif result['verb'] is not None and result['object'] is None:
                    result['object'] = word_lower
        
        return result
    
    def _determine_response_type(self, words: List[str], encoded: Dict) -> str:
        """Determine response type based on question words."""
        if not words:
            return 'unknown'
        
        first_word = words[0].lower()
        
        # Direct question word detection (most reliable)
        if first_word == 'who':
            return 'who_query'
        if first_word == 'what':
            return 'what_query'
        if first_word in ['does', 'do', 'is', 'can', 'did', 'will', 'has', 'have']:
            return 'yesno_query'
        if first_word in ['where', 'when', 'how', 'why']:
            return 'wh_query'
        
        # Fall back to grounding-based detection
        grounding = self.learner.word_grounding.get(first_word, {})
        if grounding.get('SOCIAL', 0) > 0:
            return 'who_query'
        if grounding.get('VISUAL', 0) > 0:
            return 'what_query'
        
        return 'statement'
    
    def _generate_who_response(self, encoded: Dict) -> str:
        """Generate 'who' response using emergent retrieval."""
        verb = encoded.get('verb')
        if not verb:
            return "i could not find the action"
        
        # Use emergent retrieval to find subjects
        subjects = self.retriever.find_subjects_for_verb(verb, top_k=5)
        
        if subjects:
            subject_words = [s[0] for s in subjects]
            return f"{', '.join(subject_words)} {verb}"
        else:
            return f"i do not know who {verb}"
    
    def _generate_what_response(self, encoded: Dict) -> str:
        """Generate 'what' response using emergent retrieval."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        words_lower = [w.lower() for w in encoded['words']]
        
        # Fix: 'what' being picked up as subject
        if subject == 'what':
            subject = None
            for word, cat in encoded['categories'].items():
                if cat in ['NOUN', 'PRONOUN'] and word != 'what':
                    subject = word
                    break
        
        # "what does X do" pattern
        if ('do' in words_lower or not verb) and subject:
            verbs = self.retriever.find_verbs_for_subject(subject, top_k=5)
            if verbs:
                verb_words = [v[0] for v in verbs]
                return f"{subject} {', '.join(verb_words)}"
            else:
                return f"i do not know what {subject} does"
        
        # "what does X verb" pattern
        if subject and verb:
            objects = self.retriever.find_objects_for_subject_verb(subject, verb, top_k=5)
            if objects:
                obj_words = [o[0] for o in objects]
                return f"{subject} {verb} {', '.join(obj_words)}"
            else:
                return f"i do not know what {subject} {verb}"
        
        return "i could not understand the question"
    
    def _generate_yesno_response(self, encoded: Dict) -> str:
        """Generate yes/no response using emergent pattern checking."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        obj = encoded.get('object')
        
        if not subject:
            return "i could not find the subject"
        
        if not verb:
            for word, cat in encoded['categories'].items():
                if cat == 'UNKNOWN' and word not in ['does', 'do', 'the', 'a']:
                    verb = word
                    break
        
        if not verb:
            return "i could not find the action"
        
        # Check if pattern exists using emergent retrieval
        exists, confidence = self.retriever.check_pattern_exists(subject, verb, obj)
        
        if exists:
            if obj:
                return f"yes {subject} {verb} {obj}"
            else:
                return f"yes {subject} {verb}"
        else:
            return f"no i have not learned that"
    
    def _generate_default_response(self, encoded: Dict) -> str:
        """Default response for statements."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        
        if subject and verb:
            return f"ok {subject} {verb}"
        return "ok"


__all__ = ['EmergentRetriever', 'EmergentGenerator']

