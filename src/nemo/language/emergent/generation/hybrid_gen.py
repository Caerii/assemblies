"""
Hybrid Emergent Generator
=========================

A pragmatic generator that combines:
- NEURAL: Assembly overlap for compatibility checking
- NEURAL: Grounding-based category inference
- SYMBOLIC: VP key parsing for word extraction

This is honest about what's emergent vs symbolic.

What's Neural (Emergent):
1. Word categories emerge from grounding
2. VP assemblies represent learned propositions
3. Compatibility is checked via assembly overlap
4. Pattern existence is checked via VP stability

What's Symbolic:
1. VP keys encode structure (subject_verb_object)
2. Word extraction uses string parsing
3. Response formatting uses templates

This hybrid approach works well in practice while we work on
making VP structure more neurally decodable.
"""

from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner

from ..areas import Area


class HybridEmergentGenerator:
    """
    Hybrid generator: neural compatibility + symbolic decoding.
    
    This is pragmatic and works well. The key insight is that
    VP keys ARE a form of learned structure - they're created
    during training based on what was learned.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
    
    def generate(self, input_words: List[str]) -> str:
        """Generate a response."""
        if not input_words:
            return ""
        
        # Encode input (neural)
        encoded = self._encode_input(input_words)
        
        # Determine response type (neural - uses grounding)
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
        """Encode input words - uses neural category inference."""
        result = {
            'words': words,
            'categories': {},
            'verb': None,
            'subject': None,
            'object': None,
        }
        
        for word in words:
            word_lower = word.lower()
            # NEURAL: Category emerges from grounding history
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
        """Determine response type - uses neural grounding."""
        if not words:
            return 'unknown'
        
        first_word = words[0].lower()
        
        # NEURAL: Check grounding patterns
        grounding = self.learner.word_grounding.get(first_word, {})
        
        if grounding.get('SOCIAL', 0) > 0:
            return 'who_query'
        if grounding.get('VISUAL', 0) > 0:
            return 'what_query'
        if grounding.get('NONE', 0) > 0 or not grounding:
            if first_word in ['does', 'do', 'is', 'can', 'did']:
                return 'yesno_query'
        
        return 'statement'
    
    def _generate_who_response(self, encoded: Dict) -> str:
        """Generate 'who' response - neural compatibility + symbolic decode."""
        verb = encoded.get('verb')
        
        if not verb:
            return "i could not find the action"
        
        # Try multiple verb forms
        verb_forms = self._get_verb_forms(verb)
        
        subjects = []
        matched_verb = verb
        
        for v in verb_forms:
            # NEURAL: Get verb assembly
            verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, v)
            
            if verb_assembly is None:
                continue
            
            # Find subjects through VP overlap (NEURAL) + key parsing (SYMBOLIC)
            for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
                # NEURAL: Check if VP contains this verb via overlap
                overlap = self.brain.get_assembly_overlap(verb_assembly, vp_assembly)
                
                if overlap > 0.1:  # Threshold
                    # SYMBOLIC: Parse VP key to get subject
                    parts = vp_key.split('_')
                    if len(parts) >= 2 and parts[1] == v:
                        subject = parts[0]
                        if subject not in subjects:
                            subjects.append(subject)
                            matched_verb = v
        
        if subjects:
            return f"{', '.join(subjects[:4])} {matched_verb}"
        else:
            return f"i do not know who {verb}"
    
    def _get_verb_forms(self, verb: str) -> List[str]:
        """Get possible verb forms (base, 3sg, etc.)."""
        forms = [verb]
        
        # Add 3sg form
        if not verb.endswith('s'):
            forms.append(verb + 's')
        
        # Add base form (remove 's')
        if verb.endswith('s') and len(verb) > 2:
            forms.append(verb[:-1])
        
        # Add 'es' form for verbs ending in consonant
        if not verb.endswith('s') and not verb.endswith('e'):
            forms.append(verb + 'es')
        
        return forms
    
    def _generate_what_response(self, encoded: Dict) -> str:
        """Generate 'what' response - neural compatibility + symbolic decode."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        words_lower = [w.lower() for w in encoded['words']]
        
        # Fix: 'what' is being picked up as subject, skip it
        if subject == 'what':
            subject = None
            for word, cat in encoded['categories'].items():
                if cat in ['NOUN', 'PRONOUN'] and word != 'what':
                    subject = word
                    break
        
        # "what does X do" pattern
        if ('do' in words_lower or not verb) and subject:
            # NEURAL: Get subject assembly
            subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
            if subj_assembly is None:
                subj_assembly = self.brain.get_learned_assembly(Area.PRON_CORE, subject)
            
            if subj_assembly is None:
                return f"i do not know {subject}"
            
            # Find verbs through VP overlap + key parsing
            verbs = []
            
            for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
                overlap = self.brain.get_assembly_overlap(subj_assembly, vp_assembly)
                
                if overlap > 0.1:
                    parts = vp_key.split('_')
                    if len(parts) >= 2 and parts[0] == subject:
                        verb_found = parts[1]
                        if verb_found not in verbs:
                            verbs.append(verb_found)
            
            if verbs:
                return f"{subject} {', '.join(verbs[:3])}"
            else:
                return f"i do not know what {subject} does"
        
        # "what does X verb" pattern
        if subject and verb:
            subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
            verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
            
            if subj_assembly is None or verb_assembly is None:
                return f"i do not know {subject} or {verb}"
            
            # Find objects
            objects = []
            
            for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
                subj_overlap = self.brain.get_assembly_overlap(subj_assembly, vp_assembly)
                verb_overlap = self.brain.get_assembly_overlap(verb_assembly, vp_assembly)
                
                if subj_overlap > 0.05 and verb_overlap > 0.05:
                    parts = vp_key.split('_')
                    if len(parts) >= 3 and parts[0] == subject and parts[1] == verb:
                        obj = parts[2]
                        if obj not in objects:
                            objects.append(obj)
            
            if objects:
                return f"{subject} {verb} {', '.join(objects[:3])}"
            else:
                return f"i do not know what {subject} {verb}"
        
        return "i could not understand the question"
    
    def _generate_yesno_response(self, encoded: Dict) -> str:
        """Generate yes/no response - uses neural VP stability."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        obj = encoded.get('object')
        
        if not subject:
            return "i could not find the subject"
        
        if not verb:
            # Try to find any word that could be a verb
            for word, cat in encoded['categories'].items():
                if cat == 'UNKNOWN' and word not in ['does', 'do', 'the', 'a']:
                    verb = word
                    break
        
        if not verb:
            return "i could not find the action"
        
        # NEURAL: Check pattern existence via VP assembly lookup + overlap
        subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, subject)
        
        # Try different verb forms
        verb_forms = [verb]
        if not verb.endswith('s'):
            verb_forms.append(verb + 's')
        if verb.endswith('s'):
            verb_forms.append(verb[:-1])
        
        for v in verb_forms:
            verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, v)
            if verb_assembly is None:
                continue
            
            # Check if matching VP exists
            for vp_key, vp_assembly in self.brain.learned_assemblies[Area.VP].items():
                parts = vp_key.split('_')
                
                # Check structure match
                if len(parts) >= 2 and parts[0] == subject and parts[1] == v:
                    if obj is None or (len(parts) >= 3 and parts[2] == obj):
                        # NEURAL: Verify via overlap
                        if subj_assembly is not None:
                            overlap = self.brain.get_assembly_overlap(subj_assembly, vp_assembly)
                            if overlap > 0.05:
                                if obj:
                                    return f"yes {subject} {v} {obj}"
                                else:
                                    return f"yes {subject} {v}"
        
        return f"no i have not learned that"
    
    def _generate_default_response(self, encoded: Dict) -> str:
        """Default response for statements."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        
        if subject and verb:
            return f"ok {subject} {verb}"
        
        return "ok"


__all__ = ['HybridEmergentGenerator']

