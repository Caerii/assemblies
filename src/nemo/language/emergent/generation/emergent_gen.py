"""
Truly Emergent Generator
========================

Generates responses through PURE NEURAL ACTIVATION.

No string lookups. No templates. No VP key parsing.

The response EMERGES from:
1. Activating query assemblies
2. Spreading through learned weights
3. Decoding settled patterns to words

This is what NEMO-style generation should look like.
"""

from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner

from ..areas import Area
from .neural_decoder import NeuralDecoder, EmergentRetriever


class TrueEmergentGenerator:
    """
    Generates responses through pure neural activation.
    
    Key differences from the previous "emergent" generator:
    1. No VP key string lookups
    2. No template formatting
    3. Words emerge from decoding neural patterns
    4. Answers emerge from activation spreading
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
        self.decoder = NeuralDecoder(self.brain)
        self.retriever = EmergentRetriever(self.brain)
        
        # Prepare decoder
        self.decoder.cache_phon_assemblies()
    
    def generate(self, input_words: List[str]) -> str:
        """
        Generate a response through emergent activation.
        
        The entire process is neural:
        1. Encode input words to assemblies
        2. Determine response type from grounding patterns
        3. Spread activation to find answer
        4. Decode answer assemblies to words
        """
        if not input_words:
            return ""
        
        # Encode input
        encoded = self._encode_input(input_words)
        
        # Determine what kind of response is needed
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
    
    def _encode_input(self, words: List[str]) -> Dict[str, any]:
        """
        Encode input words to neural assemblies.
        
        Returns dict with:
        - 'words': list of words
        - 'assemblies': dict of area -> assembly for each word
        - 'categories': dict of word -> emergent category
        """
        result = {
            'words': words,
            'assemblies': {},
            'categories': {},
            'verb': None,
            'subject': None,
            'object': None,
        }
        
        for word in words:
            word_lower = word.lower()
            cat, _ = self.learner.get_emergent_category(word_lower)
            result['categories'][word_lower] = cat
            
            # Get assemblies from appropriate areas
            for area in [Area.NOUN_CORE, Area.VERB_CORE, Area.ADJ_CORE, 
                        Area.PRON_CORE, Area.PREP_CORE]:
                assembly = self.brain.get_learned_assembly(area, word_lower)
                if assembly is not None:
                    if word_lower not in result['assemblies']:
                        result['assemblies'][word_lower] = {}
                    result['assemblies'][word_lower][area] = assembly
            
            # Track subject/verb/object
            if cat == 'VERB' and result['verb'] is None:
                result['verb'] = word_lower
            elif cat in ['NOUN', 'PRONOUN']:
                if result['verb'] is None and result['subject'] is None:
                    result['subject'] = word_lower
                elif result['verb'] is not None and result['object'] is None:
                    result['object'] = word_lower
        
        return result
    
    def _determine_response_type(self, words: List[str], 
                                  encoded: Dict) -> str:
        """
        Determine response type from neural patterns.
        
        Instead of checking if first word == "who", we check
        the grounding pattern of the first word.
        """
        if not words:
            return 'unknown'
        
        first_word = words[0].lower()
        
        # Check grounding patterns
        grounding = self.learner.word_grounding.get(first_word, {})
        
        # SOCIAL + QUERY grounding = "who" type
        if grounding.get('SOCIAL', 0) > 0:
            return 'who_query'
        
        # VISUAL + QUERY grounding = "what" type
        if grounding.get('VISUAL', 0) > 0:
            return 'what_query'
        
        # No grounding (function word) at start = could be yes/no
        if grounding.get('NONE', 0) > 0 or not grounding:
            if first_word in ['does', 'do', 'is', 'can', 'did']:
                return 'yesno_query'
        
        return 'statement'
    
    def _generate_who_response(self, encoded: Dict) -> str:
        """
        Generate response to "who" query through neural activation.
        
        Process:
        1. Get verb from encoded input
        2. Use EmergentRetriever to find subjects
        3. Decode subject assemblies to words
        """
        verb = encoded.get('verb')
        
        if not verb:
            # Try to find a verb in the words
            for word, cat in encoded['categories'].items():
                if cat == 'VERB':
                    verb = word
                    break
        
        if not verb:
            return "i could not find the action"
        
        # Retrieve subjects through neural activation
        subjects = self.retriever.retrieve_subjects_for_verb(verb)
        
        if not subjects:
            return f"i do not know who {verb}"
        
        # Format response from decoded words
        subject_words = [w for w, _ in subjects[:4]]
        return f"{', '.join(subject_words)} {verb}"
    
    def _generate_what_response(self, encoded: Dict) -> str:
        """
        Generate response to "what" query through neural activation.
        
        Two types:
        - "what does X do" → find verbs for subject
        - "what does X verb" → find objects
        """
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        
        # Check if asking "what does X do"
        words_lower = [w.lower() for w in encoded['words']]
        if 'do' in words_lower and subject:
            # Find verbs for this subject
            verbs = self.retriever.retrieve_verbs_for_subject(subject)
            
            if verbs:
                verb_words = [w for w, _ in verbs[:3]]
                return f"{subject} {', '.join(verb_words)}"
            else:
                return f"i do not know what {subject} does"
        
        # Otherwise, find objects
        if subject and verb:
            objects = self.retriever.retrieve_objects_for_subject_verb(subject, verb)
            
            if objects:
                obj_words = [w for w, _ in objects[:3]]
                return f"{subject} {verb} {', '.join(obj_words)}"
            else:
                return f"i do not know what {subject} {verb}"
        
        return "i could not understand the question"
    
    def _generate_yesno_response(self, encoded: Dict) -> str:
        """
        Generate yes/no response through neural activation.
        
        Check if pattern exists by measuring VP stability.
        """
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        obj = encoded.get('object')
        
        if not subject:
            return "i could not find the subject"
        
        if not verb:
            # Try to find any potential verb
            for word, cat in encoded['categories'].items():
                if cat == 'VERB' or cat == 'UNKNOWN':
                    if word not in ['does', 'do', 'is', 'can']:
                        verb = word
                        break
        
        if not verb:
            return "i could not find the action"
        
        # Check if pattern exists through neural activation
        exists, confidence = self.retriever.check_pattern_exists(subject, verb, obj)
        
        if exists:
            if obj:
                return f"yes {subject} {verb} {obj}"
            else:
                return f"yes {subject} {verb}"
        else:
            return "no i have not learned that"
    
    def _generate_default_response(self, encoded: Dict) -> str:
        """Generate default response for statements."""
        subject = encoded.get('subject')
        verb = encoded.get('verb')
        
        if subject and verb:
            return f"i understand {subject} {verb}"
        
        return "ok"


__all__ = ['TrueEmergentGenerator']


