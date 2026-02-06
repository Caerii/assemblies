"""
Emergent Response Generator
===========================

Generates responses through neural activation, not templates.

Key Insight: Generation is the reverse of comprehension.
- Comprehension: words → assemblies → meaning
- Generation: meaning → assemblies → words

The VP assembly is the unit of meaning. To generate:
1. Activate relevant VP assemblies
2. Decode them to word sequences
3. The response EMERGES from activation patterns
"""

from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner

from ..areas import Area
from .vp_decoder import VPDecoder
from .activation import ActivationSpreader


class EmergentGenerator:
    """
    Generates responses through emergent activation patterns.
    
    Two modes of operation:
    1. VP-key mode: Use VP keys directly (faster, more reliable)
    2. Assembly mode: Use activation spreading (more NEMO-like)
    
    The assembly mode is experimental but more neurobiologically plausible.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
        self.decoder = VPDecoder(learner)
        self.spreader = ActivationSpreader(self.brain)
    
    # =========================================================================
    # HIGH-LEVEL GENERATION API
    # =========================================================================
    
    def generate_answer(self, question_words: List[str]) -> str:
        """
        Generate an answer to a question through emergent activation.
        
        Args:
            question_words: Tokenized question
        
        Returns:
            Generated answer string
        """
        if not question_words:
            return ""
        
        first_word = question_words[0].lower()
        
        # Dispatch based on question type
        if first_word == 'who':
            return self._generate_who_answer(question_words)
        elif first_word == 'what':
            return self._generate_what_answer(question_words)
        elif first_word == 'does':
            return self._generate_does_answer(question_words)
        else:
            return self._generate_default_answer(question_words)
    
    # =========================================================================
    # QUESTION-SPECIFIC GENERATION
    # =========================================================================
    
    def _generate_who_answer(self, words: List[str]) -> str:
        """
        Generate answer to "who" question.
        
        Strategy:
        1. Find the verb in the question
        2. Find VP assemblies containing that verb
        3. Extract and return subjects
        """
        # Extract verb from question
        verb = self._extract_verb(words)
        
        if not verb:
            return "i could not find the verb"
        
        # Find VPs with this verb
        vp_keys = self.decoder.find_vp_by_verb(verb)
        
        if not vp_keys:
            # Try spreading activation to find related patterns
            activation = self.spreader.spread_from_verb(verb)
            
            if Area.NOUN_CORE in activation:
                # Decode nouns from activation
                nouns = self._decode_words_from_activation(
                    activation[Area.NOUN_CORE], Area.NOUN_CORE
                )
                if nouns:
                    return f"{', '.join(nouns)} {verb}"
            
            return f"i do not know who {verb}"
        
        # Extract subjects from VP keys
        subjects = set()
        for vp_key in vp_keys:
            parts = self.decoder.decode_vp_key(vp_key)
            if parts['subject']:
                subjects.add(parts['subject'])
        
        if subjects:
            return f"{', '.join(subjects)} {verb}"
        
        return f"i do not know who {verb}"
    
    def _generate_what_answer(self, words: List[str]) -> str:
        """
        Generate answer to "what" question.
        
        Two types:
        - "what does X do?" → find verbs for subject
        - "what does X verb?" → find objects for subject-verb
        """
        # Extract subject and verb
        subject = self._extract_subject(words)
        verb = self._extract_verb(words)
        
        # "what does X do?" pattern
        if words[-1].lower() == 'do' or (subject and not verb):
            if subject:
                vp_keys = self.decoder.find_vp_by_subject(subject)
                verbs = set()
                for vp_key in vp_keys:
                    parts = self.decoder.decode_vp_key(vp_key)
                    if parts['verb']:
                        verbs.add(parts['verb'])
                
                if verbs:
                    return f"{subject} {', '.join(verbs)}"
                
                # Try activation spreading
                activation = self.spreader.spread_from_noun(subject)
                if Area.VERB_CORE in activation:
                    verbs = self._decode_words_from_activation(
                        activation[Area.VERB_CORE], Area.VERB_CORE
                    )
                    if verbs:
                        return f"{subject} {', '.join(verbs)}"
                
                return f"i do not know what {subject} does"
            
            return "i could not find the subject"
        
        # "what does X verb?" pattern
        if subject and verb:
            vp_keys = self.decoder.find_vp_by_pattern(subject=subject, verb=verb)
            objects = set()
            for vp_key in vp_keys:
                parts = self.decoder.decode_vp_key(vp_key)
                if parts['object']:
                    objects.add(parts['object'])
            
            if objects:
                return f"{subject} {verb} {', '.join(objects)}"
            
            return f"i do not know what {subject} {verb}"
        
        return "i could not understand the question"
    
    def _generate_does_answer(self, words: List[str]) -> str:
        """
        Generate answer to "does" question (yes/no).
        
        Check if the pattern exists in learned VP assemblies.
        Also handles verb form variations (run/runs).
        """
        subject = self._extract_subject(words)
        verb = self._extract_verb(words)
        obj = self._extract_object(words)
        
        if not subject:
            return "i could not find the subject"
        
        if not verb:
            # Try to find any word that could be a verb
            skip_words = {'does', 'do', 'the', 'a', 'an', subject}
            for word in words:
                word_lower = word.lower()
                if word_lower not in skip_words:
                    verb = word_lower
                    break
        
        if not verb:
            return "i could not find the verb"
        
        # Try multiple verb forms
        verb_forms = [verb]
        if not verb.endswith('s'):
            verb_forms.append(verb + 's')
        if verb.endswith('s'):
            verb_forms.append(verb[:-1])
        
        # Build VP key to check
        for v in verb_forms:
            if obj:
                vp_key = f"{subject}_{v}_{obj}"
                if self.brain.has_learned_assembly(Area.VP, vp_key):
                    return f"yes {subject} {v} {obj}"
            else:
                vp_key = f"{subject}_{v}"
                if self.brain.has_learned_assembly(Area.VP, vp_key):
                    return f"yes {subject} {v}"
        
        # Check partial match (subject-verb without object)
        for v in verb_forms:
            sv_key = f"{subject}_{v}"
            if self.brain.has_learned_assembly(Area.VP, sv_key):
                if obj:
                    return f"i know {subject} {v} but not {obj}"
        
        return f"no i have not learned that {subject} {verb}"
    
    def _generate_default_answer(self, words: List[str]) -> str:
        """Generate a default answer when question type is unclear."""
        # Try to find any relevant VP
        for word in words:
            vp_keys = self.decoder.find_vp_by_verb(word)
            if vp_keys:
                # Return first match
                return ' '.join(self.decoder.decode_vp_key_to_sentence(vp_keys[0]))
        
        return "i do not understand"
    
    # =========================================================================
    # WORD EXTRACTION
    # =========================================================================
    
    def _extract_verb(self, words: List[str]) -> Optional[str]:
        """Extract the main verb from a question."""
        skip_words = {'what', 'who', 'where', 'when', 'why', 'how',
                      'do', 'does', 'did', 'is', 'are', 'the', 'a', 'an'}
        
        for word in words:
            word_lower = word.lower()
            if word_lower in skip_words:
                continue
            
            cat, _ = self.learner.get_emergent_category(word_lower)
            if cat == 'VERB':
                return word_lower
            
            # Check if adding 's' makes it a known verb
            if cat == 'UNKNOWN':
                cat_s, _ = self.learner.get_emergent_category(word_lower + 's')
                if cat_s == 'VERB':
                    return word_lower + 's'  # Return inflected form
            
            # Check if base form (without 's') is a verb
            if word_lower.endswith('s'):
                base = word_lower[:-1]
                cat_base, _ = self.learner.get_emergent_category(base)
                if cat_base == 'VERB':
                    return word_lower  # Return inflected form
            
            # Check if it's in VP keys as a verb
            for vp_key in self.brain.learned_assemblies[Area.VP].keys():
                parts = vp_key.split('_')
                if len(parts) >= 2 and parts[1] == word_lower:
                    return word_lower
        
        return None
    
    def _extract_subject(self, words: List[str]) -> Optional[str]:
        """Extract the subject from a question."""
        skip_words = {'what', 'who', 'where', 'when', 'why', 'how',
                      'do', 'does', 'did', 'is', 'are', 'the', 'a', 'an'}
        
        for word in words:
            word_lower = word.lower()
            if word_lower in skip_words:
                continue
            
            cat, _ = self.learner.get_emergent_category(word_lower)
            if cat in ['NOUN', 'PRONOUN']:
                return word_lower
        
        return None
    
    def _extract_object(self, words: List[str]) -> Optional[str]:
        """Extract the object from a question (noun after verb)."""
        found_verb = False
        skip_words = {'the', 'a', 'an'}
        
        for word in words:
            word_lower = word.lower()
            
            cat, _ = self.learner.get_emergent_category(word_lower)
            
            if cat == 'VERB':
                found_verb = True
                continue
            
            if found_verb and cat == 'NOUN' and word_lower not in skip_words:
                return word_lower
        
        return None
    
    # =========================================================================
    # ASSEMBLY-BASED DECODING
    # =========================================================================
    
    def _decode_words_from_activation(self, assembly: cp.ndarray,
                                       area: Area,
                                       max_words: int = 5) -> List[str]:
        """
        Decode words from an activated assembly.
        
        Finds words whose learned assemblies overlap with the activation.
        """
        words = []
        
        for word, word_assembly in self.brain.learned_assemblies[area].items():
            overlap = self.brain.get_assembly_overlap(assembly, word_assembly)
            if overlap > 0.1:
                words.append((word, overlap))
        
        # Sort by overlap and take top
        words.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in words[:max_words]]
    
    # =========================================================================
    # FULL EMERGENT GENERATION (experimental)
    # =========================================================================
    
    def generate_emergent(self, seeds: Dict[Area, cp.ndarray],
                          max_words: int = 10) -> str:
        """
        Generate a response through pure activation spreading.
        
        This is the most NEMO-like generation mode:
        1. Seed activation in relevant areas
        2. Let activation spread through learned weights
        3. Decode the settled pattern to words
        4. Order words by category transitions
        
        EXPERIMENTAL - may produce less coherent output than VP-based generation.
        """
        # Spread activation
        settled = self.spreader.spread(seeds, max_rounds=10)
        
        # Decode each area
        decoded = {}
        
        area_category = {
            Area.NOUN_CORE: 'NOUN',
            Area.VERB_CORE: 'VERB',
            Area.ADJ_CORE: 'ADJECTIVE',
            Area.PRON_CORE: 'PRONOUN',
        }
        
        for area, category in area_category.items():
            if area in settled:
                words = self._decode_words_from_activation(settled[area], area, max_words=3)
                if words:
                    decoded[category] = words[0]  # Take best match
        
        # Order by typical word order
        word_order = ['PRONOUN', 'NOUN', 'VERB', 'ADJECTIVE']
        
        output = []
        for category in word_order:
            if category in decoded:
                word = decoded[category]
                # Add determiner before nouns
                if category == 'NOUN':
                    output.append('the')
                output.append(word)
        
        return ' '.join(output) if output else "i do not know"


__all__ = ['EmergentGenerator']

