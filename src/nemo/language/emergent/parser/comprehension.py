"""
NEMO Emergent Question Answering
================================

Version: 1.0.0
Date: 2025-11-30

Answer questions using learned knowledge stored in VP assemblies.

Supports:
- "what does X verb?" → find typical objects
- "who verbs X?" → find typical subjects/agents
- "does X verb Y?" → yes/no verification
"""

from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner

from ..areas import Area
from .core import SentenceParser

__all__ = ['QuestionAnswerer']


class QuestionAnswerer:
    """
    Answers questions using learned knowledge.
    
    NEMO approach:
    - Parse the question to identify known parts
    - Query learned VP assemblies to find answers
    - Use assembly overlap for uncertain matches
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
        self.parser = SentenceParser(learner)
    
    def answer(self, question_words: List[str]) -> str:
        """
        Answer a question.
        
        Args:
            question_words: List of words in the question
            
        Returns:
            Answer string
        """
        if not question_words:
            return "Empty question"
        
        first_word = question_words[0].lower()
        
        if first_word == 'what':
            return self._answer_what(question_words)
        elif first_word == 'who':
            return self._answer_who(question_words)
        elif first_word == 'does':
            return self._answer_does(question_words)
        elif first_word == 'is':
            return self._answer_is(question_words)
        elif first_word == 'can':
            return self._answer_can(question_words)
        else:
            return "I don't understand that question type"
    
    def _answer_what(self, words: List[str]) -> str:
        """Answer 'what' questions (find objects or actions)."""
        words_lower = [w.lower() for w in words]
        
        # Extract subject from question (look for nouns after "does"/"do")
        subject = self._extract_subject_from_question(words_lower)
        
        # Extract verb from question (look for known verbs)
        verb = self._extract_verb_from_question(words_lower)
        
        # Special case: "what does X do" - asking for actions, not objects
        if words_lower[-1] == "do" or (not verb and subject):
            # Find all verbs this subject performs
            verbs = self._find_learned_verbs_for_subject(subject) if subject else set()
            if verbs:
                return f"{subject} {', '.join(verbs)}"
            elif subject:
                return f"I don't know what {subject} does"
            return "I couldn't understand the question"
        
        if not verb:
            # No verb found - try parsing
            parse = self.parser.parse(words)
            verb = parse.verb
            subject = subject or parse.subject
        
        if not verb:
            if subject:
                verbs = self._find_learned_verbs_for_subject(subject)
                if verbs:
                    return f"{subject} {', '.join(verbs)}"
                else:
                    return f"I don't know what {subject} does"
            return "I couldn't identify the verb in your question"
        
        if subject:
            # "What does X verb?" - find objects for this subject-verb
            # Try both forms of verb (e.g., "chase" and "chases")
            objects = self._find_learned_objects(subject, verb)
            if not objects and not verb.endswith('s'):
                objects = self._find_learned_objects(subject, verb + 's')
            if not objects and verb.endswith('s'):
                objects = self._find_learned_objects(subject, verb[:-1])
            
            if objects:
                return f"{subject} {verb} {', '.join(objects)}"
            else:
                return f"I don't know what {subject} {verb}"
        else:
            # "What verbs?" - find all objects for this verb
            objects = self._find_all_objects_for_verb(verb)
            
            if objects:
                return f"Things that get {verb}: {', '.join(objects)}"
            else:
                return f"I don't know what gets {verb}"
    
    def _extract_subject_from_question(self, words: List[str]) -> Optional[str]:
        """Extract subject from a question like 'what does the dog do'."""
        # Look for nouns in the question
        for i, word in enumerate(words):
            cat, _ = self.learner.get_emergent_category(word)
            if cat in ['NOUN', 'PRONOUN']:
                return word
        return None
    
    def _extract_verb_from_question(self, words: List[str]) -> Optional[str]:
        """Extract content verb from a question (not 'do/does')."""
        skip_words = {'what', 'who', 'where', 'when', 'why', 'how', 
                      'do', 'does', 'did', 'is', 'are', 'the', 'a', 'an'}
        
        for word in words:
            if word in skip_words:
                continue
            cat, _ = self.learner.get_emergent_category(word)
            if cat == 'VERB':
                return word
            # Also check if adding 's' makes it a known verb
            if cat == 'UNKNOWN' and not word.endswith('s'):
                cat_s, _ = self.learner.get_emergent_category(word + 's')
                if cat_s == 'VERB':
                    return word + 's'
        return None
    
    def _answer_who(self, words: List[str]) -> str:
        """Answer 'who' questions (find subjects/agents)."""
        parse = self.parser.parse(words)
        
        verb = parse.verb
        obj = parse.object
        
        if not verb:
            return "I couldn't identify the verb in your question"
        
        # "Who verbs X?" - find subjects
        subjects = self._find_learned_subjects(verb, obj)
        
        if subjects:
            answer = f"{', '.join(subjects)} {verb}"
            if obj:
                answer += f" {obj}"
            return answer
        else:
            answer = f"I don't know who {verb}"
            if obj:
                answer += f" {obj}"
            return answer
    
    def _answer_does(self, words: List[str]) -> str:
        """Answer 'does' questions (yes/no verification)."""
        parse = self.parser.parse(words)
        
        subject = parse.subject
        verb = parse.verb
        obj = parse.object
        
        if not subject or not verb:
            return "I couldn't parse the question"
        
        # Check if we learned this pattern
        if obj:
            vp_key = f"{subject}_{verb}_{obj}"
            if self.brain.has_learned_assembly(Area.VP, vp_key):
                return f"Yes, {subject} {verb} {obj}"
            else:
                # Check if subject-verb exists at all
                sv_key = f"{subject}_{verb}"
                if self.brain.has_learned_assembly(Area.VP, sv_key):
                    return f"I know {subject} {verb}, but not specifically {obj}"
                else:
                    return f"I haven't learned that {subject} {verb} {obj}"
        else:
            vp_key = f"{subject}_{verb}"
            if self.brain.has_learned_assembly(Area.VP, vp_key):
                return f"Yes, {subject} {verb}"
            else:
                return f"I haven't learned that {subject} {verb}"
    
    def _answer_is(self, words: List[str]) -> str:
        """Answer 'is' questions (property/identity)."""
        # Simple handling for now
        parse = self.parser.parse(words)
        
        if parse.subject:
            cat, _ = self.learner.get_emergent_category(parse.subject)
            return f"{parse.subject} is a {cat}"
        
        return "I don't understand that question"
    
    def _answer_can(self, words: List[str]) -> str:
        """Answer 'can' questions (ability)."""
        parse = self.parser.parse(words)
        
        subject = parse.subject
        verb = parse.verb
        
        if not subject or not verb:
            return "I couldn't parse the question"
        
        # Check if we've seen this subject do this action
        vp_patterns = [key for key in self.brain.learned_assemblies[Area.VP].keys()
                       if key.startswith(f"{subject}_{verb}")]
        
        if vp_patterns:
            return f"Yes, {subject} can {verb}"
        else:
            # Check if ANY animate thing does this
            any_patterns = [key for key in self.brain.learned_assemblies[Area.VP].keys()
                           if f"_{verb}" in key]
            if any_patterns:
                return f"I haven't seen {subject} {verb}, but others do"
            else:
                return f"I don't know if {subject} can {verb}"
    
    def _find_learned_objects(self, subject: str, verb: str) -> Set[str]:
        """Find all objects learned with this subject-verb pair."""
        objects = set()
        
        prefix = f"{subject}_{verb}_"
        for key in self.brain.learned_assemblies[Area.VP].keys():
            if key.startswith(prefix):
                parts = key.split('_')
                if len(parts) == 3:
                    objects.add(parts[2])
        
        return objects
    
    def _find_all_objects_for_verb(self, verb: str) -> Set[str]:
        """Find all objects that appear with this verb."""
        objects = set()
        
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            if len(parts) == 3 and parts[1] == verb:
                objects.add(parts[2])
        
        return objects
    
    def _find_learned_subjects(self, verb: str, obj: Optional[str] = None) -> Set[str]:
        """Find all subjects learned with this verb (and optionally object)."""
        subjects = set()
        
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            
            if len(parts) >= 2 and parts[1] == verb:
                if obj is None:
                    subjects.add(parts[0])
                elif len(parts) == 3 and parts[2] == obj:
                    subjects.add(parts[0])
        
        return subjects
    
    def _find_learned_verbs_for_subject(self, subject: str) -> Set[str]:
        """Find all verbs learned for this subject (what does X do?)."""
        verbs = set()
        
        for key in self.brain.learned_assemblies[Area.VP].keys():
            parts = key.split('_')
            
            if len(parts) >= 2 and parts[0] == subject:
                verbs.add(parts[1])
        
        return verbs
    
    def get_knowledge_summary(self) -> dict:
        """Get a summary of what the system knows."""
        vp_keys = list(self.brain.learned_assemblies[Area.VP].keys())
        
        # Count patterns
        intransitive = [k for k in vp_keys if k.count('_') == 1]
        transitive = [k for k in vp_keys if k.count('_') == 2]
        
        # Extract unique subjects, verbs, objects
        subjects = set()
        verbs = set()
        objects = set()
        
        for key in vp_keys:
            parts = key.split('_')
            if len(parts) >= 2:
                subjects.add(parts[0])
                verbs.add(parts[1])
            if len(parts) == 3:
                objects.add(parts[2])
        
        return {
            'total_patterns': len(vp_keys),
            'intransitive_patterns': len(intransitive),
            'transitive_patterns': len(transitive),
            'unique_subjects': list(subjects),
            'unique_verbs': list(verbs),
            'unique_objects': list(objects),
        }

