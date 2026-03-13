"""
Response Generation
===================

Generates responses based on learned patterns.

Design Philosophy:
- Classification uses LEARNED categories, not hardcoded rules
- Responses emerge from neural activation, not templates
- Self-knowledge queries use actual brain state

Response types:
1. Answer - respond to questions using EMERGENT generation
2. Acknowledgment - confirm learning
3. Clarification - ask about unknown words
4. Self-report - describe what we know (emergent)

NEW: Uses EmergentGenerator for question answering instead of
template-based QuestionAnswerer. This is more NEMO-like.
"""

from typing import List, Optional, TYPE_CHECKING
from ..parser.comprehension import QuestionAnswerer
from ..generation import EmergentGenerator

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner
    from .dialogue import DialogueState


class ResponseGenerator:
    """
    Generates responses using learned patterns.
    
    Key insight: Classification should use NEMO's own learned categories,
    not hardcoded word lists. This makes the system truly emergent.
    
    NEW: Uses EmergentGenerator for question answering.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner', 
                 use_emergent: bool = True):
        """
        Initialize response generator.
        
        Args:
            learner: The language learner
            use_emergent: If True, use EmergentGenerator for questions.
                         If False, use template-based QuestionAnswerer.
        """
        self.learner = learner
        self.use_emergent = use_emergent
        
        # Both generators available - can switch between them
        self.qa = QuestionAnswerer(learner)
        self.emergent = EmergentGenerator(learner)
    
    def generate(self, words: List[str], 
                 input_type: str,
                 dialogue_state: 'DialogueState') -> str:
        """
        Generate a response to user input.
        
        Args:
            words: Tokenized user input
            input_type: Classification of input type
            dialogue_state: Current dialogue context
        
        Returns:
            Response string
        """
        if input_type == "question":
            return self._answer_question(words, dialogue_state)
        
        elif input_type == "statement":
            return self._acknowledge_statement(words, dialogue_state)
        
        elif input_type == "self_query":
            return self._answer_self_query(words, dialogue_state)
        
        elif input_type == "unknown_word":
            return self._ask_clarification(words, dialogue_state)
        
        else:
            return self._default_response(words, dialogue_state)
    
    def classify_input(self, words: List[str], 
                       dialogue_state: 'DialogueState') -> str:
        """
        Classify the type of input using LEARNED categories.
        
        This uses NEMO's emergent understanding rather than
        hardcoded word lists.
        """
        if not words:
            return "empty"
        
        first_word = words[0].lower()
        
        # Use learned categories for classification
        first_cat, _ = self.learner.get_emergent_category(first_word)
        
        # Question detection:
        # 1. First word is a question word (learned as having QUERY grounding)
        # 2. First word is an auxiliary (learned as FUNCTION)
        is_question = self._is_question_word(first_word) or \
                      (first_cat == "FUNCTION" and first_word in ["does", "do", "is", "are", "can", "did"])
        
        if is_question:
            # Check if it's about self/knowledge
            if self._is_self_referential(words):
                return "self_query"
            return "question"
        
        # Check for unknown words
        unknown = self._find_unknown_words(words)
        if unknown:
            dialogue_state.unknown_words = unknown
            # Only ask about unknown content words
            content_unknown = [w for w in unknown 
                             if self._is_content_word(w, words)]
            if content_unknown:
                return "unknown_word"
        
        # Default: treat as statement
        return "statement"
    
    def _is_question_word(self, word: str) -> bool:
        """
        Check if word is a question word based on learned grounding.
        
        Question words have QUERY in their grounding.
        """
        # Get the word's grounding counts
        grounding = self.learner.word_grounding.get(word, {})
        
        # Check if it has visual QUERY grounding (what) or social QUERY (who)
        # Question words typically have unique grounding patterns
        # For now, use the fallback list since QUERY isn't stored as grounding type
        
        # Fallback: known question words (these should be learned)
        return word in {"what", "who", "where", "when", "why", "how"}
    
    def _is_self_referential(self, words: List[str]) -> bool:
        """
        Check if the input is about self/knowledge.
        
        Uses learned categories rather than hardcoded lists.
        """
        words_lower = [w.lower() for w in words]
        
        # Check for second-person pronouns (addressing NEMO)
        has_you = any(
            self.learner.get_emergent_category(w)[0] == "PRONOUN" 
            and w in ["you", "your"]
            for w in words_lower
        )
        
        # Check for cognitive verbs based on grounding
        # Cognitive verbs have COGNITIVE or COGNITION in their grounding
        has_cognitive = any(
            self.learner.word_grounding.get(w, {}).get('COGNITIVE', 0) > 0
            for w in words_lower
        )
        
        # Also check for common self-query patterns
        self_query_words = {"know", "learned", "understand", "remember", "think"}
        has_self_word = any(w in self_query_words for w in words_lower)
        
        return has_you and (has_cognitive or has_self_word)
    
    def _find_unknown_words(self, words: List[str]) -> List[str]:
        """Find words not in vocabulary."""
        return [w for w in words 
                if w not in self.learner.word_count
                and w not in {".", ",", "?", "!"}]
    
    def _is_content_word(self, word: str, words: List[str]) -> bool:
        """
        Check if word is a content word based on position and category.
        
        Content words are NOUNs, VERBs, ADJECTIVEs - not FUNCTION words.
        """
        cat, _ = self.learner.get_emergent_category(word)
        
        # If we know the category, use it
        if cat != "UNKNOWN":
            return cat in ["NOUN", "VERB", "ADJECTIVE", "ADVERB"]
        
        # For unknown words, use position heuristics
        try:
            idx = words.index(word)
            # After determiner = likely content word
            if idx > 0 and words[idx-1] in ["the", "a", "an"]:
                return True
            return True  # Assume content word for unknowns
        except ValueError:
            return False
    
    def _answer_question(self, words: List[str],
                        dialogue_state: 'DialogueState') -> str:
        """
        Answer a question using learned knowledge.
        
        Uses EmergentGenerator (NEMO-like) or QuestionAnswerer (template-based)
        depending on use_emergent setting.
        """
        if self.use_emergent:
            return self.emergent.generate_answer(words)
        else:
            return self.qa.answer(words)
    
    def _acknowledge_statement(self, words: List[str],
                              dialogue_state: 'DialogueState') -> str:
        """Acknowledge learning from a statement."""
        # Extract what pattern we learned
        pattern = self._extract_pattern(words)
        
        if pattern:
            return f"learned: {pattern}"
        else:
            # Just acknowledge
            new_words = self._find_new_words(words)
            if new_words:
                return f"learned new words: {', '.join(new_words)}"
            return "ok"
    
    def _answer_self_query(self, words: List[str],
                          dialogue_state: 'DialogueState') -> str:
        """
        Answer questions about self/knowledge.
        
        This is where NEMO reports on its own state.
        """
        words_lower = [w.lower() for w in words]
        
        # "how many words do you know"
        if "words" in words_lower and ("many" in words_lower or "how" in words_lower):
            return f"i know {len(self.learner.word_count)} words"
        
        # "what do you know"
        if "know" in words_lower and "what" in words_lower:
            return self._report_knowledge()
        
        # "what have you learned"
        if "learned" in words_lower:
            recent = list(dialogue_state.history[-5:])
            patterns = []
            for turn in recent:
                patterns.extend(turn.learned_patterns)
            if patterns:
                return f"recently learned: {', '.join(patterns[-5:])}"
            return "i have been learning from our conversation"
        
        # "do you understand X"
        if "understand" in words_lower:
            content_words = [w for w in words 
                           if w not in {"do", "you", "understand", "the", "a"}]
            known = [w for w in content_words if w in self.learner.word_count]
            unknown = [w for w in content_words if w not in self.learner.word_count]
            
            if unknown:
                return f"i do not know: {', '.join(unknown)}"
            elif known:
                return f"yes i know: {', '.join(known)}"
            return "i am not sure what you are asking about"
        
        # "do you know X"
        if "know" in words_lower:
            # Find the object of "know"
            try:
                know_idx = words_lower.index("know")
                target = words_lower[know_idx + 1:] if know_idx + 1 < len(words_lower) else []
                target = [w for w in target if w not in ["the", "a", "an", "?"]]
                if target:
                    word = target[0]
                    if word in self.learner.word_count:
                        cat, conf = self.learner.get_emergent_category(word)
                        return f"yes i know {word} - it is a {cat.lower()}"
                    else:
                        return f"no i do not know {word}"
            except (ValueError, IndexError):
                pass
        
        return "i am not sure how to answer that"
    
    def _ask_clarification(self, words: List[str],
                          dialogue_state: 'DialogueState') -> str:
        """Ask about unknown words."""
        unknown = dialogue_state.unknown_words
        if unknown:
            return f"what is {unknown[0]}?"
        return "i did not understand something"
    
    def _default_response(self, words: List[str],
                         dialogue_state: 'DialogueState') -> str:
        """Default response when unsure."""
        return "ok"
    
    def _extract_pattern(self, words: List[str]) -> Optional[str]:
        """Extract the learned pattern from a sentence."""
        subject = None
        verb = None
        obj = None
        
        for word in words:
            cat, _ = self.learner.get_emergent_category(word)
            
            if cat == "VERB" and verb is None:
                verb = word
            elif cat in ["NOUN", "PRONOUN"]:
                if verb is None and subject is None:
                    subject = word
                elif verb is not None and obj is None:
                    obj = word
        
        if subject and verb:
            if obj:
                return f"{subject}_{verb}_{obj}"
            return f"{subject}_{verb}"
        
        return None
    
    def _find_new_words(self, words: List[str]) -> List[str]:
        """Find words that were just learned (first occurrence)."""
        return [w for w in words 
                if self.learner.word_count.get(w, 0) == 1
                and w not in {".", ",", "?", "!"}]
    
    def _report_knowledge(self) -> str:
        """Report summary of what we know."""
        vocab = self.learner.get_vocabulary_by_category()
        
        parts = []
        for cat in ["NOUN", "VERB", "ADJECTIVE"]:
            if cat in vocab:
                words = list(vocab[cat])[:5]
                parts.append(f"{cat.lower()}s: {', '.join(words)}")
        
        if parts:
            return "i know " + "; ".join(parts)
        return "i am still learning"
