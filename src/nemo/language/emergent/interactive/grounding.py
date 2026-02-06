"""
Grounding Inference
===================

Infers grounding contexts for words based on:
1. Known word categories (from prior learning)
2. Positional patterns (where the word appears)
3. Dialogue context (what was recently discussed)

This allows NEMO to learn new words from context,
without explicit grounding being provided.
"""

from typing import List, Optional, TYPE_CHECKING
from ..params import GroundingContext

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner


class GroundingInference:
    """
    Infers grounding for words based on context and learned patterns.
    
    This enables interactive learning where grounding isn't
    explicitly provided - it's inferred from experience.
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
    
    def infer_grounding(self, word: str, 
                        sentence_words: List[str],
                        position: int) -> GroundingContext:
        """
        Infer grounding context for a word.
        
        Strategy:
        1. If known word, use learned grounding
        2. If unknown, infer from position/context
        """
        # Check if we know this word
        if word in self.learner.word_count:
            return self._get_learned_grounding(word)
        
        # Unknown word - infer from context
        return self._infer_from_context(word, sentence_words, position)
    
    def _get_learned_grounding(self, word: str) -> GroundingContext:
        """Get grounding based on what we've learned about this word."""
        ctx = GroundingContext()
        
        # Get the word's emergent category
        category, scores = self.learner.get_emergent_category(word)
        
        # Map category back to likely grounding
        if category == "NOUN":
            ctx.visual = [word]  # Nouns are visually grounded
        elif category == "VERB":
            ctx.motor = [word]  # Verbs are motor grounded
        elif category == "ADJECTIVE":
            ctx.property = [word]  # Adjectives are property grounded
        elif category == "ADVERB":
            ctx.property = [word]  # Adverbs modify properties
        elif category == "PREPOSITION":
            ctx.spatial = [word]  # Prepositions are spatial
        elif category == "PRONOUN":
            ctx.social = [word]  # Pronouns are socially grounded
        # FUNCTION words get no grounding (that's what makes them function words)
        
        return ctx
    
    def _infer_from_context(self, word: str,
                            sentence_words: List[str],
                            position: int) -> GroundingContext:
        """
        Infer grounding for unknown word from context.
        
        Uses positional patterns learned from experience.
        """
        ctx = GroundingContext()
        
        # Get typical word order
        word_order = self.learner.get_word_order()
        
        # Analyze position in sentence
        relative_position = position / max(len(sentence_words), 1)
        
        # Look at surrounding words
        prev_word = sentence_words[position - 1] if position > 0 else None
        next_word = sentence_words[position + 1] if position < len(sentence_words) - 1 else None
        
        # Infer based on patterns
        if prev_word:
            prev_cat, _ = self.learner.get_emergent_category(prev_word)
            
            # After determiner → likely NOUN
            if prev_cat == "FUNCTION" and prev_word in ["the", "a", "an"]:
                ctx.visual = [word]  # Assume noun-like grounding
                
            # After NOUN → likely VERB
            elif prev_cat == "NOUN":
                ctx.motor = [word]  # Assume verb-like grounding
                
            # After VERB → could be NOUN (object) or ADVERB
            elif prev_cat == "VERB":
                if next_word and next_word in ["the", "a", "an"]:
                    ctx.property = [word]  # Adverb before determiner
                else:
                    ctx.visual = [word]  # Object noun
        
        # Default: no grounding (will be classified as FUNCTION)
        return ctx
    
    def infer_sentence_grounding(self, words: List[str]) -> List[GroundingContext]:
        """Infer grounding for all words in a sentence."""
        return [
            self.infer_grounding(word, words, i)
            for i, word in enumerate(words)
        ]
    
    def infer_roles(self, words: List[str]) -> List[Optional[str]]:
        """
        Infer thematic roles for words in a sentence.
        
        Uses learned patterns about word order and categories.
        """
        roles = [None] * len(words)
        
        found_verb = False
        found_agent = False
        
        for i, word in enumerate(words):
            cat, _ = self.learner.get_emergent_category(word)
            
            if cat == "VERB":
                roles[i] = "action"
                found_verb = True
                
            elif cat in ["NOUN", "PRONOUN"]:
                if not found_verb and not found_agent:
                    # Noun before verb → AGENT
                    roles[i] = "agent"
                    found_agent = True
                elif found_verb:
                    # Noun after verb → PATIENT
                    roles[i] = "patient"
        
        return roles

