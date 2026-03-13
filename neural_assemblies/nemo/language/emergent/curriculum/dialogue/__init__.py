"""
Dialogue Patterns
=================

Training data for learning dialogue patterns.

Key insight: Dialogue patterns should be LEARNED, not hardcoded.
By training on question-answer pairs, NEMO learns:
- What types of questions exist
- How to map questions to answers
- Response patterns

Structure:
- qa_patterns.py: Question-answer pair generators
- response_patterns.py: Response type patterns
"""

from typing import List
from dataclasses import dataclass

from ...params import GroundedSentence, GroundingContext
from ..generators import SentenceGenerator
from ..vocabulary import get_vocabulary


@dataclass
class DialogueTurn:
    """A single turn in a dialogue."""
    speaker: str  # 'user' or 'system'
    sentence: GroundedSentence


@dataclass
class DialoguePair:
    """A question-answer pair for training."""
    question: GroundedSentence
    answer: GroundedSentence
    pattern_type: str  # 'who_query', 'what_query', 'yes_no', etc.


class DialoguePatternGenerator:
    """
    Generates dialogue patterns for training.
    
    These patterns teach NEMO:
    1. Question → Answer mappings
    2. Response generation patterns
    3. Self-referential dialogue
    """
    
    def __init__(self, seed: int = 42):
        self.gen = SentenceGenerator(seed=seed)
        self.vocab = get_vocabulary()
    
    def generate_who_qa_pairs(self, n: int = 10) -> List[DialoguePair]:
        """
        Generate 'who' question-answer pairs.
        
        Pattern: "who runs" → "the dog runs"
        """
        pairs = []
        
        for _ in range(n):
            subject = self.gen._intransitive.random_choice(self.gen.animate_nouns)
            verb = self.gen._intransitive.random_choice(self.gen._intransitive.intransitive_verbs)
            
            if not subject or not verb:
                continue
            
            verb_form = verb.get_form('3sg')
            
            # Question: "who runs"
            question = GroundedSentence(
                words=['who', verb_form],
                contexts=[
                    GroundingContext(social=['QUERY', 'PERSON']),
                    verb.grounding,
                ],
                roles=['agent', 'action'],
                mood='interrogative'
            )
            
            # Answer: "the dog runs"
            answer = GroundedSentence(
                words=['the', subject.lemma, verb_form],
                contexts=[
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=[None, 'agent', 'action'],
                mood='declarative'
            )
            
            pairs.append(DialoguePair(
                question=question,
                answer=answer,
                pattern_type='who_query'
            ))
        
        return pairs
    
    def generate_what_qa_pairs(self, n: int = 10) -> List[DialoguePair]:
        """
        Generate 'what does X do' question-answer pairs.
        
        Pattern: "what does the dog do" → "the dog runs"
        """
        pairs = []
        
        for _ in range(n):
            subject = self.gen._intransitive.random_choice(self.gen.animate_nouns)
            verb = self.gen._intransitive.random_choice(self.gen._intransitive.intransitive_verbs)
            
            if not subject or not verb:
                continue
            
            verb_form = verb.get_form('3sg')
            
            # Question: "what does the dog do"
            question = GroundedSentence(
                words=['what', 'does', 'the', subject.lemma, 'do'],
                contexts=[
                    GroundingContext(visual=['QUERY', 'ACTION']),
                    GroundingContext(),
                    GroundingContext(),
                    subject.grounding,
                    GroundingContext(),
                ],
                roles=['patient', None, None, 'agent', 'action'],
                mood='interrogative'
            )
            
            # Answer: "the dog runs"
            answer = GroundedSentence(
                words=['the', subject.lemma, verb_form],
                contexts=[
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=[None, 'agent', 'action'],
                mood='declarative'
            )
            
            pairs.append(DialoguePair(
                question=question,
                answer=answer,
                pattern_type='what_do_query'
            ))
        
        return pairs
    
    def generate_yesno_qa_pairs(self, n: int = 10) -> List[DialoguePair]:
        """
        Generate yes/no question-answer pairs.
        
        Pattern: "does the dog run" → "yes the dog runs"
        """
        pairs = []
        
        for _ in range(n):
            subject = self.gen._intransitive.random_choice(self.gen.animate_nouns)
            verb = self.gen._intransitive.random_choice(self.gen._intransitive.intransitive_verbs)
            
            if not subject or not verb:
                continue
            
            verb_form = verb.get_form('3sg')
            
            # Question: "does the dog run"
            question = GroundedSentence(
                words=['does', 'the', subject.lemma, verb.lemma],
                contexts=[
                    GroundingContext(),
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=[None, None, 'agent', 'action'],
                mood='interrogative'
            )
            
            # Answer: "yes the dog runs"
            answer = GroundedSentence(
                words=['yes', 'the', subject.lemma, verb_form],
                contexts=[
                    GroundingContext(emotional=['AFFIRM']),
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=[None, None, 'agent', 'action'],
                mood='declarative'
            )
            
            pairs.append(DialoguePair(
                question=question,
                answer=answer,
                pattern_type='yesno_affirm'
            ))
        
        return pairs
    
    def generate_self_knowledge_qa(self, n: int = 10) -> List[DialoguePair]:
        """
        Generate self-knowledge question-answer pairs.
        
        Pattern: "do you know the dog" → "yes i know the dog"
        """
        pairs = []
        
        you_info = self.vocab.get('you')
        i_info = self.vocab.get('i')
        
        if not you_info or not i_info:
            return pairs
        
        cognitive_verbs = self.gen._cognitive.cognitive_verbs
        
        for _ in range(n):
            verb = self.gen._intransitive.random_choice(cognitive_verbs)
            obj = self.gen._intransitive.random_choice(self.gen.nouns)
            
            if not verb or not obj:
                continue
            
            # Question: "do you know the dog"
            question = GroundedSentence(
                words=['do', 'you', verb.lemma, 'the', obj.lemma],
                contexts=[
                    GroundingContext(),
                    you_info.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=[None, 'agent', 'action', None, 'patient'],
                mood='interrogative'
            )
            
            # Answer: "yes i know the dog"
            answer = GroundedSentence(
                words=['yes', 'i', verb.lemma, 'the', obj.lemma],
                contexts=[
                    GroundingContext(emotional=['AFFIRM']),
                    i_info.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=[None, 'agent', 'action', None, 'patient'],
                mood='declarative'
            )
            
            pairs.append(DialoguePair(
                question=question,
                answer=answer,
                pattern_type='self_knowledge'
            ))
        
        return pairs
    
    def generate_all_pairs(self, n_each: int = 10) -> List[DialoguePair]:
        """Generate all types of dialogue pairs."""
        pairs = []
        pairs.extend(self.generate_who_qa_pairs(n_each))
        pairs.extend(self.generate_what_qa_pairs(n_each))
        pairs.extend(self.generate_yesno_qa_pairs(n_each))
        pairs.extend(self.generate_self_knowledge_qa(n_each))
        return pairs
    
    def pairs_to_sentences(self, pairs: List[DialoguePair]) -> List[GroundedSentence]:
        """Convert dialogue pairs to training sentences."""
        sentences = []
        for pair in pairs:
            sentences.append(pair.question)
            sentences.append(pair.answer)
        return sentences


def get_dialogue_curriculum(seed: int = 42, n_each: int = 15) -> List[GroundedSentence]:
    """
    Get dialogue training curriculum.
    
    Returns sentences from question-answer pairs.
    """
    gen = DialoguePatternGenerator(seed=seed)
    pairs = gen.generate_all_pairs(n_each=n_each)
    return gen.pairs_to_sentences(pairs)


__all__ = [
    'DialogueTurn',
    'DialoguePair',
    'DialoguePatternGenerator',
    'get_dialogue_curriculum',
]
