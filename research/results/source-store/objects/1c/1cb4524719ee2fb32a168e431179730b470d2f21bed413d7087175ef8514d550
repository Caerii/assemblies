"""
Interrogative Sentence Generators
=================================

Generators for question sentences:
- Who questions: "who runs"
- What questions: "what does the dog see"
- Yes/No questions: "does the dog run"
- Where questions: "where is the dog"
"""

from typing import List

from ...params import GroundingContext, GroundedSentence
from ..vocabulary import create_question_word_grounding
from .base import BaseGenerator


class WhoQuestionGenerator(BaseGenerator):
    """Generate 'who' questions: 'who runs'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        who_grounding = create_question_word_grounding('who')
        
        for _ in range(n):
            verb = self.random_choice(self.intransitive_verbs)
            
            if not verb:
                continue
            
            verb_form = verb.get_form('3sg')
            
            sentences.append(GroundedSentence(
                words=['who', verb_form],
                contexts=[
                    who_grounding,
                    verb.grounding,
                ],
                roles=['agent', 'action'],
                mood='interrogative'
            ))
        
        return sentences


class WhatQuestionGenerator(BaseGenerator):
    """Generate 'what' questions: 'what does the dog see'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        what_grounding = create_question_word_grounding('what')
        
        for _ in range(n):
            subject = self.random_choice(self.animate_nouns)
            verb = self.random_choice(self.transitive_verbs)
            
            if not subject or not verb:
                continue
            
            sentences.append(GroundedSentence(
                words=['what', 'does', 'the', subject.lemma, verb.lemma],
                contexts=[
                    what_grounding,
                    GroundingContext(),  # 'does' - auxiliary
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=['patient', None, None, 'agent', 'action'],
                mood='interrogative'
            ))
        
        return sentences


class YesNoQuestionGenerator(BaseGenerator):
    """Generate yes/no questions: 'does the dog run'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        for _ in range(n):
            subject = self.random_choice(self.animate_nouns)
            verb = self.random_choice(self.intransitive_verbs)
            
            if not subject or not verb:
                continue
            
            sentences.append(GroundedSentence(
                words=['does', 'the', subject.lemma, verb.lemma],
                contexts=[
                    GroundingContext(),  # 'does' - auxiliary
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=[None, None, 'agent', 'action'],
                mood='interrogative'
            ))
        
        return sentences


class WhereQuestionGenerator(BaseGenerator):
    """Generate 'where' questions: 'where is the dog'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        where_grounding = create_question_word_grounding('where')
        
        for _ in range(n):
            noun = self.random_choice(self.nouns)
            
            if not noun:
                continue
            
            sentences.append(GroundedSentence(
                words=['where', 'is', 'the', noun.lemma],
                contexts=[
                    where_grounding,
                    GroundingContext(),  # 'is' - copula
                    GroundingContext(),
                    noun.grounding,
                ],
                roles=[None, 'action', None, 'agent'],
                mood='interrogative'
            ))
        
        return sentences


__all__ = [
    'WhoQuestionGenerator',
    'WhatQuestionGenerator',
    'YesNoQuestionGenerator',
    'WhereQuestionGenerator',
]


