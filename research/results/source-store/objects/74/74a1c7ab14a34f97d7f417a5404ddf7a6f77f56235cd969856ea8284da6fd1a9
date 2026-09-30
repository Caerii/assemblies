"""
Declarative Sentence Generators
===============================

Generators for declarative (statement) sentences:
- Intransitive: "the dog runs"
- Transitive: "the dog chases the cat"
- Copular: "the dog is big"
- Adjective-noun: "the big dog"
"""

from typing import List

from ...params import GroundingContext, GroundedSentence
from .base import BaseGenerator


class IntransitiveGenerator(BaseGenerator):
    """Generate intransitive sentences: 'the dog runs'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        for _ in range(n):
            subject = self.random_choice(self.animate_nouns)
            verb = self.random_choice(self.intransitive_verbs)
            
            if not subject or not verb:
                continue
            
            verb_form = verb.get_form('3sg')
            
            sentences.append(GroundedSentence(
                words=['the', subject.lemma, verb_form],
                contexts=[
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                ],
                roles=[None, 'agent', 'action'],
                mood='declarative'
            ))
        
        return sentences


class TransitiveGenerator(BaseGenerator):
    """Generate transitive sentences: 'the dog chases the cat'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        for _ in range(n):
            subject = self.random_choice(self.animate_nouns)
            verb = self.random_choice(self.transitive_verbs)
            obj = self.random_choice(self.nouns)
            
            if not subject or not verb or not obj:
                continue
            
            # Avoid reflexive
            if obj.lemma == subject.lemma:
                continue
            
            verb_form = verb.get_form('3sg')
            
            sentences.append(GroundedSentence(
                words=['the', subject.lemma, verb_form, 'the', obj.lemma],
                contexts=[
                    GroundingContext(),
                    subject.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=[None, 'agent', 'action', None, 'patient'],
                mood='declarative'
            ))
        
        return sentences


class CopularGenerator(BaseGenerator):
    """Generate copular sentences: 'the dog is big'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        for _ in range(n):
            noun = self.random_choice(self.nouns)
            adj = self.random_choice(self.adjectives)
            
            if not noun or not adj:
                continue
            
            sentences.append(GroundedSentence(
                words=['the', noun.lemma, 'is', adj.lemma],
                contexts=[
                    GroundingContext(),
                    noun.grounding,
                    GroundingContext(),  # 'is' - copula
                    adj.grounding,
                ],
                roles=[None, 'agent', 'action', None],
                mood='declarative'
            ))
        
        return sentences


class AdjectiveNounGenerator(BaseGenerator):
    """Generate adjective-noun phrases: 'the big dog'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        for _ in range(n):
            noun = self.random_choice(self.nouns)
            adj = self.random_choice(self.adjectives)
            
            if not noun or not adj:
                continue
            
            sentences.append(GroundedSentence(
                words=['the', adj.lemma, noun.lemma],
                contexts=[
                    GroundingContext(),
                    adj.grounding,
                    noun.grounding,
                ],
                roles=[None, None, None],
                mood='declarative'
            ))
        
        return sentences


__all__ = [
    'IntransitiveGenerator',
    'TransitiveGenerator', 
    'CopularGenerator',
    'AdjectiveNounGenerator',
]


