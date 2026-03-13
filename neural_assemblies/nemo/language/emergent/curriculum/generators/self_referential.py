"""
Self-Referential Sentence Generators
====================================

Generators for sentences involving self-reference:
- First person: "I see the dog"
- Second person: "you see the dog"
- Cognitive verbs: "I know the dog"
- Meta-cognitive: "I think you know"
"""

from typing import List

from ...params import GroundingContext, GroundedSentence
from .base import BaseGenerator


class FirstPersonGenerator(BaseGenerator):
    """Generate first-person sentences: 'I see the dog'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        i_info = self.vocab.get('i')
        if not i_info:
            return sentences
        
        for _ in range(n):
            verb = self.random_choice(self.transitive_verbs)
            obj = self.random_choice(self.nouns)
            
            if not verb or not obj:
                continue
            
            sentences.append(GroundedSentence(
                words=['i', verb.lemma, 'the', obj.lemma],
                contexts=[
                    i_info.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=['agent', 'action', None, 'patient'],
                mood='declarative'
            ))
        
        return sentences


class SecondPersonGenerator(BaseGenerator):
    """Generate second-person sentences: 'you see the dog'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        you_info = self.vocab.get('you')
        if not you_info:
            return sentences
        
        for _ in range(n):
            verb = self.random_choice(self.transitive_verbs)
            obj = self.random_choice(self.nouns)
            
            if not verb or not obj:
                continue
            
            sentences.append(GroundedSentence(
                words=['you', verb.lemma, 'the', obj.lemma],
                contexts=[
                    you_info.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=['agent', 'action', None, 'patient'],
                mood='declarative'
            ))
        
        return sentences


class CognitiveVerbGenerator(BaseGenerator):
    """Generate cognitive verb sentences: 'I know the dog'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        i_info = self.vocab.get('i')
        you_info = self.vocab.get('you')
        
        if not i_info and not you_info:
            return sentences
        
        pronouns = [p for p in [i_info, you_info] if p]
        
        for _ in range(n):
            pronoun = self.random_choice(pronouns)
            verb = self.random_choice(self.cognitive_verbs)
            obj = self.random_choice(self.nouns)
            
            if not pronoun or not verb or not obj:
                continue
            
            sentences.append(GroundedSentence(
                words=[pronoun.lemma, verb.lemma, 'the', obj.lemma],
                contexts=[
                    pronoun.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=['agent', 'action', None, 'patient'],
                mood='declarative'
            ))
        
        return sentences


class MetaCognitiveGenerator(BaseGenerator):
    """Generate meta-cognitive sentences: 'I think you know'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        i_info = self.vocab.get('i')
        you_info = self.vocab.get('you')
        
        if not i_info or not you_info:
            return sentences
        
        for _ in range(n):
            verb1 = self.random_choice(self.cognitive_verbs)
            verb2 = self.random_choice(self.cognitive_verbs)
            obj = self.random_choice(self.nouns)
            
            if not verb1 or not verb2 or not obj:
                continue
            
            # "I think you know the dog"
            sentences.append(GroundedSentence(
                words=['i', verb1.lemma, 'you', verb2.lemma, 'the', obj.lemma],
                contexts=[
                    i_info.grounding,
                    verb1.grounding,
                    you_info.grounding,
                    verb2.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=['agent', 'action', 'agent', 'action', None, 'patient'],
                mood='declarative'
            ))
        
        return sentences


class SelfQueryGenerator(BaseGenerator):
    """Generate self-query sentences: 'do you know the dog'"""
    
    def generate(self, n: int = 10) -> List[GroundedSentence]:
        sentences = []
        
        you_info = self.vocab.get('you')
        if not you_info:
            return sentences
        
        for _ in range(n):
            verb = self.random_choice(self.cognitive_verbs)
            obj = self.random_choice(self.nouns)
            
            if not verb or not obj:
                continue
            
            # "do you know the dog"
            sentences.append(GroundedSentence(
                words=['do', 'you', verb.lemma, 'the', obj.lemma],
                contexts=[
                    GroundingContext(),  # 'do' - auxiliary
                    you_info.grounding,
                    verb.grounding,
                    GroundingContext(),
                    obj.grounding,
                ],
                roles=[None, 'agent', 'action', None, 'patient'],
                mood='interrogative'
            ))
        
        return sentences


__all__ = [
    'FirstPersonGenerator',
    'SecondPersonGenerator',
    'CognitiveVerbGenerator',
    'MetaCognitiveGenerator',
    'SelfQueryGenerator',
]


