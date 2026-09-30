"""
Sentence Generators
===================

Modular sentence pattern generators for NEMO training.

Structure:
- base.py: BaseGenerator with common functionality
- declarative.py: Statement generators (SVO, copular, etc.)
- interrogative.py: Question generators (who, what, yes/no)
- self_referential.py: Self-reference generators (I, you, cognitive)

Usage:
    from curriculum.generators import SentenceGenerator
    
    gen = SentenceGenerator(seed=42)
    sentences = gen.generate_mixed(100)
"""

from typing import List, Optional
import random

from ...params import GroundedSentence
from .base import BaseGenerator
from .declarative import (
    IntransitiveGenerator,
    TransitiveGenerator,
    CopularGenerator,
    AdjectiveNounGenerator,
)
from .interrogative import (
    WhoQuestionGenerator,
    WhatQuestionGenerator,
    YesNoQuestionGenerator,
    WhereQuestionGenerator,
)
from .self_referential import (
    FirstPersonGenerator,
    SecondPersonGenerator,
    CognitiveVerbGenerator,
    MetaCognitiveGenerator,
    SelfQueryGenerator,
)


class SentenceGenerator:
    """
    Unified sentence generator that combines all pattern generators.
    
    Provides convenient methods for generating various sentence types
    and mixed curricula.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Initialize all sub-generators with same seed
        self._intransitive = IntransitiveGenerator(seed)
        self._transitive = TransitiveGenerator(seed)
        self._copular = CopularGenerator(seed)
        self._adjective_noun = AdjectiveNounGenerator(seed)
        self._who_question = WhoQuestionGenerator(seed)
        self._what_question = WhatQuestionGenerator(seed)
        self._yesno_question = YesNoQuestionGenerator(seed)
        self._where_question = WhereQuestionGenerator(seed)
        self._first_person = FirstPersonGenerator(seed)
        self._second_person = SecondPersonGenerator(seed)
        self._cognitive = CognitiveVerbGenerator(seed)
        self._meta_cognitive = MetaCognitiveGenerator(seed)
        self._self_query = SelfQueryGenerator(seed)
        
        # Access vocabulary through base generator
        self.vocab = self._intransitive.vocab
    
    # === Declarative ===
    
    def generate_intransitive(self, n: int = 10) -> List[GroundedSentence]:
        """Generate intransitive sentences: 'the dog runs'"""
        return self._intransitive.generate(n)
    
    def generate_transitive(self, n: int = 10) -> List[GroundedSentence]:
        """Generate transitive sentences: 'the dog chases the cat'"""
        return self._transitive.generate(n)
    
    def generate_copular(self, n: int = 10) -> List[GroundedSentence]:
        """Generate copular sentences: 'the dog is big'"""
        return self._copular.generate(n)
    
    def generate_adjective_noun(self, n: int = 10) -> List[GroundedSentence]:
        """Generate adjective-noun phrases: 'the big dog'"""
        return self._adjective_noun.generate(n)
    
    # === Interrogative ===
    
    def generate_question_who(self, n: int = 10) -> List[GroundedSentence]:
        """Generate 'who' questions: 'who runs'"""
        return self._who_question.generate(n)
    
    def generate_question_what(self, n: int = 10) -> List[GroundedSentence]:
        """Generate 'what' questions: 'what does the dog see'"""
        return self._what_question.generate(n)
    
    def generate_question_yesno(self, n: int = 10) -> List[GroundedSentence]:
        """Generate yes/no questions: 'does the dog run'"""
        return self._yesno_question.generate(n)
    
    def generate_question_where(self, n: int = 10) -> List[GroundedSentence]:
        """Generate 'where' questions: 'where is the dog'"""
        return self._where_question.generate(n)
    
    # === Self-Referential ===
    
    def generate_first_person(self, n: int = 10) -> List[GroundedSentence]:
        """Generate first-person sentences: 'I see the dog'"""
        return self._first_person.generate(n)
    
    def generate_second_person(self, n: int = 10) -> List[GroundedSentence]:
        """Generate second-person sentences: 'you see the dog'"""
        return self._second_person.generate(n)
    
    def generate_self_referential(self, n: int = 10) -> List[GroundedSentence]:
        """Generate mixed self-referential sentences."""
        sentences = []
        sentences.extend(self._first_person.generate(n // 2))
        sentences.extend(self._second_person.generate(n // 2))
        self.rng.shuffle(sentences)
        return sentences
    
    def generate_cognitive(self, n: int = 10) -> List[GroundedSentence]:
        """Generate cognitive verb sentences: 'I know the dog'"""
        return self._cognitive.generate(n)
    
    def generate_meta_cognitive(self, n: int = 10) -> List[GroundedSentence]:
        """Generate meta-cognitive sentences: 'I think you know'"""
        return self._meta_cognitive.generate(n)
    
    def generate_self_query(self, n: int = 10) -> List[GroundedSentence]:
        """Generate self-query sentences: 'do you know the dog'"""
        return self._self_query.generate(n)
    
    # === Mixed ===
    
    def generate_mixed(self, n: int = 100) -> List[GroundedSentence]:
        """Generate a balanced mix of all sentence types."""
        sentences = []
        
        # Proportions (roughly)
        sentences.extend(self.generate_intransitive(n // 8))
        sentences.extend(self.generate_transitive(n // 8))
        sentences.extend(self.generate_copular(n // 10))
        sentences.extend(self.generate_adjective_noun(n // 10))
        sentences.extend(self.generate_question_who(n // 12))
        sentences.extend(self.generate_question_what(n // 12))
        sentences.extend(self.generate_question_yesno(n // 12))
        sentences.extend(self.generate_first_person(n // 10))
        sentences.extend(self.generate_second_person(n // 10))
        sentences.extend(self.generate_cognitive(n // 10))
        sentences.extend(self.generate_self_query(n // 12))
        
        self.rng.shuffle(sentences)
        return sentences
    
    # === Properties for backward compatibility ===
    
    @property
    def nouns(self):
        return self._intransitive.nouns
    
    @property
    def animate_nouns(self):
        return self._intransitive.animate_nouns
    
    @property
    def verbs(self):
        return self._intransitive.verbs
    
    @property
    def transitive_verbs(self):
        return self._intransitive.transitive_verbs
    
    @property
    def adjectives(self):
        return self._intransitive.adjectives


__all__ = [
    'SentenceGenerator',
    'BaseGenerator',
    'IntransitiveGenerator',
    'TransitiveGenerator',
    'CopularGenerator',
    'AdjectiveNounGenerator',
    'WhoQuestionGenerator',
    'WhatQuestionGenerator',
    'YesNoQuestionGenerator',
    'WhereQuestionGenerator',
    'FirstPersonGenerator',
    'SecondPersonGenerator',
    'CognitiveVerbGenerator',
    'MetaCognitiveGenerator',
    'SelfQueryGenerator',
]
