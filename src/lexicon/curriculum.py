"""
Curriculum-Based Learning
=========================

Defines learning stages and progressions for vocabulary acquisition.
Based on child language acquisition research and frequency statistics.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Callable
import random

from .lexicon_manager import Word, WordCategory, SemanticDomain, LexiconManager


class LearningStage(Enum):
    """Stages of vocabulary acquisition (based on child development)"""
    
    # Stage 1: First words (12-18 months equivalent)
    FIRST_WORDS = 1
    # ~50 words: mama, dada, ball, dog, cat, no, more, up, hi, bye
    
    # Stage 2: Vocabulary spurt (18-24 months)
    VOCABULARY_SPURT = 2
    # ~200-300 words: common nouns, verbs, adjectives
    
    # Stage 3: Two-word combinations (24-30 months)
    TWO_WORD = 3
    # ~500 words: more verbs, prepositions, pronouns
    
    # Stage 4: Sentence formation (30-36 months)
    SENTENCES = 4
    # ~1000 words: auxiliaries, conjunctions, more abstract words
    
    # Stage 5: Complex grammar (3-4 years)
    COMPLEX_GRAMMAR = 5
    # ~2000 words: relative clauses, passives, more vocabulary
    
    # Stage 6: School-age (5-7 years)
    SCHOOL_AGE = 6
    # ~5000 words: academic vocabulary, literacy-related
    
    # Stage 7: Adolescent/Adult
    ADVANCED = 7
    # ~10000+ words: specialized vocabulary, rare words


@dataclass
class CurriculumStage:
    """Definition of a learning stage"""
    stage: LearningStage
    name: str
    description: str
    
    # Word selection criteria
    max_aoa: float                      # Maximum age of acquisition
    min_frequency: float                # Minimum frequency (log scale)
    target_vocab_size: int              # Target number of words
    
    # Category distribution (proportions)
    category_weights: Dict[WordCategory, float] = field(default_factory=dict)
    
    # Semantic domain focus
    focus_domains: List[SemanticDomain] = field(default_factory=list)
    
    # Syntactic complexity
    max_arguments: int = 1              # Max verb arguments
    allow_abstract: bool = False        # Allow abstract words
    
    # Training parameters
    repetitions_per_word: int = 10      # How many times to see each word
    sentence_complexity: int = 1        # 1=single word, 2=two word, etc.


@dataclass
class TrainingExample:
    """A training example for the curriculum"""
    words: List[str]                    # Words in order
    categories: List[WordCategory]      # Categories in order
    stage: LearningStage
    context: Optional[str] = None       # Optional context description


class Curriculum:
    """
    Manages curriculum-based vocabulary learning.
    
    Provides:
    - Stage-appropriate word selection
    - Proper frequency distribution
    - Gradual complexity increase
    - Spaced repetition
    """
    
    def __init__(self, lexicon: LexiconManager):
        self.lexicon = lexicon
        self.stages = self._define_stages()
        self.current_stage = LearningStage.FIRST_WORDS
        
        # Tracking
        self.words_learned: Set[str] = set()
        self.word_exposures: Dict[str, int] = {}
        self.stage_progress: Dict[LearningStage, float] = {}
    
    def _define_stages(self) -> Dict[LearningStage, CurriculumStage]:
        """Define all curriculum stages"""
        return {
            LearningStage.FIRST_WORDS: CurriculumStage(
                stage=LearningStage.FIRST_WORDS,
                name="First Words",
                description="Basic nouns, social words, simple verbs",
                max_aoa=2.0,
                min_frequency=5.0,  # Very frequent words
                target_vocab_size=50,
                category_weights={
                    WordCategory.NOUN: 0.50,
                    WordCategory.VERB: 0.20,
                    WordCategory.INTERJECTION: 0.15,
                    WordCategory.ADJECTIVE: 0.10,
                    WordCategory.ADVERB: 0.05,
                },
                focus_domains=[
                    SemanticDomain.PERSON, SemanticDomain.ANIMAL,
                    SemanticDomain.FOOD, SemanticDomain.BODY_PART,
                    SemanticDomain.SOCIAL,
                ],
                max_arguments=1,
                allow_abstract=False,
                repetitions_per_word=20,
                sentence_complexity=1,
            ),
            
            LearningStage.VOCABULARY_SPURT: CurriculumStage(
                stage=LearningStage.VOCABULARY_SPURT,
                name="Vocabulary Spurt",
                description="Rapid noun acquisition, basic verbs and adjectives",
                max_aoa=2.5,
                min_frequency=4.0,
                target_vocab_size=300,
                category_weights={
                    WordCategory.NOUN: 0.55,
                    WordCategory.VERB: 0.25,
                    WordCategory.ADJECTIVE: 0.15,
                    WordCategory.ADVERB: 0.05,
                },
                focus_domains=[
                    SemanticDomain.ANIMAL, SemanticDomain.OBJECT,
                    SemanticDomain.FOOD, SemanticDomain.CLOTHING,
                    SemanticDomain.MOTION, SemanticDomain.PERCEPTION,
                ],
                max_arguments=1,
                allow_abstract=False,
                repetitions_per_word=15,
                sentence_complexity=1,
            ),
            
            LearningStage.TWO_WORD: CurriculumStage(
                stage=LearningStage.TWO_WORD,
                name="Two-Word Stage",
                description="Combining words, more verbs and prepositions",
                max_aoa=3.0,
                min_frequency=3.5,
                target_vocab_size=500,
                category_weights={
                    WordCategory.NOUN: 0.40,
                    WordCategory.VERB: 0.30,
                    WordCategory.ADJECTIVE: 0.10,
                    WordCategory.PREPOSITION: 0.10,
                    WordCategory.PRONOUN: 0.05,
                    WordCategory.ADVERB: 0.05,
                },
                focus_domains=[
                    SemanticDomain.MOTION, SemanticDomain.SPACE,
                    SemanticDomain.POSSESSION, SemanticDomain.PERCEPTION,
                ],
                max_arguments=2,
                allow_abstract=False,
                repetitions_per_word=12,
                sentence_complexity=2,
            ),
            
            LearningStage.SENTENCES: CurriculumStage(
                stage=LearningStage.SENTENCES,
                name="Sentence Formation",
                description="Full sentences, auxiliaries, more complex verbs",
                max_aoa=4.0,
                min_frequency=3.0,
                target_vocab_size=1000,
                category_weights={
                    WordCategory.NOUN: 0.35,
                    WordCategory.VERB: 0.30,
                    WordCategory.ADJECTIVE: 0.10,
                    WordCategory.PREPOSITION: 0.08,
                    WordCategory.AUXILIARY: 0.05,
                    WordCategory.PRONOUN: 0.05,
                    WordCategory.CONJUNCTION: 0.04,
                    WordCategory.ADVERB: 0.03,
                },
                max_arguments=3,
                allow_abstract=True,
                repetitions_per_word=10,
                sentence_complexity=4,
            ),
            
            LearningStage.COMPLEX_GRAMMAR: CurriculumStage(
                stage=LearningStage.COMPLEX_GRAMMAR,
                name="Complex Grammar",
                description="Complex sentences, relative clauses, passives",
                max_aoa=5.0,
                min_frequency=2.5,
                target_vocab_size=2000,
                category_weights={
                    WordCategory.NOUN: 0.30,
                    WordCategory.VERB: 0.30,
                    WordCategory.ADJECTIVE: 0.12,
                    WordCategory.ADVERB: 0.08,
                    WordCategory.PREPOSITION: 0.08,
                    WordCategory.CONJUNCTION: 0.06,
                    WordCategory.PRONOUN: 0.04,
                    WordCategory.AUXILIARY: 0.02,
                },
                max_arguments=3,
                allow_abstract=True,
                repetitions_per_word=8,
                sentence_complexity=6,
            ),
            
            LearningStage.SCHOOL_AGE: CurriculumStage(
                stage=LearningStage.SCHOOL_AGE,
                name="School Age",
                description="Academic vocabulary, literacy-related words",
                max_aoa=7.0,
                min_frequency=2.0,
                target_vocab_size=5000,
                category_weights={
                    WordCategory.NOUN: 0.35,
                    WordCategory.VERB: 0.25,
                    WordCategory.ADJECTIVE: 0.15,
                    WordCategory.ADVERB: 0.10,
                    WordCategory.PREPOSITION: 0.05,
                    WordCategory.CONJUNCTION: 0.05,
                    WordCategory.PRONOUN: 0.03,
                    WordCategory.AUXILIARY: 0.02,
                },
                max_arguments=3,
                allow_abstract=True,
                repetitions_per_word=5,
                sentence_complexity=8,
            ),
            
            LearningStage.ADVANCED: CurriculumStage(
                stage=LearningStage.ADVANCED,
                name="Advanced",
                description="Full adult vocabulary, specialized terms",
                max_aoa=18.0,
                min_frequency=0.0,
                target_vocab_size=10000,
                category_weights={
                    WordCategory.NOUN: 0.35,
                    WordCategory.VERB: 0.25,
                    WordCategory.ADJECTIVE: 0.15,
                    WordCategory.ADVERB: 0.10,
                    WordCategory.PREPOSITION: 0.05,
                    WordCategory.CONJUNCTION: 0.04,
                    WordCategory.PRONOUN: 0.03,
                    WordCategory.AUXILIARY: 0.02,
                    WordCategory.INTERJECTION: 0.01,
                },
                max_arguments=3,
                allow_abstract=True,
                repetitions_per_word=3,
                sentence_complexity=10,
            ),
        }
    
    def get_stage_words(self, stage: LearningStage) -> List[Word]:
        """Get words appropriate for a stage"""
        stage_def = self.stages[stage]
        
        # Get words meeting criteria
        candidates = self.lexicon.get_by_aoa(stage_def.max_aoa)
        candidates = [w for w in candidates if w.frequency >= stage_def.min_frequency]
        
        # Filter by abstract if needed
        if not stage_def.allow_abstract:
            candidates = [w for w in candidates 
                         if SemanticDomain.COGNITION not in w.semantic_domains]
        
        # Sample according to category weights
        result = []
        for category, weight in stage_def.category_weights.items():
            cat_words = [w for w in candidates if w.category == category]
            n_words = int(stage_def.target_vocab_size * weight)
            if len(cat_words) >= n_words:
                result.extend(random.sample(cat_words, n_words))
            else:
                result.extend(cat_words)
        
        return result[:stage_def.target_vocab_size]
    
    def generate_training_examples(self, stage: LearningStage, 
                                   n_examples: int = 100) -> List[TrainingExample]:
        """Generate training examples for a stage"""
        stage_def = self.stages[stage]
        words = self.get_stage_words(stage)
        
        examples = []
        complexity = stage_def.sentence_complexity
        
        for _ in range(n_examples):
            # Select words based on complexity
            n_words = min(complexity, random.randint(1, complexity))
            
            if n_words == 1:
                # Single word
                word = random.choice(words)
                examples.append(TrainingExample(
                    words=[word.lemma],
                    categories=[word.category],
                    stage=stage,
                ))
            else:
                # Multi-word combination
                selected = []
                categories = []
                
                # Try to make grammatical combinations
                if n_words >= 2:
                    # DET + NOUN or ADJ + NOUN
                    nouns = [w for w in words if w.category == WordCategory.NOUN]
                    dets = [w for w in words if w.category == WordCategory.DETERMINER]
                    adjs = [w for w in words if w.category == WordCategory.ADJECTIVE]
                    
                    if nouns and (dets or adjs):
                        noun = random.choice(nouns)
                        if dets and random.random() < 0.7:
                            det = random.choice(dets)
                            selected = [det.lemma, noun.lemma]
                            categories = [det.category, noun.category]
                        elif adjs:
                            adj = random.choice(adjs)
                            selected = [adj.lemma, noun.lemma]
                            categories = [adj.category, noun.category]
                
                if n_words >= 3 and len(selected) == 2:
                    # Add verb: NOUN + VERB or NOUN + VERB + NOUN
                    verbs = [w for w in words if w.category == WordCategory.VERB]
                    if verbs:
                        verb = random.choice(verbs)
                        selected.append(verb.lemma)
                        categories.append(verb.category)
                
                if n_words >= 4 and len(selected) == 3:
                    # Add object
                    nouns = [w for w in words if w.category == WordCategory.NOUN]
                    if nouns:
                        obj = random.choice(nouns)
                        selected.append(obj.lemma)
                        categories.append(obj.category)
                
                if selected:
                    examples.append(TrainingExample(
                        words=selected,
                        categories=categories,
                        stage=stage,
                    ))
        
        return examples
    
    def record_exposure(self, word: str):
        """Record that a word was seen"""
        self.word_exposures[word] = self.word_exposures.get(word, 0) + 1
        
        # Check if word is "learned" (enough exposures)
        stage_def = self.stages[self.current_stage]
        if self.word_exposures[word] >= stage_def.repetitions_per_word:
            self.words_learned.add(word)
    
    def get_progress(self) -> Dict:
        """Get learning progress"""
        stage_def = self.stages[self.current_stage]
        stage_words = set(w.lemma for w in self.get_stage_words(self.current_stage))
        
        learned_in_stage = len(self.words_learned & stage_words)
        progress = learned_in_stage / max(len(stage_words), 1)
        
        return {
            'current_stage': self.current_stage.name,
            'words_learned': len(self.words_learned),
            'stage_progress': progress,
            'target_vocab': stage_def.target_vocab_size,
        }
    
    def should_advance(self) -> bool:
        """Check if ready to advance to next stage"""
        progress = self.get_progress()
        return progress['stage_progress'] >= 0.8  # 80% mastery
    
    def advance_stage(self) -> bool:
        """Advance to next stage if possible"""
        stages = list(LearningStage)
        current_idx = stages.index(self.current_stage)
        
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            return True
        return False

