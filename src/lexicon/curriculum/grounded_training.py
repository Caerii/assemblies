"""
Grounded Language Training
==========================

Training data where every sentence is grounded in context.
This is how children actually learn language.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum, auto
import random


class GroundingType(Enum):
    """Types of perceptual/conceptual grounding"""
    VISUAL = auto()      # What you see
    MOTOR = auto()       # Actions you do
    AUDITORY = auto()    # What you hear
    TACTILE = auto()     # What you touch
    EMOTIONAL = auto()   # How you feel
    SOCIAL = auto()      # Social context
    SPATIAL = auto()     # Where things are
    TEMPORAL = auto()    # When things happen


class SpeechAct(Enum):
    """Pragmatic function of utterance"""
    NAMING = auto()         # "That's a dog"
    DESCRIBING = auto()     # "The dog is big"
    REQUESTING = auto()     # "Give me the ball"
    COMMANDING = auto()     # "Sit down"
    QUESTIONING = auto()    # "What's that?"
    ANSWERING = auto()      # "It's a dog"
    GREETING = auto()       # "Hi!"
    AFFIRMING = auto()      # "Yes!"
    DENYING = auto()        # "No!"
    EXPRESSING = auto()     # "Ouch!" / "Yay!"


@dataclass
class GroundedContext:
    """The perceptual/conceptual context for an utterance"""
    
    # Visual scene elements (what's visible)
    visual_objects: List[str] = field(default_factory=list)
    visual_properties: Dict[str, List[str]] = field(default_factory=dict)  # object -> properties
    visual_relations: List[tuple] = field(default_factory=list)  # (obj1, relation, obj2)
    
    # Actions happening
    actions: List[tuple] = field(default_factory=list)  # (agent, action, patient)
    
    # Emotional state
    emotions: List[str] = field(default_factory=list)
    
    # Social context
    speaker: str = "CAREGIVER"
    addressee: str = "CHILD"
    joint_attention: Optional[str] = None  # What both are looking at
    
    # Spatial context
    locations: Dict[str, str] = field(default_factory=dict)  # object -> location


@dataclass
class GroundedUtterance:
    """A single training example with full grounding"""
    
    # The actual words
    words: List[str]
    
    # Grammatical structure
    pos_tags: List[str]  # Part of speech for each word
    
    # Pragmatic function
    speech_act: SpeechAct
    
    # Grounding context
    context: GroundedContext
    
    # Learning metadata
    complexity_level: int = 1  # 1-10
    target_words: List[str] = field(default_factory=list)  # Words being taught
    requires_response: bool = False
    expected_response: Optional[str] = None
    
    @property
    def sentence(self) -> str:
        return ' '.join(self.words)
    
    def get_grounding_activations(self) -> Dict[str, List[str]]:
        """Get brain areas that should be activated for grounding"""
        activations = {
            'VISUAL': self.context.visual_objects.copy(),
            'MOTOR': [a[1] for a in self.context.actions],  # Just the actions
            'EMOTION': self.context.emotions.copy(),
        }
        
        # Add properties
        for obj, props in self.context.visual_properties.items():
            activations['VISUAL'].extend(props)
        
        return activations


class GroundedCorpus:
    """
    A corpus of grounded training examples.
    Allows adding free sentences with automatic grounding inference.
    """
    
    def __init__(self):
        self.examples: List[GroundedUtterance] = []
        self.word_exposures: Dict[str, int] = {}
        self.structure_exposures: Dict[str, int] = {}
        
        # Simple POS tagging rules (can be expanded)
        self.pos_rules = {
            'the': 'DET', 'a': 'DET', 'an': 'DET',
            'my': 'DET', 'your': 'DET', 'his': 'DET', 'her': 'DET',
            'this': 'DET', 'that': 'DET',
            'is': 'AUX', 'are': 'AUX', 'was': 'AUX', 'were': 'AUX',
            'am': 'AUX', 'be': 'AUX', 'been': 'AUX', 'being': 'AUX',
            'has': 'AUX', 'have': 'AUX', 'had': 'AUX',
            'do': 'AUX', 'does': 'AUX', 'did': 'AUX',
            'can': 'MODAL', 'could': 'MODAL', 'will': 'MODAL', 'would': 'MODAL',
            'shall': 'MODAL', 'should': 'MODAL', 'may': 'MODAL', 'might': 'MODAL',
            'must': 'MODAL',
            'and': 'CONJ', 'or': 'CONJ', 'but': 'CONJ',
            'in': 'PREP', 'on': 'PREP', 'at': 'PREP', 'to': 'PREP',
            'from': 'PREP', 'with': 'PREP', 'by': 'PREP', 'for': 'PREP',
            'of': 'PREP', 'about': 'PREP', 'under': 'PREP', 'over': 'PREP',
            'i': 'PRON', 'you': 'PRON', 'he': 'PRON', 'she': 'PRON',
            'it': 'PRON', 'we': 'PRON', 'they': 'PRON',
            'me': 'PRON', 'him': 'PRON', 'her': 'PRON', 'us': 'PRON', 'them': 'PRON',
            'not': 'NEG', "n't": 'NEG',
            'very': 'ADV', 'really': 'ADV', 'quickly': 'ADV', 'slowly': 'ADV',
            'now': 'ADV', 'then': 'ADV', 'here': 'ADV', 'there': 'ADV',
        }
        
        # Known nouns, verbs, adjectives (will be expanded from lexicon)
        self.known_nouns: Set[str] = set()
        self.known_verbs: Set[str] = set()
        self.known_adjectives: Set[str] = set()
    
    def load_vocabulary(self, lexicon):
        """Load vocabulary from lexicon for POS tagging"""
        from src.lexicon.lexicon_manager import WordCategory
        
        for word in lexicon.words.values():
            if word.category == WordCategory.NOUN:
                self.known_nouns.add(word.lemma)
                for form in word.forms.values():
                    self.known_nouns.add(form)
            elif word.category == WordCategory.VERB:
                self.known_verbs.add(word.lemma)
                for form in word.forms.values():
                    self.known_verbs.add(form)
            elif word.category == WordCategory.ADJECTIVE:
                self.known_adjectives.add(word.lemma)
                for form in word.forms.values():
                    self.known_adjectives.add(form)
    
    def infer_pos(self, word: str) -> str:
        """Infer part of speech for a word"""
        word_lower = word.lower()
        
        if word_lower in self.pos_rules:
            return self.pos_rules[word_lower]
        if word_lower in self.known_nouns:
            return 'NOUN'
        if word_lower in self.known_verbs:
            return 'VERB'
        if word_lower in self.known_adjectives:
            return 'ADJ'
        
        # Heuristics
        if word_lower.endswith('ly'):
            return 'ADV'
        if word_lower.endswith('ing'):
            return 'VERB'  # Could be noun (gerund) but usually verb
        if word_lower.endswith('ed'):
            return 'VERB'
        if word_lower.endswith('s') and len(word_lower) > 3:
            return 'NOUN'  # Plural noun (rough heuristic)
        
        return 'NOUN'  # Default to noun
    
    def infer_context(self, words: List[str], pos_tags: List[str]) -> GroundedContext:
        """Infer grounding context from sentence structure"""
        context = GroundedContext()
        
        # Extract nouns as visual objects
        for word, pos in zip(words, pos_tags):
            if pos == 'NOUN':
                context.visual_objects.append(word.upper())
        
        # Extract adjectives as properties
        current_noun = None
        for i, (word, pos) in enumerate(zip(words, pos_tags)):
            if pos == 'NOUN':
                current_noun = word.upper()
            elif pos == 'ADJ' and current_noun is None:
                # Adjective before noun - look ahead
                for j in range(i + 1, len(pos_tags)):
                    if pos_tags[j] == 'NOUN':
                        noun = words[j].upper()
                        if noun not in context.visual_properties:
                            context.visual_properties[noun] = []
                        context.visual_properties[noun].append(word.upper())
                        break
        
        # Extract actions (simple SVO pattern)
        for i, (word, pos) in enumerate(zip(words, pos_tags)):
            if pos == 'VERB':
                agent = None
                patient = None
                
                # Look for agent before verb
                for j in range(i - 1, -1, -1):
                    if pos_tags[j] == 'NOUN':
                        agent = words[j].upper()
                        break
                    elif pos_tags[j] == 'PRON':
                        agent = words[j].upper()
                        break
                
                # Look for patient after verb
                for j in range(i + 1, len(pos_tags)):
                    if pos_tags[j] == 'NOUN':
                        patient = words[j].upper()
                        break
                
                context.actions.append((agent, word.upper(), patient))
        
        return context
    
    def infer_speech_act(self, words: List[str]) -> SpeechAct:
        """Infer the speech act from sentence form"""
        sentence = ' '.join(words).lower()
        
        if sentence.endswith('?'):
            return SpeechAct.QUESTIONING
        if sentence.startswith(('what', 'where', 'who', 'when', 'why', 'how')):
            return SpeechAct.QUESTIONING
        if sentence.startswith(('is ', 'are ', 'do ', 'does ', 'can ', 'will ')):
            return SpeechAct.QUESTIONING
        if sentence.startswith(('give', 'show', 'bring', 'get', 'please')):
            return SpeechAct.REQUESTING
        if any(w in sentence for w in ['hi', 'hello', 'bye', 'goodbye']):
            return SpeechAct.GREETING
        if sentence in ['yes', 'yeah', 'yep', 'ok', 'okay']:
            return SpeechAct.AFFIRMING
        if sentence in ['no', 'nope', 'nah']:
            return SpeechAct.DENYING
        if 'is' in words or 'are' in words:
            return SpeechAct.DESCRIBING
        
        return SpeechAct.NAMING
    
    def add_sentence(self, sentence: str, 
                     context: Optional[GroundedContext] = None,
                     speech_act: Optional[SpeechAct] = None,
                     complexity: int = 1,
                     target_words: Optional[List[str]] = None,
                     requires_response: bool = False,
                     expected_response: Optional[str] = None) -> GroundedUtterance:
        """
        Add a free sentence to the corpus.
        Automatically infers grounding if not provided.
        """
        # Tokenize
        words = sentence.lower().replace('?', '').replace('!', '').replace('.', '').split()
        
        # POS tag
        pos_tags = [self.infer_pos(w) for w in words]
        
        # Infer context if not provided
        if context is None:
            context = self.infer_context(words, pos_tags)
        
        # Infer speech act if not provided
        if speech_act is None:
            speech_act = self.infer_speech_act(words)
        
        # Create utterance
        utterance = GroundedUtterance(
            words=words,
            pos_tags=pos_tags,
            speech_act=speech_act,
            context=context,
            complexity_level=complexity,
            target_words=target_words or [],
            requires_response=requires_response,
            expected_response=expected_response,
        )
        
        self.examples.append(utterance)
        
        # Track exposures
        for word in words:
            self.word_exposures[word] = self.word_exposures.get(word, 0) + 1
        
        structure = ' '.join(pos_tags)
        self.structure_exposures[structure] = self.structure_exposures.get(structure, 0) + 1
        
        return utterance
    
    def add_grounded_word(self, word: str, visual_context: List[str], 
                          n_examples: int = 10) -> List[GroundedUtterance]:
        """
        Add grounded examples for learning a single word.
        Creates ~10 varied examples as per learning protocol.
        """
        examples = []
        
        # Naming examples (3x)
        for _ in range(3):
            ctx = GroundedContext(
                visual_objects=visual_context,
                joint_attention=visual_context[0] if visual_context else None,
            )
            ex = self.add_sentence(
                f"this is a {word}",
                context=ctx,
                speech_act=SpeechAct.NAMING,
                complexity=1,
                target_words=[word],
            )
            examples.append(ex)
        
        # Descriptive examples (2x)
        adjectives = ['big', 'little', 'nice', 'good']
        for adj in random.sample(adjectives, min(2, len(adjectives))):
            ctx = GroundedContext(
                visual_objects=visual_context,
                visual_properties={visual_context[0]: [adj.upper()]} if visual_context else {},
            )
            ex = self.add_sentence(
                f"the {word} is {adj}",
                context=ctx,
                speech_act=SpeechAct.DESCRIBING,
                complexity=2,
                target_words=[word],
            )
            examples.append(ex)
        
        # Action examples (2x)
        if visual_context:
            actions = ['see', 'look at', 'like', 'want']
            for action in random.sample(actions, min(2, len(actions))):
                ctx = GroundedContext(
                    visual_objects=visual_context,
                    actions=[('I', action.upper(), visual_context[0])],
                )
                ex = self.add_sentence(
                    f"i {action} the {word}",
                    context=ctx,
                    speech_act=SpeechAct.DESCRIBING,
                    complexity=3,
                    target_words=[word],
                )
                examples.append(ex)
        
        # Question examples (2x)
        ctx = GroundedContext(
            visual_objects=visual_context,
            joint_attention=visual_context[0] if visual_context else None,
        )
        ex = self.add_sentence(
            f"where is the {word}?",
            context=ctx,
            speech_act=SpeechAct.QUESTIONING,
            complexity=2,
            target_words=[word],
            requires_response=True,
        )
        examples.append(ex)
        
        ex = self.add_sentence(
            f"do you see the {word}?",
            context=ctx,
            speech_act=SpeechAct.QUESTIONING,
            complexity=3,
            target_words=[word],
            requires_response=True,
        )
        examples.append(ex)
        
        # Command example (1x)
        ctx = GroundedContext(
            visual_objects=visual_context,
        )
        ex = self.add_sentence(
            f"look at the {word}",
            context=ctx,
            speech_act=SpeechAct.COMMANDING,
            complexity=2,
            target_words=[word],
        )
        examples.append(ex)
        
        return examples
    
    def get_training_batch(self, complexity_max: int = 10, 
                           batch_size: int = 32) -> List[GroundedUtterance]:
        """Get a batch of training examples up to given complexity"""
        eligible = [ex for ex in self.examples if ex.complexity_level <= complexity_max]
        if len(eligible) <= batch_size:
            return eligible
        return random.sample(eligible, batch_size)
    
    def get_word_exposure_count(self, word: str) -> int:
        """How many times has this word been seen?"""
        return self.word_exposures.get(word.lower(), 0)
    
    def get_structure_exposure_count(self, structure: str) -> int:
        """How many times has this structure been seen?"""
        return self.structure_exposures.get(structure, 0)
    
    def get_statistics(self) -> Dict:
        """Get corpus statistics"""
        return {
            'total_examples': len(self.examples),
            'unique_words': len(self.word_exposures),
            'unique_structures': len(self.structure_exposures),
            'by_complexity': {
                i: len([ex for ex in self.examples if ex.complexity_level == i])
                for i in range(1, 11)
            },
            'by_speech_act': {
                act.name: len([ex for ex in self.examples if ex.speech_act == act])
                for act in SpeechAct
            },
            'avg_exposures_per_word': (
                sum(self.word_exposures.values()) / max(len(self.word_exposures), 1)
            ),
        }


# Pre-built grounded corpus for Stage 1
def create_stage1_corpus() -> GroundedCorpus:
    """Create a grounded corpus for Stage 1 (First Words)"""
    corpus = GroundedCorpus()
    
    # Core vocabulary with grounding
    words_with_grounding = [
        ('dog', ['DOG']),
        ('cat', ['CAT']),
        ('ball', ['BALL']),
        ('book', ['BOOK']),
        ('mom', ['MOM', 'PERSON']),
        ('dad', ['DAD', 'PERSON']),
        ('baby', ['BABY', 'PERSON']),
        ('milk', ['MILK', 'DRINK']),
        ('juice', ['JUICE', 'DRINK']),
        ('apple', ['APPLE', 'FOOD']),
        ('cookie', ['COOKIE', 'FOOD']),
        ('shoe', ['SHOE', 'CLOTHING']),
        ('cup', ['CUP', 'CONTAINER']),
        ('bird', ['BIRD', 'ANIMAL']),
    ]
    
    # Add grounded examples for each word
    for word, visual in words_with_grounding:
        corpus.add_grounded_word(word, visual, n_examples=10)
    
    return corpus


def create_stage2_corpus() -> GroundedCorpus:
    """Create a grounded corpus for Stage 2 (Vocabulary Spurt)"""
    corpus = GroundedCorpus()
    
    # More words, more complex sentences
    sentences = [
        # Naming with properties
        "the big dog runs",
        "the little cat sleeps",
        "the red ball bounces",
        "the blue car goes fast",
        
        # Actions
        "the dog eats food",
        "the cat drinks milk",
        "the baby plays with toys",
        "mom reads a book",
        
        # Locations
        "the ball is on the table",
        "the cat is under the bed",
        "the dog is in the house",
        
        # Possessives
        "my ball is big",
        "your dog is nice",
        "the baby has a toy",
    ]
    
    for sentence in sentences:
        corpus.add_sentence(sentence, complexity=3)
    
    return corpus

