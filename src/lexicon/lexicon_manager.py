"""
Lexicon Manager
===============

Manages a massive lexicon with proper organization and features.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple
import json
import os

class WordCategory(Enum):
    """Grammatical categories"""
    # Core categories
    NOUN = auto()
    VERB = auto()
    ADJECTIVE = auto()
    ADVERB = auto()
    
    # Function words
    DETERMINER = auto()
    PRONOUN = auto()
    PREPOSITION = auto()
    CONJUNCTION = auto()
    INTERJECTION = auto()
    
    # Verb subtypes
    AUXILIARY = auto()
    MODAL = auto()
    
    # Special
    PARTICLE = auto()
    NUMERAL = auto()


class SemanticDomain(Enum):
    """Semantic domains for grounding"""
    # Concrete
    ANIMAL = auto()
    PERSON = auto()
    BODY_PART = auto()
    FOOD = auto()
    PLANT = auto()
    OBJECT = auto()
    BUILDING = auto()
    VEHICLE = auto()
    CLOTHING = auto()
    TOOL = auto()
    
    # Abstract
    EMOTION = auto()
    COGNITION = auto()
    TIME = auto()
    SPACE = auto()
    QUANTITY = auto()
    QUALITY = auto()
    SOCIAL = auto()
    
    # Actions
    MOTION = auto()
    PERCEPTION = auto()
    COMMUNICATION = auto()
    CONSUMPTION = auto()
    CREATION = auto()
    DESTRUCTION = auto()
    POSSESSION = auto()
    
    # Function
    FUNCTION_WORD = auto()


@dataclass
class Word:
    """A word entry with all its features"""
    lemma: str                          # Base form
    category: WordCategory              # Grammatical category
    forms: Dict[str, str] = field(default_factory=dict)  # Inflected forms
    
    # Semantic features
    semantic_domains: List[SemanticDomain] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Syntactic features
    subcategory: Optional[str] = None   # e.g., "transitive", "mass_noun"
    arguments: List[str] = field(default_factory=list)  # Argument structure
    
    # Statistics
    frequency: float = 0.0              # Log frequency
    age_of_acquisition: float = 0.0     # When typically learned
    
    # Relationships
    synonyms: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    hypernyms: List[str] = field(default_factory=list)  # More general
    hyponyms: List[str] = field(default_factory=list)   # More specific
    
    def get_all_forms(self) -> List[str]:
        """Get all surface forms of this word"""
        return [self.lemma] + list(self.forms.values())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'lemma': self.lemma,
            'category': self.category.name,
            'forms': self.forms,
            'semantic_domains': [d.name for d in self.semantic_domains],
            'features': self.features,
            'subcategory': self.subcategory,
            'arguments': self.arguments,
            'frequency': self.frequency,
            'age_of_acquisition': self.age_of_acquisition,
            'synonyms': self.synonyms,
            'antonyms': self.antonyms,
            'hypernyms': self.hypernyms,
            'hyponyms': self.hyponyms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Word':
        """Create from dictionary"""
        return cls(
            lemma=data['lemma'],
            category=WordCategory[data['category']],
            forms=data.get('forms', {}),
            semantic_domains=[SemanticDomain[d] for d in data.get('semantic_domains', [])],
            features=data.get('features', {}),
            subcategory=data.get('subcategory'),
            arguments=data.get('arguments', []),
            frequency=data.get('frequency', 0.0),
            age_of_acquisition=data.get('age_of_acquisition', 0.0),
            synonyms=data.get('synonyms', []),
            antonyms=data.get('antonyms', []),
            hypernyms=data.get('hypernyms', []),
            hyponyms=data.get('hyponyms', []),
        )


class LexiconManager:
    """
    Manages a massive lexicon with:
    - Organized word storage by category
    - Fast lookup by form or lemma
    - Frequency-based sampling
    - Semantic domain filtering
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        
        # Storage
        self.words: Dict[str, Word] = {}  # lemma -> Word
        self.form_to_lemma: Dict[str, str] = {}  # surface form -> lemma
        
        # Indices
        self.by_category: Dict[WordCategory, List[str]] = {cat: [] for cat in WordCategory}
        self.by_domain: Dict[SemanticDomain, List[str]] = {dom: [] for dom in SemanticDomain}
        
        # Statistics
        self.total_words = 0
        self.category_counts: Dict[WordCategory, int] = {}
    
    def add_word(self, word: Word):
        """Add a word to the lexicon"""
        self.words[word.lemma] = word
        self.form_to_lemma[word.lemma] = word.lemma
        
        for form in word.forms.values():
            self.form_to_lemma[form] = word.lemma
        
        self.by_category[word.category].append(word.lemma)
        
        for domain in word.semantic_domains:
            self.by_domain[domain].append(word.lemma)
        
        self.total_words += 1
        self.category_counts[word.category] = self.category_counts.get(word.category, 0) + 1
    
    def get_word(self, form: str) -> Optional[Word]:
        """Get word by any surface form"""
        lemma = self.form_to_lemma.get(form.lower())
        if lemma:
            return self.words.get(lemma)
        return None
    
    def get_by_category(self, category: WordCategory) -> List[Word]:
        """Get all words of a category"""
        return [self.words[lemma] for lemma in self.by_category[category]]
    
    def get_by_domain(self, domain: SemanticDomain) -> List[Word]:
        """Get all words in a semantic domain"""
        return [self.words[lemma] for lemma in self.by_domain[domain]]
    
    def get_by_frequency(self, min_freq: float = 0.0, max_freq: float = float('inf'),
                         category: Optional[WordCategory] = None) -> List[Word]:
        """Get words within a frequency range"""
        words = self.words.values() if category is None else self.get_by_category(category)
        return [w for w in words if min_freq <= w.frequency <= max_freq]
    
    def get_by_aoa(self, max_aoa: float, category: Optional[WordCategory] = None) -> List[Word]:
        """Get words up to a certain age of acquisition"""
        words = self.words.values() if category is None else self.get_by_category(category)
        return [w for w in words if w.age_of_acquisition <= max_aoa]
    
    def save(self, filename: str = 'lexicon.json'):
        """Save lexicon to file"""
        filepath = os.path.join(self.data_dir, filename)
        data = {
            'words': [w.to_dict() for w in self.words.values()],
            'metadata': {
                'total_words': self.total_words,
                'category_counts': {k.name: v for k, v in self.category_counts.items()},
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str = 'lexicon.json'):
        """Load lexicon from file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for word_data in data['words']:
            word = Word.from_dict(word_data)
            self.add_word(word)
    
    def get_stats(self) -> Dict:
        """Get lexicon statistics"""
        return {
            'total_words': self.total_words,
            'by_category': {cat.name: len(lemmas) for cat, lemmas in self.by_category.items() if lemmas},
            'by_domain': {dom.name: len(lemmas) for dom, lemmas in self.by_domain.items() if lemmas},
        }

