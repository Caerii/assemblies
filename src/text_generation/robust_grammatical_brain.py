#!/usr/bin/env python3
"""
Robust Grammatical Brain
========================

A comprehensive implementation of grammatical structure learning using Assembly Calculus.
Features:
- Multiple CORE areas for different grammatical categories
- SEQ area for learning word order from corpus
- Error detection via assembly stability ("wobbly" assemblies signal impossible parses)
- Rich lexicon with many word types
- MOOD area for sentence types (declarative, interrogative, imperative)
- Context areas (VISUAL, MOTOR, ABSTRACT) for grounding

Based on parser.py, recursive_parser.py, learner.py, and language_brain_simulation.md
"""

from collections import namedtuple, defaultdict
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple, Any
import numpy as np
import random

import brain

# =============================================================================
# BRAIN AREAS - Comprehensive Architecture
# =============================================================================

# Lexical Areas (Explicit - fixed assemblies for words)
LEX = "LEX"           # Main lexicon
PHON = "PHON"         # Phonological forms

# Syntactic Role Areas (Non-explicit - emerge dynamically)
SUBJ = "SUBJ"         # Subject
OBJ = "OBJ"           # Direct object
IOBJ = "IOBJ"         # Indirect object
VERB = "VERB"         # Verb phrase
COMP = "COMP"         # Complement clause

# Modifier Areas
DET = "DET"           # Determiner
ADJ = "ADJ"           # Adjective
ADV = "ADV"           # Adverb
PREP = "PREP"         # Preposition
PREP_P = "PREP_P"     # Prepositional phrase

# CORE Areas (Explicit - grammatical category representations)
# These are crucial for word order learning and category detection
NOUN_CORE = "NOUN_CORE"
VERB_CORE = "VERB_CORE"
ADJ_CORE = "ADJ_CORE"
DET_CORE = "DET_CORE"
PREP_CORE = "PREP_CORE"
ADV_CORE = "ADV_CORE"

# Sequence and Control Areas
SEQ = "SEQ"           # Sequence memory for word order
MOOD = "MOOD"         # Sentence mood (declarative, interrogative, etc.)
TENSE = "TENSE"       # Tense marking
ASPECT = "ASPECT"     # Aspect (progressive, perfect, etc.)

# Context/Grounding Areas (Explicit - connect words to meaning)
VISUAL = "VISUAL"     # Visual concepts (concrete nouns)
MOTOR = "MOTOR"       # Motor concepts (action verbs)
ABSTRACT = "ABSTRACT" # Abstract concepts
EMOTION = "EMOTION"   # Emotional concepts

# Error Detection Area
ERROR = "ERROR"       # Signals parsing errors via unstable assemblies

# Area groupings
ALL_AREAS = [
    LEX, SUBJ, OBJ, IOBJ, VERB, COMP,
    DET, ADJ, ADV, PREP, PREP_P,
    NOUN_CORE, VERB_CORE, ADJ_CORE, DET_CORE, PREP_CORE, ADV_CORE,
    SEQ, MOOD, TENSE, ASPECT,
    VISUAL, MOTOR, ABSTRACT, EMOTION,
    ERROR
]

EXPLICIT_AREAS = [LEX, NOUN_CORE, VERB_CORE, ADJ_CORE, DET_CORE, PREP_CORE, ADV_CORE, 
                  MOOD, TENSE, ASPECT, VISUAL, MOTOR, ABSTRACT, EMOTION]

SYNTACTIC_AREAS = [SUBJ, OBJ, IOBJ, VERB, COMP, DET, ADJ, ADV, PREP, PREP_P]

CORE_AREAS = [NOUN_CORE, VERB_CORE, ADJ_CORE, DET_CORE, PREP_CORE, ADV_CORE]

CONTEXT_AREAS = [VISUAL, MOTOR, ABSTRACT, EMOTION]

# =============================================================================
# MOOD TYPES
# =============================================================================

class MoodType(Enum):
    DECLARATIVE = 0      # "The cat runs"
    INTERROGATIVE = 1    # "Does the cat run?"
    IMPERATIVE = 2       # "Run!"
    EXCLAMATORY = 3      # "What a cat!"
    CONDITIONAL = 4      # "If the cat runs..."

# =============================================================================
# TENSE TYPES
# =============================================================================

class TenseType(Enum):
    PRESENT = 0
    PAST = 1
    FUTURE = 2

# =============================================================================
# ACTIONS
# =============================================================================

DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"

AreaRule = namedtuple("AreaRule", ["action", "area", "index"])
FiberRule = namedtuple("FiberRule", ["action", "area1", "area2", "index"])

# =============================================================================
# LEXICON - Comprehensive with categories and features
# =============================================================================

def create_word_entry(index: int, word_type: str, 
                      context_area: str = None,
                      context_index: int = None,
                      features: Dict = None) -> Dict:
    """Create a lexicon entry with full features"""
    entry = {
        "index": index,
        "type": word_type,
        "context_area": context_area,
        "context_index": context_index,
        "features": features or {},
        "PRE_RULES": [],
        "POST_RULES": [],
    }
    
    # Add type-specific rules
    if word_type == "NOUN":
        entry["PRE_RULES"] = [
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            FiberRule(DISINHIBIT, LEX, IOBJ, 0),
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
            FiberRule(DISINHIBIT, DET, OBJ, 0),
            FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, OBJ, 0),
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, PREP, 0),
        ]
        entry["POST_RULES"] = [
            AreaRule(INHIBIT, DET, 0),
            AreaRule(INHIBIT, ADJ, 0),
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            FiberRule(INHIBIT, ADJ, SUBJ, 0),
            FiberRule(INHIBIT, ADJ, OBJ, 0),
            FiberRule(INHIBIT, DET, SUBJ, 0),
            FiberRule(INHIBIT, DET, OBJ, 0),
        ]
        entry["core_area"] = NOUN_CORE
        
    elif word_type == "VERB":
        is_transitive = features.get("transitive", True)
        is_ditransitive = features.get("ditransitive", False)
        
        entry["PRE_RULES"] = [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADV, 0),
            AreaRule(DISINHIBIT, ADV, 1),
        ]
        post_rules = [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(INHIBIT, SUBJ, 0),
            AreaRule(INHIBIT, ADV, 0),
        ]
        if is_transitive:
            post_rules.append(AreaRule(DISINHIBIT, OBJ, 0))
        if is_ditransitive:
            post_rules.append(AreaRule(DISINHIBIT, IOBJ, 0))
        entry["POST_RULES"] = post_rules
        entry["core_area"] = VERB_CORE
        
    elif word_type == "DET":
        entry["PRE_RULES"] = [
            AreaRule(DISINHIBIT, DET, 0),
            FiberRule(DISINHIBIT, LEX, DET, 0),
        ]
        entry["POST_RULES"] = [
            FiberRule(INHIBIT, LEX, DET, 0),
        ]
        entry["core_area"] = DET_CORE
        
    elif word_type == "ADJ":
        entry["PRE_RULES"] = [
            AreaRule(DISINHIBIT, ADJ, 0),
            FiberRule(DISINHIBIT, LEX, ADJ, 0),
        ]
        entry["POST_RULES"] = [
            FiberRule(INHIBIT, LEX, ADJ, 0),
        ]
        entry["core_area"] = ADJ_CORE
        
    elif word_type == "ADV":
        entry["PRE_RULES"] = [
            AreaRule(DISINHIBIT, ADV, 0),
            FiberRule(DISINHIBIT, LEX, ADV, 0),
        ]
        entry["POST_RULES"] = [
            FiberRule(INHIBIT, LEX, ADV, 0),
            AreaRule(INHIBIT, ADV, 1),
        ]
        entry["core_area"] = ADV_CORE
        
    elif word_type == "PREP":
        entry["PRE_RULES"] = [
            AreaRule(DISINHIBIT, PREP, 0),
            FiberRule(DISINHIBIT, LEX, PREP, 0),
        ]
        entry["POST_RULES"] = [
            FiberRule(INHIBIT, LEX, PREP, 0),
            AreaRule(DISINHIBIT, PREP_P, 0),
        ]
        entry["core_area"] = PREP_CORE
    
    return entry


# Build comprehensive lexicon
def build_lexicon() -> Dict[str, Dict]:
    """Build a comprehensive lexicon with many word types"""
    lexicon = {}
    idx = 0
    
    # Determiners
    determiners = [
        ("the", {"definite": True}),
        ("a", {"definite": False}),
        ("an", {"definite": False}),
        ("this", {"demonstrative": True, "proximal": True}),
        ("that", {"demonstrative": True, "proximal": False}),
        ("these", {"demonstrative": True, "proximal": True, "plural": True}),
        ("those", {"demonstrative": True, "proximal": False, "plural": True}),
        ("my", {"possessive": True, "person": 1}),
        ("your", {"possessive": True, "person": 2}),
        ("his", {"possessive": True, "person": 3, "gender": "m"}),
        ("her", {"possessive": True, "person": 3, "gender": "f"}),
        ("some", {"quantifier": True}),
        ("any", {"quantifier": True}),
        ("every", {"quantifier": True, "universal": True}),
        ("no", {"quantifier": True, "negative": True}),
    ]
    for word, features in determiners:
        lexicon[word] = create_word_entry(idx, "DET", features=features)
        idx += 1
    
    # Nouns - Concrete (visual)
    concrete_nouns = [
        ("dog", {"animate": True, "animal": True}),
        ("cat", {"animate": True, "animal": True}),
        ("bird", {"animate": True, "animal": True}),
        ("fish", {"animate": True, "animal": True}),
        ("mouse", {"animate": True, "animal": True}),
        ("horse", {"animate": True, "animal": True}),
        ("man", {"animate": True, "human": True, "gender": "m"}),
        ("woman", {"animate": True, "human": True, "gender": "f"}),
        ("child", {"animate": True, "human": True}),
        ("boy", {"animate": True, "human": True, "gender": "m"}),
        ("girl", {"animate": True, "human": True, "gender": "f"}),
        ("book", {"animate": False, "object": True}),
        ("table", {"animate": False, "object": True, "furniture": True}),
        ("chair", {"animate": False, "object": True, "furniture": True}),
        ("house", {"animate": False, "building": True}),
        ("car", {"animate": False, "vehicle": True}),
        ("tree", {"animate": False, "plant": True}),
        ("flower", {"animate": False, "plant": True}),
        ("water", {"animate": False, "mass": True}),
        ("food", {"animate": False, "mass": True}),
    ]
    for i, (word, features) in enumerate(concrete_nouns):
        lexicon[word] = create_word_entry(idx, "NOUN", 
                                          context_area=VISUAL, 
                                          context_index=i,
                                          features=features)
        idx += 1
    
    # Nouns - Abstract
    abstract_nouns = [
        ("love", {"abstract": True, "emotion": True}),
        ("hate", {"abstract": True, "emotion": True}),
        ("fear", {"abstract": True, "emotion": True}),
        ("joy", {"abstract": True, "emotion": True}),
        ("idea", {"abstract": True}),
        ("thought", {"abstract": True}),
        ("time", {"abstract": True}),
        ("place", {"abstract": True}),
        ("way", {"abstract": True}),
        ("thing", {"abstract": True}),
    ]
    for i, (word, features) in enumerate(abstract_nouns):
        lexicon[word] = create_word_entry(idx, "NOUN",
                                          context_area=ABSTRACT,
                                          context_index=i,
                                          features=features)
        idx += 1
    
    # Verbs - Transitive (motor)
    transitive_verbs = [
        ("sees", {"transitive": True, "perception": True, "tense": "present"}),
        ("saw", {"transitive": True, "perception": True, "tense": "past"}),
        ("hears", {"transitive": True, "perception": True, "tense": "present"}),
        ("heard", {"transitive": True, "perception": True, "tense": "past"}),
        ("chases", {"transitive": True, "motion": True, "tense": "present"}),
        ("chased", {"transitive": True, "motion": True, "tense": "past"}),
        ("catches", {"transitive": True, "motion": True, "tense": "present"}),
        ("caught", {"transitive": True, "motion": True, "tense": "past"}),
        ("eats", {"transitive": True, "consumption": True, "tense": "present"}),
        ("ate", {"transitive": True, "consumption": True, "tense": "past"}),
        ("drinks", {"transitive": True, "consumption": True, "tense": "present"}),
        ("drank", {"transitive": True, "consumption": True, "tense": "past"}),
        ("loves", {"transitive": True, "emotion": True, "tense": "present"}),
        ("loved", {"transitive": True, "emotion": True, "tense": "past"}),
        ("hates", {"transitive": True, "emotion": True, "tense": "present"}),
        ("hated", {"transitive": True, "emotion": True, "tense": "past"}),
        ("reads", {"transitive": True, "cognitive": True, "tense": "present"}),
        ("read", {"transitive": True, "cognitive": True, "tense": "past"}),
        ("writes", {"transitive": True, "cognitive": True, "tense": "present"}),
        ("wrote", {"transitive": True, "cognitive": True, "tense": "past"}),
    ]
    for i, (word, features) in enumerate(transitive_verbs):
        lexicon[word] = create_word_entry(idx, "VERB",
                                          context_area=MOTOR,
                                          context_index=i,
                                          features=features)
        idx += 1
    
    # Verbs - Intransitive
    intransitive_verbs = [
        ("runs", {"transitive": False, "motion": True, "tense": "present"}),
        ("ran", {"transitive": False, "motion": True, "tense": "past"}),
        ("walks", {"transitive": False, "motion": True, "tense": "present"}),
        ("walked", {"transitive": False, "motion": True, "tense": "past"}),
        ("jumps", {"transitive": False, "motion": True, "tense": "present"}),
        ("jumped", {"transitive": False, "motion": True, "tense": "past"}),
        ("sleeps", {"transitive": False, "state": True, "tense": "present"}),
        ("slept", {"transitive": False, "state": True, "tense": "past"}),
        ("sits", {"transitive": False, "posture": True, "tense": "present"}),
        ("sat", {"transitive": False, "posture": True, "tense": "past"}),
        ("stands", {"transitive": False, "posture": True, "tense": "present"}),
        ("stood", {"transitive": False, "posture": True, "tense": "past"}),
    ]
    for i, (word, features) in enumerate(intransitive_verbs):
        lexicon[word] = create_word_entry(idx, "VERB",
                                          context_area=MOTOR,
                                          context_index=len(transitive_verbs) + i,
                                          features=features)
        idx += 1
    
    # Verbs - Ditransitive
    ditransitive_verbs = [
        ("gives", {"transitive": True, "ditransitive": True, "transfer": True, "tense": "present"}),
        ("gave", {"transitive": True, "ditransitive": True, "transfer": True, "tense": "past"}),
        ("sends", {"transitive": True, "ditransitive": True, "transfer": True, "tense": "present"}),
        ("sent", {"transitive": True, "ditransitive": True, "transfer": True, "tense": "past"}),
        ("tells", {"transitive": True, "ditransitive": True, "communication": True, "tense": "present"}),
        ("told", {"transitive": True, "ditransitive": True, "communication": True, "tense": "past"}),
        ("shows", {"transitive": True, "ditransitive": True, "perception": True, "tense": "present"}),
        ("showed", {"transitive": True, "ditransitive": True, "perception": True, "tense": "past"}),
    ]
    for i, (word, features) in enumerate(ditransitive_verbs):
        lexicon[word] = create_word_entry(idx, "VERB",
                                          context_area=MOTOR,
                                          context_index=len(transitive_verbs) + len(intransitive_verbs) + i,
                                          features=features)
        idx += 1
    
    # Adjectives
    adjectives = [
        ("big", {"size": True, "positive": True}),
        ("small", {"size": True, "negative": True}),
        ("large", {"size": True, "positive": True}),
        ("tiny", {"size": True, "negative": True}),
        ("fast", {"speed": True, "positive": True}),
        ("slow", {"speed": True, "negative": True}),
        ("quick", {"speed": True, "positive": True}),
        ("good", {"quality": True, "positive": True}),
        ("bad", {"quality": True, "negative": True}),
        ("beautiful", {"aesthetic": True, "positive": True}),
        ("ugly", {"aesthetic": True, "negative": True}),
        ("happy", {"emotion": True, "positive": True}),
        ("sad", {"emotion": True, "negative": True}),
        ("angry", {"emotion": True, "negative": True}),
        ("old", {"age": True}),
        ("young", {"age": True}),
        ("new", {"age": True}),
        ("red", {"color": True}),
        ("blue", {"color": True}),
        ("green", {"color": True}),
    ]
    for i, (word, features) in enumerate(adjectives):
        lexicon[word] = create_word_entry(idx, "ADJ", features=features)
        idx += 1
    
    # Adverbs
    adverbs = [
        ("quickly", {"manner": True, "speed": True}),
        ("slowly", {"manner": True, "speed": True}),
        ("carefully", {"manner": True}),
        ("happily", {"manner": True, "emotion": True}),
        ("sadly", {"manner": True, "emotion": True}),
        ("loudly", {"manner": True, "volume": True}),
        ("quietly", {"manner": True, "volume": True}),
        ("always", {"frequency": True}),
        ("never", {"frequency": True, "negative": True}),
        ("often", {"frequency": True}),
        ("sometimes", {"frequency": True}),
        ("here", {"location": True, "proximal": True}),
        ("there", {"location": True, "distal": True}),
        ("now", {"time": True, "present": True}),
        ("then", {"time": True, "past": True}),
    ]
    for i, (word, features) in enumerate(adverbs):
        lexicon[word] = create_word_entry(idx, "ADV", features=features)
        idx += 1
    
    # Prepositions
    prepositions = [
        ("in", {"spatial": True, "containment": True}),
        ("on", {"spatial": True, "contact": True}),
        ("at", {"spatial": True, "location": True}),
        ("to", {"spatial": True, "direction": True}),
        ("from", {"spatial": True, "source": True}),
        ("with", {"accompaniment": True}),
        ("without", {"accompaniment": True, "negative": True}),
        ("for", {"purpose": True}),
        ("about", {"topic": True}),
        ("by", {"agent": True}),
        ("of", {"possession": True}),
        ("under", {"spatial": True, "below": True}),
        ("over", {"spatial": True, "above": True}),
        ("between", {"spatial": True}),
        ("through", {"spatial": True, "path": True}),
    ]
    for i, (word, features) in enumerate(prepositions):
        lexicon[word] = create_word_entry(idx, "PREP", features=features)
        idx += 1
    
    return lexicon


LEXICON = build_lexicon()
LEX_SIZE = len(LEXICON)

# =============================================================================
# ROBUST GRAMMATICAL BRAIN
# =============================================================================

class RobustGrammaticalBrain(brain.Brain):
    """
    A robust grammatical brain with:
    - Multiple CORE areas for grammatical categories
    - SEQ area for learning word order from corpus
    - Error detection via assembly stability
    - Rich lexicon with semantic grounding
    """
    
    def __init__(self, 
                 p: float = 0.1,
                 LEX_k: int = 20,
                 CORE_k: int = 15,
                 non_LEX_n: int = 50000,
                 non_LEX_k: int = 100,
                 SEQ_n: int = 100000,
                 SEQ_k: int = 100,
                 CONTEXT_k: int = 50,
                 default_beta: float = 0.1,
                 LEX_beta: float = 0.5,
                 CORE_beta: float = 0.3,
                 recurrent_beta: float = 0.05,
                 interarea_beta: float = 0.2,
                 verbose: bool = False):
        
        brain.Brain.__init__(self, p)
        
        self.lexicon = LEXICON
        self.verbose = verbose
        self.LEX_k = LEX_k
        self.CORE_k = CORE_k
        
        # Track states
        self.fiber_states = defaultdict(lambda: defaultdict(set))
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        
        # Track training
        self.sentences_trained = 0
        self.word_order_stats = defaultdict(lambda: defaultdict(int))
        
        # =====================================================================
        # CREATE AREAS
        # =====================================================================
        
        # Main Lexicon (explicit)
        LEX_n = LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)
        
        # CORE areas (explicit) - one for each grammatical category
        # Each CORE area has slots for different grammatical features
        num_cores = 6  # NOUN, VERB, ADJ, DET, PREP, ADV
        for core_name in CORE_AREAS:
            self.add_explicit_area(core_name, CORE_k, CORE_k, CORE_beta,
                                   custom_inner_p=0.9, custom_out_p=0.5, custom_in_p=0.5)
        
        # Syntactic role areas (non-explicit)
        for area in SYNTACTIC_AREAS:
            self.add_area(area, non_LEX_n, non_LEX_k, default_beta)
        
        # Sequence area for word order (non-explicit, large)
        self.add_area(SEQ, SEQ_n, SEQ_k, default_beta)
        
        # Control areas (explicit)
        num_moods = len(MoodType)
        self.add_explicit_area(MOOD, num_moods * LEX_k, LEX_k, default_beta)
        
        num_tenses = len(TenseType)
        self.add_explicit_area(TENSE, num_tenses * LEX_k, LEX_k, default_beta)
        
        # Context areas (explicit) - for grounding
        num_visual = 30  # number of visual concepts
        self.add_explicit_area(VISUAL, num_visual * CONTEXT_k, CONTEXT_k, default_beta)
        
        num_motor = 50  # number of motor concepts
        self.add_explicit_area(MOTOR, num_motor * CONTEXT_k, CONTEXT_k, default_beta)
        
        num_abstract = 20  # number of abstract concepts
        self.add_explicit_area(ABSTRACT, num_abstract * CONTEXT_k, CONTEXT_k, default_beta)
        
        num_emotion = 10  # number of emotion concepts
        self.add_explicit_area(EMOTION, num_emotion * CONTEXT_k, CONTEXT_k, default_beta)
        
        # Error area (non-explicit) - unstable assemblies signal errors
        self.add_area(ERROR, non_LEX_n, non_LEX_k, default_beta * 0.5)  # Lower plasticity
        
        # =====================================================================
        # SET PLASTICITIES
        # =====================================================================
        
        custom_plasticities = defaultdict(list)
        
        # LEX <-> syntactic areas: strong
        for area in SYNTACTIC_AREAS:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
        
        # CORE <-> LEX: strong
        for core in CORE_AREAS:
            custom_plasticities[core].append((LEX, CORE_beta))
            custom_plasticities[LEX].append((core, CORE_beta))
        
        # CORE <-> SEQ: strong (for word order learning)
        for core in CORE_AREAS:
            custom_plasticities[core].append((SEQ, CORE_beta))
            custom_plasticities[SEQ].append((core, CORE_beta))
        
        # SEQ recurrent: moderate
        custom_plasticities[SEQ].append((SEQ, recurrent_beta))
        
        # MOOD -> SEQ: strong
        custom_plasticities[MOOD].append((SEQ, LEX_beta))
        
        # Context <-> LEX: moderate
        for ctx in CONTEXT_AREAS:
            custom_plasticities[ctx].append((LEX, interarea_beta))
            custom_plasticities[LEX].append((ctx, interarea_beta))
        
        self.update_plasticities(area_update_map=custom_plasticities)
        
        # Initialize states
        self.initialize_states()
        
        if verbose:
            print(f"Created RobustGrammaticalBrain:")
            print(f"  Lexicon size: {LEX_SIZE} words")
            print(f"  CORE areas: {CORE_AREAS}")
            print(f"  Syntactic areas: {SYNTACTIC_AREAS}")
    
    def initialize_states(self):
        """Initialize fiber and area inhibition states"""
        all_areas = list(self.area_by_name.keys())
        
        for from_area in all_areas:
            self.fiber_states[from_area] = defaultdict(set)
            for to_area in all_areas:
                self.fiber_states[from_area][to_area].add(0)  # Initially inhibited
        
        for area in all_areas:
            self.area_states[area].add(0)  # Initially inhibited
        
        # Disinhibit initial areas
        initial_areas = [LEX, SUBJ, VERB, SEQ, MOOD]
        for area in initial_areas:
            self.area_states[area].discard(0)
    
    def apply_fiber_rule(self, rule: FiberRule):
        """Apply a fiber rule"""
        if rule.action == INHIBIT:
            self.fiber_states[rule.area1][rule.area2].add(rule.index)
            self.fiber_states[rule.area2][rule.area1].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.fiber_states[rule.area1][rule.area2].discard(rule.index)
            self.fiber_states[rule.area2][rule.area1].discard(rule.index)
    
    def apply_area_rule(self, rule: AreaRule):
        """Apply an area rule"""
        if rule.action == INHIBIT:
            self.area_states[rule.area].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.area_states[rule.area].discard(rule.index)
    
    def apply_rule(self, rule):
        """Apply any rule"""
        if isinstance(rule, FiberRule):
            self.apply_fiber_rule(rule)
        elif isinstance(rule, AreaRule):
            self.apply_area_rule(rule)
    
    def activate_word(self, word: str):
        """Activate a word in LEX and its CORE"""
        if word not in self.lexicon:
            if self.verbose:
                print(f"Unknown word: {word}")
            return False
        
        lexeme = self.lexicon[word]
        area = self.area_by_name[LEX]
        k = area.k
        assembly_start = lexeme["index"] * k
        area.winners = np.array(list(range(assembly_start, assembly_start + k)), dtype=np.uint32)
        area.fix_assembly()
        
        # Also activate the corresponding CORE
        if "core_area" in lexeme:
            core_area = lexeme["core_area"]
            if core_area in self.area_by_name:
                core = self.area_by_name[core_area]
                # CORE has single assembly (all neurons)
                core.winners = np.array(list(range(core.k)), dtype=np.uint32)
                core.fix_assembly()
        
        # Activate context if available
        if lexeme.get("context_area") and lexeme.get("context_index") is not None:
            ctx_area = lexeme["context_area"]
            ctx_idx = lexeme["context_index"]
            if ctx_area in self.area_by_name:
                ctx = self.area_by_name[ctx_area]
                ctx_start = ctx_idx * ctx.k
                ctx_end = ctx_start + ctx.k
                if ctx_end <= ctx.n:
                    ctx.winners = np.array(list(range(ctx_start, ctx_end)), dtype=np.uint32)
                    ctx.fix_assembly()
        
        return True
    
    def get_word(self, area_name: str = LEX, min_overlap: float = 0.7) -> Optional[str]:
        """Get word from current assembly"""
        area_winners = self.area_by_name[area_name].winners
        if area_winners is None or len(area_winners) == 0:
            return None
        
        winners = set(area_winners)
        area_k = self.area_by_name[area_name].k
        threshold = min_overlap * area_k
        
        for word, lexeme in self.lexicon.items():
            word_index = lexeme["index"]
            word_assembly_start = word_index * area_k
            word_assembly = set(range(word_assembly_start, word_assembly_start + area_k))
            if len(winners & word_assembly) >= threshold:
                return word
        
        return None
    
    def get_active_core(self) -> Optional[str]:
        """Determine which CORE area is most active"""
        best_core = None
        best_activation = 0
        
        for core in CORE_AREAS:
            if core in self.area_by_name:
                winners = self.area_by_name[core].winners
                if winners is not None and len(winners) > best_activation:
                    best_activation = len(winners)
                    best_core = core
        
        return best_core
    
    def measure_assembly_stability(self, area_name: str, rounds: int = 3) -> float:
        """
        Measure assembly stability by projecting recurrently and checking overlap.
        Low stability (< 0.5) indicates possible parsing error.
        """
        if area_name not in self.area_by_name:
            return 0.0
        
        area = self.area_by_name[area_name]
        if area.winners is None or len(area.winners) == 0:
            return 0.0
        
        initial_winners = set(area.winners)
        
        # Temporarily disable plasticity
        old_plasticity = self.disable_plasticity
        self.disable_plasticity = True
        
        # Project recurrently
        for _ in range(rounds):
            self.project({}, {area_name: [area_name]})
        
        final_winners = set(area.winners) if area.winners is not None else set()
        
        self.disable_plasticity = old_plasticity
        
        if len(initial_winners) == 0:
            return 0.0
        
        overlap = len(initial_winners & final_winners) / len(initial_winners)
        return overlap
    
    def check_parse_validity(self) -> Tuple[bool, float]:
        """
        Check if current parse is valid by measuring assembly stability.
        Returns (is_valid, confidence).
        """
        stabilities = []
        
        for area in SYNTACTIC_AREAS:
            if area in self.area_by_name:
                winners = self.area_by_name[area].winners
                if winners is not None and len(winners) > 0:
                    stability = self.measure_assembly_stability(area)
                    stabilities.append(stability)
        
        if not stabilities:
            return True, 1.0  # No assemblies to check
        
        avg_stability = sum(stabilities) / len(stabilities)
        is_valid = avg_stability > 0.5
        
        return is_valid, avg_stability
    
    # =========================================================================
    # TRAINING: Learn word order from corpus
    # =========================================================================
    
    def train_word_order(self, sentences: List[str], 
                        mood: MoodType = MoodType.DECLARATIVE,
                        rounds_per_sentence: int = 5):
        """
        Train the SEQ area to learn word order from a corpus.
        
        The key mechanism:
        1. Activate MOOD -> SEQ (establish mood context)
        2. For each word: activate word -> CORE -> SEQ
        3. SEQ learns the sequence of COREs (grammatical categories)
        """
        for sentence in sentences:
            self._train_single_sentence(sentence, mood, rounds_per_sentence)
            self.sentences_trained += 1
        
        if self.verbose:
            print(f"Trained on {len(sentences)} sentences")
            print(f"Total sentences trained: {self.sentences_trained}")
    
    def _train_single_sentence(self, sentence: str, 
                               mood: MoodType,
                               proj_rounds: int):
        """Train on a single sentence"""
        words = sentence.lower().split()
        
        # Reset areas
        for area_name in self.area_by_name:
            if area_name not in [LEX] + CORE_AREAS:
                self.area_by_name[area_name].winners = np.array([], dtype=np.uint32)
        
        # 1. Activate MOOD -> SEQ
        mood_area = self.area_by_name[MOOD]
        mood_start = mood.value * mood_area.k
        mood_area.winners = np.array(list(range(mood_start, mood_start + mood_area.k)), dtype=np.uint32)
        mood_area.fix_assembly()
        
        self.project({}, {MOOD: [SEQ]})
        for _ in range(proj_rounds):
            self.project({}, {MOOD: [SEQ], SEQ: [SEQ]})
        
        # 2. Process each word
        prev_core = None
        for word in words:
            if word not in self.lexicon:
                continue
            
            lexeme = self.lexicon[word]
            core_area = lexeme.get("core_area")
            
            if core_area is None:
                continue
            
            # Activate word and its CORE
            self.activate_word(word)
            
            # Project: SEQ -> CORE (predict), CORE -> SEQ (update)
            self.project({}, {SEQ: [core_area], core_area: [SEQ]})
            
            # Let CORE update SEQ
            self.project({}, {core_area: [SEQ]})
            for _ in range(proj_rounds):
                self.project({}, {core_area: [SEQ], SEQ: [SEQ]})
            
            # Track word order statistics
            if prev_core:
                self.word_order_stats[prev_core][core_area] += 1
            
            prev_core = core_area
        
        # Unfix assemblies
        for area_name in CORE_AREAS + [MOOD]:
            if area_name in self.area_by_name:
                self.area_by_name[area_name].unfix_assembly()
    
    def predict_next_category(self) -> Optional[str]:
        """
        Given current SEQ state, predict the next grammatical category.
        """
        self.disable_plasticity = True
        
        best_core = None
        best_activation = 0
        
        for core in CORE_AREAS:
            if core in self.area_by_name:
                # Project SEQ -> CORE
                self.project({}, {SEQ: [core]})
                
                winners = self.area_by_name[core].winners
                if winners is not None:
                    activation = len(winners)
                    if activation > best_activation:
                        best_activation = activation
                        best_core = core
        
        self.disable_plasticity = False
        return best_core
    
    # =========================================================================
    # GENERATION
    # =========================================================================
    
    def generate_sentence(self, 
                         mood: MoodType = MoodType.DECLARATIVE,
                         max_words: int = 10,
                         temperature: float = 1.0) -> str:
        """
        Generate a sentence using learned word order.
        """
        generated_words = []
        
        # Reset areas
        for area_name in self.area_by_name:
            self.area_by_name[area_name].winners = np.array([], dtype=np.uint32)
        
        # Activate MOOD
        mood_area = self.area_by_name[MOOD]
        mood_start = mood.value * mood_area.k
        mood_area.winners = np.array(list(range(mood_start, mood_start + mood_area.k)), dtype=np.uint32)
        mood_area.fix_assembly()
        
        # Project MOOD -> SEQ
        self.project({}, {MOOD: [SEQ]})
        
        for _ in range(max_words):
            # Predict next category
            next_core = self.predict_next_category()
            
            if next_core is None:
                break
            
            # Sample a word from that category
            word = self._sample_word_from_category(next_core, temperature)
            
            if word is None:
                break
            
            generated_words.append(word)
            
            # Update SEQ with the chosen word
            self.activate_word(word)
            self.project({}, {next_core: [SEQ]})
            self.project({}, {SEQ: [SEQ]})
        
        return " ".join(generated_words)
    
    def _sample_word_from_category(self, core_area: str, temperature: float = 1.0) -> Optional[str]:
        """Sample a word from a grammatical category"""
        # Map CORE area to word type
        core_to_type = {
            NOUN_CORE: "NOUN",
            VERB_CORE: "VERB",
            ADJ_CORE: "ADJ",
            DET_CORE: "DET",
            PREP_CORE: "PREP",
            ADV_CORE: "ADV",
        }
        
        word_type = core_to_type.get(core_area)
        if word_type is None:
            return None
        
        # Get all words of this type
        candidates = [w for w, l in self.lexicon.items() if l["type"] == word_type]
        
        if not candidates:
            return None
        
        # Random sampling (could be improved with context)
        return random.choice(candidates)
    
    def get_word_order_stats(self) -> Dict:
        """Get learned word order statistics"""
        return dict(self.word_order_stats)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the robust grammatical brain"""
    print("=" * 70)
    print("ROBUST GRAMMATICAL BRAIN DEMO")
    print("=" * 70)
    print()
    
    # Create brain
    brain_obj = RobustGrammaticalBrain(
        p=0.1,
        LEX_k=15,
        CORE_k=10,
        non_LEX_n=10000,
        non_LEX_k=50,
        SEQ_n=20000,
        SEQ_k=50,
        verbose=True
    )
    
    # Training corpus - SVO sentences
    training_corpus = [
        "the dog chases the cat",
        "the cat sees the bird",
        "a man reads a book",
        "the woman loves the child",
        "the boy eats food",
        "a girl drinks water",
        "the dog runs quickly",
        "the cat sleeps quietly",
        "the big dog chases the small cat",
        "a happy child runs fast",
    ]
    
    print("\nTraining corpus:")
    for s in training_corpus:
        print(f"  - {s}")
    
    print("\nTraining on word order...")
    brain_obj.train_word_order(training_corpus, 
                               mood=MoodType.DECLARATIVE,
                               rounds_per_sentence=3)
    
    print("\nLearned word order statistics:")
    stats = brain_obj.get_word_order_stats()
    for from_core, to_cores in stats.items():
        print(f"  {from_core}:")
        for to_core, count in to_cores.items():
            print(f"    -> {to_core}: {count}")
    
    print("\n" + "=" * 70)
    print("GENERATION")
    print("=" * 70)
    
    print("\nGenerating sentences:")
    for i in range(5):
        sentence = brain_obj.generate_sentence(
            mood=MoodType.DECLARATIVE,
            max_words=6
        )
        print(f"  {i+1}. {sentence}")
    
    print("\nDone!")


if __name__ == "__main__":
    demo()

