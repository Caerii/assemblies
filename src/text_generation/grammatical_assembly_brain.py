#!/usr/bin/env python3
"""
Grammatical Assembly Brain
==========================

A proper implementation of grammatical structure learning using Assembly Calculus,
based on the architecture from parser.py, recursive_parser.py, and learner.py.

Key Concepts from the Original Code:
1. **Fiber States**: Connections between areas can be INHIBITED or DISINHIBITED
2. **Area States**: Areas can be INHIBITED (won't fire) or DISINHIBITED (can fire)
3. **PRE_RULES / POST_RULES**: Each word type has rules that control which fibers/areas activate
4. **Explicit vs Non-Explicit Areas**: LEX is explicit (fixed assemblies), syntactic areas are not
5. **Readout**: Recover parse structure by projecting between areas

Architecture:
- LEX: Lexicon area with explicit word assemblies
- SUBJ: Subject noun phrase
- OBJ: Object noun phrase
- VERB: Verb phrase
- DET: Determiner
- ADJ: Adjective
- PREP: Preposition
- PREP_P: Prepositional phrase
- SEQ: Sequence area (for word order)
- MOOD: Sentence mood (declarative, interrogative, etc.)
"""

from collections import namedtuple, defaultdict
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
import numpy as np

import brain

# =============================================================================
# BRAIN AREAS
# =============================================================================

# Core areas
LEX = "LEX"           # Lexicon - explicit area with word assemblies
PHON = "PHON"         # Phonological form

# Syntactic areas (non-explicit, form assemblies dynamically)
SUBJ = "SUBJ"         # Subject
OBJ = "OBJ"           # Object
VERB = "VERB"         # Verb phrase
DET = "DET"           # Determiner
ADJ = "ADJ"           # Adjective
PREP = "PREP"         # Preposition
PREP_P = "PREP_P"     # Prepositional phrase
ADVERB = "ADVERB"     # Adverb

# Sequence/Control areas
SEQ = "SEQ"           # Sequence memory
MOOD = "MOOD"         # Sentence mood

# All areas
AREAS = [LEX, SUBJ, OBJ, VERB, DET, ADJ, PREP, PREP_P, ADVERB]
EXPLICIT_AREAS = [LEX]
RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]

# =============================================================================
# ACTIONS FOR FIBER/AREA CONTROL
# =============================================================================

DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"
ACTIVATE_ONLY = "ACTIVATE_ONLY"

# =============================================================================
# RULES (from parser.py)
# =============================================================================

AreaRule = namedtuple("AreaRule", ["action", "area", "index"])
FiberRule = namedtuple("FiberRule", ["action", "area1", "area2", "index"])


def generic_noun(index: int) -> dict:
    """Rules for processing a noun"""
    return {
        "index": index,
        "type": "NOUN",
        "PRE_RULES": [
            # Enable LEX -> SUBJ/OBJ/PREP_P
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            FiberRule(DISINHIBIT, LEX, PREP_P, 0),
            # Enable DET -> noun areas
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
            FiberRule(DISINHIBIT, DET, OBJ, 0),
            FiberRule(DISINHIBIT, DET, PREP_P, 0),
            # Enable ADJ -> noun areas
            FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, OBJ, 0),
            FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
            # Enable VERB -> OBJ
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
        ],
        "POST_RULES": [
            # After noun, inhibit modifiers
            AreaRule(INHIBIT, DET, 0),
            AreaRule(INHIBIT, ADJ, 0),
            # Inhibit LEX -> noun areas
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            FiberRule(INHIBIT, LEX, PREP_P, 0),
            FiberRule(INHIBIT, ADJ, SUBJ, 0),
            FiberRule(INHIBIT, ADJ, OBJ, 0),
            FiberRule(INHIBIT, DET, SUBJ, 0),
            FiberRule(INHIBIT, DET, OBJ, 0),
        ],
    }


def generic_verb(index: int, transitive: bool = True) -> dict:
    """Rules for processing a verb"""
    post_rules = [
        FiberRule(INHIBIT, LEX, VERB, 0),
        AreaRule(INHIBIT, SUBJ, 0),
        AreaRule(INHIBIT, ADVERB, 0),
    ]
    if transitive:
        post_rules.append(AreaRule(DISINHIBIT, OBJ, 0))
    
    return {
        "index": index,
        "type": "VERB",
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADVERB, 0),
            AreaRule(DISINHIBIT, ADVERB, 1),
        ],
        "POST_RULES": post_rules,
    }


def generic_determiner(index: int) -> dict:
    """Rules for processing a determiner (the, a, an)"""
    return {
        "index": index,
        "type": "DET",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, DET, 0),
            FiberRule(DISINHIBIT, LEX, DET, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, DET, 0),
        ],
    }


def generic_adjective(index: int) -> dict:
    """Rules for processing an adjective"""
    return {
        "index": index,
        "type": "ADJ",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADJ, 0),
            FiberRule(DISINHIBIT, LEX, ADJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADJ, 0),
        ],
    }


def generic_preposition(index: int) -> dict:
    """Rules for processing a preposition"""
    return {
        "index": index,
        "type": "PREP",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, PREP, 0),
            FiberRule(DISINHIBIT, LEX, PREP, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, PREP, 0),
            AreaRule(DISINHIBIT, PREP_P, 0),
        ],
    }


def generic_adverb(index: int) -> dict:
    """Rules for processing an adverb"""
    return {
        "index": index,
        "type": "ADVERB",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, LEX, ADVERB, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADVERB, 0),
            AreaRule(INHIBIT, ADVERB, 1),
        ],
    }


# =============================================================================
# LEXICON
# =============================================================================

LEXEME_DICT = {
    # Determiners
    "the": generic_determiner(0),
    "a": generic_determiner(1),
    "an": generic_determiner(2),
    
    # Nouns
    "dog": generic_noun(3),
    "cat": generic_noun(4),
    "mouse": generic_noun(5),
    "bird": generic_noun(6),
    "man": generic_noun(7),
    "woman": generic_noun(8),
    "child": generic_noun(9),
    "book": generic_noun(10),
    
    # Transitive verbs
    "chases": generic_verb(11, transitive=True),
    "sees": generic_verb(12, transitive=True),
    "loves": generic_verb(13, transitive=True),
    "bites": generic_verb(14, transitive=True),
    "reads": generic_verb(15, transitive=True),
    
    # Intransitive verbs
    "runs": generic_verb(16, transitive=False),
    "jumps": generic_verb(17, transitive=False),
    "sleeps": generic_verb(18, transitive=False),
    
    # Adjectives
    "big": generic_adjective(19),
    "small": generic_adjective(20),
    "fast": generic_adjective(21),
    "slow": generic_adjective(22),
    
    # Prepositions
    "in": generic_preposition(23),
    "on": generic_preposition(24),
    "with": generic_preposition(25),
    
    # Adverbs
    "quickly": generic_adverb(26),
    "slowly": generic_adverb(27),
}

LEX_SIZE = len(LEXEME_DICT)

# Readout rules: which areas can be read from which
READOUT_RULES = {
    VERB: [LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ],
    SUBJ: [LEX, DET, ADJ, PREP_P],
    OBJ: [LEX, DET, ADJ, PREP_P],
    PREP_P: [LEX, PREP, ADJ, DET],
    PREP: [LEX],
    ADJ: [LEX],
    DET: [LEX],
    ADVERB: [LEX],
    LEX: [],
}


# =============================================================================
# PARSER BRAIN (from parser.py)
# =============================================================================

class GrammaticalBrain(brain.Brain):
    """
    A brain that learns grammatical structure through Assembly Calculus.
    
    Based on ParserBrain from parser.py but extended for generation.
    """
    
    def __init__(self, p: float = 0.1, 
                 LEX_k: int = 20,
                 non_LEX_n: int = 100000,
                 non_LEX_k: int = 100,
                 default_beta: float = 0.2,
                 LEX_beta: float = 1.0,
                 recurrent_beta: float = 0.05,
                 interarea_beta: float = 0.5,
                 verbose: bool = False):
        
        brain.Brain.__init__(self, p)
        
        self.lexeme_dict = LEXEME_DICT
        self.all_areas = AREAS
        self.recurrent_areas = RECURRENT_AREAS
        self.initial_areas = [LEX, SUBJ, VERB]  # Initially disinhibited
        self.readout_rules = READOUT_RULES
        self.verbose = verbose
        
        # Fiber and area states for inhibition control
        self.fiber_states = defaultdict(lambda: defaultdict(set))
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        
        # Create areas
        LEX_n = LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)
        
        self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DET, non_LEX_n, LEX_k, default_beta)  # DET uses LEX_k
        self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)
        
        # Set custom plasticities (from parser.py)
        custom_plasticities = defaultdict(list)
        for area in RECURRENT_AREAS:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other_area in RECURRENT_AREAS:
                if other_area != area:
                    custom_plasticities[area].append((other_area, interarea_beta))
        
        self.update_plasticities(area_update_map=custom_plasticities)
        
        # Initialize states
        self.initialize_states()
    
    def initialize_states(self):
        """Initialize fiber and area states"""
        for from_area in self.all_areas:
            self.fiber_states[from_area] = defaultdict(set)
            for to_area in self.all_areas:
                self.fiber_states[from_area][to_area].add(0)  # Initially inhibited
        
        for area in self.all_areas:
            self.area_states[area].add(0)  # Initially inhibited
        
        for area in self.initial_areas:
            self.area_states[area].discard(0)  # Disinhibit initial areas
    
    def apply_fiber_rule(self, rule: FiberRule):
        """Apply a fiber rule (inhibit/disinhibit connection)"""
        if rule.action == INHIBIT:
            self.fiber_states[rule.area1][rule.area2].add(rule.index)
            self.fiber_states[rule.area2][rule.area1].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.fiber_states[rule.area1][rule.area2].discard(rule.index)
            self.fiber_states[rule.area2][rule.area1].discard(rule.index)
    
    def apply_area_rule(self, rule: AreaRule):
        """Apply an area rule (inhibit/disinhibit area)"""
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
    
    def get_project_map(self) -> Dict[str, Set[str]]:
        """Get current projection map based on inhibition states"""
        proj_map = defaultdict(set)
        for area1 in self.all_areas:
            if len(self.area_states[area1]) == 0:  # Area is disinhibited
                for area2 in self.all_areas:
                    if area1 == LEX and area2 == LEX:
                        continue  # Don't project LEX to itself
                    if len(self.area_states[area2]) == 0:  # Target disinhibited
                        if len(self.fiber_states[area1][area2]) == 0:  # Fiber disinhibited
                            # Check if area has winners (handle both list and numpy array)
                            area1_winners = self.area_by_name[area1].winners
                            area2_winners = self.area_by_name[area2].winners
                            has_area1_winners = len(area1_winners) > 0 if hasattr(area1_winners, '__len__') else bool(area1_winners)
                            has_area2_winners = len(area2_winners) > 0 if hasattr(area2_winners, '__len__') else bool(area2_winners)
                            
                            if has_area1_winners:
                                proj_map[area1].add(area2)
                            if has_area2_winners:
                                proj_map[area2].add(area2)  # Recurrent
        return proj_map
    
    def parse_project(self):
        """Execute one projection step"""
        project_map = self.get_project_map()
        self.remember_fibers(project_map)
        self.project({}, project_map)
    
    def remember_fibers(self, project_map):
        """Remember which fibers were activated (for readout)"""
        for from_area, to_areas in project_map.items():
            self.activated_fibers[from_area].update(to_areas)
    
    def activate_word(self, word: str):
        """Activate a word in LEX"""
        if word not in self.lexeme_dict:
            raise ValueError(f"Unknown word: {word}")
        
        lexeme = self.lexeme_dict[word]
        area = self.area_by_name[LEX]
        k = area.k
        assembly_start = lexeme["index"] * k
        # Use numpy array for compatibility with brain.py
        area.winners = np.array(list(range(assembly_start, assembly_start + k)), dtype=np.uint32)
        area.fix_assembly()
    
    def get_word(self, area_name: str = LEX, min_overlap: float = 0.7) -> Optional[str]:
        """Get word from current assembly in area"""
        area_winners = self.area_by_name[area_name].winners
        if area_winners is None or len(area_winners) == 0:
            return None
        
        winners = set(area_winners)
        area_k = self.area_by_name[area_name].k
        threshold = min_overlap * area_k
        
        for word, lexeme in self.lexeme_dict.items():
            word_index = lexeme["index"]
            word_assembly_start = word_index * area_k
            word_assembly = set(range(word_assembly_start, word_assembly_start + area_k))
            if len(winners & word_assembly) >= threshold:
                return word
        
        return None
    
    def parse_sentence(self, sentence: str, project_rounds: int = 20) -> Dict:
        """Parse a sentence and return its structure"""
        self.initialize_states()
        words = sentence.lower().split()
        
        for word in words:
            if word not in self.lexeme_dict:
                print(f"Unknown word: {word}")
                continue
            
            lexeme = self.lexeme_dict[word]
            
            # Activate word in LEX
            self.activate_word(word)
            if self.verbose:
                print(f"Activated: {word}")
            
            # Apply PRE_RULES
            for rule in lexeme["PRE_RULES"]:
                self.apply_rule(rule)
            
            # Get projection map and fix/unfix assemblies
            proj_map = self.get_project_map()
            for area in proj_map:
                if area not in proj_map.get(LEX, set()):
                    self.area_by_name[area].fix_assembly()
                elif area != LEX:
                    self.area_by_name[area].unfix_assembly()
                    self.area_by_name[area].winners = []
            
            # Project for multiple rounds
            for _ in range(project_rounds):
                self.parse_project()
            
            # Apply POST_RULES
            for rule in lexeme["POST_RULES"]:
                self.apply_rule(rule)
        
        # Readout
        return self.readout()
    
    def readout(self) -> Dict:
        """Read out the parse structure (simplified, non-recursive)"""
        self.disable_plasticity = True
        for area in self.all_areas:
            self.area_by_name[area].unfix_assembly()
        
        dependencies = []
        activated_fibers = {k: list(v) for k, v in self.activated_fibers.items()}
        
        # Simple readout: just check which words are in which areas
        # Project from VERB to LEX to get the verb
        if VERB in activated_fibers:
            try:
                self.project({}, {VERB: [LEX]})
                verb_word = self.get_word(LEX)
                
                # Project from SUBJ to LEX to get subject
                if SUBJ in activated_fibers:
                    self.project({}, {SUBJ: [LEX]})
                    subj_word = self.get_word(LEX)
                    if verb_word and subj_word:
                        dependencies.append({
                            "head": verb_word,
                            "dependent": subj_word,
                            "relation": "SUBJ"
                        })
                
                # Project from OBJ to LEX to get object
                if OBJ in activated_fibers:
                    self.project({}, {OBJ: [LEX]})
                    obj_word = self.get_word(LEX)
                    if verb_word and obj_word:
                        dependencies.append({
                            "head": verb_word,
                            "dependent": obj_word,
                            "relation": "OBJ"
                        })
            except Exception as e:
                print(f"Readout error: {e}")
        
        self.disable_plasticity = False
        
        return {"dependencies": dependencies}
    
    def generate_sentence(self, structure: str = "SVO", 
                         subject: str = None,
                         verb: str = None,
                         obj: str = None) -> str:
        """Generate a sentence with given structure"""
        # Get random words if not specified
        nouns = [w for w, l in self.lexeme_dict.items() if l.get("type") == "NOUN"]
        verbs = [w for w, l in self.lexeme_dict.items() if l.get("type") == "VERB"]
        
        if subject is None:
            subject = np.random.choice(nouns)
        if verb is None:
            verb = np.random.choice(verbs)
        if obj is None:
            obj = np.random.choice([n for n in nouns if n != subject])
        
        # Build sentence based on structure
        if structure == "SVO":
            return f"{subject} {verb} {obj}"
        elif structure == "SOV":
            return f"{subject} {obj} {verb}"
        elif structure == "VSO":
            return f"{verb} {subject} {obj}"
        elif structure == "SV":  # Intransitive
            return f"{subject} {verb}"
        else:
            return f"{subject} {verb} {obj}"


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the grammatical brain"""
    print("=" * 70)
    print("GRAMMATICAL ASSEMBLY BRAIN DEMO")
    print("=" * 70)
    print()
    
    # Create brain with smaller parameters for testing
    brain_obj = GrammaticalBrain(
        p=0.1, 
        LEX_k=20, 
        non_LEX_n=10000,  # Smaller for testing
        non_LEX_k=50,
        verbose=False  # Less verbose
    )
    print("Created GrammaticalBrain")
    print(f"  Vocabulary size: {len(LEXEME_DICT)}")
    print(f"  Areas: {AREAS}")
    print()
    
    # Parse sentences with fewer rounds
    sentences = [
        "the dog chases the cat",
        "a big cat runs",
    ]
    
    for sentence in sentences:
        print(f"\nParsing: '{sentence}'")
        print("-" * 50)
        try:
            result = brain_obj.parse_sentence(sentence, project_rounds=5)
            print(f"Dependencies: {result['dependencies']}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Generate sentences
    print("\n" + "=" * 70)
    print("GENERATION")
    print("=" * 70)
    
    for structure in ["SVO", "SV"]:
        sentence = brain_obj.generate_sentence(structure=structure)
        print(f"Generated ({structure}): {sentence}")


if __name__ == "__main__":
    demo()

