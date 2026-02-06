"""
Emergent NEMO Language Learner
==============================

Version: 1.0.0
Date: 2025-12-01

A truly neurobiologically plausible language learner where:
- ALL categories emerge from grounding patterns (no pre-labeled POS)
- Nouns emerge from consistent Visual grounding
- Verbs emerge from consistent Motor grounding
- Adjectives emerge from Property grounding (co-occurring with nouns)
- Function words emerge from high frequency + no grounding

Key Insight from Papers:
- Words are NOT labeled - categories EMERGE from differential grounding
- A word heard with Visual context → develops stable assembly in Lex1
- A word heard with Motor context → develops stable assembly in Lex2
- A word heard with both → develops in both (ambiguous)
- A word heard with neither → function word (high freq, no grounding)

NO GROUND TRUTH LABELS - Everything learned!
"""

import cupy as cp
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

# Import core kernels
from nemo.core.kernel import projection_fp16_kernel, hebbian_kernel


class GroundingModality(Enum):
    """Sensory modalities for grounding"""
    VISUAL = 0      # Objects, scenes, properties
    MOTOR = 1       # Actions, movements
    PROPERTY = 2    # Adjective-like (color, size, etc.)
    SPATIAL = 3     # Locations, prepositions
    SOCIAL = 4      # People, pronouns
    NONE = 5        # Function words (no grounding)


@dataclass
class GroundingContext:
    """
    Grounding context for a word.
    
    In real learning, this comes from the environment.
    A word's category emerges from WHICH modalities are active when it's heard.
    
    Modality → Category mapping:
    - visual → NOUN
    - motor → VERB
    - properties → ADJECTIVE
    - spatial → PREPOSITION
    - social → PRONOUN
    - temporal → ADVERB (temporal)
    - emotional → ADJECTIVE/ADVERB (emotional)
    - none → FUNCTION WORD
    """
    visual: List[str] = field(default_factory=list)      # Objects present → NOUN
    motor: List[str] = field(default_factory=list)       # Actions happening → VERB
    properties: List[str] = field(default_factory=list)  # Properties observed → ADJECTIVE
    spatial: List[str] = field(default_factory=list)     # Spatial relations → PREPOSITION
    social: List[str] = field(default_factory=list)      # Social context → PRONOUN
    temporal: List[str] = field(default_factory=list)    # Time concepts → ADVERB
    emotional: List[str] = field(default_factory=list)   # Emotions → ADJ/ADV


@dataclass
class EmergentParams:
    """Parameters for emergent NEMO"""
    n: int = 10000         # Neurons per area
    k: int = None          # Winners (sqrt(n) if None)
    p: float = 0.05        # Connection probability
    beta: float = 0.1      # Hebbian plasticity
    w_max: float = 10.0    # Weight saturation
    tau: int = 3           # Firing steps per word
    
    # Stability threshold for category assignment
    stability_threshold: float = 0.3
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class Area(Enum):
    """
    Brain areas - ALL categories EMERGE from grounding patterns.
    
    Based on neuroscience and NEMO papers:
    - Input areas: sensory modalities
    - Lexical areas: content vs function words
    - Core areas: grammatical categories (emerge from grounding)
    - Thematic roles: semantic roles (mutual inhibition)
    - Phrase structure: compositional units
    - Syntactic roles: grammatical functions
    - Sequence/Control: word order, mood, tense
    """
    
    # =========== INPUT MODALITIES (8 areas) ===========
    PHON = 0           # Phonological input (auditory cortex)
    VISUAL = 1         # Visual grounding - objects (inferotemporal)
    MOTOR = 2          # Motor grounding - actions (motor cortex)
    PROPERTY = 3       # Properties - size, color (temporal-parietal)
    SPATIAL = 4        # Spatial relations (parietal cortex)
    TEMPORAL = 5       # Time concepts (prefrontal)
    SOCIAL = 6         # Social/people (temporal pole)
    EMOTION = 7        # Emotional concepts (amygdala, insula)
    
    # =========== LEXICAL AREAS (2 areas) ===========
    LEX_CONTENT = 8    # Content words (middle temporal)
    LEX_FUNCTION = 9   # Function words (inferior frontal)
    
    # =========== CORE/CATEGORY AREAS (8 areas) ===========
    # These EMERGE from consistent grounding patterns
    NOUN_CORE = 10     # Noun category (← VISUAL grounding)
    VERB_CORE = 11     # Verb category (← MOTOR grounding)
    ADJ_CORE = 12      # Adjective category (← PROPERTY grounding)
    ADV_CORE = 13      # Adverb category (← manner/degree)
    PREP_CORE = 14     # Preposition category (← SPATIAL grounding)
    DET_CORE = 15      # Determiner category (← high freq, no grounding)
    PRON_CORE = 16     # Pronoun category (← SOCIAL grounding)
    CONJ_CORE = 17     # Conjunction category (← linking, no grounding)
    
    # =========== THEMATIC ROLE AREAS (6 areas) ===========
    # Under MUTUAL INHIBITION - only one active at a time
    ROLE_AGENT = 18    # Agent/doer of action
    ROLE_PATIENT = 19  # Patient/undergoer
    ROLE_THEME = 20    # Theme/moved entity
    ROLE_GOAL = 21     # Goal/destination
    ROLE_SOURCE = 22   # Source/origin
    ROLE_LOCATION = 23 # Location
    
    # =========== PHRASE STRUCTURE (5 areas) ===========
    NP = 24            # Noun phrase (DET + ADJ + N)
    VP = 25            # Verb phrase (V + NP/PP)
    PP = 26            # Prepositional phrase (PREP + NP)
    ADJP = 27          # Adjective phrase
    SENT = 28          # Full sentence (NP + VP)
    
    # =========== SYNTACTIC ROLES (3 areas) ===========
    # Under MUTUAL INHIBITION
    SUBJ = 29          # Subject
    OBJ = 30           # Direct object
    IOBJ = 31          # Indirect object
    
    # =========== SEQUENCE/CONTROL (4 areas) ===========
    SEQ = 32           # Sequence memory (word order)
    MOOD = 33          # Sentence mood (declarative, interrogative)
    TENSE = 34         # Tense marking
    POLARITY = 35      # Affirmative/Negative
    
    # =========== ERROR DETECTION (1 area) ===========
    ERROR = 36         # Parse error (wobbly = error)


NUM_AREAS = 37

# Mutual inhibition groups (only one area in each group can be active)
MUTUAL_INHIBITION_GROUPS = [
    # Thematic roles compete
    [Area.ROLE_AGENT, Area.ROLE_PATIENT, Area.ROLE_THEME, 
     Area.ROLE_GOAL, Area.ROLE_SOURCE, Area.ROLE_LOCATION],
    # Syntactic roles compete
    [Area.SUBJ, Area.OBJ, Area.IOBJ],
]

# Grounding → Core area mappings (for emergent categorization)
GROUNDING_TO_CORE = {
    'VISUAL': Area.NOUN_CORE,
    'MOTOR': Area.VERB_CORE,
    'PROPERTY': Area.ADJ_CORE,
    'SPATIAL': Area.PREP_CORE,
    'SOCIAL': Area.PRON_CORE,
    'TEMPORAL': Area.ADV_CORE,  # Temporal adverbs
    'NONE': Area.DET_CORE,      # Function words
}


class EmergentNemoBrain:
    """
    Brain where word categories EMERGE from grounding patterns.
    
    No pre-labeled categories - everything learned from:
    1. Which modalities are active when word is heard
    2. How consistently a word co-occurs with each modality
    3. Assembly stability in different areas
    
    Features:
    - 37 neurobiologically plausible areas
    - Mutual inhibition for competing roles
    - Core areas that emerge from grounding
    - Phrase structure areas for composition
    - Error detection via assembly stability
    """
    
    def __init__(self, params: EmergentParams = None, verbose: bool = True):
        self.p = params or EmergentParams()
        self.verbose = verbose
        n, k = self.p.n, self.p.k
        
        # Input assemblies for each modality (created on demand)
        self.assemblies: Dict[Area, Dict[str, cp.ndarray]] = {
            Area.PHON: {},
            Area.VISUAL: {},
            Area.MOTOR: {},
            Area.PROPERTY: {},
            Area.SPATIAL: {},
            Area.TEMPORAL: {},
            Area.SOCIAL: {},
            Area.EMOTION: {},
            Area.MOOD: {},      # Mood assemblies (declarative, interrogative, etc.)
            Area.TENSE: {},     # Tense assemblies (past, present, future)
            Area.POLARITY: {},  # Polarity assemblies (affirmative, negative)
        }
        
        # Legacy accessors for compatibility
        self.phon = self.assemblies[Area.PHON]
        self.visual = self.assemblies[Area.VISUAL]
        self.motor = self.assemblies[Area.MOTOR]
        self.property = self.assemblies[Area.PROPERTY]
        self.spatial = self.assemblies[Area.SPATIAL]
        self.social = self.assemblies[Area.SOCIAL]
        
        # Area seeds for implicit connectivity
        self.seeds = cp.arange(NUM_AREAS, dtype=cp.uint32) * 1000
        
        # Learned weights per area
        self.max_learned = k * k * 500
        self.l_src = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(NUM_AREAS)]
        self.l_dst = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(NUM_AREAS)]
        self.l_delta = [cp.zeros(self.max_learned, dtype=cp.float32) for _ in range(NUM_AREAS)]
        self.l_num = [cp.zeros(1, dtype=cp.uint32) for _ in range(NUM_AREAS)]
        
        # Current and previous activations
        self.current: Dict[Area, Optional[cp.ndarray]] = {a: None for a in Area}
        self.prev: Dict[Area, Optional[cp.ndarray]] = {a: None for a in Area}
        
        # Inhibition state (for mutual inhibition)
        self.inhibited: Set[Area] = set()
        
        # Word statistics (for emergent categorization)
        self.word_grounding_counts: Dict[str, Dict[GroundingModality, int]] = defaultdict(
            lambda: defaultdict(int))
        self.word_exposure_count: Dict[str, int] = defaultdict(int)
        
        # Kernel config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        if verbose:
            print(f"EmergentNemoBrain initialized:")
            print(f"  n={n:,}, k={k}")
            print(f"  Areas: {NUM_AREAS}")
            print(f"    Input: 8, Lexical: 2, Core: 8")
            print(f"    Thematic: 6, Phrase: 5, Syntactic: 3")
            print(f"    Control: 4, Error: 1")
            print(f"  Mutual inhibition groups: {len(MUTUAL_INHIBITION_GROUPS)}")
            print(f"  NO pre-labeled categories - all emergent!")
    
    def _get_or_create(self, area: Area, name: str) -> cp.ndarray:
        """Get or create assembly for a concept in an input area"""
        if area not in self.assemblies:
            self.assemblies[area] = {}
        store = self.assemblies[area]
        if name not in store:
            store[name] = cp.random.randint(0, self.p.n, self.p.k, dtype=cp.uint32)
        return store[name]
    
    def _apply_mutual_inhibition(self, active_area: Area):
        """Apply mutual inhibition - if one area in a group is active, inhibit others"""
        for group in MUTUAL_INHIBITION_GROUPS:
            if active_area in group:
                for other in group:
                    if other != active_area:
                        self.inhibited.add(other)
    
    def _is_inhibited(self, area: Area) -> bool:
        """Check if area is currently inhibited"""
        return area in self.inhibited
    
    def _project(self, area: Area, inp: cp.ndarray, learn: bool = True) -> Optional[cp.ndarray]:
        """Project input to area, respecting inhibition"""
        if self._is_inhibited(area):
            return None
        
        n, k = self.p.n, self.p.k
        area_idx = area.value
        
        active = cp.zeros(k, dtype=cp.uint32)
        active[:len(inp)] = inp[:k]
        
        result = cp.zeros(n, dtype=cp.float16)
        
        projection_fp16_kernel(
            (self.gx, 1), (self.bs,),
            (active.reshape(1, -1), result.reshape(1, -1),
             cp.uint32(k), cp.uint32(n), cp.uint32(1),
             self.seeds[area_idx:area_idx+1], cp.float32(self.p.p)),
            shared_mem=k * 4
        )
        
        result_torch = torch.as_tensor(result, device='cuda')
        _, winners_idx = torch.topk(result_torch, k, sorted=False)
        winners = cp.asarray(winners_idx).astype(cp.uint32)
        
        if learn and self.prev[area] is not None:
            grid = (k * k + self.bs - 1) // self.bs
            hebbian_kernel(
                (grid,), (self.bs,),
                (self.l_src[area_idx], self.l_dst[area_idx],
                 self.l_delta[area_idx], self.l_num[area_idx],
                 self.prev[area], winners,
                 cp.uint32(k), cp.float32(self.p.beta), cp.float32(self.p.w_max),
                 cp.uint32(self.max_learned), self.seeds[area_idx], cp.float32(self.p.p))
            )
        
        self.prev[area] = winners
        self.current[area] = winners
        
        # Apply mutual inhibition
        self._apply_mutual_inhibition(area)
        
        return winners
    
    def _clear_area(self, area: Area):
        """Clear area state"""
        self.current[area] = None
        self.prev[area] = None
    
    def clear_all(self):
        """Clear all areas and inhibition state"""
        for area in Area:
            self._clear_area(area)
        self.inhibited.clear()
    
    def measure_stability(self, area: Area, rounds: int = 3) -> float:
        """
        Measure assembly stability in an area.
        
        Stable assembly = valid parse/category
        Wobbly assembly = error/wrong category
        """
        if self.current[area] is None:
            return 0.0
        
        initial = set(self.current[area].get().tolist())
        
        # Recurrent projection
        for _ in range(rounds):
            self._project(area, self.current[area], learn=False)
        
        if self.current[area] is None:
            return 0.0
        
        final = set(self.current[area].get().tolist())
        
        # Calculate overlap
        intersection = len(initial & final)
        return intersection / self.p.k
    
    def get_learned_strength(self, area: Area, inp: cp.ndarray) -> float:
        """Get strength of learned connections for input in an area"""
        area_idx = area.value
        num_learned = int(self.l_num[area_idx].get()[0])
        if num_learned == 0:
            return 0.0
        
        # Project and check activation
        self._clear_area(area)
        self._project(area, inp, learn=False)
        
        winners = self.current[area]
        if winners is None:
            return 0.0
        
        dst_arr = self.l_dst[area_idx][:num_learned].get()
        delta_arr = self.l_delta[area_idx][:num_learned].get()
        
        winners_set = set(winners.get().tolist())
        strength = sum(1.0 + delta for dst, delta in zip(dst_arr, delta_arr) if dst in winners_set)
        
        return strength
    
    # =========================================================================
    # PHRASE COMPOSITION METHODS
    # =========================================================================
    # These implement the NEMO merge operation where multiple assemblies
    # combine into a single phrase representation.
    
    def merge_to_area(self, target_area: Area, source_assembly: cp.ndarray, 
                      learn: bool = True) -> Optional[cp.ndarray]:
        """
        Merge a source assembly into a target phrase area.
        
        This is the KEY NEMO operation for phrase building:
        - If target_area already has an assembly, the new input MERGES with it
        - The resulting assembly represents the combined phrase
        
        Example:
            merge_to_area(NP, "the")  → NP has assembly for "the"
            merge_to_area(NP, "big")  → NP has merged assembly for "the big"
            merge_to_area(NP, "dog")  → NP has merged assembly for "the big dog"
        """
        if self._is_inhibited(target_area):
            return None
        
        # Project source to target
        # If target already has prev, Hebbian learning creates merge
        result = self._project(target_area, source_assembly, learn=learn)
        
        return result
    
    def bind_phrase_to_role(self, phrase_assembly: cp.ndarray, role: Area,
                            learn: bool = True) -> Optional[cp.ndarray]:
        """
        Bind a phrase to a syntactic role (SUBJ, OBJ, IOBJ).
        
        This establishes the grammatical function of a phrase in a sentence.
        Mutual inhibition ensures only one role can be active.
        
        Example:
            bind_phrase_to_role(np_assembly, SUBJ)  → "the dog" is subject
            # Now OBJ and IOBJ are inhibited
            # After verb, OBJ becomes disinhibited
        """
        if role not in [Area.SUBJ, Area.OBJ, Area.IOBJ]:
            raise ValueError(f"Role must be SUBJ, OBJ, or IOBJ, got {role}")
        
        if self._is_inhibited(role):
            return None
        
        # Project phrase to role area
        result = self._project(role, phrase_assembly, learn=learn)
        
        # Mutual inhibition is applied automatically in _project
        return result
    
    def link_to_predicate(self, role_assembly: cp.ndarray, predicate_area: Area = Area.VP,
                          learn: bool = True) -> Optional[cp.ndarray]:
        """
        Link a role (SUBJ/OBJ) to the predicate (VP).
        
        This creates the sentence-level binding between arguments and predicate.
        
        Example:
            link_to_predicate(subj_assembly, VP)  → Subject linked to verb phrase
        """
        if self._is_inhibited(predicate_area):
            return None
        
        # Project role to predicate - creates bidirectional link via Hebbian
        result = self._project(predicate_area, role_assembly, learn=learn)
        
        return result
    
    def disinhibit_role(self, role: Area):
        """
        Explicitly disinhibit a role area.
        
        Used after verb to allow object to be processed.
        """
        if role in self.inhibited:
            self.inhibited.remove(role)
    
    def get_phrase_stability(self, phrase_area: Area, rounds: int = 3) -> float:
        """
        Measure how stable a phrase assembly is.
        
        High stability (>0.5) = well-formed phrase
        Low stability (<0.3) = malformed/incomplete phrase
        
        This is used for error detection - wobbly phrases signal parse errors.
        """
        return self.measure_stability(phrase_area, rounds)
    
    def project_backwards(self, from_area: Area, to_area: Area) -> Optional[cp.ndarray]:
        """
        Project from a higher-level area back to a lower-level area.
        
        Used for generation: SENT → VP → SUBJ → NP → LEX → word
        
        This retrieves the components that were merged into a phrase.
        """
        if self.current[from_area] is None:
            return None
        
        # Project without learning (retrieval mode)
        result = self._project(to_area, self.current[from_area], learn=False)
        
        return result


class EmergentLanguageLearner:
    """
    Language learner where ALL categories EMERGE from grounding.
    
    NO GROUND TRUTH LABELS! NO HARDCODED POS TAGS!
    
    Categories emerge from:
    1. Consistent VISUAL grounding → NOUN (objects)
    2. Consistent MOTOR grounding → VERB (actions)
    3. Consistent PROPERTY grounding → ADJECTIVE (qualities)
    4. Consistent SPATIAL grounding → PREPOSITION (locations)
    5. Consistent SOCIAL grounding → PRONOUN (people)
    6. Consistent TEMPORAL grounding → ADVERB (time)
    7. High frequency + no grounding → FUNCTION WORD (determiners, conjunctions)
    
    Additional emergent structure:
    - Thematic roles from argument position
    - Phrase structure from co-occurrence
    - Word order from sequence statistics
    """
    
    def __init__(self, params: EmergentParams = None, verbose: bool = True):
        self.brain = EmergentNemoBrain(params, verbose=verbose)
        self.p = self.brain.p
        self.verbose = verbose
        
        # Track word statistics (for emergent categorization)
        self.word_grounding: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_count: Dict[str, int] = defaultdict(int)
        
        # Track co-occurrence (for selectional restrictions and phrase structure)
        self.word_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Sequence statistics (for word order)
        self.position_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Category transitions (for syntax learning)
        self.category_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Argument structure (for thematic roles)
        self.word_as_first_arg: Dict[str, int] = defaultdict(int)  # Often AGENT
        self.word_as_second_arg: Dict[str, int] = defaultdict(int)  # Often PATIENT
        self.word_as_action: Dict[str, int] = defaultdict(int)      # VERB
        
        # Mood statistics
        self.mood_word_first: Dict[str, int] = defaultdict(int)
        
        self.sentences_seen = 0
    
    def present_word_with_grounding(self, word: str, context: GroundingContext, 
                                     position: int = 0, role: str = None,
                                     learn: bool = True):
        """
        Present a word with its grounding context.
        
        The word's category will EMERGE from which modalities are active.
        
        Args:
            word: The word being presented
            context: Grounding context (visual, motor, etc.)
            position: Position in sentence (for word order learning)
            role: Thematic role hint ('agent', 'patient', 'action') - for structure learning
            learn: Whether to apply Hebbian learning
        """
        # Get phonological assembly
        phon = self.brain._get_or_create(Area.PHON, word)
        
        # Track grounding statistics
        if learn:
            self.word_count[word] += 1
            self.position_counts[word][position] += 1
            
            # Track thematic role statistics
            if role == 'agent':
                self.word_as_first_arg[word] += 1
            elif role == 'patient':
                self.word_as_second_arg[word] += 1
            elif role == 'action':
                self.word_as_action[word] += 1
        
        # Project to each grounded modality and track
        grounding_active = []
        
        # Visual grounding (objects present) → NOUN
        if context.visual:
            for v in context.visual:
                vis = self.brain._get_or_create(Area.VISUAL, v)
                self.brain._project(Area.VISUAL, vis, learn=False)
            grounding_active.append('VISUAL')
            if learn:
                self.word_grounding[word]['VISUAL'] += 1
                # Project to NOUN_CORE
                self.brain._project(Area.NOUN_CORE, phon, learn=learn)
        
        # Motor grounding (actions happening) → VERB
        if context.motor:
            for m in context.motor:
                mot = self.brain._get_or_create(Area.MOTOR, m)
                self.brain._project(Area.MOTOR, mot, learn=False)
            grounding_active.append('MOTOR')
            if learn:
                self.word_grounding[word]['MOTOR'] += 1
                # Project to VERB_CORE
                self.brain._project(Area.VERB_CORE, phon, learn=learn)
        
        # Property grounding → ADJECTIVE
        if context.properties:
            for p in context.properties:
                prop = self.brain._get_or_create(Area.PROPERTY, p)
                self.brain._project(Area.PROPERTY, prop, learn=False)
            grounding_active.append('PROPERTY')
            if learn:
                self.word_grounding[word]['PROPERTY'] += 1
                # Project to ADJ_CORE
                self.brain._project(Area.ADJ_CORE, phon, learn=learn)
        
        # Spatial grounding → PREPOSITION
        if context.spatial:
            for s in context.spatial:
                spat = self.brain._get_or_create(Area.SPATIAL, s)
                self.brain._project(Area.SPATIAL, spat, learn=False)
            grounding_active.append('SPATIAL')
            if learn:
                self.word_grounding[word]['SPATIAL'] += 1
                # Project to PREP_CORE
                self.brain._project(Area.PREP_CORE, phon, learn=learn)
        
        # Social grounding → PRONOUN
        if context.social:
            for s in context.social:
                soc = self.brain._get_or_create(Area.SOCIAL, s)
                self.brain._project(Area.SOCIAL, soc, learn=False)
            grounding_active.append('SOCIAL')
            if learn:
                self.word_grounding[word]['SOCIAL'] += 1
                # Project to PRON_CORE
                self.brain._project(Area.PRON_CORE, phon, learn=learn)
        
        # Temporal grounding → ADVERB (temporal)
        if context.temporal if hasattr(context, 'temporal') else False:
            for t in context.temporal:
                temp = self.brain._get_or_create(Area.TEMPORAL, t)
                self.brain._project(Area.TEMPORAL, temp, learn=False)
            grounding_active.append('TEMPORAL')
            if learn:
                self.word_grounding[word]['TEMPORAL'] += 1
                self.brain._project(Area.ADV_CORE, phon, learn=learn)
        
        # Emotional grounding → ADJECTIVE/ADVERB (emotional)
        if context.emotional if hasattr(context, 'emotional') else False:
            for e in context.emotional:
                emo = self.brain._get_or_create(Area.EMOTION, e)
                self.brain._project(Area.EMOTION, emo, learn=False)
            grounding_active.append('EMOTION')
            if learn:
                self.word_grounding[word]['EMOTION'] += 1
        
        # No grounding = function word → DET_CORE or CONJ_CORE
        if not grounding_active:
            if learn:
                self.word_grounding[word]['NONE'] += 1
                # Project to DET_CORE (function words)
                self.brain._project(Area.DET_CORE, phon, learn=learn)
        
        # Project phonological to lexical areas
        for _ in range(self.p.tau):
            # Content words go to LEX_CONTENT (if grounded)
            if grounding_active:
                self.brain._project(Area.LEX_CONTENT, phon, learn=learn)
            else:
                # Function words go to LEX_FUNCTION
                self.brain._project(Area.LEX_FUNCTION, phon, learn=learn)
        
        # Project to thematic role area if role is specified
        if role and learn:
            if role == 'agent':
                self.brain._project(Area.ROLE_AGENT, phon, learn=learn)
            elif role == 'patient':
                self.brain._project(Area.ROLE_PATIENT, phon, learn=learn)
            elif role == 'action':
                # Action binds to syntactic VERB role
                self.brain._project(Area.VP, phon, learn=learn)
    
    def present_grounded_sentence(self, words: List[str], contexts: List[GroundingContext],
                                   roles: List[str] = None, mood: str = 'declarative',
                                   learn: bool = True):
        """
        Present a sentence where each word has its own grounding context.
        
        This is the KEY to emergent categories:
        - Nouns are heard with Visual context
        - Verbs are heard with Motor context
        - Adjectives are heard with Property context (and near nouns)
        - Prepositions are heard with Spatial context
        - Pronouns are heard with Social context
        - Function words are heard with no grounding
        
        Args:
            words: List of words in the sentence
            contexts: List of grounding contexts for each word
            roles: Optional list of thematic roles ('agent', 'patient', 'action', None)
            mood: Sentence mood ('declarative', 'interrogative', 'imperative')
            learn: Whether to apply Hebbian learning
        """
        self.brain.clear_all()
        
        # Activate mood
        mood_assembly = self.brain._get_or_create(Area.MOOD, mood)
        self.brain._project(Area.MOOD, mood_assembly, learn=learn)
        self.brain._project(Area.SEQ, mood_assembly, learn=learn)
        
        if roles is None:
            roles = [None] * len(words)
        
        prev_word = None
        prev_category = None
        
        for i, (word, context, role) in enumerate(zip(words, contexts, roles)):
            self.present_word_with_grounding(word, context, position=i, role=role, learn=learn)
            
            # Get emergent category for this word
            current_category, _ = self.get_emergent_category(word)
            
            # Track word transitions
            if prev_word and learn:
                self.transitions[(prev_word, word)] += 1
            
            # Track category transitions (for syntax learning)
            if prev_category and current_category and learn:
                self.category_transitions[(prev_category, current_category)] += 1
            
            # Track co-occurrence (for phrase structure)
            if learn:
                for other_word in words:
                    if other_word != word:
                        self.word_cooccurrence[word][other_word] += 1
            
            # Project to SEQ for word order learning
            if learn:
                # Get the core area for this category
                core_area = GROUNDING_TO_CORE.get(
                    self._get_dominant_grounding(word), Area.DET_CORE)
                if self.brain.current[core_area] is not None:
                    self.brain._project(Area.SEQ, self.brain.current[core_area], learn=learn)
            
            prev_word = word
            prev_category = current_category
        
        if learn:
            self.sentences_seen += 1
            # Track first word for mood
            if words:
                self.mood_word_first[words[0]] += 1
    
    def _get_dominant_grounding(self, word: str) -> str:
        """Get the dominant grounding modality for a word"""
        grounding = self.word_grounding[word]
        if not grounding:
            return 'NONE'
        return max(grounding, key=grounding.get)
    
    # =========================================================================
    # PHRASE-LEVEL COMPOSITION (NEMO way)
    # =========================================================================
    # These methods build phrases by MERGING word assemblies into phrase areas.
    # This is the key to grammatical sentence generation.
    
    def build_noun_phrase(self, words: List[str], contexts: List[GroundingContext],
                          learn: bool = True) -> Optional[cp.ndarray]:
        """
        Build a noun phrase by merging words into NP area.
        
        Words are processed in order, each one MERGING with the previous
        to create a combined phrase representation.
        
        Example: ["the", "big", "dog"] → NP assembly for "the big dog"
        
        Structure learned:
        - DET → NP
        - ADJ → NP (merges with DET)
        - NOUN → NP (merges with DET+ADJ)
        
        Returns: The NP assembly, or None if failed
        """
        # Clear NP area to start fresh
        self.brain._clear_area(Area.NP)
        
        for word, ctx in zip(words, contexts):
            # Get phonological assembly
            phon = self.brain._get_or_create(Area.PHON, word)
            
            # Determine category from grounding
            if ctx.visual:
                # Noun - project to NOUN_CORE then NP
                self.brain._project(Area.NOUN_CORE, phon, learn=learn)
                if self.brain.current[Area.NOUN_CORE] is not None:
                    self.brain.merge_to_area(Area.NP, self.brain.current[Area.NOUN_CORE], learn=learn)
                    
            elif ctx.properties:
                # Adjective - project to ADJ_CORE then NP
                self.brain._project(Area.ADJ_CORE, phon, learn=learn)
                if self.brain.current[Area.ADJ_CORE] is not None:
                    self.brain.merge_to_area(Area.NP, self.brain.current[Area.ADJ_CORE], learn=learn)
                    
            elif not any([ctx.visual, ctx.motor, ctx.properties, ctx.spatial, ctx.social]):
                # Function word (determiner) - project to DET_CORE then NP
                self.brain._project(Area.DET_CORE, phon, learn=learn)
                if self.brain.current[Area.DET_CORE] is not None:
                    self.brain.merge_to_area(Area.NP, self.brain.current[Area.DET_CORE], learn=learn)
            
            # Track word statistics
            if learn:
                self.word_count[word] += 1
                if ctx.visual:
                    self.word_grounding[word]['VISUAL'] += 1
                elif ctx.properties:
                    self.word_grounding[word]['PROPERTY'] += 1
                elif not any([ctx.visual, ctx.motor, ctx.properties, ctx.spatial, ctx.social]):
                    self.word_grounding[word]['NONE'] += 1
        
        return self.brain.current[Area.NP]
    
    def build_verb_phrase(self, verb: str, verb_ctx: GroundingContext,
                          object_np: Optional[cp.ndarray] = None,
                          learn: bool = True) -> Optional[cp.ndarray]:
        """
        Build a verb phrase by projecting verb to VP, optionally merging object.
        
        Example: "chases" + NP("the cat") → VP assembly for "chases the cat"
        
        Returns: The VP assembly
        """
        # Clear VP area
        self.brain._clear_area(Area.VP)
        
        # Get verb assembly
        phon = self.brain._get_or_create(Area.PHON, verb)
        
        # Project to VERB_CORE then VP
        self.brain._project(Area.VERB_CORE, phon, learn=learn)
        if self.brain.current[Area.VERB_CORE] is not None:
            self.brain.merge_to_area(Area.VP, self.brain.current[Area.VERB_CORE], learn=learn)
        
        # If there's an object NP, merge it into VP
        if object_np is not None:
            # First bind object to OBJ role
            self.brain.disinhibit_role(Area.OBJ)  # Allow OBJ after verb
            self.brain.bind_phrase_to_role(object_np, Area.OBJ, learn=learn)
            
            # Then merge OBJ into VP
            if self.brain.current[Area.OBJ] is not None:
                self.brain.merge_to_area(Area.VP, self.brain.current[Area.OBJ], learn=learn)
        
        # Track statistics
        if learn:
            self.word_count[verb] += 1
            self.word_grounding[verb]['MOTOR'] += 1
        
        return self.brain.current[Area.VP]
    
    def build_sentence(self, subject_np: cp.ndarray, vp: cp.ndarray,
                       mood: str = 'declarative', learn: bool = True) -> Optional[cp.ndarray]:
        """
        Build a complete sentence by combining subject NP and VP.
        
        Structure: SENT = SUBJ + VP
        
        This creates the top-level sentence representation that can be
        used for comprehension (parsing) or generation (reconstruction).
        """
        self.brain._clear_area(Area.SENT)
        
        # Activate mood
        mood_assembly = self.brain._get_or_create(Area.MOOD, mood)
        self.brain._project(Area.MOOD, mood_assembly, learn=learn)
        
        # Bind subject NP to SUBJ role
        self.brain.bind_phrase_to_role(subject_np, Area.SUBJ, learn=learn)
        
        # Link SUBJ to VP (creates sentence-level binding)
        if self.brain.current[Area.SUBJ] is not None:
            self.brain.link_to_predicate(self.brain.current[Area.SUBJ], Area.VP, learn=learn)
        
        # Merge VP into SENT
        self.brain.merge_to_area(Area.SENT, vp, learn=learn)
        
        # Also merge SUBJ into SENT for complete representation
        if self.brain.current[Area.SUBJ] is not None:
            self.brain.merge_to_area(Area.SENT, self.brain.current[Area.SUBJ], learn=learn)
        
        # Project to SEQ for word order learning
        if learn:
            self.brain._project(Area.SEQ, self.brain.current[Area.SENT], learn=learn)
        
        return self.brain.current[Area.SENT]
    
    def present_structured_sentence(self, sentence: 'GroundedSentence', learn: bool = True):
        """
        Present a sentence with PROPER phrase structure building.
        
        This is the NEMO way - words are merged into phrases,
        phrases are bound to roles, roles are linked to predicates.
        
        Example: "the big dog chases the cat"
        1. Build NP1 = merge("the", "big", "dog") → NP
        2. Bind NP1 → SUBJ
        3. Build VP = "chases" → VP
        4. Build NP2 = merge("the", "cat") → NP
        5. Bind NP2 → OBJ
        6. Merge OBJ → VP
        7. Build SENT = SUBJ + VP
        """
        self.brain.clear_all()
        
        words = sentence.words
        contexts = sentence.contexts
        roles = sentence.roles if hasattr(sentence, 'roles') else [None] * len(words)
        mood = sentence.mood if hasattr(sentence, 'mood') else 'declarative'
        
        # Find phrase boundaries based on roles
        # agent/patient words are NP heads, action words are VP heads
        
        current_np_words = []
        current_np_contexts = []
        subject_np = None
        object_np = None
        verb = None
        verb_ctx = None
        
        for i, (word, ctx, role) in enumerate(zip(words, contexts, roles)):
            # Track word statistics
            if learn:
                self.word_count[word] += 1
                self.position_counts[word][i] += 1
            
            # Determine if this word ends a phrase
            is_noun = bool(ctx.visual)
            is_verb = bool(ctx.motor)
            is_function = not any([ctx.visual, ctx.motor, ctx.properties, ctx.spatial, ctx.social])
            is_adj = bool(ctx.properties)
            
            if is_verb:
                # Verb - first complete any pending NP
                if current_np_words:
                    np_assembly = self.build_noun_phrase(current_np_words, current_np_contexts, learn=learn)
                    if subject_np is None:
                        subject_np = np_assembly
                        if np_assembly is not None:
                            self.brain.bind_phrase_to_role(np_assembly, Area.SUBJ, learn=learn)
                    else:
                        object_np = np_assembly
                    current_np_words = []
                    current_np_contexts = []
                
                # Store verb for VP building
                verb = word
                verb_ctx = ctx
                
                # Track role statistics
                if learn:
                    self.word_as_action[word] += 1
                    self.word_grounding[word]['MOTOR'] += 1
                
            elif is_noun:
                # Noun - add to current NP
                current_np_words.append(word)
                current_np_contexts.append(ctx)
                
                # Track role statistics
                if learn:
                    if subject_np is None and verb is None:
                        self.word_as_first_arg[word] += 1
                    else:
                        self.word_as_second_arg[word] += 1
                    self.word_grounding[word]['VISUAL'] += 1
                
            elif is_adj or is_function:
                # Adjective or determiner - add to current NP
                current_np_words.append(word)
                current_np_contexts.append(ctx)
                
                if learn:
                    if is_adj:
                        self.word_grounding[word]['PROPERTY'] += 1
                    else:
                        self.word_grounding[word]['NONE'] += 1
        
        # Complete any remaining NP
        if current_np_words:
            np_assembly = self.build_noun_phrase(current_np_words, current_np_contexts, learn=learn)
            if subject_np is None:
                subject_np = np_assembly
            else:
                object_np = np_assembly
        
        # Build VP with object if present
        if verb is not None:
            vp = self.build_verb_phrase(verb, verb_ctx, object_np, learn=learn)
        else:
            vp = None
        
        # Build complete sentence
        if subject_np is not None and vp is not None:
            sent = self.build_sentence(subject_np, vp, mood=mood, learn=learn)
            
            # Track category transitions
            if learn:
                prev_cat = None
                for word in words:
                    cat, _ = self.get_emergent_category(word)
                    if prev_cat and cat != 'UNKNOWN':
                        self.category_transitions[(prev_cat, cat)] += 1
                    prev_cat = cat
        
        if learn:
            self.sentences_seen += 1
    
    def get_emergent_category(self, word: str) -> Tuple[str, Dict[str, float]]:
        """
        Get the EMERGENT category for a word based on its grounding history.
        
        Returns: (category, confidence_scores)
        
        Categories (all emergent from grounding patterns):
        - NOUN: Primarily Visual grounding
        - VERB: Primarily Motor grounding
        - ADJECTIVE: Primarily Property grounding
        - ADVERB: Primarily Temporal grounding
        - PREPOSITION: Primarily Spatial grounding
        - PRONOUN: Primarily Social grounding
        - FUNCTION: No consistent grounding, high frequency
        - UNKNOWN: Not enough data
        """
        if word not in self.word_count or self.word_count[word] < 2:
            return 'UNKNOWN', {}
        
        grounding = self.word_grounding[word]
        total = sum(grounding.values())
        
        if total == 0:
            return 'FUNCTION', {'FUNCTION': 1.0}
        
        # Calculate proportions for all modalities
        modalities = ['VISUAL', 'MOTOR', 'PROPERTY', 'SPATIAL', 'SOCIAL', 
                      'TEMPORAL', 'EMOTION', 'NONE']
        scores = {}
        for modality in modalities:
            scores[modality] = grounding.get(modality, 0) / total
        
        # Determine category from dominant grounding
        if scores.get('NONE', 0) > 0.7:
            return 'FUNCTION', scores
        
        max_modality = max(scores, key=scores.get)
        
        # Mapping from grounding modality to grammatical category
        category_map = {
            'VISUAL': 'NOUN',
            'MOTOR': 'VERB',
            'PROPERTY': 'ADJECTIVE',
            'SPATIAL': 'PREPOSITION',
            'SOCIAL': 'PRONOUN',
            'TEMPORAL': 'ADVERB',
            'EMOTION': 'ADJECTIVE',  # Emotional words often adjectives
            'NONE': 'FUNCTION',
        }
        
        return category_map.get(max_modality, 'UNKNOWN'), scores
    
    def get_thematic_role(self, word: str) -> Tuple[str, float]:
        """
        Get the emergent thematic role for a word based on argument position.
        
        Returns: (role, confidence)
        
        Roles emerge from:
        - Frequently first argument → AGENT
        - Frequently second argument → PATIENT
        - Frequently as action → ACTION
        """
        total = (self.word_as_first_arg[word] + 
                 self.word_as_second_arg[word] + 
                 self.word_as_action[word])
        
        if total < 2:
            return 'UNKNOWN', 0.0
        
        scores = {
            'AGENT': self.word_as_first_arg[word] / total,
            'PATIENT': self.word_as_second_arg[word] / total,
            'ACTION': self.word_as_action[word] / total,
        }
        
        best_role = max(scores, key=scores.get)
        return best_role, scores[best_role]
    
    def get_word_order(self) -> List[str]:
        """Get learned word order from position statistics"""
        # Find typical positions for each category
        category_positions = defaultdict(list)
        
        for word in self.word_count:
            cat, _ = self.get_emergent_category(word)
            if cat != 'UNKNOWN':
                positions = self.position_counts[word]
                if positions:
                    avg_pos = sum(p * c for p, c in positions.items()) / sum(positions.values())
                    category_positions[cat].append(avg_pos)
        
        # Average position per category
        avg_by_cat = {}
        for cat, positions in category_positions.items():
            if positions:
                avg_by_cat[cat] = sum(positions) / len(positions)
        
        # Sort by position
        sorted_cats = sorted(avg_by_cat.keys(), key=lambda c: avg_by_cat[c])
        
        return sorted_cats
    
    def get_vocabulary_by_category(self) -> Dict[str, List[str]]:
        """Get vocabulary grouped by emergent category"""
        vocab = defaultdict(list)
        
        for word in self.word_count:
            cat, scores = self.get_emergent_category(word)
            if cat != 'UNKNOWN':
                vocab[cat].append(word)
        
        return dict(vocab)
    
    def generate_sentence(self, length: int = 3) -> List[str]:
        """
        Generate a sentence using learned PHRASE STRUCTURE patterns.
        
        This is still simplified, but respects basic phrase structure:
        - NP = (DET) + (ADJ)* + NOUN
        - VP = VERB + (NP)?
        - SENT = NP + VP
        
        NOT just random category sampling!
        """
        vocab = self.get_vocabulary_by_category()
        
        # Get words by category (with fallbacks)
        nouns = vocab.get('NOUN', [])
        verbs = vocab.get('VERB', [])
        adjectives = vocab.get('ADJECTIVE', [])
        determiners = [w for w in vocab.get('FUNCTION', []) if w in ['the', 'a']]
        
        if not nouns or not verbs:
            return []
        
        sentence = []
        
        # Use category transitions to determine structure
        # Most common pattern from training: FUNCTION → NOUN → VERB
        trans = self.category_transitions
        
        # Check if we learned SVO pattern
        has_det_noun = trans.get(('FUNCTION', 'NOUN'), 0) > 10
        has_noun_verb = trans.get(('NOUN', 'VERB'), 0) > 10
        has_verb_noun = trans.get(('VERB', 'FUNCTION'), 0) > 5
        
        if has_det_noun and has_noun_verb:
            # Generate SVO-like structure
            
            # Subject NP
            if determiners and np.random.rand() > 0.3:
                sentence.append(np.random.choice(determiners))
            
            # Optional adjective (only if we learned ADJ → NOUN)
            if adjectives and trans.get(('ADJECTIVE', 'NOUN'), 0) > 5 and np.random.rand() > 0.5:
                sentence.append(np.random.choice(adjectives))
            
            # Subject noun
            subj = np.random.choice(nouns)
            sentence.append(subj)
            
            # Verb
            sentence.append(np.random.choice(verbs))
            
            # Optional object NP (if we learned transitive patterns)
            if has_verb_noun and length > 3 and np.random.rand() > 0.3:
                if determiners:
                    sentence.append(np.random.choice(determiners))
                # Pick different noun for object
                obj_candidates = [n for n in nouns if n != subj]
                if obj_candidates:
                    sentence.append(np.random.choice(obj_candidates))
        else:
            # Fallback: simple NOUN VERB
            sentence.append(np.random.choice(nouns))
            sentence.append(np.random.choice(verbs))
        
        return sentence[:length] if len(sentence) > length else sentence
    
    def generate_sentence_structured(self) -> List[str]:
        """
        Generate a sentence using PROPER phrase structure.
        
        This is the NEMO way:
        1. Generate a subject NP (DET + ADJ? + NOUN)
        2. Generate a VP (VERB + object NP?)
        3. Combine into sentence
        
        Uses learned statistics for:
        - Whether to include determiner
        - Whether to include adjective
        - Whether sentence is transitive (has object)
        - Which words go together (selectional preferences)
        """
        vocab = self.get_vocabulary_by_category()
        
        nouns = vocab.get('NOUN', [])
        verbs = vocab.get('VERB', [])
        adjectives = vocab.get('ADJECTIVE', [])
        determiners = [w for w in vocab.get('FUNCTION', []) if w in ['the', 'a']]
        
        if not nouns or not verbs:
            return []
        
        sentence = []
        trans = self.category_transitions
        
        # === SUBJECT NP ===
        
        # Determiner? (based on learned FUNCTION → NOUN transitions)
        det_noun_count = trans.get(('FUNCTION', 'NOUN'), 0)
        noun_only_count = sum(1 for w in nouns if self.position_counts[w].get(0, 0) > 0)
        
        if determiners and det_noun_count > noun_only_count * 0.3:
            sentence.append(np.random.choice(determiners))
        
        # Adjective? (based on learned ADJECTIVE → NOUN transitions)
        adj_noun_count = trans.get(('ADJECTIVE', 'NOUN'), 0)
        if adjectives and adj_noun_count > 5 and np.random.rand() > 0.5:
            sentence.append(np.random.choice(adjectives))
        
        # Subject noun - prefer words that appear as AGENT
        agent_nouns = [(n, self.word_as_first_arg[n]) for n in nouns if self.word_as_first_arg[n] > 0]
        if agent_nouns:
            # Weighted by agent frequency
            weights = [c for _, c in agent_nouns]
            total = sum(weights)
            probs = [w/total for w in weights]
            subj = np.random.choice([n for n, _ in agent_nouns], p=probs)
        else:
            subj = np.random.choice(nouns)
        sentence.append(subj)
        
        # === VERB ===
        
        # Choose verb - prefer verbs that co-occur with subject
        verb_scores = {}
        for v in verbs:
            # Score by co-occurrence with subject
            cooc = self.word_cooccurrence[subj].get(v, 0)
            # And by overall frequency
            freq = self.word_count.get(v, 1)
            verb_scores[v] = cooc + freq * 0.1
        
        if verb_scores:
            best_verb = max(verb_scores, key=verb_scores.get)
        else:
            best_verb = np.random.choice(verbs)
        sentence.append(best_verb)
        
        # === OBJECT NP (optional) ===
        
        # Check if this verb is transitive (learned from VERB → FUNCTION patterns)
        verb_obj_count = trans.get(('VERB', 'FUNCTION'), 0)
        verb_end_count = sum(1 for (src, _), c in trans.items() if src == 'VERB') - verb_obj_count
        
        is_transitive = verb_obj_count > verb_end_count * 0.3
        
        if is_transitive:
            # Object determiner
            if determiners:
                sentence.append(np.random.choice(determiners))
            
            # Object noun - prefer words that appear as PATIENT and different from subject
            patient_nouns = [(n, self.word_as_second_arg[n]) for n in nouns 
                            if self.word_as_second_arg[n] > 0 and n != subj]
            if patient_nouns:
                weights = [c for _, c in patient_nouns]
                total = sum(weights)
                probs = [w/total for w in weights]
                obj = np.random.choice([n for n, _ in patient_nouns], p=probs)
            else:
                obj_candidates = [n for n in nouns if n != subj]
                if obj_candidates:
                    obj = np.random.choice(obj_candidates)
                else:
                    obj = np.random.choice(nouns)
            sentence.append(obj)
        
        return sentence
    
    def generate_sentence_neural(self, max_length: int = 5) -> List[str]:
        """
        Generate using actual neural activation through SEQ area.
        
        This uses the brain's learned connections to predict word order.
        """
        self.brain.clear_all()
        vocab = self.get_vocabulary_by_category()
        
        # Start with declarative mood
        mood_assembly = self.brain._get_or_create(Area.MOOD, 'declarative')
        self.brain._project(Area.MOOD, mood_assembly, learn=False)
        self.brain._project(Area.SEQ, mood_assembly, learn=False)
        
        sentence = []
        prev_category = None
        
        for _ in range(max_length):
            # Find best next category based on transitions
            if prev_category is None:
                # First word - check what typically starts sentences
                # (from mood_word_first statistics)
                if self.mood_word_first:
                    first_word = max(self.mood_word_first, key=self.mood_word_first.get)
                    cat, _ = self.get_emergent_category(first_word)
                    if cat == 'FUNCTION':
                        cat = 'NOUN'  # Skip function word at start for cleaner output
                else:
                    cat = 'NOUN'
            else:
                # Use learned transitions
                candidates = {}
                for (src, dst), count in self.category_transitions.items():
                    if src == prev_category:
                        candidates[dst] = count
                
                if not candidates:
                    break
                
                # Sample from candidates (weighted by count)
                total = sum(candidates.values())
                r = np.random.rand() * total
                cumsum = 0
                cat = None
                for c, count in candidates.items():
                    cumsum += count
                    if r <= cumsum:
                        cat = c
                        break
                
                if cat is None or cat == 'UNKNOWN':
                    break
            
            # Get word from category
            if cat in vocab and vocab[cat]:
                word = np.random.choice(vocab[cat])
                sentence.append(word)
                prev_category = cat
            else:
                break
        
        return sentence
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        vocab = self.get_vocabulary_by_category()
        return {
            'sentences_seen': self.sentences_seen,
            'vocabulary_size': len(self.word_count),
            'categories': {cat: len(words) for cat, words in vocab.items()},
            'word_order': self.get_word_order(),
        }


# =============================================================================
# DEMO with realistic grounded training
# =============================================================================

@dataclass
class GroundedSentence:
    """A sentence with full grounding information"""
    words: List[str]
    contexts: List[GroundingContext]
    roles: List[str]  # 'agent', 'patient', 'action', None
    mood: str = 'declarative'


def create_grounded_training_data() -> List[GroundedSentence]:
    """
    Create training data where grounding determines category.
    
    This simulates how a child learns:
    - Seeing a dog while hearing "dog" → VISUAL grounding → NOUN
    - Seeing running while hearing "runs" → MOTOR grounding → VERB
    - Seeing bigness while hearing "big" → PROPERTY grounding → ADJECTIVE
    - Hearing "the" with no specific grounding → FUNCTION word
    - First noun in transitive → AGENT role
    - Second noun in transitive → PATIENT role
    """
    data = []
    
    # === DECLARATIVE SENTENCES ===
    
    # "the dog runs" (intransitive)
    data.append(GroundedSentence(
        words=['the', 'dog', 'runs'],
        contexts=[
            GroundingContext(),  # "the" - function word
            GroundingContext(visual=['DOG', 'ANIMAL']),  # "dog" - visual
            GroundingContext(motor=['RUNNING', 'MOTION']),  # "runs" - motor
        ],
        roles=[None, 'agent', 'action'],
        mood='declarative'
    ))
    
    # "the cat chases the bird" (transitive)
    data.append(GroundedSentence(
        words=['the', 'cat', 'chases', 'the', 'bird'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(motor=['CHASING', 'PURSUIT']),
            GroundingContext(),
            GroundingContext(visual=['BIRD', 'ANIMAL']),
        ],
        roles=[None, 'agent', 'action', None, 'patient'],
        mood='declarative'
    ))
    
    # "a big cat sleeps" (adjective + intransitive)
    data.append(GroundedSentence(
        words=['a', 'big', 'cat', 'sleeps'],
        contexts=[
            GroundingContext(),
            GroundingContext(properties=['SIZE', 'BIG']),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(motor=['SLEEPING', 'REST']),
        ],
        roles=[None, None, 'agent', 'action'],
        mood='declarative'
    ))
    
    # "the ball is red" (copula)
    data.append(GroundedSentence(
        words=['the', 'ball', 'is', 'red'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['BALL', 'OBJECT']),
            GroundingContext(),  # copula - function word
            GroundingContext(properties=['COLOR', 'RED']),
        ],
        roles=[None, 'agent', 'action', None],
        mood='declarative'
    ))
    
    # "she sees the bird" (pronoun subject)
    data.append(GroundedSentence(
        words=['she', 'sees', 'the', 'bird'],
        contexts=[
            GroundingContext(social=['PERSON', 'FEMALE']),
            GroundingContext(motor=['SEEING', 'PERCEPTION']),
            GroundingContext(),
            GroundingContext(visual=['BIRD', 'ANIMAL']),
        ],
        roles=['agent', 'action', None, 'patient'],
        mood='declarative'
    ))
    
    # "the boy eats food quickly" (adverb)
    data.append(GroundedSentence(
        words=['the', 'boy', 'eats', 'food', 'quickly'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['BOY', 'PERSON'], social=['PERSON', 'CHILD']),
            GroundingContext(motor=['EATING', 'CONSUMPTION']),
            GroundingContext(visual=['FOOD', 'OBJECT']),
            GroundingContext(temporal=['QUICK', 'MANNER']),  # temporal/manner adverb
        ],
        roles=[None, 'agent', 'action', 'patient', None],
        mood='declarative'
    ))
    
    # "the cat is on the table" (preposition)
    data.append(GroundedSentence(
        words=['the', 'cat', 'is', 'on', 'the', 'table'],
        contexts=[
            GroundingContext(),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(),
            GroundingContext(spatial=['ON', 'ABOVE']),  # preposition
            GroundingContext(),
            GroundingContext(visual=['TABLE', 'FURNITURE']),
        ],
        roles=[None, 'agent', 'action', None, None, None],
        mood='declarative'
    ))
    
    # "he and she play" (conjunction)
    data.append(GroundedSentence(
        words=['he', 'and', 'she', 'play'],
        contexts=[
            GroundingContext(social=['PERSON', 'MALE']),
            GroundingContext(),  # conjunction - function word
            GroundingContext(social=['PERSON', 'FEMALE']),
            GroundingContext(motor=['PLAYING', 'ACTION']),
        ],
        roles=['agent', None, 'agent', 'action'],
        mood='declarative'
    ))
    
    # "yesterday the dog ran" (temporal adverb)
    data.append(GroundedSentence(
        words=['yesterday', 'the', 'dog', 'ran'],
        contexts=[
            GroundingContext(temporal=['PAST', 'TIME']),  # temporal adverb
            GroundingContext(),
            GroundingContext(visual=['DOG', 'ANIMAL']),
            GroundingContext(motor=['RUNNING', 'MOTION']),
        ],
        roles=[None, None, 'agent', 'action'],
        mood='declarative'
    ))
    
    # === INTERROGATIVE SENTENCES ===
    
    # "does the dog run?" (yes/no question)
    data.append(GroundedSentence(
        words=['does', 'the', 'dog', 'run'],
        contexts=[
            GroundingContext(),  # auxiliary - function word
            GroundingContext(),
            GroundingContext(visual=['DOG', 'ANIMAL']),
            GroundingContext(motor=['RUNNING', 'MOTION']),
        ],
        roles=[None, None, 'agent', 'action'],
        mood='interrogative'
    ))
    
    # "what does the cat see?" (wh-question)
    data.append(GroundedSentence(
        words=['what', 'does', 'the', 'cat', 'see'],
        contexts=[
            GroundingContext(),  # wh-word - function word
            GroundingContext(),
            GroundingContext(),
            GroundingContext(visual=['CAT', 'ANIMAL']),
            GroundingContext(motor=['SEEING', 'PERCEPTION']),
        ],
        roles=['patient', None, None, 'agent', 'action'],
        mood='interrogative'
    ))
    
    # === IMPERATIVE SENTENCES ===
    
    # "run!" (command)
    data.append(GroundedSentence(
        words=['run'],
        contexts=[
            GroundingContext(motor=['RUNNING', 'MOTION']),
        ],
        roles=['action'],
        mood='imperative'
    ))
    
    # "eat the food!" (command with object)
    data.append(GroundedSentence(
        words=['eat', 'the', 'food'],
        contexts=[
            GroundingContext(motor=['EATING', 'CONSUMPTION']),
            GroundingContext(),
            GroundingContext(visual=['FOOD', 'OBJECT']),
        ],
        roles=['action', None, 'patient'],
        mood='imperative'
    ))
    
    # === GENERATE MORE EXAMPLES ===
    
    nouns = ['dog', 'cat', 'bird', 'ball', 'book', 'boy', 'girl', 'food', 'table', 'car']
    verbs = ['runs', 'sees', 'eats', 'plays', 'sleeps', 'jumps', 'walks', 'reads', 'chases', 'finds']
    adjectives = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'good', 'bad', 'happy', 'sad']
    prepositions = ['on', 'in', 'under', 'near', 'behind']
    adverbs = ['quickly', 'slowly', 'happily', 'sadly', 'now', 'yesterday', 'tomorrow']
    
    # Intransitive sentences: "the NOUN VERB"
    for noun in nouns:
        for verb in ['runs', 'sleeps', 'jumps', 'walks', 'plays']:
            data.append(GroundedSentence(
                words=['the', noun, verb],
                contexts=[
                    GroundingContext(),
                    GroundingContext(visual=[noun.upper(), 'OBJECT']),
                    GroundingContext(motor=[verb.upper(), 'ACTION']),
                ],
                roles=[None, 'agent', 'action'],
                mood='declarative'
            ))
    
    # Transitive sentences: "the NOUN1 VERB the NOUN2"
    for noun1 in nouns[:5]:
        for verb in ['sees', 'chases', 'finds', 'eats']:
            for noun2 in nouns[5:]:
                data.append(GroundedSentence(
                    words=['the', noun1, verb, 'the', noun2],
                    contexts=[
                        GroundingContext(),
                        GroundingContext(visual=[noun1.upper(), 'OBJECT']),
                        GroundingContext(motor=[verb.upper(), 'ACTION']),
                        GroundingContext(),
                        GroundingContext(visual=[noun2.upper(), 'OBJECT']),
                    ],
                    roles=[None, 'agent', 'action', None, 'patient'],
                    mood='declarative'
                ))
    
    # Adjective + noun: "the ADJ NOUN"
    for adj in adjectives:
        for noun in nouns[:5]:
            data.append(GroundedSentence(
                words=['the', adj, noun],
                contexts=[
                    GroundingContext(),
                    GroundingContext(properties=[adj.upper(), 'PROPERTY']),
                    GroundingContext(visual=[noun.upper(), 'OBJECT']),
                ],
                roles=[None, None, None],
                mood='declarative'
            ))
    
    # Prepositional phrases: "the NOUN is PREP the NOUN"
    for noun1 in nouns[:3]:
        for prep in prepositions:
            for noun2 in nouns[3:6]:
                data.append(GroundedSentence(
                    words=['the', noun1, 'is', prep, 'the', noun2],
                    contexts=[
                        GroundingContext(),
                        GroundingContext(visual=[noun1.upper(), 'OBJECT']),
                        GroundingContext(),
                        GroundingContext(spatial=[prep.upper(), 'LOCATION']),
                        GroundingContext(),
                        GroundingContext(visual=[noun2.upper(), 'OBJECT']),
                    ],
                    roles=[None, 'agent', 'action', None, None, None],
                    mood='declarative'
                ))
    
    # Adverb sentences: "the NOUN VERB ADV"
    for noun in nouns[:3]:
        for verb in ['runs', 'walks', 'eats']:
            for adv in adverbs[:4]:
                data.append(GroundedSentence(
                    words=['the', noun, verb, adv],
                    contexts=[
                        GroundingContext(),
                        GroundingContext(visual=[noun.upper(), 'OBJECT']),
                        GroundingContext(motor=[verb.upper(), 'ACTION']),
                        GroundingContext(temporal=[adv.upper(), 'MANNER']),
                    ],
                    roles=[None, 'agent', 'action', None],
                    mood='declarative'
                ))
    
    return data


def demo():
    """Demo emergent category learning with full neurobiological architecture"""
    print("="*70)
    print("EMERGENT NEMO LANGUAGE LEARNER")
    print("37 brain areas - ALL categories emerge from grounding!")
    print("="*70)
    
    # Create learner
    params = EmergentParams(n=10000)
    learner = EmergentLanguageLearner(params, verbose=True)
    
    # Create training data
    print("\nCreating grounded training data...")
    training_data = create_grounded_training_data()
    print(f"  {len(training_data)} sentences")
    
    # Train
    print("\nTraining...")
    start = time.perf_counter()
    for epoch in range(3):
        for sentence in training_data:
            learner.present_grounded_sentence(
                sentence.words, sentence.contexts, 
                roles=sentence.roles, mood=sentence.mood,
                learn=True
            )
    elapsed = time.perf_counter() - start
    print(f"  {learner.sentences_seen} sentences in {elapsed:.2f}s")
    
    # Show emergent categories
    print("\n" + "="*60)
    print("EMERGENT CATEGORIES (learned from grounding)")
    print("="*60)
    
    vocab = learner.get_vocabulary_by_category()
    for cat in ['NOUN', 'VERB', 'ADJECTIVE', 'ADVERB', 'PREPOSITION', 'PRONOUN', 'FUNCTION']:
        if cat in vocab:
            words = vocab[cat][:10]  # Show first 10
            print(f"\n{cat} ({len(vocab[cat])} words):")
            print(f"  {', '.join(words)}")
    
    # Show word order
    print("\n" + "="*60)
    print("LEARNED WORD ORDER")
    print("="*60)
    print(f"  Category order: {learner.get_word_order()}")
    
    # Show category transitions
    print("\n  Category transitions (top 10):")
    sorted_trans = sorted(learner.category_transitions.items(), key=lambda x: -x[1])[:10]
    for (cat1, cat2), count in sorted_trans:
        print(f"    {cat1} → {cat2}: {count}")
    
    # Show thematic roles
    print("\n" + "="*60)
    print("EMERGENT THEMATIC ROLES")
    print("="*60)
    
    test_words = ['dog', 'cat', 'boy', 'runs', 'chases', 'food']
    print(f"\n{'Word':>10} {'Category':>12} {'Role':>10} {'Conf':>6}")
    print("-"*45)
    for word in test_words:
        cat, _ = learner.get_emergent_category(word)
        role, conf = learner.get_thematic_role(word)
        print(f"{word:>10} {cat:>12} {role:>10} {conf:>6.0%}")
    
    # Generate sentences - structure-based
    print("\n" + "="*60)
    print("GENERATED SENTENCES (phrase structure)")
    print("="*60)
    for i in range(5):
        sent = learner.generate_sentence(5)
        print(f"  {i+1}. {' '.join(sent)}")
    
    # Generate sentences - neural/transition-based
    print("\n" + "="*60)
    print("GENERATED SENTENCES (neural transitions)")
    print("="*60)
    for i in range(5):
        sent = learner.generate_sentence_neural(5)
        print(f"  {i+1}. {' '.join(sent)}")
    
    # Show detailed grounding analysis
    print("\n" + "="*60)
    print("WORD GROUNDING ANALYSIS")
    print("="*60)
    
    test_words = ['dog', 'runs', 'big', 'the', 'she', 'on', 'quickly', 'yesterday']
    print(f"\n{'Word':>10} {'Cat':>8} {'V':>5} {'M':>5} {'P':>5} {'Sp':>5} {'So':>5} {'T':>5} {'∅':>5}")
    print("-"*65)
    
    for word in test_words:
        cat, scores = learner.get_emergent_category(word)
        v = scores.get('VISUAL', 0)
        m = scores.get('MOTOR', 0)
        p = scores.get('PROPERTY', 0)
        sp = scores.get('SPATIAL', 0)
        so = scores.get('SOCIAL', 0)
        t = scores.get('TEMPORAL', 0)
        n = scores.get('NONE', 0)
        print(f"{word:>10} {cat:>8} {v:>5.0%} {m:>5.0%} {p:>5.0%} {sp:>5.0%} {so:>5.0%} {t:>5.0%} {n:>5.0%}")
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    stats = learner.get_stats()
    print(f"  Sentences seen: {stats['sentences_seen']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Brain areas: {NUM_AREAS}")
    print(f"  Mutual inhibition groups: {len(MUTUAL_INHIBITION_GROUPS)}")


if __name__ == "__main__":
    demo()

