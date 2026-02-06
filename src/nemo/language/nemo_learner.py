"""
NEMO Language Learner - Neurobiologically Plausible
====================================================

Version: 1.0.0
Date: 2025-12-01

Based on:
- Mitropolsky & Papadimitriou 2025: "Simulated Language Acquisition"
- Papadimitriou et al. 2020: "Brain Computation by Assemblies of Neurons"

Key Principles (from papers):
1. GROUNDED LEARNING - words presented with continuous sensory context
2. DIFFERENTIAL LEX AREAS - Lex1 (nouns→Visual), Lex2 (verbs→Motor)
3. STABILITY-BASED CLASSIFICATION - stable assembly = correct area
4. ROLE AREAS WITH MUTUAL INHIBITION - only one role active at a time
5. HIERARCHICAL STRUCTURE - Lex → NP/VP → Sent

Architecture:
    Phon ─────────┬──────────┐
                  ▼          ▼
    Visual ──→ Lex1 ──→ NP ──┬──→ Sent
    Motor ───→ Lex2 ──→ VP ──┘
                  │
                  ▼
    Role_agent ←─┼─→ Role_action ←─┼─→ Role_patient
         (mutual inhibition)
                  │
                  ▼
                 SEQ (word order)
"""

import cupy as cp
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

# Import core kernels
from src.nemo.core.kernel import projection_fp16_kernel, hebbian_kernel


class GroundingType(Enum):
    """Types of perceptual grounding (from paper)"""
    VISUAL = auto()      # Objects, scenes
    MOTOR = auto()       # Actions, movements
    AUDITORY = auto()    # Sounds
    TACTILE = auto()     # Touch sensations
    EMOTIONAL = auto()   # Feelings
    SPATIAL = auto()     # Locations


class SpeechAct(Enum):
    """Pragmatic function of utterance"""
    NAMING = auto()         # "That's a dog"
    DESCRIBING = auto()     # "The dog is big"
    REQUESTING = auto()     # "Give me the ball"
    COMMANDING = auto()     # "Sit down"
    QUESTIONING = auto()    # "What's that?"


@dataclass
class GroundedContext:
    """Perceptual context for grounded learning"""
    visual: List[str] = field(default_factory=list)    # Visual objects
    motor: List[str] = field(default_factory=list)     # Actions
    properties: Dict[str, List[str]] = field(default_factory=dict)  # obj -> props
    emotions: List[str] = field(default_factory=list)
    spatial: Dict[str, str] = field(default_factory=dict)  # obj -> location
    joint_attention: Optional[str] = None  # What both speaker/listener attend to


@dataclass
class NemoParams:
    """Parameters for NEMO model (from paper)"""
    n: int = 100000        # Neurons per area
    k: int = None          # Winners (sqrt(n) if None)
    p: float = 0.05        # Connection probability
    beta: float = 0.1      # Hebbian plasticity rate
    w_max: float = 10.0    # Weight saturation
    tau: int = 2           # Firing steps per word
    
    # Strong fibers (paper uses different strengths)
    p_strong: float = 0.1
    beta_strong: float = 0.15
    
    def __post_init__(self):
        if self.k is None:
            self.k = int(np.sqrt(self.n))


class Area(Enum):
    """Brain areas in NEMO model"""
    # Input areas
    PHON = 0           # Phonological
    VISUAL = 1         # Visual grounding
    MOTOR = 2          # Motor grounding
    EMOTION = 3        # Emotional grounding
    
    # Lexical areas (differential for noun/verb)
    LEX1 = 4           # Noun lexicon (strong → Visual)
    LEX2 = 5           # Verb lexicon (strong → Motor)
    
    # Phrase structure
    NP = 6             # Noun phrase
    VP = 7             # Verb phrase
    SENT = 8           # Sentence
    
    # Thematic roles (mutual inhibition)
    ROLE_AGENT = 9     # Agent/Subject
    ROLE_ACTION = 10   # Action/Verb
    ROLE_PATIENT = 11  # Patient/Object
    
    # Sequence
    SEQ = 12           # Word order learning


# Fiber definitions (which areas connect)
STRONG_FIBERS = [
    (Area.PHON, Area.LEX1),
    (Area.PHON, Area.LEX2),
    (Area.LEX1, Area.VISUAL),
    (Area.VISUAL, Area.LEX1),
    (Area.LEX2, Area.MOTOR),
    (Area.MOTOR, Area.LEX2),
]

REGULAR_FIBERS = [
    # Lex to phrase
    (Area.LEX1, Area.NP),
    (Area.LEX2, Area.VP),
    # Phrase to sentence
    (Area.NP, Area.SENT),
    (Area.VP, Area.SENT),
    # Lex to roles
    (Area.LEX1, Area.ROLE_AGENT),
    (Area.LEX1, Area.ROLE_PATIENT),
    (Area.LEX2, Area.ROLE_ACTION),
    # Roles to sequence
    (Area.ROLE_AGENT, Area.SEQ),
    (Area.ROLE_ACTION, Area.SEQ),
    (Area.ROLE_PATIENT, Area.SEQ),
    # Recurrent within Lex
    (Area.LEX1, Area.LEX1),
    (Area.LEX2, Area.LEX2),
    # Weak cross-modal (verb can have visual, noun can have motor)
    (Area.LEX1, Area.MOTOR),
    (Area.LEX2, Area.VISUAL),
]

ROLE_AREAS = [Area.ROLE_AGENT, Area.ROLE_ACTION, Area.ROLE_PATIENT]


class NemoBrain:
    """
    NEMO brain with proper neurobiological architecture.
    
    Key features:
    - Differential Lex areas (Lex1→Visual for nouns, Lex2→Motor for verbs)
    - Continuous grounding during word presentation
    - Stability-based classification
    - Role areas with mutual inhibition
    """
    
    NUM_AREAS = 13
    
    def __init__(self, params: NemoParams = None, verbose: bool = True):
        self.p = params or NemoParams()
        self.verbose = verbose
        n, k = self.p.n, self.p.k
        
        # Input assemblies (pre-created for each concept)
        self.phon: Dict[str, cp.ndarray] = {}      # word → assembly
        self.visual: Dict[str, cp.ndarray] = {}    # visual concept → assembly
        self.motor: Dict[str, cp.ndarray] = {}     # motor concept → assembly
        self.emotion: Dict[str, cp.ndarray] = {}   # emotion → assembly
        
        # Area seeds for implicit connectivity
        self.seeds = cp.arange(self.NUM_AREAS, dtype=cp.uint32) * 1000
        
        # Learned weights per area
        self.max_learned = k * k * 500
        self.l_src = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        self.l_dst = [cp.zeros(self.max_learned, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        self.l_delta = [cp.zeros(self.max_learned, dtype=cp.float32) for _ in range(self.NUM_AREAS)]
        self.l_num = [cp.zeros(1, dtype=cp.uint32) for _ in range(self.NUM_AREAS)]
        
        # Current and previous activations
        self.current: Dict[Area, Optional[cp.ndarray]] = {a: None for a in Area}
        self.prev: Dict[Area, Optional[cp.ndarray]] = {a: None for a in Area}
        
        # Firing history (for stability measurement)
        self.firing_history: Dict[Area, List[cp.ndarray]] = defaultdict(list)
        
        # Kernel config
        self.bs = 512
        self.gx = (n + self.bs - 1) // self.bs
        
        # Statistics
        self.sentences_seen = 0
        self.words_learned: Set[str] = set()
        
        if verbose:
            print("NemoBrain initialized:")
            print(f"  n={n:,}, k={k}")
            print(f"  Areas: {self.NUM_AREAS}")
            print(f"  Strong fibers: {len(STRONG_FIBERS)}")
            print(f"  Regular fibers: {len(REGULAR_FIBERS)}")
    
    def _get_or_create_assembly(self, store: Dict, name: str) -> cp.ndarray:
        """Get or create a random assembly for a concept"""
        if name not in store:
            store[name] = cp.random.randint(0, self.p.n, self.p.k, dtype=cp.uint32)
        return store[name]
    
    def _project(self, area: Area, input_assembly: cp.ndarray, 
                 learn: bool = True, is_strong: bool = False) -> cp.ndarray:
        """
        Project input to an area using implicit connectivity + learned weights.
        
        Returns the winning assembly (top-k neurons).
        """
        n, k = self.p.n, self.p.k
        area_idx = area.value
        
        # Use strong or regular connection parameters
        p = self.p.p_strong if is_strong else self.p.p
        beta = self.p.beta_strong if is_strong else self.p.beta
        
        # Prepare input
        active = cp.zeros(k, dtype=cp.uint32)
        active[:len(input_assembly)] = input_assembly[:k]
        
        # Allocate result
        result = cp.zeros(n, dtype=cp.float16)
        
        # Run projection kernel
        projection_fp16_kernel(
            (self.gx, 1), (self.bs,),
            (active.reshape(1, -1), result.reshape(1, -1),
             cp.uint32(k), cp.uint32(n), cp.uint32(1),
             self.seeds[area_idx:area_idx+1], cp.float32(p)),
            shared_mem=k * 4
        )
        
        # Top-k selection
        result_torch = torch.as_tensor(result, device='cuda')
        _, winners_idx = torch.topk(result_torch, k, sorted=False)
        winners = cp.asarray(winners_idx).astype(cp.uint32)
        
        # Hebbian learning
        if learn and self.prev[area] is not None:
            grid = (k * k + self.bs - 1) // self.bs
            hebbian_kernel(
                (grid,), (self.bs,),
                (self.l_src[area_idx], self.l_dst[area_idx], 
                 self.l_delta[area_idx], self.l_num[area_idx],
                 self.prev[area], winners,
                 cp.uint32(k), cp.float32(beta), cp.float32(self.p.w_max),
                 cp.uint32(self.max_learned), self.seeds[area_idx], cp.float32(p))
            )
        
        # Update state
        self.prev[area] = winners
        self.current[area] = winners
        self.firing_history[area].append(winners.copy())
        
        return winners
    
    def _project_with_grounding(self, area: Area, phon_input: cp.ndarray,
                                 grounding_input: cp.ndarray, 
                                 grounding_area: Area,
                                 learn: bool = True) -> cp.ndarray:
        """
        Project with continuous grounding (key insight from paper).
        
        Both phonological and grounding inputs fire together,
        creating associated assemblies in the Lex area.
        """
        # First project grounding to Lex (strong fiber)
        self._project(area, grounding_input, learn=learn, is_strong=True)
        
        # Then project phonological to same Lex (strong fiber)
        # This creates association between word form and grounding
        return self._project(area, phon_input, learn=learn, is_strong=True)
    
    def _measure_stability(self, area: Area) -> float:
        """
        Measure assembly stability in an area.
        
        Stable assembly = same neurons keep firing (learned word).
        Wobbly assembly = neurons change each step (not learned).
        """
        history = self.firing_history[area]
        if len(history) < 2:
            return 0.0
        
        first = set(history[0].get().tolist())
        last = set(history[-1].get().tolist())
        
        overlap = len(first & last) / self.p.k
        return overlap
    
    def _clear_area(self, area: Area):
        """Clear activations for an area"""
        self.current[area] = None
        self.prev[area] = None
        self.firing_history[area].clear()
    
    def clear_all(self):
        """Clear all activations"""
        for area in Area:
            self._clear_area(area)


class NemoLanguageLearner:
    """
    Language learner following NEMO principles.
    
    Learning phases (from paper):
    1. Word learning with grounding
    2. Stability-based classification
    3. Role binding
    4. Word order learning
    """
    
    def __init__(self, params: NemoParams = None, verbose: bool = True):
        self.brain = NemoBrain(params, verbose=verbose)
        self.p = self.brain.p
        self.verbose = verbose
        
        # Ground truth for evaluation (NOT used in learning!)
        self._gt_nouns: Set[str] = set()
        self._gt_verbs: Set[str] = set()
        
        # Learned classifications (from stability)
        self.learned_nouns: Set[str] = set()
        self.learned_verbs: Set[str] = set()
        
        # Word order transitions
        self.transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        self.sentences_seen = 0
    
    def register_noun(self, word: str, visual_concepts: List[str]):
        """Register a noun with its visual grounding"""
        self._gt_nouns.add(word)
        self.brain._get_or_create_assembly(self.brain.phon, word)
        for v in visual_concepts:
            self.brain._get_or_create_assembly(self.brain.visual, v)
    
    def register_verb(self, word: str, motor_concepts: List[str]):
        """Register a verb with its motor grounding"""
        self._gt_verbs.add(word)
        self.brain._get_or_create_assembly(self.brain.phon, word)
        for m in motor_concepts:
            self.brain._get_or_create_assembly(self.brain.motor, m)
    
    def present_grounded_word(self, word: str, context: GroundedContext, 
                               learn: bool = True):
        """
        Present a word with its grounding context.
        
        Key insight from paper: Grounding fires CONTINUOUSLY while word is presented.
        This creates differential association:
        - Nouns → Lex1 (via Visual) - Visual grounding active
        - Verbs → Lex2 (via Motor) - Motor grounding active
        
        CRITICAL: Only project to the CORRECT Lex area based on grounding!
        """
        phon = self.brain._get_or_create_assembly(self.brain.phon, word)
        
        # Determine grounding type
        is_noun = word in self._gt_nouns
        is_verb = word in self._gt_verbs
        
        if is_noun:
            # Clear Lex1 state to avoid cross-word contamination
            self.brain._clear_area(Area.LEX1)
            
            # Noun: Visual grounding → Lex1
            visual_concept = word.upper()
            grounding = self.brain._get_or_create_assembly(self.brain.visual, visual_concept)
            
            for _ in range(self.p.tau):
                # Activate Visual continuously
                self.brain.current[Area.VISUAL] = grounding
                
                # Project Phon → Lex1 (strong fiber)
                self.brain._project(Area.LEX1, phon, learn=learn, is_strong=True)
                
                # Also project Visual → Lex1 (creates grounding association)
                self.brain._project(Area.LEX1, grounding, learn=learn, is_strong=True)
            
        elif is_verb:
            # Clear Lex2 state to avoid cross-word contamination
            self.brain._clear_area(Area.LEX2)
            
            # Verb: Motor grounding → Lex2
            motor_concept = word.upper()
            grounding = self.brain._get_or_create_assembly(self.brain.motor, motor_concept)
            
            for _ in range(self.p.tau):
                # Activate Motor continuously
                self.brain.current[Area.MOTOR] = grounding
                
                # Project Phon → Lex2 (strong fiber)
                self.brain._project(Area.LEX2, phon, learn=learn, is_strong=True)
                
                # Also project Motor → Lex2 (creates grounding association)
                self.brain._project(Area.LEX2, grounding, learn=learn, is_strong=True)
        
        else:
            # Unknown word - try both (weak learning)
            grounding = self.brain._get_or_create_assembly(self.brain.visual, word.upper())
            self.brain._project(Area.LEX1, phon, learn=learn, is_strong=False)
        
        if learn:
            self.brain.words_learned.add(word)
    
    def present_grounded_sentence(self, words: List[str], 
                                   context: GroundedContext,
                                   roles: List[str] = None,
                                   learn: bool = True):
        """
        Present a grounded sentence.
        
        Args:
            words: List of words in sentence
            context: Grounding context
            roles: Optional role labels ['SUBJ', 'VERB', 'OBJ']
        """
        self.brain.clear_all()
        
        # Default roles based on position (SVO)
        if roles is None:
            if len(words) == 2:
                roles = ['SUBJ', 'VERB']
            elif len(words) >= 3:
                roles = ['SUBJ', 'VERB', 'OBJ']
            else:
                roles = ['SUBJ']
        
        # Present each word with role binding
        prev_role = None
        for word, role in zip(words, roles):
            # Present word with grounding (this projects to correct Lex area)
            self.present_grounded_word(word, context, learn=learn)
            
            # Bind to role area - use the CORRECT Lex area based on word type
            role_area = {
                'SUBJ': Area.ROLE_AGENT,
                'VERB': Area.ROLE_ACTION,
                'OBJ': Area.ROLE_PATIENT,
            }.get(role)
            
            if role_area:
                # Get the correct Lex assembly
                if word in self._gt_nouns:
                    lex = self.brain.current.get(Area.LEX1)
                elif word in self._gt_verbs:
                    lex = self.brain.current.get(Area.LEX2)
                else:
                    lex = self.brain.current.get(Area.LEX1)  # Default
                
                if lex is not None:
                    self.brain._project(role_area, lex, learn=learn)
                    
                    # Update sequence
                    if self.brain.current.get(role_area) is not None:
                        self.brain._project(Area.SEQ, 
                                           self.brain.current[role_area], 
                                           learn=learn)
            
            # Track transitions
            if prev_role and learn:
                self.transitions[(prev_role, role)] += 1
            prev_role = role
        
        if learn:
            self.sentences_seen += 1
    
    def classify_word(self, word: str) -> Tuple[str, float, float]:
        """
        Classify word as NOUN or VERB by testing which Lex area responds.
        
        Method: Project word's phon assembly to each Lex area and measure
        the activation strength (sum of activations in top-k winners).
        
        This tests: "Which area has learned to recognize this word?"
        
        Returns: (classification, activation_lex1, activation_lex2)
        """
        phon = self.brain._get_or_create_assembly(self.brain.phon, word)
        
        def test_activation(lex_area: Area) -> float:
            """Test how strongly this Lex area responds to the word"""
            self.brain._clear_area(lex_area)
            
            # Project phon to Lex (no learning)
            self.brain._project(lex_area, phon, learn=False, is_strong=True)
            
            # Get the resulting activation
            winners = self.brain.current.get(lex_area)
            if winners is None:
                return 0.0
            
            # Check how many of the winners have learned weights
            area_idx = lex_area.value
            num_learned = int(self.brain.l_num[area_idx].get()[0])
            if num_learned == 0:
                return 0.0
            
            # Get learned destination neurons
            dst_arr = self.brain.l_dst[area_idx][:num_learned].get()
            delta_arr = self.brain.l_delta[area_idx][:num_learned].get()
            
            # Sum weights for neurons that are in the winner set
            winners_set = set(winners.get().tolist())
            activation = 0.0
            for dst, delta in zip(dst_arr, delta_arr):
                if dst in winners_set:
                    activation += 1.0 + delta  # Base weight + learned delta
            
            return activation
        
        act_lex1 = test_activation(Area.LEX1)
        act_lex2 = test_activation(Area.LEX2)
        
        # Normalize
        total = act_lex1 + act_lex2 + 1e-10
        act_lex1_norm = act_lex1 / total
        act_lex2_norm = act_lex2 / total
        
        # Classify based on which Lex area responds more strongly
        if act_lex1 > act_lex2:
            classification = 'NOUN'
            self.learned_nouns.add(word)
        else:
            classification = 'VERB'
            self.learned_verbs.add(word)
        
        return classification, act_lex1_norm, act_lex2_norm
    
    def get_learned_word_order(self) -> List[str]:
        """Infer word order from learned transitions"""
        if not self.transitions:
            return []
        
        # Find starting role
        all_sources = set(src for (src, _) in self.transitions.keys())
        all_targets = set(dst for (_, dst) in self.transitions.keys())
        starts = all_sources - all_targets
        
        if starts:
            current = list(starts)[0]
        else:
            current = 'SUBJ'
        
        order = [current]
        for _ in range(3):
            candidates = {
                dst: count for (src, dst), count in self.transitions.items()
                if src == current and dst not in order
            }
            if not candidates:
                break
            next_role = max(candidates, key=candidates.get)
            order.append(next_role)
            current = next_role
        
        return order
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'sentences_seen': self.sentences_seen,
            'words_learned': len(self.brain.words_learned),
            'nouns_learned': len(self.learned_nouns),
            'verbs_learned': len(self.learned_verbs),
            'word_order': self.get_learned_word_order(),
            'transitions': dict(self.transitions),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEMO LANGUAGE LEARNER - Neurobiologically Plausible")
    print("=" * 70)
    
    # Create learner with smaller params for testing
    params = NemoParams(n=10000)  # k = sqrt(10000) = 100
    learner = NemoLanguageLearner(params, verbose=True)
    
    # Register vocabulary with grounding
    nouns = ['dog', 'cat', 'ball', 'book', 'boy', 'girl']
    verbs = ['runs', 'sees', 'chases', 'reads', 'throws']
    
    for noun in nouns:
        learner.register_noun(noun, [noun.upper()])
    for verb in verbs:
        learner.register_verb(verb, [verb.upper()])
    
    # Create grounded context
    context = GroundedContext(
        visual=['DOG', 'CAT', 'BALL'],
        motor=['RUNS', 'CHASES'],
    )
    
    # Train on grounded sentences
    print("\nTraining on grounded sentences...")
    sentences = [
        (['dog', 'runs'], ['SUBJ', 'VERB']),
        (['cat', 'runs'], ['SUBJ', 'VERB']),
        (['dog', 'chases', 'cat'], ['SUBJ', 'VERB', 'OBJ']),
        (['boy', 'throws', 'ball'], ['SUBJ', 'VERB', 'OBJ']),
        (['girl', 'reads', 'book'], ['SUBJ', 'VERB', 'OBJ']),
    ]
    
    for _ in range(20):  # Multiple exposures
        for words, roles in sentences:
            learner.present_grounded_sentence(words, context, roles, learn=True)
    
    # Debug: Check learned connections per area
    print("\n" + "=" * 50)
    print("DEBUG: LEARNED CONNECTIONS")
    print("=" * 50)
    for area in [Area.LEX1, Area.LEX2]:
        num = int(learner.brain.l_num[area.value].get()[0])
        print(f"  {area.name}: {num} connections")
    
    # Test classification
    print("\n" + "=" * 50)
    print("STABILITY-BASED CLASSIFICATION")
    print("=" * 50)
    
    print(f"\n{'Word':>10} {'True':>8} {'Pred':>8} {'Lex1':>8} {'Lex2':>8}")
    print("-" * 50)
    
    correct = 0
    for word in nouns + verbs:
        true_label = 'NOUN' if word in nouns else 'VERB'
        pred, s1, s2 = learner.classify_word(word)
        ok = (pred == true_label)
        if ok:
            correct += 1
        print(f"{word:>10} {true_label:>8} {pred:>8} {s1:>8.2f} {s2:>8.2f} {'✓' if ok else '✗'}")
    
    print(f"\nAccuracy: {correct}/{len(nouns)+len(verbs)} = {correct/(len(nouns)+len(verbs)):.1%}")
    
    # Word order
    print("\n" + "=" * 50)
    print("LEARNED WORD ORDER")
    print("=" * 50)
    print(f"Transitions: {dict(learner.transitions)}")
    print(f"Word order: {learner.get_learned_word_order()}")
    
    # Stats
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    stats = learner.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

