"""
NEMO Full Language System

Based on Mitropolsky & Papadimitriou 2025

Implements:
1. Word learning (semantics + noun/verb distinction)
2. Word order learning (syntax)
3. Sentence generation

Key components:
- Phon: Phonological input
- Visual, Motor: Semantic grounding
- Lex1, Lex2: Lexical areas (nouns, verbs)
- Role_agent, Role_action, Role_patient: Thematic roles (MUTUAL INHIBITION)
- Subj, Verb, Obj: Syntactic positions
- Mood: Grammatical mood

GPU-accelerated with proper saturating weight updates.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Check CUDA
assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name()}")


@dataclass
class NemoParams:
    """Parameters from the paper"""
    n: int = 100000          # Neurons per area (paper: 10^5)
    k_lex: int = 50          # Winners in Lex areas
    k_context: int = 20      # Winners in context areas
    k_other: int = 100       # Winners in other areas
    p: float = 0.05          # Connection probability
    beta: float = 0.06       # Plasticity
    tau: int = 2             # Firing steps per word
    w_max: float = 10.0      # Weight saturation


class NemoFullBrain:
    """
    Complete NEMO brain for language acquisition.
    
    Phase 1: Word learning (semantics + POS)
    Phase 2: Word order learning (syntax)
    """
    
    def __init__(self, params: NemoParams = None, verbose: bool = True):
        self.p = params or NemoParams()
        self.verbose = verbose
        
        # Use smaller n for faster iteration
        n = self.p.n
        
        # ========== AREAS ==========
        
        # Input areas (pre-initialized assemblies)
        self.phon_assemblies: Dict[str, torch.Tensor] = {}
        self.visual_assemblies: Dict[str, torch.Tensor] = {}
        self.motor_assemblies: Dict[str, torch.Tensor] = {}
        
        # Lexical areas
        self.lex1_activated: Optional[torch.Tensor] = None  # Nouns
        self.lex2_activated: Optional[torch.Tensor] = None  # Verbs
        
        # Role areas (UNDER MUTUAL INHIBITION)
        self.role_agent_activated: Optional[torch.Tensor] = None
        self.role_action_activated: Optional[torch.Tensor] = None
        self.role_patient_activated: Optional[torch.Tensor] = None
        
        # Syntactic areas
        self.subj_activated: Optional[torch.Tensor] = None
        self.verb_activated: Optional[torch.Tensor] = None
        self.obj_activated: Optional[torch.Tensor] = None
        
        # Mood area
        self.mood_assemblies: Dict[str, torch.Tensor] = {}
        self.mood_activated: Optional[torch.Tensor] = None
        
        # ========== WEIGHT MATRICES ==========
        
        # Phon -> Lex (strong)
        self.W_phon_lex1 = self._init_weights(n, n, self.p.p * 2)
        self.W_phon_lex2 = self._init_weights(n, n, self.p.p * 2)
        
        # Semantic -> Lex (differential: Visual->Lex1 strong, Motor->Lex2 strong)
        self.W_visual_lex1 = self._init_weights(n, n, self.p.p * 2)  # Strong
        self.W_motor_lex2 = self._init_weights(n, n, self.p.p * 2)   # Strong
        self.W_visual_lex2 = self._init_weights(n, n, self.p.p * 0.5)  # Weak
        self.W_motor_lex1 = self._init_weights(n, n, self.p.p * 0.5)   # Weak
        
        # Lex -> Semantic (for pathway)
        self.W_lex1_visual = self._init_weights(n, n, self.p.p * 2)
        self.W_lex2_motor = self._init_weights(n, n, self.p.p * 2)
        
        # Recurrent within Lex
        self.W_lex1_rec = self._init_weights(n, n, self.p.p)
        self.W_lex2_rec = self._init_weights(n, n, self.p.p)
        
        # Lex -> Role (for word order learning)
        self.W_lex1_role_agent = self._init_weights(n, n, self.p.p)
        self.W_lex1_role_patient = self._init_weights(n, n, self.p.p)
        self.W_lex2_role_action = self._init_weights(n, n, self.p.p)
        
        # Role -> Syntax
        self.W_role_agent_subj = self._init_weights(n, n, self.p.p)
        self.W_role_action_verb = self._init_weights(n, n, self.p.p)
        self.W_role_patient_obj = self._init_weights(n, n, self.p.p)
        
        # Syntax -> Role (for word order: which role comes after which syntax)
        self.W_subj_role_action = self._init_weights(n, n, self.p.p)
        self.W_subj_role_patient = self._init_weights(n, n, self.p.p)
        self.W_verb_role_agent = self._init_weights(n, n, self.p.p)
        self.W_verb_role_patient = self._init_weights(n, n, self.p.p)
        self.W_obj_role_action = self._init_weights(n, n, self.p.p)
        
        # Mood -> Role (for different word orders per mood)
        self.W_mood_role_agent = self._init_weights(n, n, self.p.p)
        self.W_mood_role_action = self._init_weights(n, n, self.p.p)
        self.W_mood_role_patient = self._init_weights(n, n, self.p.p)
        
        # Recurrent within Role and Syntax
        self.W_role_agent_rec = self._init_weights(n, n, self.p.p)
        self.W_role_action_rec = self._init_weights(n, n, self.p.p)
        self.W_role_patient_rec = self._init_weights(n, n, self.p.p)
        
        self.sentences_seen = 0
        
        if verbose:
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"NemoFullBrain initialized: n={n}")
            print(f"  GPU Memory: {mem:.2f} GB")
    
    def _init_weights(self, n_src: int, n_dst: int, p: float) -> torch.Tensor:
        """Initialize sparse weight matrix - always use dense for simplicity"""
        W = torch.zeros(n_src, n_dst, device=DEVICE)
        mask = torch.rand(n_src, n_dst, device=DEVICE) < p
        W[mask] = 1.0
        del mask  # Free memory
        return W
    
    def saturating_update(self, W: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        """Saturating weight update: delta = beta * (1 - w/w_max)"""
        idx = (src.unsqueeze(1), dst.unsqueeze(0))
        current_w = W[idx]
        delta = self.p.beta * (1.0 - current_w / self.p.w_max)
        delta = torch.clamp(delta, min=0)
        W[idx] = current_w + delta
    
    def create_assembly(self, area: str, name: str) -> torch.Tensor:
        """Create random assembly"""
        k = self.p.k_lex if area in ['Phon', 'Visual', 'Motor'] else self.p.k_other
        indices = torch.randperm(self.p.n, device=DEVICE)[:k]
        
        if area == 'Phon':
            self.phon_assemblies[name] = indices
        elif area == 'Visual':
            self.visual_assemblies[name] = indices
        elif area == 'Motor':
            self.motor_assemblies[name] = indices
        elif area == 'Mood':
            self.mood_assemblies[name] = indices
        
        return indices
    
    def inhibit_all(self):
        """Reset all activations"""
        self.lex1_activated = None
        self.lex2_activated = None
        self.role_agent_activated = None
        self.role_action_activated = None
        self.role_patient_activated = None
        self.subj_activated = None
        self.verb_activated = None
        self.obj_activated = None
        self.mood_activated = None
    
    def inhibit_roles(self):
        """Inhibit only Role areas (for generation trigger)"""
        self.role_agent_activated = None
        self.role_action_activated = None
        self.role_patient_activated = None
    
    # ========== PHASE 1: WORD LEARNING ==========
    
    def present_word(self, word: str, is_noun: bool, learn: bool = True):
        """
        Present a single word with grounding.
        
        Nouns: Phon + Visual -> Lex1
        Verbs: Phon + Motor -> Lex2
        """
        if word not in self.phon_assemblies:
            self.create_assembly('Phon', word)
        
        phon = self.phon_assemblies[word]
        
        if is_noun:
            if word not in self.visual_assemblies:
                self.create_assembly('Visual', word)
            visual = self.visual_assemblies[word]
            
            # Compute Lex1 input
            total_input = self.W_phon_lex1[phon].sum(dim=0)
            total_input += self.W_visual_lex1[visual].sum(dim=0)
            if self.lex1_activated is not None:
                total_input += self.W_lex1_rec[self.lex1_activated].sum(dim=0)
            
            _, winners = torch.topk(total_input, self.p.k_lex)
            
            if learn:
                if self.lex1_activated is not None:
                    self.saturating_update(self.W_lex1_rec, self.lex1_activated, winners)
                self.saturating_update(self.W_visual_lex1, visual, winners)
                self.saturating_update(self.W_lex1_visual, winners, visual)
            
            self.lex1_activated = winners
            
        else:  # Verb
            if word not in self.motor_assemblies:
                self.create_assembly('Motor', word)
            motor = self.motor_assemblies[word]
            
            # Compute Lex2 input
            total_input = self.W_phon_lex2[phon].sum(dim=0)
            total_input += self.W_motor_lex2[motor].sum(dim=0)
            if self.lex2_activated is not None:
                total_input += self.W_lex2_rec[self.lex2_activated].sum(dim=0)
            
            _, winners = torch.topk(total_input, self.p.k_lex)
            
            if learn:
                if self.lex2_activated is not None:
                    self.saturating_update(self.W_lex2_rec, self.lex2_activated, winners)
                self.saturating_update(self.W_motor_lex2, motor, winners)
                self.saturating_update(self.W_lex2_motor, winners, motor)
            
            self.lex2_activated = winners
    
    def present_grounded_sentence_phase1(self, subject: str, verb: str, 
                                          obj: str = None, learn: bool = True):
        """
        Phase 1: Learn word semantics and noun/verb distinction.
        
        Present: "the dog runs" or "the dog eats the food"
        
        Key insight from paper: Words that appear with more "complements" 
        (different contexts) have less stable representations.
        """
        self.inhibit_all()
        
        # Present subject (noun)
        for _ in range(self.p.tau):
            self.present_word(subject, is_noun=True, learn=learn)
        
        # Reset between words (as per paper)
        self.lex1_activated = None
        self.lex2_activated = None
        
        # Present verb
        for _ in range(self.p.tau):
            self.present_word(verb, is_noun=False, learn=learn)
        
        # Present object if transitive
        if obj is not None:
            # Reset between words
            self.lex1_activated = None
            self.lex2_activated = None
            
            for _ in range(self.p.tau):
                self.present_word(obj, is_noun=True, learn=learn)
        
        if learn:
            self.sentences_seen += 1
    
    # ========== PHASE 2: WORD ORDER LEARNING ==========
    
    def present_grounded_sentence_phase2(self, subject: str, verb: str,
                                          obj: str = None, mood: str = 'declarative',
                                          word_order: str = 'SVO', learn: bool = True):
        """
        Phase 2: Learn word order.
        
        The scene (Role areas) is prepared with thematic roles.
        Words are presented in the language's word order.
        Plasticity learns which Role follows which Syntax position.
        """
        self.inhibit_all()
        
        # Ensure mood assembly exists
        if mood not in self.mood_assemblies:
            self.create_assembly('Mood', mood)
        self.mood_activated = self.mood_assemblies[mood]
        
        # Prepare scene in Role areas (from visual/motor grounding)
        # Role_agent <- subject's visual representation
        # Role_action <- verb's motor representation
        # Role_patient <- object's visual representation (if transitive)
        
        if subject in self.visual_assemblies:
            visual_subj = self.visual_assemblies[subject]
            # Project to Role_agent
            input_agent = self.W_lex1_role_agent[visual_subj].sum(dim=0)
            _, self.role_agent_activated = torch.topk(input_agent, self.p.k_other)
        
        if verb in self.motor_assemblies:
            motor_verb = self.motor_assemblies[verb]
            # Project to Role_action
            input_action = self.W_lex2_role_action[motor_verb].sum(dim=0)
            _, self.role_action_activated = torch.topk(input_action, self.p.k_other)
        
        if obj is not None and obj in self.visual_assemblies:
            visual_obj = self.visual_assemblies[obj]
            # Project to Role_patient
            input_patient = self.W_lex1_role_patient[visual_obj].sum(dim=0)
            _, self.role_patient_activated = torch.topk(input_patient, self.p.k_other)
        
        # Present words in order, learning Role -> Syntax connections
        if word_order == 'SVO':
            order = [(subject, 'SUBJ'), (verb, 'VERB')]
            if obj:
                order.append((obj, 'OBJ'))
        elif word_order == 'SOV':
            order = [(subject, 'SUBJ')]
            if obj:
                order.append((obj, 'OBJ'))
            order.append((verb, 'VERB'))
        elif word_order == 'VSO':
            order = [(verb, 'VERB'), (subject, 'SUBJ')]
            if obj:
                order.append((obj, 'OBJ'))
        else:
            order = [(subject, 'SUBJ'), (verb, 'VERB')]
            if obj:
                order.append((obj, 'OBJ'))
        
        prev_syntax = None
        
        for word, syntax_role in order:
            # Present word in Phon
            phon = self.phon_assemblies.get(word)
            if phon is None:
                continue
            
            # Fire through Lex to Role
            is_noun = word in self.visual_assemblies
            
            if is_noun:
                # Noun -> Lex1 -> Role_agent or Role_patient
                total_input = self.W_phon_lex1[phon].sum(dim=0)
                if self.lex1_activated is not None:
                    total_input += self.W_lex1_rec[self.lex1_activated].sum(dim=0)
                _, self.lex1_activated = torch.topk(total_input, self.p.k_lex)
                
                # Determine which Role fires (mutual inhibition)
                # First noun = agent, second noun = patient
                if syntax_role == 'SUBJ':
                    role_activated = self.role_agent_activated
                    W_role_syntax = self.W_role_agent_subj
                    self.subj_activated = self._fire_syntax('SUBJ', role_activated, learn)
                else:  # OBJ
                    role_activated = self.role_patient_activated
                    W_role_syntax = self.W_role_patient_obj
                    self.obj_activated = self._fire_syntax('OBJ', role_activated, learn)
            else:
                # Verb -> Lex2 -> Role_action
                total_input = self.W_phon_lex2[phon].sum(dim=0)
                if self.lex2_activated is not None:
                    total_input += self.W_lex2_rec[self.lex2_activated].sum(dim=0)
                _, self.lex2_activated = torch.topk(total_input, self.p.k_lex)
                
                role_activated = self.role_action_activated
                self.verb_activated = self._fire_syntax('VERB', role_activated, learn)
            
            # Learn: previous syntax -> current role
            if learn and prev_syntax is not None:
                self._learn_word_order(prev_syntax, syntax_role)
            
            prev_syntax = syntax_role
        
        if learn:
            self.sentences_seen += 1
    
    def _fire_syntax(self, syntax: str, role_activated: torch.Tensor, 
                     learn: bool) -> torch.Tensor:
        """Fire from Role to Syntax area"""
        if role_activated is None:
            return None
        
        if syntax == 'SUBJ':
            W = self.W_role_agent_subj
        elif syntax == 'VERB':
            W = self.W_role_action_verb
        else:  # OBJ
            W = self.W_role_patient_obj
        
        total_input = W[role_activated].sum(dim=0)
        _, winners = torch.topk(total_input, self.p.k_other)
        
        if learn:
            self.saturating_update(W, role_activated, winners)
        
        return winners
    
    def _learn_word_order(self, prev_syntax: str, curr_role: str):
        """Learn which Role follows which Syntax position"""
        # Get the appropriate weight matrix
        if prev_syntax == 'SUBJ':
            if curr_role == 'VERB':
                W = self.W_subj_role_action
                src = self.subj_activated
                dst = self.role_action_activated
            elif curr_role == 'OBJ':
                W = self.W_subj_role_patient
                src = self.subj_activated
                dst = self.role_patient_activated
            else:
                return
        elif prev_syntax == 'VERB':
            if curr_role == 'SUBJ':
                W = self.W_verb_role_agent
                src = self.verb_activated
                dst = self.role_agent_activated
            elif curr_role == 'OBJ':
                W = self.W_verb_role_patient
                src = self.verb_activated
                dst = self.role_patient_activated
            else:
                return
        elif prev_syntax == 'OBJ':
            if curr_role == 'VERB':
                W = self.W_obj_role_action
                src = self.obj_activated
                dst = self.role_action_activated
            else:
                return
        else:
            return
        
        if src is not None and dst is not None:
            self.saturating_update(W, src, dst)
    
    # ========== GENERATION ==========
    
    def generate_sentence(self, subject: str, verb: str, obj: str = None,
                          mood: str = 'declarative') -> List[str]:
        """
        Generate a sentence given a scene.
        
        Uses the trigger mechanism from the paper:
        1. Prepare scene (activate Role areas from semantic input)
        2. For each word position:
           a. Inhibit all Role areas (trigger)
           b. Compute input to each Role from Mood and previous Syntax
           c. The Role with max input wins (mutual inhibition)
           d. Fire corresponding Syntax area
           e. Output corresponding word
        """
        self.inhibit_all()
        
        # Set mood
        if mood in self.mood_assemblies:
            self.mood_activated = self.mood_assemblies[mood]
        
        # Store the semantic assemblies for the scene
        # These represent the "meaning" we want to express
        scene = {
            'agent': self.visual_assemblies.get(subject),
            'action': self.motor_assemblies.get(verb),
            'patient': self.visual_assemblies.get(obj) if obj else None
        }
        
        # Word mapping
        role_to_word = {
            'agent': subject,
            'action': verb,
            'patient': obj
        }
        
        # Generate words
        output = []
        used_roles = set()
        max_words = 3 if obj else 2
        
        prev_syntax = None
        
        for position in range(max_words):
            # Compute input to each Role area
            role_scores = {}
            
            for role in ['agent', 'action', 'patient']:
                if role in used_roles:
                    continue
                if scene[role] is None:
                    continue
                
                score = 0.0
                
                # 1. Input from the scene (semantic grounding)
                # This is the "what to say" - always present
                if role == 'agent':
                    score += self.W_lex1_role_agent[scene['agent']].sum().item()
                elif role == 'action':
                    score += self.W_lex2_role_action[scene['action']].sum().item()
                elif role == 'patient' and scene['patient'] is not None:
                    score += self.W_lex1_role_patient[scene['patient']].sum().item()
                
                # 2. Input from previous Syntax position (word order)
                # This is "what comes next" based on LEARNED order
                if prev_syntax is not None:
                    if prev_syntax == 'SUBJ' and self.subj_activated is not None:
                        if role == 'action':
                            score += self.W_subj_role_action[self.subj_activated].sum().item()
                        elif role == 'patient':
                            score += self.W_subj_role_patient[self.subj_activated].sum().item()
                    elif prev_syntax == 'VERB' and self.verb_activated is not None:
                        if role == 'patient':
                            score += self.W_verb_role_patient[self.verb_activated].sum().item()
                        elif role == 'agent':
                            score += self.W_verb_role_agent[self.verb_activated].sum().item()
                    elif prev_syntax == 'OBJ' and self.obj_activated is not None:
                        if role == 'action':
                            score += self.W_obj_role_action[self.obj_activated].sum().item()
                else:
                    # First position - use learned Mood -> Role weights
                    # For now, bias toward agent (subject-first is common)
                    if role == 'agent':
                        score *= 1.2
                
                role_scores[role] = score
            
            if not role_scores:
                break
            
            # Mutual inhibition: winner takes all
            winner_role = max(role_scores, key=role_scores.get)
            used_roles.add(winner_role)
            
            # Output word
            word = role_to_word[winner_role]
            if word:
                output.append(word)
            
            # Update syntax state
            if winner_role == 'agent':
                # Fire Subj area
                if scene['agent'] is not None:
                    input_subj = self.W_role_agent_subj[scene['agent']].sum(dim=0)
                    _, self.subj_activated = torch.topk(input_subj, self.p.k_other)
                prev_syntax = 'SUBJ'
            elif winner_role == 'action':
                # Fire Verb area
                if scene['action'] is not None:
                    input_verb = self.W_role_action_verb[scene['action']].sum(dim=0)
                    _, self.verb_activated = torch.topk(input_verb, self.p.k_other)
                prev_syntax = 'VERB'
            elif winner_role == 'patient':
                # Fire Obj area
                if scene['patient'] is not None:
                    input_obj = self.W_role_patient_obj[scene['patient']].sum(dim=0)
                    _, self.obj_activated = torch.topk(input_obj, self.p.k_other)
                prev_syntax = 'OBJ'
        
        return output
    
    # ========== TESTING ==========
    
    def measure_stability(self, word: str, area: str, n_rounds: int = 10) -> float:
        """Measure assembly stability (for classification)"""
        if word not in self.phon_assemblies:
            return 0.0
        
        phon = self.phon_assemblies[word]
        
        # Get semantic grounding
        visual = self.visual_assemblies.get(word)
        motor = self.motor_assemblies.get(word)
        
        first_activation = None
        activated = None
        
        for i in range(n_rounds):
            if area == 'Lex1':
                total_input = self.W_phon_lex1[phon].sum(dim=0)
                if visual is not None:
                    total_input += self.W_visual_lex1[visual].sum(dim=0)
                if activated is not None:
                    total_input += self.W_lex1_rec[activated].sum(dim=0)
                _, activated = torch.topk(total_input, self.p.k_lex)
            else:  # Lex2
                total_input = self.W_phon_lex2[phon].sum(dim=0)
                if motor is not None:
                    total_input += self.W_motor_lex2[motor].sum(dim=0)
                if activated is not None:
                    total_input += self.W_lex2_rec[activated].sum(dim=0)
                _, activated = torch.topk(total_input, self.p.k_lex)
            
            if i == 0:
                first_activation = set(activated.cpu().numpy())
        
        if first_activation is None or activated is None:
            return 0.0
        
        last_activation = set(activated.cpu().numpy())
        return len(first_activation & last_activation) / self.p.k_lex
    
    def is_wobbly(self, word: str, area: str, threshold: float = 0.5) -> bool:
        """
        Check if a word's assembly is "wobbly" in the given area.
        
        From the paper: "firing this set again recurrently, even with the 
        input of Phon[w], results in a quite different set"
        
        A wobbly assembly indicates the word doesn't belong in that area.
        This can be used to detect parsing errors.
        
        Args:
            word: The word to test
            area: 'Lex1' or 'Lex2'
            threshold: Stability below this is considered wobbly
            
        Returns:
            True if the assembly is wobbly (unstable)
        """
        stability = self.measure_stability(word, area)
        return stability < threshold
    
    def detect_pos(self, word: str) -> str:
        """
        Detect part-of-speech using stability comparison.
        
        Returns 'NOUN', 'VERB', or 'UNKNOWN'
        """
        s1 = self.measure_stability(word, 'Lex1')
        s2 = self.measure_stability(word, 'Lex2')
        
        # If both are wobbly, it's an unknown word
        if s1 < 0.5 and s2 < 0.5:
            return 'UNKNOWN'
        
        return 'NOUN' if s1 > s2 else 'VERB'
    
    def validate_parse(self, words: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate a parse by checking for wobbly assemblies.
        
        Returns:
            (is_valid, list of errors)
        """
        errors = []
        
        for word in words:
            if word not in self.phon_assemblies:
                errors.append(f"Unknown word: {word}")
                continue
            
            pos = self.detect_pos(word)
            if pos == 'UNKNOWN':
                errors.append(f"Wobbly word (no clear POS): {word}")
        
        return len(errors) == 0, errors


def run_full_experiment(scale: str = 'small'):
    """
    Run the full experiment.
    
    Args:
        scale: 'small' (n=10k), 'medium' (n=50k), or 'paper' (n=100k)
    """
    print("=" * 70)
    print("NEMO FULL LANGUAGE SYSTEM")
    print("Based on Mitropolsky & Papadimitriou 2025")
    print("=" * 70)
    
    # Memory calculation: 27 matrices × n² × 4 bytes
    # n=10,000: 10.8 GB (fits in 16GB)
    # n=100,000: 1080 GB (requires sparse matrices)
    
    if scale == 'paper':
        # Paper parameters with our max dense size
        # Paper uses n=100,000 but we're limited by VRAM
        params = NemoParams(n=10000, k_lex=50, k_other=100, p=0.05, beta=0.06)
        print(f"\nUsing PAPER parameters (β=0.06) with n={params.n:,}")
        print("  Note: Paper uses n=100,000 which requires sparse matrices")
    elif scale == 'medium':
        # Slightly more training data
        params = NemoParams(n=10000, k_lex=50, k_other=100, p=0.05, beta=0.08)
        print(f"\nUsing MEDIUM parameters: n={params.n:,}")
    else:
        params = NemoParams(n=10000, k_lex=50, k_other=100, p=0.05, beta=0.1)
        print(f"\nUsing SMALL parameters: n={params.n:,}")
    
    brain = NemoFullBrain(params, verbose=True)
    
    # Define vocabulary
    nouns = ['dog', 'cat', 'boy', 'girl', 'ball', 'food', 'bird', 'man', 'woman', 'baby']
    intransitive_verbs = ['runs', 'sleeps', 'jumps', 'walks', 'flies']
    transitive_verbs = ['eats', 'sees', 'has', 'wants', 'likes']
    all_verbs = intransitive_verbs + transitive_verbs
    
    # ========== PHASE 1: WORD LEARNING ==========
    print("\n" + "=" * 70)
    print("PHASE 1: WORD LEARNING (Mixed Sentences)")
    print("=" * 70)
    
    # Train on MIXED sentences (intransitive + transitive)
    # Paper: "if transitive sentences are not more than roughly a half"
    n_sentences = 200
    n_intransitive = n_sentences // 2
    n_transitive = n_sentences - n_intransitive
    
    print(f"\nTraining on {n_intransitive} intransitive + {n_transitive} transitive sentences...")
    
    start = time.perf_counter()
    
    # Shuffle order
    sentence_types = ['intransitive'] * n_intransitive + ['transitive'] * n_transitive
    np.random.shuffle(sentence_types)
    
    for sent_type in sentence_types:
        if sent_type == 'intransitive':
            noun = np.random.choice(nouns)
            verb = np.random.choice(intransitive_verbs)
            brain.present_grounded_sentence_phase1(noun, verb)
        else:
            subj = np.random.choice(nouns)
            verb = np.random.choice(transitive_verbs)
            obj = np.random.choice([n for n in nouns if n != subj])
            brain.present_grounded_sentence_phase1(subj, verb, obj=obj)
    
    train_time = time.perf_counter() - start
    print(f"Training: {train_time:.2f}s ({n_sentences/train_time:.1f} sentences/sec)")
    
    # Test classification
    print("\nClassification results:")
    print(f"{'Word':<10} {'Type':<6} {'Lex1':<6} {'Lex2':<6} {'Pred':<6} {'OK'}")
    print("-" * 50)
    
    correct = 0
    for word in nouns + all_verbs:
        is_noun = word in nouns
        s1 = brain.measure_stability(word, 'Lex1')
        s2 = brain.measure_stability(word, 'Lex2')
        pred_noun = s1 > s2
        ok = pred_noun == is_noun
        if ok:
            correct += 1
        expected = 'NOUN' if is_noun else 'VERB'
        predicted = 'NOUN' if pred_noun else 'VERB'
        print(f"{word:<10} {expected:<6} {s1:<6.2f} {s2:<6.2f} {predicted:<6} {'✓' if ok else '✗'}")
    
    print(f"\nClassification Accuracy: {correct}/{len(nouns)+len(all_verbs)} = {correct/(len(nouns)+len(all_verbs)):.1%}")
    
    # ========== PHASE 2: WORD ORDER LEARNING ==========
    print("\n" + "=" * 70)
    print("PHASE 2: WORD ORDER LEARNING (SVO)")
    print("=" * 70)
    
    # Train on mixed sentences with SVO order
    n_sentences = 200
    print(f"\nTraining on {n_sentences} sentences (SVO word order)...")
    
    start = time.perf_counter()
    
    for _ in range(n_sentences):
        subj = np.random.choice(nouns)
        
        # Mix intransitive and transitive
        if np.random.random() < 0.5:
            # Intransitive: just Subject + Verb
            verb = np.random.choice(intransitive_verbs)
            brain.present_grounded_sentence_phase2(subj, verb, obj=None, word_order='SVO')
        else:
            # Transitive: Subject + Verb + Object
            verb = np.random.choice(transitive_verbs)
            obj = np.random.choice([n for n in nouns if n != subj])
            brain.present_grounded_sentence_phase2(subj, verb, obj, word_order='SVO')
    
    train_time = time.perf_counter() - start
    print(f"Training: {train_time:.2f}s ({n_sentences/train_time:.1f} sentences/sec)")
    
    # Test generation - transitive
    print("\nGeneration test (Transitive SVO):")
    test_cases_trans = [
        ('dog', 'eats', 'food'),
        ('cat', 'sees', 'ball'),
        ('boy', 'has', 'dog'),
        ('girl', 'wants', 'bird'),
        ('man', 'likes', 'woman'),
    ]
    
    correct_trans = 0
    for subj, verb, obj in test_cases_trans:
        output = brain.generate_sentence(subj, verb, obj)
        expected = [subj, verb, obj]
        match = output == expected
        if match:
            correct_trans += 1
        print(f"  {subj} {verb} {obj} -> {' '.join(output)} {'✓' if match else '✗'}")
    
    print(f"\nTransitive accuracy: {correct_trans}/{len(test_cases_trans)} = {correct_trans/len(test_cases_trans):.1%}")
    
    # Test generation - intransitive
    print("\nGeneration test (Intransitive SV):")
    test_cases_intrans = [
        ('dog', 'runs', None),
        ('cat', 'sleeps', None),
        ('bird', 'flies', None),
        ('baby', 'walks', None),
    ]
    
    correct_intrans = 0
    for subj, verb, obj in test_cases_intrans:
        output = brain.generate_sentence(subj, verb, obj)
        expected = [subj, verb]
        match = output == expected
        if match:
            correct_intrans += 1
        print(f"  {subj} {verb} -> {' '.join(output)} {'✓' if match else '✗'}")
    
    print(f"\nIntransitive accuracy: {correct_intrans}/{len(test_cases_intrans)} = {correct_intrans/len(test_cases_intrans):.1%}")
    
    # ========== PHASE 3: TEST DIFFERENT WORD ORDERS ==========
    print("\n" + "=" * 70)
    print("PHASE 3: LEARNING DIFFERENT WORD ORDER (SOV)")
    print("=" * 70)
    
    # Create a new brain for SOV language
    brain_sov = NemoFullBrain(params, verbose=False)
    
    # Phase 1: Word learning
    print("\nPhase 1: Word learning...")
    for _ in range(200):
        if np.random.random() < 0.5:
            noun = np.random.choice(nouns)
            verb = np.random.choice(intransitive_verbs)
            brain_sov.present_grounded_sentence_phase1(noun, verb)
        else:
            subj = np.random.choice(nouns)
            verb = np.random.choice(transitive_verbs)
            obj = np.random.choice([n for n in nouns if n != subj])
            brain_sov.present_grounded_sentence_phase1(subj, verb, obj=obj)
    
    # Phase 2: SOV word order - MORE training
    print("Phase 2: Training on SOV word order (500 sentences)...")
    for _ in range(500):
        subj = np.random.choice(nouns)
        verb = np.random.choice(transitive_verbs)
        obj = np.random.choice([n for n in nouns if n != subj])
        brain_sov.present_grounded_sentence_phase2(subj, verb, obj, word_order='SOV')
    
    # Debug: Check learned weights
    print("\nWeight analysis (SOV brain):")
    print(f"  W_subj_role_action max: {brain_sov.W_subj_role_action.max():.2f}")
    print(f"  W_subj_role_patient max: {brain_sov.W_subj_role_patient.max():.2f}")
    print(f"  W_obj_role_action max: {brain_sov.W_obj_role_action.max():.2f}")
    print(f"  W_verb_role_patient max: {brain_sov.W_verb_role_patient.max():.2f}")
    
    # For SOV: SUBJ -> OBJ (patient), OBJ -> VERB (action)
    # So W_subj_role_patient and W_obj_role_action should be high
    
    # Test SOV generation
    print("\nGeneration test (SOV):")
    correct_sov = 0
    for subj, verb, obj in test_cases_trans:
        output = brain_sov.generate_sentence(subj, verb, obj)
        expected = [subj, obj, verb]  # SOV order!
        match = output == expected
        if match:
            correct_sov += 1
        print(f"  Scene: {subj} {verb} {obj}")
        print(f"  Generated: {' '.join(output)}")
        print(f"  Expected (SOV): {' '.join(expected)} {'✓' if match else '✗'}")
    
    print(f"\nSOV accuracy: {correct_sov}/{len(test_cases_trans)} = {correct_sov/len(test_cases_trans):.1%}")
    
    # ========== PHASE 4: WOBBLY TEST ==========
    print("\n" + "=" * 70)
    print("PHASE 4: WOBBLY TEST (Parsing Validation)")
    print("=" * 70)
    
    print("\nTesting wobbly detection on known words:")
    for word in ['dog', 'runs', 'cat', 'eats']:
        s1 = brain.measure_stability(word, 'Lex1')
        s2 = brain.measure_stability(word, 'Lex2')
        pos = brain.detect_pos(word)
        wobbly1 = brain.is_wobbly(word, 'Lex1')
        wobbly2 = brain.is_wobbly(word, 'Lex2')
        print(f"  {word}: Lex1={s1:.2f} ({'wobbly' if wobbly1 else 'stable'}), "
              f"Lex2={s2:.2f} ({'wobbly' if wobbly2 else 'stable'}) -> {pos}")
    
    # Test with unknown word
    print("\nTesting with unknown word 'xyz':")
    brain.create_assembly('Phon', 'xyz')  # Just phonology, no grounding
    s1 = brain.measure_stability('xyz', 'Lex1')
    s2 = brain.measure_stability('xyz', 'Lex2')
    pos = brain.detect_pos('xyz')
    print(f"  xyz: Lex1={s1:.2f}, Lex2={s2:.2f} -> {pos}")
    
    # Validate a parse
    print("\nValidating parses:")
    valid, errors = brain.validate_parse(['dog', 'eats', 'food'])
    print(f"  'dog eats food': {'valid' if valid else 'invalid'} {errors}")
    valid, errors = brain.validate_parse(['runs', 'dog', 'eats'])  # Wrong order
    print(f"  'runs dog eats': {'valid' if valid else 'invalid'} {errors}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Scale: {scale}")
    print(f"  Parameters: n={params.n:,}, k_lex={params.k_lex}, p={params.p}, β={params.beta}")
    print(f"  Classification: {correct}/{len(nouns)+len(all_verbs)} = {correct/(len(nouns)+len(all_verbs)):.1%}")
    print(f"  SVO Generation: {correct_trans + correct_intrans}/{len(test_cases_trans) + len(test_cases_intrans)}")
    print(f"  SOV Generation: {correct_sov}/{len(test_cases_trans)}")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU Memory: {mem:.2f} GB")


if __name__ == "__main__":
    import sys
    scale = sys.argv[1] if len(sys.argv) > 1 else 'small'
    run_full_experiment(scale=scale)

