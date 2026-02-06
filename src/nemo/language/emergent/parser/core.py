"""
NEMO Emergent Parser Core
=========================

Version: 1.0.0
Date: 2025-11-30

Core parsing logic using learned assemblies.

Key insight from parser.py:
- Parse by activating words, applying rules, and projecting
- Track fiber activations for readout
- Use assembly overlap to determine roles
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from ..learner import EmergentLanguageLearner

from ..areas import Area

__all__ = ['SentenceParser', 'ParseResult']


@dataclass
class ParseResult:
    """Result of parsing a sentence."""
    
    # Syntactic roles
    subject: Optional[str] = None
    verb: Optional[str] = None
    object: Optional[str] = None
    
    # Thematic roles
    agent: Optional[str] = None
    patient: Optional[str] = None
    
    # Modifiers
    adjectives: List[str] = field(default_factory=list)
    adverbs: List[str] = field(default_factory=list)
    determiners: List[str] = field(default_factory=list)
    prepositions: List[str] = field(default_factory=list)
    
    # Confidence and metadata
    confidence: float = 0.0
    word_roles: Dict[str, List[str]] = field(default_factory=dict)
    word_categories: Dict[str, str] = field(default_factory=dict)
    
    # Activated fibers (for readout, inspired by parser.py)
    activated_fibers: Dict[str, List[str]] = field(default_factory=dict)
    
    # Error info
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'subject': self.subject,
            'verb': self.verb,
            'object': self.object,
            'agent': self.agent,
            'patient': self.patient,
            'adjectives': self.adjectives,
            'adverbs': self.adverbs,
            'determiners': self.determiners,
            'prepositions': self.prepositions,
            'confidence': self.confidence,
            'word_roles': self.word_roles,
            'word_categories': self.word_categories,
            'error': self.error,
        }
    
    def __str__(self) -> str:
        if self.error:
            return f"ParseError: {self.error}"
        
        parts = []
        if self.subject:
            parts.append(f"SUBJ={self.subject}")
        if self.verb:
            parts.append(f"VERB={self.verb}")
        if self.object:
            parts.append(f"OBJ={self.object}")
        parts.append(f"conf={self.confidence:.2f}")
        
        return f"Parse({', '.join(parts)})"


class SentenceParser:
    """
    Parses sentences using learned assemblies.
    
    NEMO approach:
    1. Categorize each word using learned grounding
    2. Identify positional candidates for roles
    3. Use learned VP assemblies to confirm subject-verb-object
    4. Track activated fibers for readout
    
    Inspired by parser.py:
    - Track which fibers were activated during parsing
    - Use assembly overlap for word matching
    - Support for modifiers (ADJ, ADV, DET, PREP)
    """
    
    def __init__(self, learner: 'EmergentLanguageLearner'):
        self.learner = learner
        self.brain = learner.brain
        
        # Track activated fibers during parsing (like parser.py)
        self.activated_fibers: Dict[Area, List[Area]] = {}
    
    def parse(self, words: List[str]) -> ParseResult:
        """
        Parse a sentence into its grammatical structure.
        
        Args:
            words: List of words in the sentence
            
        Returns:
            ParseResult with subject, verb, object, etc.
        """
        self.brain.clear_all()
        self.activated_fibers = {}
        
        result = ParseResult()
        
        # === Step 1: Categorize words ===
        word_categories = {}
        for word in words:
            cat, _ = self.learner.get_emergent_category(word)
            word_categories[word] = cat
        result.word_categories = word_categories
        
        # === Step 2: Extract by category ===
        nouns = [w for w in words if word_categories[w] in ['NOUN', 'PRONOUN']]
        verbs = [w for w in words if word_categories[w] == 'VERB']
        adjectives = [w for w in words if word_categories[w] == 'ADJECTIVE']
        adverbs = [w for w in words if word_categories[w] == 'ADVERB']
        determiners = [w for w in words if word_categories[w] == 'FUNCTION' 
                       and w in ['the', 'a', 'an']]
        prepositions = [w for w in words if word_categories[w] == 'PREPOSITION']
        
        result.adjectives = adjectives
        result.adverbs = adverbs
        result.determiners = determiners
        result.prepositions = prepositions
        
        if not verbs:
            result.error = 'No verb found'
            return result
        
        # === Step 3: Find verb position ===
        verb_position = None
        for i, word in enumerate(words):
            if word_categories[word] == 'VERB':
                verb_position = i
                break
        
        result.verb = verbs[0]
        
        # === Step 4: Identify subject/object candidates by position ===
        subject_candidates = []
        object_candidates = []
        
        for i, word in enumerate(words):
            if word_categories[word] in ['NOUN', 'PRONOUN']:
                if verb_position is not None and i < verb_position:
                    subject_candidates.append(word)
                elif verb_position is not None and i > verb_position:
                    object_candidates.append(word)
        
        # === Step 5: Use learned assemblies to confirm roles ===
        result.subject = self._find_best_subject(
            subject_candidates, result.verb, word_categories)
        
        result.object = self._find_best_object(
            object_candidates, result.subject, result.verb)
        
        # === Step 6: Assign thematic roles ===
        result.agent = result.subject
        result.patient = result.object
        
        # === Step 7: Calculate confidence ===
        result.confidence = self._calculate_confidence(result)
        
        # === Step 8: Build word roles ===
        result.word_roles = self._build_word_roles(words, result, word_categories)
        
        return result
    
    def _find_best_subject(self, candidates: List[str], verb: str,
                           word_categories: Dict[str, str]) -> Optional[str]:
        """Find the best subject from candidates using learned assemblies."""
        if not candidates:
            return None
        
        best_subject = None
        best_score = 0.0
        
        for candidate in candidates:
            # Check for learned subject-verb VP assembly
            vp_key = f"{candidate}_{verb}"
            vp_assembly = self.brain.get_learned_assembly(Area.VP, vp_key)
            
            if vp_assembly is not None:
                score = 1.0  # Perfect match
            else:
                # Check assembly overlap
                subj_assembly = self._get_noun_assembly(candidate)
                verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
                
                if subj_assembly is not None and verb_assembly is not None:
                    self.brain._clear_area(Area.VP)
                    self.brain._project(Area.VP, subj_assembly, learn=False)
                    
                    # Track fiber activation
                    self._record_fiber(Area.NOUN_CORE, Area.VP)
                    
                    if self.brain.current[Area.VP] is not None:
                        score = self.brain.get_assembly_overlap(
                            self.brain.current[Area.VP], verb_assembly)
                    else:
                        score = 0.0
                else:
                    # Fallback: positional score
                    score = 0.5 if word_categories.get(candidate) in ['NOUN', 'PRONOUN'] else 0.0
            
            if score > best_score:
                best_score = score
                best_subject = candidate
        
        # Fallback to last noun before verb
        if best_subject is None and candidates:
            best_subject = candidates[-1]
        
        return best_subject
    
    def _find_best_object(self, candidates: List[str], subject: Optional[str],
                          verb: str) -> Optional[str]:
        """Find the best object from candidates using learned assemblies."""
        if not candidates:
            return None
        
        best_object = None
        best_score = 0.0
        
        for candidate in candidates:
            # Check for learned subject-verb-object VP assembly
            if subject:
                vp_key = f"{subject}_{verb}_{candidate}"
                vp_assembly = self.brain.get_learned_assembly(Area.VP, vp_key)
                
                if vp_assembly is not None:
                    score = 1.0
                else:
                    # Check overlap
                    obj_assembly = self._get_noun_assembly(candidate)
                    verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, verb)
                    
                    if obj_assembly is not None and verb_assembly is not None:
                        self.brain._clear_area(Area.VP)
                        self.brain._project(Area.VP, verb_assembly, learn=False)
                        
                        # Track fiber activation
                        self._record_fiber(Area.VERB_CORE, Area.VP)
                        
                        if self.brain.current[Area.VP] is not None:
                            score = self.brain.get_assembly_overlap(
                                self.brain.current[Area.VP], obj_assembly)
                        else:
                            score = 0.0
                    else:
                        score = 0.0
            else:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_object = candidate
        
        # Fallback to first noun after verb
        if best_object is None and candidates:
            best_object = candidates[0]
        
        return best_object
    
    def _get_noun_assembly(self, word: str) -> Optional[cp.ndarray]:
        """Get noun assembly from NOUN_CORE or PRON_CORE."""
        assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, word)
        if assembly is None:
            assembly = self.brain.get_learned_assembly(Area.PRON_CORE, word)
        return assembly
    
    def _record_fiber(self, from_area: Area, to_area: Area):
        """Record that a fiber was activated (for readout)."""
        if from_area not in self.activated_fibers:
            self.activated_fibers[from_area] = []
        if to_area not in self.activated_fibers[from_area]:
            self.activated_fibers[from_area].append(to_area)
    
    def _calculate_confidence(self, result: ParseResult) -> float:
        """Calculate parse confidence based on learned matches."""
        scores = []
        
        if result.subject and result.verb:
            vp_key = f"{result.subject}_{result.verb}"
            if self.brain.has_learned_assembly(Area.VP, vp_key):
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        if result.object and result.subject and result.verb:
            vp_key = f"{result.subject}_{result.verb}_{result.object}"
            if self.brain.has_learned_assembly(Area.VP, vp_key):
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _build_word_roles(self, words: List[str], result: ParseResult,
                          word_categories: Dict[str, str]) -> Dict[str, List[str]]:
        """Build mapping of words to their grammatical roles."""
        word_roles = {}
        
        for word in words:
            roles = []
            
            if word == result.subject:
                roles.extend(['SUBJECT', 'AGENT'])
            if word == result.object:
                roles.extend(['OBJECT', 'PATIENT'])
            if word == result.verb:
                roles.append('VERB')
            
            cat = word_categories.get(word, 'UNKNOWN')
            if cat == 'FUNCTION':
                roles.append('DETERMINER')
            elif cat == 'ADJECTIVE':
                roles.append('MODIFIER')
            elif cat == 'ADVERB':
                roles.append('ADVERB')
            elif cat == 'PREPOSITION':
                roles.append('PREPOSITION')
            
            word_roles[word] = roles if roles else ['UNKNOWN']
        
        return word_roles
    
    def get_activated_fibers(self) -> Dict[str, List[str]]:
        """Get fibers that were activated during parsing (for readout)."""
        return {
            area.name: [a.name for a in to_areas]
            for area, to_areas in self.activated_fibers.items()
        }

