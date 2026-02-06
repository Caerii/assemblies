"""
Emergent Language Learner
=========================

Version: 2.2.0
Date: 2025-11-30

Language learner where ALL categories emerge from grounding patterns.
NO pre-labeled categories - everything learned from experience.

Note: Parsing and comprehension have been moved to the parser submodule.
Use SentenceParser and QuestionAnswerer from .parser for those features.
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .areas import Area, GROUNDING_TO_CORE
from .params import EmergentParams, GroundingContext, GroundedSentence
from .brain import EmergentNemoBrain

__all__ = ['EmergentLanguageLearner']


class EmergentLanguageLearner:
    """
    Language learner where ALL categories EMERGE from grounding.
    
    NO GROUND TRUTH LABELS! NO HARDCODED POS TAGS!
    
    Categories emerge from:
    1. Consistent VISUAL grounding → NOUN
    2. Consistent MOTOR grounding → VERB
    3. Consistent PROPERTY grounding → ADJECTIVE
    4. Consistent SPATIAL grounding → PREPOSITION
    5. Consistent SOCIAL grounding → PRONOUN
    6. Consistent TEMPORAL grounding → ADVERB
    7. High frequency + no grounding → FUNCTION WORD
    """
    
    def __init__(self, params: EmergentParams = None, verbose: bool = True):
        self.brain = EmergentNemoBrain(params, verbose=verbose)
        self.p = self.brain.p
        self.verbose = verbose
        
        # Word statistics (for emergent categorization)
        self.word_grounding: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_count: Dict[str, int] = defaultdict(int)
        
        # Co-occurrence (for selectional restrictions)
        self.word_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Sequence statistics (for word order)
        self.position_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Category transitions (for syntax)
        self.category_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Thematic roles
        self.word_as_first_arg: Dict[str, int] = defaultdict(int)
        self.word_as_second_arg: Dict[str, int] = defaultdict(int)
        self.word_as_action: Dict[str, int] = defaultdict(int)
        
        # Mood statistics
        self.mood_word_first: Dict[str, int] = defaultdict(int)
        
        self.sentences_seen = 0
    
    # =========================================================================
    # WORD PRESENTATION
    # =========================================================================
    
    def present_word_with_grounding(self, word: str, context: GroundingContext, 
                                     position: int = 0, role: str = None,
                                     learn: bool = True):
        """Present a word with its grounding context."""
        phon = self.brain._get_or_create(Area.PHON, word)
        
        if learn:
            self.word_count[word] += 1
            self.position_counts[word][position] += 1
            
            if role == 'agent':
                self.word_as_first_arg[word] += 1
            elif role == 'patient':
                self.word_as_second_arg[word] += 1
            elif role == 'action':
                self.word_as_action[word] += 1
        
        grounding_active = []
        
        # Visual → NOUN
        if context.visual:
            for v in context.visual:
                vis = self.brain._get_or_create(Area.VISUAL, v)
                self.brain._project(Area.VISUAL, vis, learn=False)
            grounding_active.append('VISUAL')
            if learn:
                self.word_grounding[word]['VISUAL'] += 1
                self.brain._project(Area.NOUN_CORE, phon, learn=learn)
                # STORE the learned assembly for this word
                if self.brain.current[Area.NOUN_CORE] is not None:
                    self.brain.store_learned_assembly(Area.NOUN_CORE, word, 
                                                       self.brain.current[Area.NOUN_CORE])
        
        # Motor → VERB
        if context.motor:
            for m in context.motor:
                mot = self.brain._get_or_create(Area.MOTOR, m)
                self.brain._project(Area.MOTOR, mot, learn=False)
            grounding_active.append('MOTOR')
            if learn:
                self.word_grounding[word]['MOTOR'] += 1
                self.brain._project(Area.VERB_CORE, phon, learn=learn)
                # STORE the learned assembly for this word
                if self.brain.current[Area.VERB_CORE] is not None:
                    self.brain.store_learned_assembly(Area.VERB_CORE, word,
                                                       self.brain.current[Area.VERB_CORE])
        
        # Property → ADJECTIVE
        if context.properties:
            for p in context.properties:
                prop = self.brain._get_or_create(Area.PROPERTY, p)
                self.brain._project(Area.PROPERTY, prop, learn=False)
            grounding_active.append('PROPERTY')
            if learn:
                self.word_grounding[word]['PROPERTY'] += 1
                self.brain._project(Area.ADJ_CORE, phon, learn=learn)
                # STORE the learned assembly
                if self.brain.current[Area.ADJ_CORE] is not None:
                    self.brain.store_learned_assembly(Area.ADJ_CORE, word,
                                                       self.brain.current[Area.ADJ_CORE])
        
        # Spatial → PREPOSITION
        if context.spatial:
            for s in context.spatial:
                spat = self.brain._get_or_create(Area.SPATIAL, s)
                self.brain._project(Area.SPATIAL, spat, learn=False)
            grounding_active.append('SPATIAL')
            if learn:
                self.word_grounding[word]['SPATIAL'] += 1
                self.brain._project(Area.PREP_CORE, phon, learn=learn)
                if self.brain.current[Area.PREP_CORE] is not None:
                    self.brain.store_learned_assembly(Area.PREP_CORE, word,
                                                       self.brain.current[Area.PREP_CORE])
        
        # Social → PRONOUN
        if context.social:
            for s in context.social:
                soc = self.brain._get_or_create(Area.SOCIAL, s)
                self.brain._project(Area.SOCIAL, soc, learn=False)
            grounding_active.append('SOCIAL')
            if learn:
                self.word_grounding[word]['SOCIAL'] += 1
                self.brain._project(Area.PRON_CORE, phon, learn=learn)
                if self.brain.current[Area.PRON_CORE] is not None:
                    self.brain.store_learned_assembly(Area.PRON_CORE, word,
                                                       self.brain.current[Area.PRON_CORE])
        
        # Temporal → ADVERB
        if context.temporal:
            for t in context.temporal:
                temp = self.brain._get_or_create(Area.TEMPORAL, t)
                self.brain._project(Area.TEMPORAL, temp, learn=False)
            grounding_active.append('TEMPORAL')
            if learn:
                self.word_grounding[word]['TEMPORAL'] += 1
                self.brain._project(Area.ADV_CORE, phon, learn=learn)
        
        # Emotional
        if context.emotional:
            for e in context.emotional:
                emo = self.brain._get_or_create(Area.EMOTION, e)
                self.brain._project(Area.EMOTION, emo, learn=False)
            grounding_active.append('EMOTION')
            if learn:
                self.word_grounding[word]['EMOTION'] += 1
        
        # No grounding → FUNCTION WORD
        if not grounding_active:
            if learn:
                self.word_grounding[word]['NONE'] += 1
                self.brain._project(Area.DET_CORE, phon, learn=learn)
        
        # Lexical projection
        for _ in range(self.p.tau):
            if grounding_active:
                self.brain._project(Area.LEX_CONTENT, phon, learn=learn)
            else:
                self.brain._project(Area.LEX_FUNCTION, phon, learn=learn)
        
        # Thematic role projection
        if role and learn:
            if role == 'agent':
                self.brain._project(Area.ROLE_AGENT, phon, learn=learn)
            elif role == 'patient':
                self.brain._project(Area.ROLE_PATIENT, phon, learn=learn)
            elif role == 'action':
                self.brain._project(Area.VP, phon, learn=learn)
    
    # =========================================================================
    # SENTENCE PRESENTATION
    # =========================================================================
    
    def present_grounded_sentence(self, words: List[str], contexts: List[GroundingContext],
                                   roles: List[str] = None, mood: str = 'declarative',
                                   learn: bool = True):
        """
        Present a sentence with grounding for each word.
        
        Key addition: We also learn VP assemblies that link subjects and verbs,
        and verbs and objects. This is what enables NEMO-style generation.
        """
        self.brain.clear_all()
        
        mood_assembly = self.brain._get_or_create(Area.MOOD, mood)
        self.brain._project(Area.MOOD, mood_assembly, learn=learn)
        self.brain._project(Area.SEQ, mood_assembly, learn=learn)
        
        if roles is None:
            roles = [None] * len(words)
        
        prev_word = None
        prev_category = None
        current_subject = None
        current_verb = None
        
        for i, (word, context, role) in enumerate(zip(words, contexts, roles)):
            self.present_word_with_grounding(word, context, position=i, role=role, learn=learn)
            
            current_category, _ = self.get_emergent_category(word)
            
            if prev_word and learn:
                self.transitions[(prev_word, word)] += 1
            
            if prev_category and current_category and learn:
                self.category_transitions[(prev_category, current_category)] += 1
            
            if learn:
                for other_word in words:
                    if other_word != word:
                        self.word_cooccurrence[word][other_word] += 1
            
            if learn:
                core_area = GROUNDING_TO_CORE.get(
                    self._get_dominant_grounding(word), Area.DET_CORE)
                if self.brain.current[core_area] is not None:
                    self.brain._project(Area.SEQ, self.brain.current[core_area], learn=learn)
            
            # Track subject and verb for VP assembly learning
            if role == 'agent':
                current_subject = word
                        
            elif role == 'action':
                current_verb = word
                
                # When we see the verb, project subject → VP → verb
                # This creates the subject-verb binding in VP (merged)
                if current_subject and learn:
                    subj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, current_subject)
                    if subj_assembly is None:
                        subj_assembly = self.brain.get_learned_assembly(Area.PRON_CORE, current_subject)
                    if subj_assembly is not None:
                        self.brain._project(Area.VP, subj_assembly, learn=True)
                        # Now project verb to VP (merges with subject)
                        verb_assembly = self.brain.get_learned_assembly(Area.VERB_CORE, word)
                        if verb_assembly is not None:
                            self.brain._project(Area.VP, verb_assembly, learn=True)
                            # Store the VP assembly for this subject-verb pair
                            vp_key = f"{current_subject}_{word}"
                            if self.brain.current[Area.VP] is not None:
                                self.brain.store_learned_assembly(Area.VP, vp_key,
                                                                   self.brain.current[Area.VP])
                            # NEW: Store VP_SUBJ and VP_VERB DIRECTLY (not projected!)
                            # This preserves the original assemblies for retrieval
                            self.brain.store_learned_assembly(Area.VP_SUBJ, vp_key,
                                                               subj_assembly)
                            self.brain.store_learned_assembly(Area.VP_VERB, vp_key,
                                                               verb_assembly)
                                                                   
            elif role == 'patient':
                current_object = word
                
                # When we see the object, add it to VP
                if current_verb and learn:
                    obj_assembly = self.brain.get_learned_assembly(Area.NOUN_CORE, word)
                    if obj_assembly is not None:
                        self.brain._project(Area.VP, obj_assembly, learn=True)
                        # Store VP with object
                        if current_subject:
                            vp_key = f"{current_subject}_{current_verb}_{word}"
                            if self.brain.current[Area.VP] is not None:
                                self.brain.store_learned_assembly(Area.VP, vp_key,
                                                                   self.brain.current[Area.VP])
                            # NEW: Store VP_OBJ DIRECTLY (not projected!)
                            self.brain.store_learned_assembly(Area.VP_OBJ, vp_key,
                                                               obj_assembly)
            
            prev_word = word
            prev_category = current_category
        
        if learn:
            self.sentences_seen += 1
            if words:
                self.mood_word_first[words[0]] += 1
    
    def _get_dominant_grounding(self, word: str) -> str:
        """Get the dominant grounding modality for a word"""
        grounding = self.word_grounding[word]
        if not grounding:
            return 'NONE'
        return max(grounding, key=grounding.get)
    
    # =========================================================================
    # PHRASE COMPOSITION
    # =========================================================================
    
    def build_noun_phrase(self, words: List[str], contexts: List[GroundingContext],
                          learn: bool = True) -> Optional[cp.ndarray]:
        """Build a noun phrase by merging words into NP area."""
        self.brain._clear_area(Area.NP)
        
        for word, ctx in zip(words, contexts):
            phon = self.brain._get_or_create(Area.PHON, word)
            
            if ctx.visual:
                self.brain._project(Area.NOUN_CORE, phon, learn=learn)
                if self.brain.current[Area.NOUN_CORE] is not None:
                    self.brain.merge_to_area(Area.NP, self.brain.current[Area.NOUN_CORE], learn=learn)
            elif ctx.properties:
                self.brain._project(Area.ADJ_CORE, phon, learn=learn)
                if self.brain.current[Area.ADJ_CORE] is not None:
                    self.brain.merge_to_area(Area.NP, self.brain.current[Area.ADJ_CORE], learn=learn)
            elif not any([ctx.visual, ctx.motor, ctx.properties, ctx.spatial, ctx.social]):
                self.brain._project(Area.DET_CORE, phon, learn=learn)
                if self.brain.current[Area.DET_CORE] is not None:
                    self.brain.merge_to_area(Area.NP, self.brain.current[Area.DET_CORE], learn=learn)
            
            if learn:
                self.word_count[word] += 1
                if ctx.visual:
                    self.word_grounding[word]['VISUAL'] += 1
                elif ctx.properties:
                    self.word_grounding[word]['PROPERTY'] += 1
                else:
                    self.word_grounding[word]['NONE'] += 1
        
        return self.brain.current[Area.NP]
    
    def build_verb_phrase(self, verb: str, verb_ctx: GroundingContext,
                          object_np: Optional[cp.ndarray] = None,
                          learn: bool = True) -> Optional[cp.ndarray]:
        """Build a verb phrase."""
        self.brain._clear_area(Area.VP)
        
        phon = self.brain._get_or_create(Area.PHON, verb)
        self.brain._project(Area.VERB_CORE, phon, learn=learn)
        if self.brain.current[Area.VERB_CORE] is not None:
            self.brain.merge_to_area(Area.VP, self.brain.current[Area.VERB_CORE], learn=learn)
        
        if object_np is not None:
            self.brain.disinhibit_role(Area.OBJ)
            self.brain.bind_phrase_to_role(object_np, Area.OBJ, learn=learn)
            if self.brain.current[Area.OBJ] is not None:
                self.brain.merge_to_area(Area.VP, self.brain.current[Area.OBJ], learn=learn)
        
        if learn:
            self.word_count[verb] += 1
            self.word_grounding[verb]['MOTOR'] += 1
        
        return self.brain.current[Area.VP]
    
    def build_sentence(self, subject_np: cp.ndarray, vp: cp.ndarray,
                       mood: str = 'declarative', learn: bool = True) -> Optional[cp.ndarray]:
        """Build a complete sentence."""
        self.brain._clear_area(Area.SENT)
        
        mood_assembly = self.brain._get_or_create(Area.MOOD, mood)
        self.brain._project(Area.MOOD, mood_assembly, learn=learn)
        
        self.brain.bind_phrase_to_role(subject_np, Area.SUBJ, learn=learn)
        
        if self.brain.current[Area.SUBJ] is not None:
            self.brain.link_to_predicate(self.brain.current[Area.SUBJ], Area.VP, learn=learn)
        
        self.brain.merge_to_area(Area.SENT, vp, learn=learn)
        
        if self.brain.current[Area.SUBJ] is not None:
            self.brain.merge_to_area(Area.SENT, self.brain.current[Area.SUBJ], learn=learn)
        
        if learn:
            self.brain._project(Area.SEQ, self.brain.current[Area.SENT], learn=learn)
        
        return self.brain.current[Area.SENT]
    
    # =========================================================================
    # CATEGORY INFERENCE
    # =========================================================================
    
    def get_emergent_category(self, word: str) -> Tuple[str, Dict[str, float]]:
        """Get the emergent category for a word based on grounding history."""
        if word not in self.word_count or self.word_count[word] < 2:
            return 'UNKNOWN', {}
        
        grounding = self.word_grounding[word]
        total = sum(grounding.values())
        
        if total == 0:
            return 'FUNCTION', {'FUNCTION': 1.0}
        
        modalities = ['VISUAL', 'MOTOR', 'PROPERTY', 'SPATIAL', 'SOCIAL', 
                      'TEMPORAL', 'EMOTION', 'COGNITIVE', 'NONE']
        scores = {m: grounding.get(m, 0) / total for m in modalities}
        
        if scores.get('NONE', 0) > 0.7:
            return 'FUNCTION', scores
        
        max_modality = max(scores, key=scores.get)
        
        category_map = {
            'VISUAL': 'NOUN',
            'MOTOR': 'VERB',
            'PROPERTY': 'ADJECTIVE',
            'SPATIAL': 'PREPOSITION',
            'SOCIAL': 'PRONOUN',
            'TEMPORAL': 'ADVERB',
            'EMOTION': 'ADJECTIVE',
            'COGNITIVE': 'VERB',  # Cognitive verbs like "know", "think"
            'NONE': 'FUNCTION',
        }
        
        return category_map.get(max_modality, 'UNKNOWN'), scores
    
    def get_thematic_role(self, word: str) -> Tuple[str, float]:
        """Get emergent thematic role for a word."""
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
    
    def get_vocabulary_by_category(self) -> Dict[str, List[str]]:
        """Get vocabulary organized by emergent category."""
        vocab = defaultdict(list)
        for word in self.word_count:
            cat, _ = self.get_emergent_category(word)
            if cat != 'UNKNOWN':
                vocab[cat].append(word)
        return dict(vocab)
    
    def get_word_order(self) -> List[str]:
        """Get learned word order from category transitions."""
        if not self.category_transitions:
            return []
        
        categories = set()
        for (src, dst) in self.category_transitions:
            categories.add(src)
            categories.add(dst)
        
        # Find starting category
        sources = {src for (src, _) in self.category_transitions}
        targets = {dst for (_, dst) in self.category_transitions}
        starts = sources - targets
        
        if not starts:
            starts = sources
        
        # Build order by following most common transitions
        order = []
        current = max(starts, key=lambda c: sum(
            self.category_transitions.get((c, d), 0) for d in categories
        )) if starts else None
        
        visited = set()
        while current and current not in visited and len(order) < len(categories):
            order.append(current)
            visited.add(current)
            
            next_candidates = {
                dst: self.category_transitions.get((current, dst), 0)
                for dst in categories if dst not in visited
            }
            
            if not next_candidates:
                break
            
            current = max(next_candidates, key=next_candidates.get)
        
        return order
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        vocab = self.get_vocabulary_by_category()
        return {
            'sentences_seen': self.sentences_seen,
            'vocabulary_size': len(self.word_count),
            'categories': {cat: len(words) for cat, words in vocab.items()},
            'word_order': self.get_word_order(),
        }
    
    # =========================================================================
    # PARSING (delegated to parser module)
    # =========================================================================
    
    def parse_sentence(self, words: List[str]) -> Dict:
        """
        Parse a sentence using learned assemblies.
        
        Convenience method that delegates to SentenceParser.
        For more control, use parser.SentenceParser directly.
        """
        from .parser import SentenceParser
        parser = SentenceParser(self)
        result = parser.parse(words)
        return result.to_dict()
    
    def comprehend_and_answer(self, question: str, question_words: List[str]) -> str:
        """
        Answer a question using learned knowledge.
        
        Convenience method that delegates to QuestionAnswerer.
        For more control, use parser.QuestionAnswerer directly.
        """
        from .parser import QuestionAnswerer
        qa = QuestionAnswerer(self)
        return qa.answer(question_words)
