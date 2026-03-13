"""
Emergent Sentence Generator
===========================

Version: 2.1.0
Date: 2025-12-01

Generates sentences using NEMO principles:
- Compatibility is tested by stability (not computed externally)
- Words are selected by projection and settling (not enumeration)
- Constraints emerge from learned weights (not hardcoded rules)
"""

import numpy as np
from typing import List

from .areas import Area
from .learner import EmergentLanguageLearner

__all__ = ['SentenceGenerator']


class SentenceGenerator:
    """
    Generates sentences using learned phrase structure.
    
    Uses:
    - Category transitions for word order
    - Thematic role statistics for word selection
    - Co-occurrence for selectional preferences
    """
    
    def __init__(self, learner: EmergentLanguageLearner):
        self.learner = learner
    
    def generate_structured(self) -> List[str]:
        """
        Generate a sentence using proper phrase structure.
        
        Structure: (DET) (ADJ) NOUN VERB (DET) (NOUN)
        """
        vocab = self.learner.get_vocabulary_by_category()
        
        nouns = vocab.get('NOUN', [])
        verbs = vocab.get('VERB', [])
        adjectives = vocab.get('ADJECTIVE', [])
        determiners = [w for w in vocab.get('FUNCTION', []) if w in ['the', 'a']]
        
        if not nouns or not verbs:
            return []
        
        sentence = []
        trans = self.learner.category_transitions
        
        # === SUBJECT NP ===
        
        # Determiner?
        det_noun_count = trans.get(('FUNCTION', 'NOUN'), 0)
        if determiners and det_noun_count > 10:
            sentence.append(np.random.choice(determiners))
        
        # Adjective?
        adj_noun_count = trans.get(('ADJECTIVE', 'NOUN'), 0)
        if adjectives and adj_noun_count > 5 and np.random.rand() > 0.5:
            sentence.append(np.random.choice(adjectives))
        
        # Subject noun (prefer AGENT words)
        agent_nouns = [(n, self.learner.word_as_first_arg[n]) 
                       for n in nouns if self.learner.word_as_first_arg[n] > 0]
        if agent_nouns:
            weights = [c for _, c in agent_nouns]
            total = sum(weights)
            probs = [w/total for w in weights]
            subj = np.random.choice([n for n, _ in agent_nouns], p=probs)
        else:
            subj = np.random.choice(nouns)
        sentence.append(subj)
        
        # === VERB ===
        
        # Prefer verbs that co-occur with subject
        verb_scores = {}
        for v in verbs:
            cooc = self.learner.word_cooccurrence[subj].get(v, 0)
            freq = self.learner.word_count.get(v, 1)
            verb_scores[v] = cooc + freq * 0.1
        
        best_verb = max(verb_scores, key=verb_scores.get) if verb_scores else np.random.choice(verbs)
        sentence.append(best_verb)
        
        # === OBJECT NP (optional) ===
        
        verb_obj_count = trans.get(('VERB', 'FUNCTION'), 0)
        is_transitive = verb_obj_count > 5
        
        if is_transitive:
            if determiners:
                sentence.append(np.random.choice(determiners))
            
            # Object noun (prefer PATIENT words, different from subject)
            patient_nouns = [(n, self.learner.word_as_second_arg[n]) 
                            for n in nouns if self.learner.word_as_second_arg[n] > 0 and n != subj]
            if patient_nouns:
                weights = [c for _, c in patient_nouns]
                total = sum(weights)
                probs = [w/total for w in weights]
                obj = np.random.choice([n for n, _ in patient_nouns], p=probs)
            else:
                obj_candidates = [n for n in nouns if n != subj]
                obj = np.random.choice(obj_candidates) if obj_candidates else np.random.choice(nouns)
            sentence.append(obj)
        
        return sentence
    
    def generate_from_transitions(self, max_length: int = 5) -> List[str]:
        """
        Generate using category transition probabilities.
        """
        vocab = self.learner.get_vocabulary_by_category()
        
        sentence = []
        prev_category = None
        
        for _ in range(max_length):
            if prev_category is None:
                # Start with noun or determiner
                if self.learner.mood_word_first:
                    first_word = max(self.learner.mood_word_first, 
                                    key=self.learner.mood_word_first.get)
                    cat, _ = self.learner.get_emergent_category(first_word)
                    if cat == 'FUNCTION':
                        cat = 'NOUN'
                else:
                    cat = 'NOUN'
            else:
                # Sample next category from transitions
                candidates = {}
                for (src, dst), count in self.learner.category_transitions.items():
                    if src == prev_category:
                        candidates[dst] = count
                
                if not candidates:
                    break
                
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
    
    def generate(self, method: str = 'structured') -> List[str]:
        """Generate a sentence using specified method."""
        if method == 'structured':
            return self.generate_structured()
        elif method == 'transitions':
            return self.generate_from_transitions()
        elif method == 'nemo':
            return self.generate_nemo_style()
        else:
            return self.generate_structured()
    
    # =========================================================================
    # NEMO-STYLE GENERATION
    # =========================================================================
    # Uses LEARNED ASSEMBLIES and overlap testing
    
    def generate_nemo_style(self, max_attempts: int = 5) -> List[str]:
        """
        Generate using TRUE NEMO principles:
        1. Pick a subject with a learned assembly
        2. Project subject's assembly to VP
        3. Find verbs whose learned assemblies overlap with VP
        4. Project verb to VP, find compatible objects
        
        This uses the ACTUAL learned assemblies, not external statistics.
        """
        vocab = self.learner.get_vocabulary_by_category()
        brain = self.learner.brain
        
        nouns = vocab.get('NOUN', [])
        verbs = vocab.get('VERB', [])
        determiners = [w for w in vocab.get('FUNCTION', []) if w in ['the', 'a']]
        
        if not nouns or not verbs:
            return []
        
        brain.clear_all()
        sentence = []
        
        # === SUBJECT NP ===
        if determiners:
            sentence.append(np.random.choice(determiners))
        
        # Pick subject from AGENT-role words that have learned assemblies
        min_agent_count = 10
        agent_nouns = [n for n in nouns 
                       if self.learner.word_as_first_arg[n] >= min_agent_count
                       and brain.has_learned_assembly(Area.NOUN_CORE, n)]
        
        if not agent_nouns:
            agent_nouns = [n for n in nouns if brain.has_learned_assembly(Area.NOUN_CORE, n)]
        
        if not agent_nouns:
            return []  # No learned nouns
        
        subj = np.random.choice(agent_nouns)
        sentence.append(subj)
        
        # Get subject's LEARNED assembly (not random!)
        subj_assembly = brain.get_learned_assembly(Area.NOUN_CORE, subj)
        
        # === VERB (by assembly overlap) ===
        # Project subject to VP
        brain._clear_area(Area.VP)
        brain._project(Area.VP, subj_assembly, learn=False)
        
        # Find verbs whose learned assemblies have good overlap with VP
        # This is the NEMO way: compatibility = assembly overlap
        verbs_with_assemblies = [v for v in verbs 
                                  if brain.has_learned_assembly(Area.VERB_CORE, v)]
        
        if not verbs_with_assemblies:
            return sentence  # No learned verbs
        
        # Check if there are VP assemblies for this subject with various verbs
        compatible_verbs = []
        for v in verbs_with_assemblies:
            # Check for learned subject-verb VP assembly
            vp_key = f"{subj}_{v}"
            vp_assembly = brain.get_learned_assembly(Area.VP, vp_key)
            
            if vp_assembly is not None:
                # We learned this combination! High compatibility.
                compatible_verbs.append((v, 1.0))
            else:
                # Check assembly overlap between verb and current VP state
                v_assembly = brain.get_learned_assembly(Area.VERB_CORE, v)
                if brain.current[Area.VP] is not None and v_assembly is not None:
                    overlap = brain.get_assembly_overlap(brain.current[Area.VP], v_assembly)
                    if overlap > 0.05:  # Some overlap
                        compatible_verbs.append((v, overlap))
        
        if compatible_verbs:
            # Sample from compatible verbs, weighted by compatibility
            verb_list = [v for v, _ in compatible_verbs]
            weights = [s for _, s in compatible_verbs]
            total = sum(weights)
            probs = [w/total for w in weights]
            best_verb = np.random.choice(verb_list, p=probs)
        else:
            # Fallback: use co-occurrence (learned signal, just different access)
            verb_scores = {v: self.learner.word_cooccurrence[subj].get(v, 0) 
                          for v in verbs_with_assemblies}
            if any(s > 0 for s in verb_scores.values()):
                verb_list = [v for v, s in verb_scores.items() if s > 0]
                weights = [verb_scores[v] for v in verb_list]
                total = sum(weights)
                probs = [w/total for w in weights]
                best_verb = np.random.choice(verb_list, p=probs)
            else:
                best_verb = np.random.choice(verbs_with_assemblies)
        
        sentence.append(best_verb)
        
        # Update VP with verb's learned assembly
        verb_assembly = brain.get_learned_assembly(Area.VERB_CORE, best_verb)
        brain._project(Area.VP, verb_assembly, learn=False)
        
        # === OBJECT (only if verb is transitive) ===
        # Check if this verb has learned transitive patterns (subj_verb_obj)
        # This is the NEMO way: the brain KNOWS which verbs take objects
        transitive_patterns = [key for key in brain.learned_assemblies[Area.VP].keys()
                               if key.count('_') == 2 and f"_{best_verb}_" in key]
        is_transitive = len(transitive_patterns) > 0
        
        if is_transitive:
            if determiners:
                sentence.append(np.random.choice(determiners))
            
            # Find objects with learned assemblies
            patient_nouns = [n for n in nouns 
                            if n != subj and brain.has_learned_assembly(Area.NOUN_CORE, n)]
            
            if not patient_nouns:
                return sentence
            
            # Check for learned subject-verb-object VP assemblies
            compatible_objects = []
            for obj_candidate in patient_nouns:
                vp_key = f"{subj}_{best_verb}_{obj_candidate}"
                vp_assembly = brain.get_learned_assembly(Area.VP, vp_key)
                
                if vp_assembly is not None:
                    # We learned this combination!
                    compatible_objects.append((obj_candidate, 1.0))
                else:
                    # Check overlap
                    obj_assembly = brain.get_learned_assembly(Area.NOUN_CORE, obj_candidate)
                    if brain.current[Area.VP] is not None and obj_assembly is not None:
                        overlap = brain.get_assembly_overlap(brain.current[Area.VP], obj_assembly)
                        if overlap > 0.05:
                            compatible_objects.append((obj_candidate, overlap))
            
            if compatible_objects:
                obj_list = [o for o, _ in compatible_objects]
                weights = [s for _, s in compatible_objects]
                total = sum(weights)
                probs = [w/total for w in weights]
                obj = np.random.choice(obj_list, p=probs)
            else:
                # Fallback: prefer PATIENT-role words
                patient_words = [n for n in patient_nouns 
                                if self.learner.word_as_second_arg[n] > 0]
                obj = np.random.choice(patient_words) if patient_words else np.random.choice(patient_nouns)
            
            sentence.append(obj)
        
        return sentence

