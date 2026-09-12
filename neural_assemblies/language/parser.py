"""
Main parser classes for language processing.

This module contains the core parser classes that extend the brain
functionality for language processing, including English and Russian parsers.
"""

import numpy as np
from neural_assemblies.core.brain import Brain
from neural_assemblies.core.backend import to_cpu, resolve_mixed_engine
from collections import defaultdict
from typing import Optional, Dict, List

from .language_areas import (
    ACC,
    ADJ,
    ADVERB,
    AREAS,
    DAT,
    DEP_CLAUSE,
    DET,
    DET_SIZE,
    DISINHIBIT,
    ENGLISH_READOUT_RULES,
    INHIBIT,
    LEX,
    LEX_SIZE,
    NOM,
    OBJ,
    PREP,
    PREP_P,
    RECURRENT_AREAS,
    RUSSIAN_AREAS,
    RUSSIAN_LEX_SIZE,
    RUSSIAN_READOUT_RULES,
    SUBJ,
    VERB,
)
from .grammar_rules import LEXEME_DICT, RUSSIAN_LEXEME_DICT, AreaRule, FiberRule

class ParserBrain(Brain):
    """Base parser brain class that extends the basic brain for language processing."""
    
    def __init__(self, p, lexeme_dict: Optional[Dict] = None,
                 all_areas: Optional[List[str]] = None,
                 recurrent_areas: Optional[List[str]] = None,
                 initial_areas: Optional[List[str]] = None,
                 readout_rules: Optional[Dict] = None, engine="auto"):
        """
        Initialize the parser brain.
        
        Args:
            p: Probability parameter for brain initialization
            lexeme_dict: Dictionary mapping words to their grammar rules
            all_areas: List of all language areas
            recurrent_areas: List of recurrent language areas
            initial_areas: List of initially active areas
            readout_rules: Rules for readout processing
            engine: Compute engine name, or ``"auto"`` for literature-safe default
        """
        if engine == "auto":
            engine = resolve_mixed_engine(engine)
        Brain.__init__(self, p, engine=engine)
        self.lexeme_dict = {} if lexeme_dict is None else lexeme_dict
        self.all_areas = [] if all_areas is None else list(all_areas)
        self.recurrent_areas = [] if recurrent_areas is None else list(recurrent_areas)
        self.initial_areas = [] if initial_areas is None else list(initial_areas)

        self.fiber_states = defaultdict()
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        self.readout_rules = {} if readout_rules is None else readout_rules
        self.area_lexeme_cache = {}
        self._outer_lexeme_cache = {}
        self._inner_lexeme_cache = {}
        self._in_dep_clause = False
        self.initialize_states()

    def initialize_states(self):
        """Initialize the fiber and area states."""
        for from_area in self.all_areas:
            self.fiber_states[from_area] = defaultdict(set)
            for to_area in self.all_areas:
                self.fiber_states[from_area][to_area].add(0)

        for area in self.all_areas:
            self.area_states[area].add(0)

        for area in self.initial_areas:
            self.area_states[area].discard(0)

    def applyFiberRule(self, rule):
        """Apply a fiber rule to update fiber states."""
        if rule.action == INHIBIT:
            self.fiber_states[rule.area1][rule.area2].add(rule.index)
            self.fiber_states[rule.area2][rule.area1].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.fiber_states[rule.area1][rule.area2].discard(rule.index)
            self.fiber_states[rule.area2][rule.area1].discard(rule.index)

    def applyAreaRule(self, rule):
        """Apply an area rule to update area states."""
        if rule.action == INHIBIT:
            self.area_states[rule.area].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.area_states[rule.area].discard(rule.index)

    def applyRule(self, rule):
        """Apply a rule (fiber or area) to update states."""
        if isinstance(rule, FiberRule):
            self.applyFiberRule(rule)
            return True
        if isinstance(rule, AreaRule):
            self.applyAreaRule(rule)
            return True
        return False

    def parse_project(self):
        """Perform a projection step for parsing."""
        # ``getProjectMap`` is a legacy defaultdict of sets; normalize it at
        # the Brain boundary so the typed projection API receives a stable,
        # ordered schedule rather than an implementation container.
        project_map = {
            source: sorted(targets)
            for source, targets in self.getProjectMap().items()
        }
        self.remember_fibers(project_map)
        self.project({}, project_map)

    def remember_fibers(self, project_map):
        """Remember activated fibers for readout."""
        for from_area, to_areas in project_map.items():
            self.activated_fibers[from_area].update(to_areas)

    def recurrent(self, area):
        """Check if an area is recurrent."""
        return (area in self.recurrent_areas)

    def getProjectMap(self):
        """Get the projection map based on current states."""
        proj_map = defaultdict(set)
        for area1 in self.all_areas:
            if len(self.area_states[area1]) == 0:
                for area2 in self.all_areas:
                    if area1 == LEX and area2 == LEX:
                        continue
                    if len(self.area_states[area2]) == 0:
                        if len(self.fiber_states[area1][area2]) == 0:
                            if len(self.area_by_name[area1].winners) > 0:
                                proj_map[area1].add(area2)
                            if len(self.area_by_name[area2].winners) > 0:
                                proj_map[area2].add(area2)
        return proj_map

    def _clear_area_winners(self, area_name):
        """Clear an area's assembly and sync to compute engines."""
        area = self.area_by_name[area_name]
        area.unfix_assembly()
        empty = np.array([], dtype=np.uint32)
        area.winners = empty
        # Write through the owning backend only.  Mirroring both primary and
        # explicit engines made the parser's state depend on private storage
        # and could leave the non-owner looking authoritative.
        self.engine_for(area_name).set_winners(area_name, empty)

    def _set_area_winners(self, area_name, winners):
        """Set winners on an area and sync to the compute engine."""
        area = self.area_by_name[area_name]
        winners_arr = np.asarray(winners, dtype=np.uint32)
        area.winners = winners_arr
        self.engine_for(area_name).set_winners(area_name, winners_arr)

    def activateWord(self, area_name, word):
        """Activate a word in the specified area."""
        area = self.area_by_name[area_name]
        k = area.k
        assembly_start = self.lexeme_dict[word]["index"] * k
        self._set_area_winners(
            area_name, range(assembly_start, assembly_start + k))
        area.fix_assembly()

    def activateIndex(self, area_name, index):
        """Activate a word by index in the specified area."""
        area = self.area_by_name[area_name]
        k = area.k
        assembly_start = index * k
        self._set_area_winners(
            area_name, range(assembly_start, assembly_start + k))
        area.fix_assembly()

    def interpretAssemblyAsString(self, area_name):
        """Interpret the assembly in an area as a string."""
        return self.getWord(area_name, 0.7)

    def begin_dep_clause(self) -> None:
        """Start center-embedded / relative clause lexical bindings."""
        self._in_dep_clause = True
        self._inner_lexeme_cache = {}

    def end_dep_clause(self) -> None:
        """Close relative clause; outer bindings resume."""
        self._in_dep_clause = False

    def reset_outer_lexeme_cache(self) -> None:
        self._outer_lexeme_cache = {}
        self._sync_lexeme_cache_view()

    def _sync_lexeme_cache_view(self) -> None:
        merged = dict(self._outer_lexeme_cache)
        merged.update(self._inner_lexeme_cache)
        self.area_lexeme_cache = merged

    def record_lexeme_bindings(self, word: str) -> None:
        """Remember which grammar areas LEX bound to for *word* this step."""
        lexeme = self.lexeme_dict[word]
        proj_map = self.getProjectMap()
        targets = proj_map.get(LEX, set()) - {LEX}
        if not targets:
            return

        cache = self._inner_lexeme_cache if self._in_dep_clause else self._outer_lexeme_cache

        pre0 = [r for r in lexeme.get("PRE_RULES", []) if r.index == 0]
        is_verb = any(
            isinstance(r, FiberRule) and r.action == DISINHIBIT
            and r.area1 == LEX and r.area2 == VERB
            for r in pre0
        )
        if is_verb:
            for area in targets:
                if area in {VERB, ADVERB, DEP_CLAUSE}:
                    cache[area] = word
            self._sync_lexeme_cache_view()
            return

        case_areas = {SUBJ, OBJ, NOM, ACC, DAT, PREP_P, DET, ADJ}
        for area in sorted(targets & case_areas):
            if (
                area == SUBJ
                and SUBJ in cache
                and OBJ in targets
            ):
                continue
            if area == SUBJ and SUBJ in cache:
                continue
            cache[area] = word
        self._sync_lexeme_cache_view()

    def getWord(self, area_name, min_overlap=0.7, cue_area=None, clause_scope="auto"):
        """Get the word represented by the assembly in an area."""
        if area_name == LEX and cue_area:
            if clause_scope in ("inner", "auto") and cue_area in self._inner_lexeme_cache:
                return self._inner_lexeme_cache[cue_area]
            if clause_scope in ("outer", "auto") and cue_area in self._outer_lexeme_cache:
                return self._outer_lexeme_cache[cue_area]
            if clause_scope == "auto" and cue_area in self.area_lexeme_cache:
                return self.area_lexeme_cache[cue_area]
        if len(self.area_by_name[area_name].winners) == 0:
            raise Exception("Cannot get word because no assembly in " + area_name)
        winners = set(int(x) for x in to_cpu(self.area_by_name[area_name].winners))
        area_k = self.area_by_name[area_name].k
        threshold = min_overlap * area_k
        best_word = None
        best_overlap = 0
        for word, lexeme in self.lexeme_dict.items():
            word_index = lexeme["index"]
            word_assembly_start = word_index * area_k
            word_assembly = set(range(word_assembly_start, word_assembly_start + area_k))
            overlap_count = len(winners & word_assembly)
            if overlap_count > best_overlap:
                best_word = word
                best_overlap = overlap_count
            if overlap_count >= threshold:
                return word
        min_absolute = max(3, int(0.25 * area_k))
        if best_overlap >= min_absolute:
            return best_word
        return None

    def getActivatedFibers(self):
        """Get activated fibers pruned by readout rules."""
        pruned_activated_fibers = defaultdict(set)
        for from_area, to_areas in self.activated_fibers.items():
            for to_area in to_areas:
                if to_area in self.readout_rules[from_area]:
                    pruned_activated_fibers[from_area].add(to_area)
        return pruned_activated_fibers


class RussianParserBrain(ParserBrain):
    """Russian language parser brain."""
    
    def __init__(self, p, non_LEX_n=1000, non_LEX_k=100, LEX_k=10, 
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, 
                 interarea_beta=0.5, verbose=False, engine="auto"):
        """
        Initialize the Russian parser brain.
        
        Args:
            p: Probability parameter
            non_LEX_n: Number of neurons in non-LEX areas
            non_LEX_k: Number of winners in non-LEX areas
            LEX_k: Number of winners in LEX area
            default_beta: Default plasticity parameter
            LEX_beta: LEX-specific plasticity parameter
            recurrent_beta: Recurrent plasticity parameter
            interarea_beta: Inter-area plasticity parameter
            verbose: Whether to print verbose output
        """
        recurrent_areas = [NOM, VERB, ACC, DAT]
        ParserBrain.__init__(self, p, 
                            lexeme_dict=RUSSIAN_LEXEME_DICT, 
                            all_areas=RUSSIAN_AREAS, 
                            recurrent_areas=recurrent_areas,
                            initial_areas=[LEX],
                            readout_rules=RUSSIAN_READOUT_RULES,
                            engine=engine)
        self.verbose = verbose

        LEX_n = RUSSIAN_LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

        self.add_area(NOM, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ACC, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DAT, non_LEX_n, non_LEX_k, default_beta)

        # Set up custom plasticities
        custom_plasticities = defaultdict(list)
        for area in recurrent_areas:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other_area in recurrent_areas:
                if other_area == area:
                    continue
                custom_plasticities[area].append((other_area, interarea_beta))

        self.update_plasticities(area_update_map=custom_plasticities)

    def getWord(self, area_name, min_overlap=0.7, cue_area=None, clause_scope="auto"):
        word = ParserBrain.getWord(
            self, area_name, min_overlap, cue_area=cue_area, clause_scope=clause_scope,
        )
        if word:
            return word
        return "<NON-WORD>"


class EnglishParserBrain(ParserBrain):
    """English language parser brain."""
    
    def __init__(self, p, non_LEX_n=1000, non_LEX_k=50, LEX_k=20,
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, 
                 interarea_beta=0.5, verbose=False, engine="auto"):
        """
        Initialize the English parser brain.
        
        Args:
            p: Probability parameter
            non_LEX_n: Number of neurons in non-LEX areas
            non_LEX_k: Number of winners in non-LEX areas
            LEX_k: Number of winners in LEX area
            default_beta: Default plasticity parameter
            LEX_beta: LEX-specific plasticity parameter
            recurrent_beta: Recurrent plasticity parameter
            interarea_beta: Inter-area plasticity parameter
            verbose: Whether to print verbose output
        """
        ParserBrain.__init__(self, p, 
                            lexeme_dict=LEXEME_DICT, 
                            all_areas=AREAS, 
                            recurrent_areas=RECURRENT_AREAS, 
                            initial_areas=[LEX, SUBJ, VERB],
                            readout_rules=ENGLISH_READOUT_RULES,
                            engine=engine)
        self.verbose = verbose

        LEX_n = LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

        DET_k = LEX_k
        for area_name in [SUBJ, OBJ, VERB, ADJ, PREP, PREP_P, DET, ADVERB, DEP_CLAUSE]:
            k = DET_k if area_name == DET else non_LEX_k
            self.add_area(area_name, non_LEX_n, k, default_beta)

        # Set up custom plasticities
        custom_plasticities = defaultdict(list)
        for area in RECURRENT_AREAS:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other_area in RECURRENT_AREAS:
                if other_area == area:
                    continue
                custom_plasticities[area].append((other_area, interarea_beta))

        self.update_plasticities(area_update_map=custom_plasticities)

    def getProjectMap(self):
        """Get projection map with English-specific constraints."""
        proj_map = ParserBrain.getProjectMap(self)
        # "War of fibers" — exclude LEX→LEX recurrent from the cap
        lex_targets = proj_map[LEX] - {LEX} if LEX in proj_map else set()
        if len(lex_targets) > 2:
            raise Exception(
                "Got that LEX projecting into many areas: " + str(proj_map[LEX])
            )
        return proj_map

    def getWord(self, area_name, min_overlap=0.7, cue_area=None, clause_scope="auto"):
        """Get word with English-specific handling."""
        word = ParserBrain.getWord(
            self, area_name, min_overlap, cue_area=cue_area, clause_scope=clause_scope,
        )
        if word:
            return word
        if not word and area_name == DET:
            winners = set(int(x) for x in to_cpu(self.area_by_name[area_name].winners))
            area_k = self.area_by_name[area_name].k
            threshold = min_overlap * area_k
            nodet_index = DET_SIZE - 1
            nodet_assembly_start = nodet_index * area_k
            nodet_assembly = set(range(nodet_assembly_start, nodet_assembly_start + area_k))
            if len((winners & nodet_assembly)) > threshold:
                return "<null-det>"
        # If nothing matched, at least we can see that in the parse output.
        return "<NON-WORD>"
