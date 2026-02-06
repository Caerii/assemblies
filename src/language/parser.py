"""
Main parser classes for language processing.

This module contains the core parser classes that extend the brain
functionality for language processing, including English and Russian parsers.
"""

import brain
from collections import defaultdict

from .language_areas import *
from .grammar_rules import LEXEME_DICT, RUSSIAN_LEXEME_DICT, AreaRule, FiberRule

class ParserBrain(brain.Brain):
    """Base parser brain class that extends the basic brain for language processing."""
    
    def __init__(self, p, lexeme_dict={}, all_areas=[], recurrent_areas=[], initial_areas=[], readout_rules={}):
        """
        Initialize the parser brain.
        
        Args:
            p: Probability parameter for brain initialization
            lexeme_dict: Dictionary mapping words to their grammar rules
            all_areas: List of all language areas
            recurrent_areas: List of recurrent language areas
            initial_areas: List of initially active areas
            readout_rules: Rules for readout processing
        """
        brain.Brain.__init__(self, p)
        self.lexeme_dict = lexeme_dict
        self.all_areas = all_areas
        self.recurrent_areas = recurrent_areas
        self.initial_areas = initial_areas

        self.fiber_states = defaultdict()
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        self.readout_rules = readout_rules
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
        project_map = self.getProjectMap()
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
                            if self.area_by_name[area1].winners:
                                proj_map[area1].add(area2)
                            if self.area_by_name[area2].winners:
                                proj_map[area2].add(area2)
        return proj_map

    def activateWord(self, area_name, word):
        """Activate a word in the specified area."""
        area = self.area_by_name[area_name]
        k = area.k
        assembly_start = self.lexeme_dict[word]["index"] * k
        area.winners = list(range(assembly_start, assembly_start + k))
        area.fix_assembly()

    def activateIndex(self, area_name, index):
        """Activate a word by index in the specified area."""
        area = self.area_by_name[area_name]
        k = area.k
        assembly_start = index * k
        area.winners = list(range(assembly_start, assembly_start + k))
        area.fix_assembly()

    def interpretAssemblyAsString(self, area_name):
        """Interpret the assembly in an area as a string."""
        return self.getWord(area_name, 0.7)

    def getWord(self, area_name, min_overlap=0.7):
        """Get the word represented by the assembly in an area."""
        if not self.area_by_name[area_name].winners:
            raise Exception("Cannot get word because no assembly in " + area_name)
        winners = set(self.area_by_name[area_name].winners)
        area_k = self.area_by_name[area_name].k
        threshold = min_overlap * area_k
        for word, lexeme in self.lexeme_dict.items():
            word_index = lexeme["index"]
            word_assembly_start = word_index * area_k
            word_assembly = set(range(word_assembly_start, word_assembly_start + area_k))
            if len((winners & word_assembly)) >= threshold:
                return word
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
    
    def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=10, 
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, 
                 interarea_beta=0.5, verbose=False):
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
                            readout_rules=RUSSIAN_READOUT_RULES)
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


class EnglishParserBrain(ParserBrain):
    """English language parser brain."""
    
    def __init__(self, p, non_LEX_n=100000, non_LEX_k=50, LEX_k=20, 
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, 
                 interarea_beta=0.5, verbose=False):
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
                            readout_rules=ENGLISH_READOUT_RULES)
        self.verbose = verbose

        LEX_n = LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

        DET_k = LEX_k
        self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DET, non_LEX_n, DET_k, default_beta)
        self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DEP_CLAUSE, non_LEX_n, non_LEX_k, default_beta)

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
        # "War of fibers"
        if LEX in proj_map and len(proj_map[LEX]) > 2:  # because LEX->LEX
            raise Exception("Got that LEX projecting into many areas: " + str(proj_map[LEX]))
        return proj_map

    def getWord(self, area_name, min_overlap=0.7):
        """Get word with English-specific handling."""
        word = ParserBrain.getWord(self, area_name, min_overlap)
        if word:
            return word
        if not word and area_name == DET:
            winners = set(self.area_by_name[area_name].winners)
            area_k = self.area_by_name[area_name].k
            threshold = min_overlap * area_k
            nodet_index = DET_SIZE - 1
            nodet_assembly_start = nodet_index * area_k
            nodet_assembly = set(range(nodet_assembly_start, nodet_assembly_start + area_k))
            if len((winners & nodet_assembly)) > threshold:
                return "<null-det>"
        # If nothing matched, at least we can see that in the parse output.
        return "<NON-WORD>"
