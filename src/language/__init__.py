"""
Language processing modules.

This module implements language processing capabilities for neural
assemblies, including parsing, syntax learning, and word acquisition.
"""

from .parser import ParserBrain, RussianParserBrain, EnglishParserBrain
from .grammar_rules import LEXEME_DICT, RUSSIAN_LEXEME_DICT
from .language_areas import *
from .readout_methods import ReadoutMethod, fixed_map_readout, fiber_readout
from .debugger import ParserDebugger

__all__ = ['ParserBrain', 'RussianParserBrain', 'EnglishParserBrain', 
           'LEXEME_DICT', 'RUSSIAN_LEXEME_DICT', 'ReadoutMethod', 
           'fixed_map_readout', 'fiber_readout', 'ParserDebugger']

def parse(sentence="cats chase mice", language="English", p=0.1, LEX_k=20, 
          project_rounds=30, verbose=True, debug=False, readout_method=ReadoutMethod.FIBER_READOUT):
    """
    Parse a sentence using the specified language model.
    
    Args:
        sentence: The sentence to parse
        language: Language to use ("English" or "Russian")
        p: Probability parameter for brain initialization
        LEX_k: Number of winners in LEX area
        project_rounds: Number of projection rounds
        verbose: Whether to print verbose output
        debug: Whether to enable debugging
        readout_method: Method for readout processing
        
    Returns:
        Parse results based on the readout method
    """
    if language == "English":
        b = EnglishParserBrain(p, LEX_k=LEX_k, verbose=verbose)
        lexeme_dict = LEXEME_DICT
        all_areas = AREAS
        explicit_areas = EXPLICIT_AREAS
        readout_rules = ENGLISH_READOUT_RULES

    if language == "Russian":
        b = RussianParserBrain(p, LEX_k=LEX_k, verbose=verbose)
        lexeme_dict = RUSSIAN_LEXEME_DICT
        all_areas = RUSSIAN_AREAS
        explicit_areas = RUSSIAN_EXPLICIT_AREAS
        readout_rules = RUSSIAN_READOUT_RULES

    return parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug, 
                      lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules)

def parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug, 
                lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules):
    """
    Helper function for parsing with detailed control.
    
    Args:
        b: Parser brain instance
        sentence: Sentence to parse
        p: Probability parameter
        LEX_k: Number of winners in LEX area
        project_rounds: Number of projection rounds
        verbose: Whether to print verbose output
        debug: Whether to enable debugging
        lexeme_dict: Dictionary of lexemes
        all_areas: List of all areas
        explicit_areas: List of explicit areas
        readout_method: Method for readout processing
        readout_rules: Rules for readout processing
        
    Returns:
        Parse results based on the readout method
    """
    from .debugger import ParserDebugger
    
    debugger = ParserDebugger(b, all_areas, explicit_areas)
    sentence = sentence.split(" ")

    extreme_debug = False
    word_index = 0
    saved_outer_start = 0 
    saved_inner_start = None
    
    while word_index < len(sentence):
        word = sentence[word_index]

        lexeme = lexeme_dict[word]
        b.activateWord(LEX, word)
        if verbose:
            print("Activated word: " + word)
            print(b.area_by_name[LEX].winners)

        for rule in lexeme["PRE_RULES"]:
            b.applyRule(rule)

        proj_map = b.getProjectMap()
        for area in proj_map:
            if area not in proj_map[LEX]:
                b.area_by_name[area].fix_assembly()
                if verbose:
                    print("FIXED assembly bc not LEX->this area in: " + area)
            elif area != LEX:
                b.area_by_name[area].unfix_assembly()
                b.area_by_name[area].winners = []
                if verbose:
                    print("ERASED assembly because LEX->this area in " + area)

        proj_map = b.getProjectMap()
        if verbose:
            print("Got proj_map = ")
            print(proj_map)

        for i in range(project_rounds):
            b.parse_project()
            if verbose:
                proj_map = b.getProjectMap()
                print("Got proj_map = ")
                print(proj_map)
            if extreme_debug and word == "a":
                print("Starting debugger after round " + str(i) + "for word" + word)
                debugger.run()

        # Handle recursive clauses
        if ((word_index+1) < len(sentence)) and (sentence[word_index+1] == "that"):
            print("Beginning of recursive clause!")
            b.applyAreaRule(AreaRule(DISINHIBIT, DEP_CLAUSE, 0))
            b.applyFiberRule(FiberRule(DISINHIBIT, SUBJ, DEP_CLAUSE, 0))
            b.applyFiberRule(FiberRule(DISINHIBIT, OBJ, DEP_CLAUSE, 0))
            b.applyAreaRule(AreaRule(INHIBIT, LEX, 0))
            b.applyAreaRule(AreaRule(INHIBIT, VERB, 0))
            proj_map = b.getProjectMap()
            print("Got recursive proj_map before fixing = ")
            print(proj_map)
            for area in proj_map:
                if area != DEP_CLAUSE:
                    b.area_by_name[area].fix_assembly()
                    print("FIXED assembly bc not DEP_CLAUSE area in: " + area)
            b.area_by_name[DEP_CLAUSE].unfix_assembly()
            for i in range(project_rounds):
                b.parse_project()
            print("Finished DEP_CLAUSE projecting")
            saved_inner_start = word_index+1 
            # refresh the machine correctly-- importantly, DEP_CLAUSE stays on 
            b.initialize_states()
            b.area_states[DEP_CLAUSE].discard(0)
            word_index = word_index+2
            continue

        if ((word_index+1) < len(sentence))  and (sentence[word_index+1] == ","):
            print("End of recursive clause!!")
            # end of dependent clause (i.e. "inner")
            # refresh everything 
            b.initialize_states()
            # from saved_outer_start to saved_inner_start, go through, apply rules, project once w/o plasticity
            b.disable_plasticity = True
            for j in range(saved_outer_start, saved_inner_start):
                word = sentence[j]
                print("TOUCHING " + word)
                lexeme = lexeme_dict[word]
                b.activateWord(LEX, word)
                for rule in lexeme["PRE_RULES"]:
                    b.applyRule(rule)
                proj_map = b.getProjectMap()
                for area in proj_map:
                    if area not in proj_map[LEX]:
                        b.area_by_name[area].fix_assembly()
                    elif area != LEX:
                        b.area_by_name[area].unfix_assembly()
                        b.area_by_name[area].winners = []
                proj_map = b.getProjectMap()
                b.parse_project()
                for rule in lexeme["POST_RULES"]:
                    b.applyRule(rule)
            b.disable_plasticity = False
            b.applyAreaRule(AreaRule(INHIBIT, DEP_CLAUSE, 0))
            word_index = word_index+2
            saved_inner_start = None
            continue

        for rule in lexeme["POST_RULES"]:
            b.applyRule(rule)

        if debug:
            print("Starting debugger after the word " + word)
            debugger.run()

        word_index += 1

    # Readout
    # For all readout methods, unfix assemblies and remove plasticity.
    b.disable_plasticity = True
    for area in all_areas:
        b.area_by_name[area].unfix_assembly()

    if readout_method == ReadoutMethod.FIXED_MAP_READOUT:
        return fixed_map_readout(b, readout_rules, verbose)
    elif readout_method == ReadoutMethod.FIBER_READOUT:
        activated_fibers = b.getActivatedFibers()
        return fiber_readout(b, activated_fibers, verbose)
    else:
        return None