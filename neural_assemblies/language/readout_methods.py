"""
Readout methods for language parsing.

This module contains the readout strategies and methods for extracting
parse trees and dependencies from neural assembly activations.
"""

from enum import Enum
try:
    import pptree
except ImportError:
    pptree = None
from .language_areas import *

class ReadoutMethod(Enum):
    """Enumeration of available readout methods."""
    FIXED_MAP_READOUT = 1
    FIBER_READOUT = 2
    NATURAL_READOUT = 3

def read_out(area, mapping, brain, dependencies, readout_rules):
    """
    Recursively read out parse structure from activated areas.
    
    Args:
        area: The current area being processed
        mapping: The mapping of areas to their targets
        brain: The parser brain instance
        dependencies: List to store discovered dependencies
        readout_rules: Rules for readout processing
    """
    to_areas = mapping[area]
    brain.project({}, {area: to_areas})
    
    if area != DEP_CLAUSE:
        this_word = brain.getWord(LEX)

    for to_area in to_areas:
        if to_area == LEX:
            continue
        if to_area == DEP_CLAUSE:
            brain.project({}, {to_area: [VERB]})
            brain.project({}, {VERB: [LEX, SUBJ]})
            dep_verb = brain.getWord(LEX)
            dependencies.append([this_word, dep_verb, "DEP-VERB"])
            brain.project({}, {SUBJ: [LEX]})
            dep_verb_subj = brain.getWord(LEX)
            dependencies.append([dep_verb, dep_verb_subj, "SUBJ"])
            continue
        brain.project({}, {to_area: [LEX]})
        other_word = brain.getWord(LEX)
        dependencies.append([this_word, other_word, to_area])

    for to_area in to_areas:
        if to_area != LEX:
            read_out(to_area, mapping, brain, dependencies, readout_rules)

def treeify(parsed_dict, parent):
    """
    Convert parsed dictionary to tree structure.
    
    Args:
        parsed_dict: Dictionary containing parse structure
        parent: Parent node in the tree
    """
    if pptree is None:
        print("pptree not available, skipping tree visualization")
        return
        
    for key, values in parsed_dict.items():
        key_node = pptree.Node(key, parent)
        if isinstance(values, str):
            _ = pptree.Node(values, key_node)
        else:
            treeify(values, key_node)

def fixed_map_readout(brain, readout_rules, verbose=False):
    """
    Perform fixed map readout of the parse structure.
    
    Args:
        brain: The parser brain instance
        readout_rules: Rules for readout processing
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing the parsed structure
    """
    dependencies = []
    parsed = {VERB: read_out(VERB, readout_rules, brain, dependencies, readout_rules)}

    if verbose:
        print("Final parse dict: ")
        print(parsed)

    if pptree is not None:
        root = pptree.Node(VERB)
        treeify(parsed[VERB], root)
    else:
        print("pptree not available, skipping tree creation")
    
    return parsed

def fiber_readout(brain, activated_fibers, verbose=False):
    """
    Perform fiber activation readout of the parse structure.
    
    Args:
        brain: The parser brain instance
        activated_fibers: Dictionary of activated fiber connections
        verbose: Whether to print verbose output
        
    Returns:
        List of dependencies discovered
    """
    dependencies = []
    
    if verbose:
        print("Got activated fibers for readout:")
        print(activated_fibers)

    read_out(VERB, activated_fibers, brain, dependencies, None)
    
    if verbose:
        print("Got dependencies: ")
        print(dependencies)
    
    return dependencies
