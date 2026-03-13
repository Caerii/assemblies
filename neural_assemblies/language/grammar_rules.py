"""
Grammar rules and lexeme dictionaries for language processing.

This module contains the grammar rules, lexeme dictionaries, and rule
definitions for English and Russian language processing.
"""

from collections import namedtuple
from .language_areas import *

# Rule definitions
AreaRule = namedtuple('AreaRule', ['action', 'area', 'index'])
FiberRule = namedtuple('FiberRule', ['action', 'area1', 'area2', 'index'])
FiringRule = namedtuple('FiringRule', ['action'])
OtherRule = namedtuple('OtherRule', ['action'])

def generic_noun(index):
    """Generate grammar rules for a generic noun."""
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, SUBJ, 0), 
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            FiberRule(DISINHIBIT, LEX, PREP_P, 0),
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
            FiberRule(DISINHIBIT, DET, OBJ, 0),
            FiberRule(DISINHIBIT, DET, PREP_P, 0),
            FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, OBJ, 0),
            FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, PREP, 0),
            FiberRule(DISINHIBIT, PREP_P, SUBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, OBJ, 0),
        ],
        "POST_RULES": [
            AreaRule(INHIBIT, DET, 0),
            AreaRule(INHIBIT, ADJ, 0),
            AreaRule(INHIBIT, PREP_P, 0),
            AreaRule(INHIBIT, PREP, 0),
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            FiberRule(INHIBIT, LEX, PREP_P, 0),
            FiberRule(INHIBIT, ADJ, SUBJ, 0),
            FiberRule(INHIBIT, ADJ, OBJ, 0),
            FiberRule(INHIBIT, ADJ, PREP_P, 0),
            FiberRule(INHIBIT, DET, SUBJ, 0),
            FiberRule(INHIBIT, DET, OBJ, 0),
            FiberRule(INHIBIT, DET, PREP_P, 0),
            FiberRule(INHIBIT, VERB, OBJ, 0),
            FiberRule(INHIBIT, PREP_P, PREP, 0),
            FiberRule(INHIBIT, PREP_P, VERB, 0),
            FiberRule(DISINHIBIT, LEX, SUBJ, 1),
            FiberRule(DISINHIBIT, LEX, OBJ, 1),
            FiberRule(DISINHIBIT, DET, SUBJ, 1),
            FiberRule(DISINHIBIT, DET, OBJ, 1),
            FiberRule(DISINHIBIT, ADJ, SUBJ, 1),
            FiberRule(DISINHIBIT, ADJ, OBJ, 1),
            FiberRule(INHIBIT, PREP_P, SUBJ, 0),
            FiberRule(INHIBIT, PREP_P, OBJ, 0),
            FiberRule(INHIBIT, VERB, ADJ, 0),
        ]
    }

def generic_trans_verb(index):
    """Generate grammar rules for a transitive verb."""
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADVERB, 0),
            FiberRule(DISINHIBIT, VERB, DEP_CLAUSE, 0),
            AreaRule(DISINHIBIT, ADVERB, 1),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, OBJ, 0),
            AreaRule(DISINHIBIT, SUBJ, 0),
            AreaRule(DISINHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, PREP_P, VERB, 0),
            FiberRule(DISINHIBIT, VERB, DEP_CLAUSE, 0),
        ]
    }

def generic_intrans_verb(index):
    """Generate grammar rules for an intransitive verb."""
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADVERB, 0),
            AreaRule(DISINHIBIT, ADVERB, 1),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, SUBJ, 0),
            AreaRule(DISINHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, PREP_P, VERB, 0),
        ]
    }

def generic_copula(index):
    """Generate grammar rules for a copula verb."""
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, OBJ, 0),
            AreaRule(DISINHIBIT, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, VERB, 0)
        ]
    }

def generic_adverb(index):
    """Generate grammar rules for an adverb."""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, LEX, ADVERB, 0)
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADVERB, 0),
            AreaRule(INHIBIT, ADVERB, 1),
        ]
    }

def generic_determinant(index):
    """Generate grammar rules for a determiner."""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, DET, 0),
            FiberRule(DISINHIBIT, LEX, DET, 0)
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, DET, 0),
            FiberRule(INHIBIT, VERB, ADJ, 0),
        ]
    }

def generic_adjective(index):
    """Generate grammar rules for an adjective."""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADJ, 0),
            FiberRule(DISINHIBIT, LEX, ADJ, 0)
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADJ, 0),
            FiberRule(INHIBIT, VERB, ADJ, 0),
        ]
    }

def generic_preposition(index):
    """Generate grammar rules for a preposition."""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, PREP, 0),
            FiberRule(DISINHIBIT, LEX, PREP, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, PREP, 0),
            AreaRule(DISINHIBIT, PREP_P, 0),
            FiberRule(INHIBIT, LEX, SUBJ, 1),
            FiberRule(INHIBIT, LEX, OBJ, 1),
            FiberRule(INHIBIT, DET, SUBJ, 1),
            FiberRule(INHIBIT, DET, OBJ, 1),
            FiberRule(INHIBIT, ADJ, SUBJ, 1),
            FiberRule(INHIBIT, ADJ, OBJ, 1),
        ]
    }

# English lexeme dictionary
LEXEME_DICT = {
    "the": generic_determinant(0),
    "a": generic_determinant(1),
    "dogs": generic_noun(2),
    "cats": generic_noun(3),
    "mice": generic_noun(4),
    "people": generic_noun(5),
    "chase": generic_trans_verb(6),
    "love": generic_trans_verb(7),
    "bite": generic_trans_verb(8),
    "of": generic_preposition(9),
    "big": generic_adjective(10),
    "bad": generic_adjective(11),
    "run": generic_intrans_verb(12),
    "fly": generic_intrans_verb(13),
    "quickly": generic_adverb(14),
    "in": generic_preposition(15),
    "are": generic_copula(16),
    "man": generic_noun(17),
    "woman": generic_noun(18),
    "saw": generic_trans_verb(19),
}

# Russian grammar functions
def generic_russian_verb(index):
    """Generate grammar rules for a Russian verb."""
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, VERB, 0),
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, NOM, 0),
            FiberRule(DISINHIBIT, VERB, ACC, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0)
        ]
    }

def generic_russian_ditransitive_verb(index):
    """Generate grammar rules for a Russian ditransitive verb."""
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, VERB, 0),
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, NOM, 0),
            FiberRule(DISINHIBIT, VERB, ACC, 0),
            FiberRule(DISINHIBIT, VERB, DAT, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0)
        ]
    }

def generic_russian_nominative_noun(index):
    """Generate grammar rules for a Russian nominative noun."""
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, NOM, 0),
            FiberRule(DISINHIBIT, LEX, NOM, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, NOM, 0)
        ]
    }

def generic_russian_accusative_noun(index):
    """Generate grammar rules for a Russian accusative noun."""
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ACC, 0),
            FiberRule(DISINHIBIT, LEX, ACC, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ACC, 0)
        ]
    }

def generic_russian_dative_noun(index):
    """Generate grammar rules for a Russian dative noun."""
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, DAT, 0),
            FiberRule(DISINHIBIT, LEX, DAT, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, DAT, 0)
        ]
    }

# Russian lexeme dictionary
RUSSIAN_LEXEME_DICT = {
    "vidit": generic_russian_verb(0),
    "lyubit": generic_russian_verb(1),
    "kot": generic_russian_nominative_noun(2),
    "kota": generic_russian_accusative_noun(2),
    "sobaka": generic_russian_nominative_noun(3),
    "sobaku": generic_russian_accusative_noun(3),
    "sobakie": generic_russian_dative_noun(3),
    "kotu": generic_russian_dative_noun(2),
    "dayet": generic_russian_ditransitive_verb(4)
}
