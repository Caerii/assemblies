"""
Language area definitions and constants.

This module defines the brain areas used for language processing,
including English and Russian language areas, and their configurations.
"""

# Brain Areas
LEX = "LEX"
DET = "DET"
SUBJ = "SUBJ"
OBJ = "OBJ"
VERB = "VERB"
PREP = "PREP"
PREP_P = "PREP_P"
ADJ = "ADJ"
ADVERB = "ADVERB"
DEP_CLAUSE = "DEPCLAUSE"

# Unique to Russian
NOM = "NOM"
ACC = "ACC"
DAT = "DAT"

# Fixed area stats for explicit areas
LEX_SIZE = 20
RUSSIAN_LEX_SIZE = 7

# Actions
DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"
ACTIVATE_ONLY = "ACTIVATE_ONLY"
CLEAR_DET = "CLEAR_DET"

# Area configurations
AREAS = [LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P, DEP_CLAUSE]
EXPLICIT_AREAS = [LEX]
RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P, DEP_CLAUSE]

RUSSIAN_AREAS = [LEX, NOM, VERB, ACC, DAT]
RUSSIAN_EXPLICIT_AREAS = [LEX]

# Readout rules
ENGLISH_READOUT_RULES = {
    VERB: [LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ],
    SUBJ: [LEX, DET, ADJ, PREP_P, DEP_CLAUSE],
    OBJ: [LEX, DET, ADJ, PREP_P, DEP_CLAUSE],
    PREP_P: [LEX, PREP, ADJ, DET],
    PREP: [LEX],
    ADJ: [LEX],
    DET: [LEX],
    ADVERB: [LEX],
    LEX: [],
    DEP_CLAUSE: [],
}

RUSSIAN_READOUT_RULES = {
    VERB: [LEX, NOM, ACC, DAT],
    NOM: [LEX],
    ACC: [LEX],
    DAT: [LEX],
    LEX: [],
}
