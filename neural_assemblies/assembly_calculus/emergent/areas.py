"""
Brain area constants for the 44-area emergent NEMO architecture.

Mirrors src/nemo/language/emergent/areas.py but uses plain strings
(the Brain API requires string area names, not enums).

References:
    Mitropolsky & Papadimitriou (2025). "Simulated Language Acquisition."
"""

# =========== INPUT MODALITIES (8 areas) ===========
PHON = "PHON"
VISUAL = "VISUAL"
MOTOR = "MOTOR"
PROPERTY = "PROPERTY"
SPATIAL = "SPATIAL"
TEMPORAL = "TEMPORAL"
SOCIAL = "SOCIAL"
EMOTION = "EMOTION"

# =========== LEXICAL AREAS (2 areas) ===========
LEX_CONTENT = "LEX_CONTENT"
LEX_FUNCTION = "LEX_FUNCTION"

# =========== CORE/CATEGORY AREAS (8 areas) ===========
NOUN_CORE = "NOUN_CORE"
VERB_CORE = "VERB_CORE"
ADJ_CORE = "ADJ_CORE"
ADV_CORE = "ADV_CORE"
PREP_CORE = "PREP_CORE"
DET_CORE = "DET_CORE"
PRON_CORE = "PRON_CORE"
CONJ_CORE = "CONJ_CORE"

# =========== THEMATIC ROLE AREAS (6 areas) ===========
ROLE_AGENT = "ROLE_AGENT"
ROLE_PATIENT = "ROLE_PATIENT"
ROLE_THEME = "ROLE_THEME"
ROLE_GOAL = "ROLE_GOAL"
ROLE_SOURCE = "ROLE_SOURCE"
ROLE_LOCATION = "ROLE_LOCATION"

# =========== PHRASE STRUCTURE (5 areas) ===========
NP = "NP"
VP = "VP"
PP = "PP"
ADJP = "ADJP"
SENT = "SENT"

# =========== SYNTACTIC ROLES (3 areas) ===========
SUBJ = "SUBJ"
OBJ = "OBJ"
IOBJ = "IOBJ"

# =========== SEQUENCE/CONTROL (5 areas) ===========
SEQ = "SEQ"
MOOD = "MOOD"
TENSE = "TENSE"
POLARITY = "POLARITY"
NUMBER = "NUMBER"          # Morphological number (SG/PL)

# =========== ERROR DETECTION (1 area) ===========
ERROR = "ERROR"

# =========== VP COMPONENT AREAS (3 areas) ===========
VP_SUBJ = "VP_SUBJ"
VP_VERB = "VP_VERB"
VP_OBJ = "VP_OBJ"

# =========== INCREMENTAL/ADVANCED AREAS (4 areas) ===========
CONTEXT = "CONTEXT"          # Running context assembly for incremental parsing
PRODUCTION = "PRODUCTION"    # Staging area for language production
PREDICTION = "PREDICTION"    # Next-token prediction area
DEP_CLAUSE = "DEP_CLAUSE"   # Embedded dependent clause area


# ---- Groupings ----

INPUT_AREAS = [PHON, VISUAL, MOTOR, PROPERTY, SPATIAL, TEMPORAL, SOCIAL, EMOTION]

LEXICAL_AREAS = [LEX_CONTENT, LEX_FUNCTION]

CORE_AREAS = [
    NOUN_CORE, VERB_CORE, ADJ_CORE, ADV_CORE,
    PREP_CORE, DET_CORE, PRON_CORE, CONJ_CORE,
]

THEMATIC_AREAS = [
    ROLE_AGENT, ROLE_PATIENT, ROLE_THEME,
    ROLE_GOAL, ROLE_SOURCE, ROLE_LOCATION,
]

PHRASE_AREAS = [NP, VP, PP, ADJP, SENT]

SYNTACTIC_AREAS = [SUBJ, OBJ, IOBJ]

CONTROL_AREAS = [SEQ, MOOD, TENSE, POLARITY, NUMBER, ERROR]

VP_COMPONENT_AREAS = [VP_SUBJ, VP_VERB, VP_OBJ]

ADVANCED_AREAS = [CONTEXT, PRODUCTION, PREDICTION, DEP_CLAUSE]

ALL_AREAS = (
    INPUT_AREAS + LEXICAL_AREAS + CORE_AREAS + THEMATIC_AREAS
    + PHRASE_AREAS + SYNTACTIC_AREAS + CONTROL_AREAS + VP_COMPONENT_AREAS
    + ADVANCED_AREAS
)

assert len(ALL_AREAS) == 45, f"Expected 45 areas, got {len(ALL_AREAS)}"


# ---- Mappings ----

GROUNDING_TO_CORE = {
    "visual": NOUN_CORE,
    "motor": VERB_CORE,
    "properties": ADJ_CORE,
    "spatial": PREP_CORE,
    "social": PRON_CORE,
    "temporal": ADV_CORE,
    "none": DET_CORE,
}

CORE_TO_CATEGORY = {
    NOUN_CORE: "NOUN",
    VERB_CORE: "VERB",
    ADJ_CORE: "ADJ",
    ADV_CORE: "ADV",
    PREP_CORE: "PREP",
    DET_CORE: "DET",
    PRON_CORE: "PRON",
    CONJ_CORE: "CONJ",
}

CATEGORY_TO_CORE = {v: k for k, v in CORE_TO_CATEGORY.items()}

MUTUAL_INHIBITION_GROUPS = [
    [ROLE_AGENT, ROLE_PATIENT, ROLE_THEME,
     ROLE_GOAL, ROLE_SOURCE, ROLE_LOCATION],
    [SUBJ, OBJ, IOBJ],
]

# ---- Function word sub-categories ----
# These are sub-types of DET (ungrounded words) discovered from
# distributional frames. They route through DET_CORE neurally but
# trigger different gating patterns during parsing.
#
# Matches the ELAN → gating model:
#   1. Rapid sub-categorization from bigram frames (ELAN, ~180ms)
#   2. Sub-category triggers learned fiber gating (Broca's top-down control)

FUNC_DET = "DET"        # Determiner: precedes NOUN/ADJ ("the", "a")
FUNC_AUX = "AUX"        # Auxiliary: between NP and VP ("was", "were")
FUNC_COMP = "COMP"      # Complementizer: after NP, opens clause ("that", "which")
FUNC_CONJ = "CONJ"      # Conjunction: between parallel structures ("and")
FUNC_MARKER = "MARKER"  # Role marker: signals upcoming role ("by" in passives)

FUNC_SUBCATEGORIES = [FUNC_DET, FUNC_AUX, FUNC_COMP, FUNC_CONJ, FUNC_MARKER]

# Maps function sub-category to core area for neural routing
# (all ungrounded function words route through DET_CORE)
FUNC_SUBCAT_TO_CORE = {
    FUNC_DET: DET_CORE,
    FUNC_AUX: DET_CORE,
    FUNC_COMP: DET_CORE,
    FUNC_CONJ: CONJ_CORE,
    FUNC_MARKER: PREP_CORE,  # "by" has spatial grounding
}
