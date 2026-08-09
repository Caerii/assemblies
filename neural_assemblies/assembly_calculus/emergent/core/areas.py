"""
Brain area constants for the emergent NEMO architecture (48 areas).

WHY THE ARCHITECTURE IS A LIST OF AREAS.  In NEMO an area is the unit of
competition: neurons within an area fight for the top-``k`` slots, neurons in
different areas do not.  So drawing an area boundary is a substantive
commitment -- it says these representations are mutually exclusive and those
are simultaneously holdable.  NOUN_CORE and VERB_CORE are separate areas
precisely because a word must be able to be a noun without suppressing the
verb representation of a different word; ROLE_AGENT and ROLE_PATIENT are
separate for the same reason.  The area list below IS the theory of what the
language organ can hold at once.

Reading the tiers, input to structure:

    INPUT      sensory/modality streams that ground words in experience
    LEXICAL    word forms, split content vs. function
    CORE       part-of-speech categories, one area per POS
    THEMATIC   who-did-what-to-whom, plus ROLE_SCENE holding the whole event
    PHRASE     constituents (NP/VP/PP/ADJP/SENT)
    SYNTACTIC  grammatical relations (SUBJ/OBJ/IOBJ/SYN_VERB)
    CONTROL    sequence position, mood, tense, polarity, number, error

The count is asserted at import (see ``ALL_AREAS``) so that adding an area
without adding it to a grouping fails loudly rather than silently dropping it
from every iteration over the architecture.

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
# The verb's thematic slot. Mitropolsky & Papadimitriou (2025) sec. 2.3 lists
# three role areas -- ROLE_agent, ROLE_action, ROLE_patient -- in mutual
# inhibition. Without ROLE_action the verb never enters the constituent chain,
# so verb POSITION is unrepresentable: OSV and OVS reduce to the same
# [patient, agent] sequence and cannot be told apart.
ROLE_ACTION = "ROLE_ACTION"
ROLE_THEME = "ROLE_THEME"
ROLE_GOAL = "ROLE_GOAL"
ROLE_SOURCE = "ROLE_SOURCE"
ROLE_LOCATION = "ROLE_LOCATION"
# Scene area: holds one assembly for the WHOLE scene, synaptically connected to
# the thematic role areas. Mitropolsky & Papadimitriou (2025) sec. 2.3: "a
# fourth area, ROLE_scene synaptically connected to all three ... ROLE_scene
# contains an assembly, representing the whole scene, synaptically connected to
# these three." During generation SCENE and MOOD "keep firing at every step",
# which is what re-supplies the fillers after the trigger transiently inhibits
# the role areas. It is deliberately NOT in the mutual-inhibition group.
ROLE_SCENE = "ROLE_SCENE"

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
# The paper's third syntactic area ("three syntactic areas, SUBJ, OBJ and
# VERB"). Named SYN_VERB to avoid confusion with the VERB_CORE lexical area.
SYN_VERB = "SYN_VERB"

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
    ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT, ROLE_THEME,
    ROLE_GOAL, ROLE_SOURCE, ROLE_LOCATION,
]

#: Role ANNOTATION string -> thematic area. The one place this mapping lives.
#: It was previously spelled out separately in `parser_mixins._shared` and
#: inline in `training/consolidation`, which is how a role label can be honoured
#: by one consumer and silently dropped by another (see #116: `goal` is in
#: THEMATIC_AREAS but in none of the maps, so those annotations `continue`).
#: Covering four of the seven areas is a REAL limit, recorded here rather
#: than rediscovered per call site (theme/source/location remain unmapped).
ROLE_LABEL_TO_AREA = {
    "agent": ROLE_AGENT,
    "action": ROLE_ACTION,
    "patient": ROLE_PATIENT,
    "goal": ROLE_GOAL,
}

# Scene area is a role-system area but not a thematic slot, so it is kept out
# of THEMATIC_AREAS (which callers iterate as the mutually-exclusive slots).
SCENE_AREAS = [ROLE_SCENE]

PHRASE_AREAS = [NP, VP, PP, ADJP, SENT]

SYNTACTIC_AREAS = [SUBJ, SYN_VERB, OBJ, IOBJ]

CONTROL_AREAS = [SEQ, MOOD, TENSE, POLARITY, NUMBER, ERROR]

VP_COMPONENT_AREAS = [VP_SUBJ, VP_VERB, VP_OBJ]

ADVANCED_AREAS = [CONTEXT, PRODUCTION, PREDICTION, DEP_CLAUSE]

ALL_AREAS = (
    INPUT_AREAS + LEXICAL_AREAS + CORE_AREAS + THEMATIC_AREAS + SCENE_AREAS
    + PHRASE_AREAS + SYNTACTIC_AREAS + CONTROL_AREAS + VP_COMPONENT_AREAS
    + ADVANCED_AREAS
)

assert len(ALL_AREAS) == 48, f"Expected 48 areas, got {len(ALL_AREAS)}"


# ---- Mappings ----

# The acquisition claim in one table.  A learner is not told that "dog" is a
# noun; it is told that "dog" co-occurs with something visual and "run" with
# something motor.  Routing each grounding modality to a different core area
# means part of speech FALLS OUT of which sensory stream a word is paired
# with, rather than being supplied as a label.  "none" -> DET_CORE is the
# residual case and carries the interesting prediction: words with no sensory
# grounding at all end up in one area together, which is roughly the function
# word class (see FUNC_SUBCATEGORIES below, where they are then separated by
# distribution rather than by grounding).
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

# Areas inside a group compete: when one wins, the others are suppressed, so
# at most one member of a group can hold an active assembly at a time.  This is
# what makes role assignment a DECISION rather than a set of independent
# scores -- a word cannot be simultaneously the agent and the patient, and the
# suppression is what forces the commitment.  ROLE_SCENE is deliberately
# excluded (see its definition above): it must keep firing while the thematic
# slots take turns, which is exactly what group membership would forbid.
MUTUAL_INHIBITION_GROUPS = [
    [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT, ROLE_THEME,
     ROLE_GOAL, ROLE_SOURCE, ROLE_LOCATION],
    [SUBJ, SYN_VERB, OBJ, IOBJ],
]

# ---- Per-value feature areas (E15, task #144) ----
# E14 measured the scaling blocker of the one-area feature design: TWO label
# values sharing ONE k-WTA area merge as total label projections grow (shared
# SG∩PL image columns 2.6 -> 17.8 of 30 at 200 frames, through fully VARIED
# sentences -- count, not replay). k-WTA amplifies shared drive, so both
# labels recruit the area's mass attractor and the images converge. No corpus
# shape fixes an architecture problem.
#
# The fix is the papers' own pattern: one area PER VALUE, in a mutual
# inhibition group -- "two or more areas can be in mutual inhibition, in which
# case there is firing only in the area that receives the greatest total
# synaptic input" (Mitropolsky & Papadimitriou 2025, sec. 2; their ROLE
# triple is "the only use of interarea inhibition in our model", and this
# extends the same device to feature values). Merging becomes structurally
# impossible: the two label images live in different areas.
#
# OPT-IN: these areas are NOT in ALL_AREAS. They are created lazily by
# MorphosyntaxMixin when `split_feature_areas` is enabled, and their MI group
# is registered at creation -- the default 48-area topology is untouched.
FEATURE_VALUE_LABELS = {
    NUMBER: ("SG", "PL"),
    TENSE: ("PRESENT", "PAST", "FUTURE", "PROGRESSIVE", "PERFECT"),
}


def feature_value_area(feature: str, label: str) -> str:
    """The per-value area name for (feature, label), e.g. NUMBER_SG."""
    return f"{feature}_{label}"

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
