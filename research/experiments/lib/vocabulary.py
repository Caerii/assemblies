"""
Vocabulary and category definitions for language experiments.

The Vocabulary class encapsulates word lists, category membership, and the
mapping from words to brain areas. Experiments configure a vocabulary once
and pass it through the pipeline — training, measurement, and grammar all
derive their behavior from the same vocabulary definition.

Design principle: adding a new word category (e.g., adjectives, determiners)
requires only extending the Vocabulary with a new category entry. Training
and measurement code operates on categories generically.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class CategoryDef:
    """Definition of a word category.

    Attributes:
        words: List of words in this category.
        core_area: Brain area where assemblies for this category live.
        role_areas: Mapping from grammatical role name to brain area name.
            E.g., {"AGENT": "ROLE_AGENT", "PATIENT": "ROLE_PATIENT"}.
            A category can fill multiple roles.
    """
    words: List[str]
    core_area: str
    role_areas: Dict[str, str] = field(default_factory=dict)


@dataclass
class Vocabulary:
    """Complete vocabulary specification for a language experiment.

    Organizes words into categories, each mapped to a core brain area and
    zero or more role slots. Provides lookup methods used by training,
    measurement, and grammar modules.

    Attributes:
        categories: Mapping from category name to its definition.
        novel_words: Words that exist in the lexicon but are never used
            during training. Used to test generalization.
    """
    categories: Dict[str, CategoryDef]
    novel_words: Dict[str, str] = field(default_factory=dict)
    # novel_words maps word -> category_name (which category it belongs to)

    @property
    def all_words(self) -> List[str]:
        """All words across all categories, including novel words."""
        words = []
        for cat in self.categories.values():
            words.extend(cat.words)
        words.extend(self.novel_words.keys())
        return words

    @property
    def all_core_areas(self) -> List[str]:
        """Unique core area names."""
        return list({cat.core_area for cat in self.categories.values()})

    @property
    def all_role_areas(self) -> List[str]:
        """Unique role area names."""
        roles: Set[str] = set()
        for cat in self.categories.values():
            roles.update(cat.role_areas.values())
        return list(roles)

    @property
    def all_areas(self) -> List[str]:
        """All brain areas: core + role + PREDICTION."""
        areas = set(self.all_core_areas)
        areas.update(self.all_role_areas)
        areas.add("PREDICTION")
        return sorted(areas)

    def core_area_for(self, word: str) -> str:
        """Return the core brain area for a word."""
        for cat in self.categories.values():
            if word in cat.words:
                return cat.core_area
        # Check novel words
        if word in self.novel_words:
            cat_name = self.novel_words[word]
            return self.categories[cat_name].core_area
        raise ValueError(f"Unknown word: {word}")

    def role_area_for(self, role: str) -> str:
        """Return the brain area for a grammatical role."""
        for cat in self.categories.values():
            if role in cat.role_areas:
                return cat.role_areas[role]
        raise ValueError(f"Unknown role: {role}")

    def words_for_category(self, cat_name: str) -> List[str]:
        """Return words in a category (excluding novel words)."""
        return list(self.categories[cat_name].words)

    def category_for_word(self, word: str) -> str:
        """Return the category name for a word."""
        for name, cat in self.categories.items():
            if word in cat.words:
                return name
        if word in self.novel_words:
            return self.novel_words[word]
        raise ValueError(f"Unknown word: {word}")


# ── Default vocabulary (SVO + PP) ──────────────────────────────────

DEFAULT_VOCAB = Vocabulary(
    categories={
        "NOUN": CategoryDef(
            words=["dog", "cat", "bird", "boy", "girl"],
            core_area="NOUN_CORE",
            role_areas={
                "AGENT": "ROLE_AGENT",
                "PATIENT": "ROLE_PATIENT",
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
        "VERB": CategoryDef(
            words=["chases", "sees", "eats", "finds", "hits"],
            core_area="VERB_CORE",
            role_areas={},  # verbs don't bind to role slots
        ),
        "PREP": CategoryDef(
            words=["in", "on", "at"],
            core_area="PREP_CORE",
            role_areas={},
        ),
        "LOCATION": CategoryDef(
            words=["garden", "park", "house"],
            core_area="NOUN_CORE",  # locations are nouns
            role_areas={
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
    },
    novel_words={
        "table": "NOUN",
        "chair": "NOUN",
    },
)

# ── Determiner vocabulary (DET + SVO + PP) ────────────────────────
#
# Adds determiners ("the", "a") before nouns. The DET-N-V-DET-N pattern
# provides distributional evidence for a 3-way category split that the
# bare SVO pattern cannot support.

DET_VOCAB = Vocabulary(
    categories={
        "DET": CategoryDef(
            words=["the", "a"],
            core_area="DET_CORE",
            role_areas={},
        ),
        "NOUN": CategoryDef(
            words=["dog", "cat", "bird", "boy", "girl"],
            core_area="NOUN_CORE",
            role_areas={
                "AGENT": "ROLE_AGENT",
                "PATIENT": "ROLE_PATIENT",
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
        "VERB": CategoryDef(
            words=["chases", "sees", "eats", "finds", "hits"],
            core_area="VERB_CORE",
            role_areas={},
        ),
        "PREP": CategoryDef(
            words=["in", "on", "at"],
            core_area="PREP_CORE",
            role_areas={},
        ),
        "LOCATION": CategoryDef(
            words=["garden", "park", "house"],
            core_area="NOUN_CORE",
            role_areas={
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
    },
    novel_words={
        "table": "NOUN",
        "chair": "NOUN",
    },
)

# ── Adjective vocabulary (DET + ADJ + SVO + PP) ───────────────────
#
# Adds adjectives between determiners and nouns. The DET-ADJ-N-V pattern
# provides distributional evidence for a 4-way category split, testing
# whether the routing mechanism scales from 3→4 categories.

ADJ_VOCAB = Vocabulary(
    categories={
        "DET": CategoryDef(
            words=["the", "a"],
            core_area="DET_CORE",
            role_areas={},
        ),
        "ADJ": CategoryDef(
            words=["big", "small", "old", "young", "fast"],
            core_area="ADJ_CORE",
            role_areas={},
        ),
        "NOUN": CategoryDef(
            words=["dog", "cat", "bird", "boy", "girl"],
            core_area="NOUN_CORE",
            role_areas={
                "AGENT": "ROLE_AGENT",
                "PATIENT": "ROLE_PATIENT",
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
        "VERB": CategoryDef(
            words=["chases", "sees", "eats", "finds", "hits"],
            core_area="VERB_CORE",
            role_areas={},
        ),
        "PREP": CategoryDef(
            words=["in", "on", "at"],
            core_area="PREP_CORE",
            role_areas={},
        ),
        "LOCATION": CategoryDef(
            words=["garden", "park", "house"],
            core_area="NOUN_CORE",
            role_areas={
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
    },
    novel_words={
        "table": "NOUN",
        "chair": "NOUN",
    },
)

# ── RC vocabulary (DET + ADJ + SVO + COMP for relative clauses) ───
#
# Extends ADJ_VOCAB with a complementizer ("that") for relative clauses.
# The COMP category provides distributional evidence for a 5-way split:
#   DET → CORE_0, ADJ → CORE_1, NOUN → CORE_2, VERB → CORE_3, COMP → CORE_4

RC_VOCAB = Vocabulary(
    categories={
        "DET": CategoryDef(
            words=["the", "a"],
            core_area="DET_CORE",
            role_areas={},
        ),
        "ADJ": CategoryDef(
            words=["big", "small", "old", "young", "fast"],
            core_area="ADJ_CORE",
            role_areas={},
        ),
        "NOUN": CategoryDef(
            words=["dog", "cat", "bird", "boy", "girl"],
            core_area="NOUN_CORE",
            role_areas={
                "AGENT": "ROLE_AGENT",
                "PATIENT": "ROLE_PATIENT",
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
        "VERB": CategoryDef(
            words=["chases", "sees", "eats", "finds", "hits"],
            core_area="VERB_CORE",
            role_areas={},
        ),
        "PREP": CategoryDef(
            words=["in", "on", "at"],
            core_area="PREP_CORE",
            role_areas={},
        ),
        "LOCATION": CategoryDef(
            words=["garden", "park", "house"],
            core_area="NOUN_CORE",
            role_areas={
                "PP_OBJ": "ROLE_PP_OBJ",
            },
        ),
        "COMP": CategoryDef(
            words=["that"],
            core_area="COMP_CORE",
            role_areas={},
        ),
    },
    novel_words={
        "table": "NOUN",
        "chair": "NOUN",
    },
)

# ── Recursive vocabulary (SVO + recursive PP + relative clauses) ───

RECURSIVE_VOCAB = Vocabulary(
    categories={
        "NOUN": CategoryDef(
            words=["dog", "cat", "bird", "boy", "girl"],
            core_area="NOUN_CORE",
            role_areas={
                "AGENT": "ROLE_AGENT",
                "PATIENT": "ROLE_PATIENT",
                "PP_OBJ": "ROLE_PP_OBJ",
                "PP_OBJ_1": "ROLE_PP_OBJ_1",
                "REL_AGENT": "ROLE_REL_AGENT",
                "REL_PATIENT": "ROLE_REL_PATIENT",
            },
        ),
        "VERB": CategoryDef(
            words=["chases", "sees", "eats", "finds", "hits"],
            core_area="VERB_CORE",
            role_areas={},
        ),
        "PREP": CategoryDef(
            words=["in", "on", "at"],
            core_area="PREP_CORE",
            role_areas={},
        ),
        "LOCATION": CategoryDef(
            words=["garden", "park", "house", "field", "river"],
            core_area="NOUN_CORE",
            role_areas={
                "PP_OBJ": "ROLE_PP_OBJ",
                "PP_OBJ_1": "ROLE_PP_OBJ_1",
            },
        ),
        "COMP": CategoryDef(
            words=["that"],
            core_area="COMP_CORE",
            role_areas={},
        ),
    },
    novel_words={
        "table": "NOUN",
        "chair": "NOUN",
    },
)


# Convenience lists for backward compatibility with existing experiments
NOUNS = DEFAULT_VOCAB.categories["NOUN"].words
VERBS = DEFAULT_VOCAB.categories["VERB"].words
PREPS = DEFAULT_VOCAB.categories["PREP"].words
LOCATIONS = DEFAULT_VOCAB.categories["LOCATION"].words
NOVEL_NOUNS = list(DEFAULT_VOCAB.novel_words.keys())
