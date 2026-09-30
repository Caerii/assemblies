"""
Shared Test Sentence Definitions

Reusable test sentence structures for P600 and agreement experiments.
"""

from typing import Dict, List, Any


def make_p600_test_sentences(vocab=None) -> List[Dict[str, Any]]:
    """SVO category violation test triples (gram/sem/cat) for P600.

    Each triple has the same context ("the SUBJ VERB the ___") with:
    - Grammatical: trained animal as object
    - Semantic violation: untrained object-category noun
    - Category violation: verb in object position (DIFFERENT from sentence verb)

    Args:
        vocab: Unused, kept for backward compatibility.
    """
    return [
        {
            "frame": "the dog chases the ___",
            "context_words": ["the", "dog", "chases", "the"],
            "grammatical": "cat",
            "semantic_violation": "table",
            "category_violation": "likes",
        },
        {
            "frame": "the cat sees the ___",
            "context_words": ["the", "cat", "sees", "the"],
            "grammatical": "bird",
            "semantic_violation": "chair",
            "category_violation": "finds",
        },
        {
            "frame": "the bird chases the ___",
            "context_words": ["the", "bird", "chases", "the"],
            "grammatical": "fish",
            "semantic_violation": "book",
            "category_violation": "sees",
        },
        {
            "frame": "the horse finds the ___",
            "context_words": ["the", "horse", "finds", "the"],
            "grammatical": "mouse",
            "semantic_violation": "ball",
            "category_violation": "chases",
        },
    ]


# -- Agreement test sentences (shared by agreement_number and
#    agreement_violations experiments) -----------------------------------------

VERB_AGREEMENT_TESTS = [
    {
        "label": "dog_chases",
        "sg_context": ["the", "dog"],
        "pl_context": ["the", "dogs"],
        "verb": "chases",
    },
    {
        "label": "cat_sees",
        "sg_context": ["the", "cat"],
        "pl_context": ["the", "cats"],
        "verb": "sees",
    },
    {
        "label": "bird_chases",
        "sg_context": ["the", "bird"],
        "pl_context": ["the", "birds"],
        "verb": "chases",
    },
    {
        "label": "horse_finds",
        "sg_context": ["the", "horse"],
        "pl_context": ["the", "horses"],
        "verb": "finds",
    },
]

OBJECT_AGREEMENT_TESTS = [
    {
        "label": "dog_chases_cat",
        "sg_context": ["the", "dog", "chases", "the"],
        "pl_context": ["the", "dogs", "chases", "the"],
        "grammatical_obj": "cat",
        "category_violation": "likes",
    },
    {
        "label": "cat_sees_bird",
        "sg_context": ["the", "cat", "sees", "the"],
        "pl_context": ["the", "cats", "sees", "the"],
        "grammatical_obj": "bird",
        "category_violation": "finds",
    },
    {
        "label": "bird_chases_fish",
        "sg_context": ["the", "bird", "chases", "the"],
        "pl_context": ["the", "birds", "chases", "the"],
        "grammatical_obj": "fish",
        "category_violation": "sees",
    },
    {
        "label": "horse_finds_mouse",
        "sg_context": ["the", "horse", "finds", "the"],
        "pl_context": ["the", "horses", "finds", "the"],
        "grammatical_obj": "mouse",
        "category_violation": "chases",
    },
]
