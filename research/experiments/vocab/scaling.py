"""
Vocabulary scaling definitions for N400 vocabulary scaling experiments.

Three vocabulary sizes with increasing number of nouns and categories:
- Small: 12 nouns (6 animals + 6 objects)
- Medium: 30 nouns (adds tools, foods, vehicles)
- Large: 50 nouns (adds clothing, extra animals, extra furniture)
"""

from typing import Dict, List, Tuple

import numpy as np

from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


def build_small_vocab() -> Dict[str, GroundingContext]:
    """12 nouns + 4 verbs + 1 function = 17 words (original scale)."""
    return {
        # Animals (share ANIMAL feature)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Objects (each pair shares a sub-category)
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "TOY"]),
        "car":    GroundingContext(visual=["CAR", "VEHICLE"]),
        "cup":    GroundingContext(visual=["CUP", "CONTAINER"]),
        # Verbs
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function
        "the":    GroundingContext(),
    }


def build_medium_vocab() -> Dict[str, GroundingContext]:
    """30 nouns + 6 verbs + 1 function = 37 words."""
    v = build_small_vocab()
    # Add tools (share TOOL feature)
    v["hammer"]  = GroundingContext(visual=["HAMMER", "TOOL"])
    v["wrench"]  = GroundingContext(visual=["WRENCH", "TOOL"])
    v["drill"]   = GroundingContext(visual=["DRILL", "TOOL"])
    v["saw"]     = GroundingContext(visual=["SAW", "TOOL"])
    v["pliers"]  = GroundingContext(visual=["PLIERS", "TOOL"])
    v["screwdriver"] = GroundingContext(visual=["SCREWDRIVER", "TOOL"])
    # Add foods (share FOOD feature)
    v["apple"]   = GroundingContext(visual=["APPLE", "FOOD"])
    v["bread"]   = GroundingContext(visual=["BREAD", "FOOD"])
    v["cheese"]  = GroundingContext(visual=["CHEESE", "FOOD"])
    v["rice"]    = GroundingContext(visual=["RICE", "FOOD"])
    v["soup"]    = GroundingContext(visual=["SOUP", "FOOD"])
    v["cake"]    = GroundingContext(visual=["CAKE", "FOOD"])
    # Add vehicles (share VEHICLE feature)
    v["truck"]   = GroundingContext(visual=["TRUCK", "VEHICLE"])
    v["bike"]    = GroundingContext(visual=["BIKE", "VEHICLE"])
    v["boat"]    = GroundingContext(visual=["BOAT", "VEHICLE"])
    v["plane"]   = GroundingContext(visual=["PLANE", "VEHICLE"])
    v["train"]   = GroundingContext(visual=["TRAIN", "VEHICLE"])
    v["bus"]     = GroundingContext(visual=["BUS", "VEHICLE"])
    # Extra verbs for diversity
    v["grabs"]   = GroundingContext(motor=["GRABBING", "ACTION"])
    v["holds"]   = GroundingContext(motor=["HOLDING", "ACTION"])
    return v


def build_large_vocab() -> Dict[str, GroundingContext]:
    """50 nouns + 8 verbs + 1 function = 59 words."""
    v = build_medium_vocab()
    # Add clothing (share CLOTHING feature)
    v["shirt"]   = GroundingContext(visual=["SHIRT", "CLOTHING"])
    v["pants"]   = GroundingContext(visual=["PANTS", "CLOTHING"])
    v["shoes"]   = GroundingContext(visual=["SHOES", "CLOTHING"])
    v["hat"]     = GroundingContext(visual=["HAT", "CLOTHING"])
    v["jacket"]  = GroundingContext(visual=["JACKET", "CLOTHING"])
    v["scarf"]   = GroundingContext(visual=["SCARF", "CLOTHING"])
    v["gloves"]  = GroundingContext(visual=["GLOVES", "CLOTHING"])
    v["belt"]    = GroundingContext(visual=["BELT", "CLOTHING"])
    # Extra animals and objects for density
    v["rabbit"]  = GroundingContext(visual=["RABBIT", "ANIMAL"])
    v["snake"]   = GroundingContext(visual=["SNAKE", "ANIMAL"])
    v["wolf"]    = GroundingContext(visual=["WOLF", "ANIMAL"])
    v["deer"]    = GroundingContext(visual=["DEER", "ANIMAL"])
    v["lamp"]    = GroundingContext(visual=["LAMP", "FURNITURE"])
    v["shelf"]   = GroundingContext(visual=["SHELF", "FURNITURE"])
    v["desk"]    = GroundingContext(visual=["DESK", "FURNITURE"])
    v["bed"]     = GroundingContext(visual=["BED", "FURNITURE"])
    # Extra verbs
    v["pushes"]  = GroundingContext(motor=["PUSHING", "ACTION"])
    v["drops"]   = GroundingContext(motor=["DROPPING", "ACTION"])
    return v


def build_training_for_vocab(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Build training sentences using available nouns and verbs.

    Generates ~20-40+ sentences covering all nouns, with random verb assignment.
    Each noun appears multiple times as both subject and object.
    """
    def ctx(w):
        return vocab[w]

    nouns = [w for w, c in vocab.items()
             if c.dominant_modality == "visual"]
    verbs = [w for w, c in vocab.items()
             if c.dominant_modality == "motor"]

    sentences = []
    rng = np.random.default_rng(42)

    # Generate sentences covering all nouns
    noun_pairs = []
    for i in range(len(nouns)):
        for j in range(len(nouns)):
            if i != j:
                noun_pairs.append((nouns[i], nouns[j]))

    rng.shuffle(noun_pairs)
    # Use enough sentences so each noun appears multiple times
    n_sentences = min(len(noun_pairs), max(40, len(nouns) * 3))

    for subj, obj in noun_pairs[:n_sentences]:
        verb = verbs[rng.integers(0, len(verbs))]
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


def make_test_pairs(
    vocab: Dict[str, GroundingContext],
) -> List[Tuple[str, str, str]]:
    """Generate within-category and cross-category test pairs.

    Returns list of (target, related_prime, unrelated_prime) where:
    - related = same category (shares feature)
    - unrelated = different category (no shared feature)
    """
    # Group nouns by their second visual feature (category)
    categories: Dict[str, List[str]] = {}
    for word, ctx in vocab.items():
        if ctx.dominant_modality != "visual":
            continue
        feats = ctx.visual
        if len(feats) >= 2:
            cat = feats[1]  # Second feature is the category
            categories.setdefault(cat, []).append(word)

    # Filter to categories with >= 2 members
    cat_names = [c for c, words in categories.items() if len(words) >= 2]
    if len(cat_names) < 2:
        return []

    pairs = []
    rng = np.random.default_rng(123)

    for cat in cat_names:
        words = categories[cat]
        # Pick unrelated category
        other_cats = [c for c in cat_names if c != cat]
        if not other_cats:
            continue
        unrel_cat = other_cats[rng.integers(0, len(other_cats))]
        unrel_words = categories[unrel_cat]

        # Generate pairs: up to 3 per category
        for i in range(min(3, len(words) - 1)):
            target = words[i]
            related = words[(i + 1) % len(words)]
            unrelated = unrel_words[rng.integers(0, len(unrel_words))]
            if target != related and target != unrelated:
                pairs.append((target, related, unrelated))

    return pairs
