"""
Standard vocabulary definitions for N400/P600 experiments.

Two vocabulary variants:
- build_standard_vocab(): 17 words for word-pair priming (N400)
- build_svo_vocab(): 15 words for SVO sentence processing (P600)
"""

from typing import Dict

from src.assembly_calculus.emergent.grounding import GroundingContext


def build_standard_vocab() -> Dict[str, GroundingContext]:
    """8-word vocabulary: 6 animals, 6 objects, 4 verbs, 1 function word.

    Used by: test_n400_pre_kwta, test_n400_controls, test_n400_parameter_sweep
    """
    return {
        # Animals (share ANIMAL feature)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Objects (unique features)
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
        # Function words
        "the":    GroundingContext(),
    }


def build_svo_vocab() -> Dict[str, GroundingContext]:
    """SVO vocabulary with determiners: 6 animals, 4 objects, 4 verbs, 1 function.

    Used by: test_p600_syntactic
    """
    return {
        # Animals (trained as subjects and objects)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Objects (never trained in sentences -- semantic violations)
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "TOY"]),
        # Verbs (trained as actions)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function
        "the":    GroundingContext(),
    }
