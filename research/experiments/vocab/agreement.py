"""
Agreement Vocabulary and Training Data

Shared vocabulary and training sentence builders for all agreement-related
experiments (agreement_violations, agreement_number, morphological_agreement).

Number is encoded as a grounding feature (SG/PL) so that singular and plural
forms of the same word share most features but differ in number marking.
"""

from typing import Dict, List

from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


def build_agreement_vocab(
    include_person_nouns: bool = False,
    include_intransitive_verbs: bool = False,
    include_untrained_objects: bool = False,
) -> Dict[str, GroundingContext]:
    """Agreement vocabulary with singular/plural noun and verb forms.

    Base vocabulary (always included):
        6 SG nouns (dog, cat, bird, horse, fish, mouse)
        4 PL nouns (dogs, cats, birds, horses)
        4 SG verbs (chases, sees, finds, likes)
        4 PL verbs (chase, see, find, like)
        1 function word (the)

    Optional extensions:
        include_person_nouns: Add boy/girl/boys/girls (SG/PL person nouns)
        include_intransitive_verbs: Add runs/run, eats/eat, sleeps/sleep
        include_untrained_objects: Add table, chair (furniture, no number)
    """
    vocab = {
        # Singular nouns
        "dog":    GroundingContext(visual=["DOG", "ANIMAL", "SG"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL", "SG"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL", "SG"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL", "SG"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL", "SG"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL", "SG"]),
        # Plural nouns
        "dogs":   GroundingContext(visual=["DOG", "ANIMAL", "PL"]),
        "cats":   GroundingContext(visual=["CAT", "ANIMAL", "PL"]),
        "birds":  GroundingContext(visual=["BIRD", "ANIMAL", "PL"]),
        "horses": GroundingContext(visual=["HORSE", "ANIMAL", "PL"]),
        # Singular verbs (3sg)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT", "SG"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION", "SG"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION", "SG"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION", "SG"]),
        # Plural/bare verbs
        "chase":  GroundingContext(motor=["CHASING", "PURSUIT", "PL"]),
        "see":    GroundingContext(motor=["SEEING", "PERCEPTION", "PL"]),
        "find":   GroundingContext(motor=["FINDING", "PERCEPTION", "PL"]),
        "like":   GroundingContext(motor=["LIKING", "EMOTION", "PL"]),
        # Function words
        "the":    GroundingContext(),
        "a":      GroundingContext(),
    }

    if include_person_nouns:
        vocab.update({
            "boy":    GroundingContext(visual=["BOY", "PERSON", "SG"]),
            "girl":   GroundingContext(visual=["GIRL", "PERSON", "SG"]),
            "boys":   GroundingContext(visual=["BOY", "PERSON", "PL"]),
            "girls":  GroundingContext(visual=["GIRL", "PERSON", "PL"]),
        })

    if include_intransitive_verbs:
        vocab.update({
            "runs":   GroundingContext(motor=["RUNNING", "MOTION", "SG"]),
            "run":    GroundingContext(motor=["RUNNING", "MOTION", "PL"]),
            "eats":   GroundingContext(motor=["EATING", "CONSUMPTION", "SG"]),
            "eat":    GroundingContext(motor=["EATING", "CONSUMPTION", "PL"]),
            "sleeps": GroundingContext(motor=["SLEEPING", "REST", "SG"]),
            "sleep":  GroundingContext(motor=["SLEEPING", "REST", "PL"]),
        })

    if include_untrained_objects:
        vocab.update({
            "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
            "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        })

    return vocab


def build_agreement_training(
    vocab: Dict[str, GroundingContext],
    n_repetitions: int = 3,
) -> List[GroundedSentence]:
    """Training on ONLY agreeing sentences (sg+sg, pl+pl).

    Generates SVO sentences where subject and verb always agree in number.
    The parser never sees disagreeing combinations during training.

    Args:
        vocab: Agreement vocabulary (from build_agreement_vocab).
        n_repetitions: How many times to repeat each sentence pattern.
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    sg_triples = [
        ("dog", "chases", "cat"), ("cat", "sees", "bird"),
        ("bird", "chases", "fish"), ("horse", "chases", "dog"),
        ("dog", "sees", "bird"), ("cat", "finds", "horse"),
    ]
    pl_triples = [
        ("dogs", "chase", "cats"), ("cats", "see", "birds"),
        ("birds", "chase", "dogs"), ("horses", "chase", "cats"),
        ("dogs", "see", "birds"), ("cats", "find", "horses"),
    ]

    for triples in [sg_triples, pl_triples]:
        for _ in range(n_repetitions):
            for subj, verb, obj in triples:
                sentences.append(GroundedSentence(
                    words=["the", subj, verb, "the", obj],
                    contexts=[ctx("the"), ctx(subj), ctx(verb),
                              ctx("the"), ctx(obj)],
                    roles=[None, "agent", "action", None, "patient"],
                ))

    return sentences
