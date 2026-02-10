"""
Grounding contexts and vocabulary for the emergent NEMO parser.

Each word in the vocabulary has a GroundingContext that specifies which
sensory modalities are active when the word is experienced. The dominant
modality determines which core area the word's assembly forms in:

    VISUAL  -> NOUN_CORE     (objects, animals, people)
    MOTOR   -> VERB_CORE     (actions, movements)
    PROPERTY -> ADJ_CORE     (size, color, speed)
    SPATIAL -> PREP_CORE     (locations, relations)
    SOCIAL  -> PRON_CORE     (people, referents)
    TEMPORAL -> ADV_CORE     (time, manner)
    none    -> DET_CORE      (function words)

References:
    Mitropolsky & Papadimitriou (2025). "Simulated Language Acquisition."
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GroundingContext:
    """Grounding context for a word — which sensory modalities are active."""

    visual: List[str] = field(default_factory=list)
    motor: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    spatial: List[str] = field(default_factory=list)
    social: List[str] = field(default_factory=list)
    temporal: List[str] = field(default_factory=list)
    emotional: List[str] = field(default_factory=list)

    @property
    def dominant_modality(self) -> str:
        """Return the name of the first non-empty modality, or 'none'."""
        for name in ("visual", "motor", "properties", "spatial",
                     "social", "temporal", "emotional"):
            if getattr(self, name):
                return name
        return "none"

    @property
    def is_grounded(self) -> bool:
        """True if any sensory modality is active."""
        return self.dominant_modality != "none"


# ======================================================================
# Vocabulary: 37 words across 7 POS categories
# ======================================================================

VOCABULARY = {
    # --- NOUNS (visual grounding) --- 10 words
    "dog": GroundingContext(visual=["DOG", "ANIMAL"]),
    "cat": GroundingContext(visual=["CAT", "ANIMAL"]),
    "bird": GroundingContext(visual=["BIRD", "ANIMAL"]),
    "boy": GroundingContext(visual=["BOY", "PERSON"]),
    "girl": GroundingContext(visual=["GIRL", "PERSON"]),
    "ball": GroundingContext(visual=["BALL", "OBJECT"]),
    "book": GroundingContext(visual=["BOOK", "OBJECT"]),
    "food": GroundingContext(visual=["FOOD", "OBJECT"]),
    "table": GroundingContext(visual=["TABLE", "FURNITURE"]),
    "car": GroundingContext(visual=["CAR", "OBJECT"]),

    # --- VERBS (motor grounding) --- 8 words
    "runs": GroundingContext(motor=["RUNNING", "MOTION"]),
    "sees": GroundingContext(motor=["SEEING", "PERCEPTION"]),
    "eats": GroundingContext(motor=["EATING", "CONSUMPTION"]),
    "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
    "plays": GroundingContext(motor=["PLAYING", "ACTION"]),
    "sleeps": GroundingContext(motor=["SLEEPING", "REST"]),
    "reads": GroundingContext(motor=["READING", "COGNITION"]),
    "finds": GroundingContext(motor=["FINDING", "PERCEPTION"]),

    # --- ADJECTIVES (property grounding) --- 5 words
    "big": GroundingContext(properties=["SIZE", "BIG"]),
    "small": GroundingContext(properties=["SIZE", "SMALL"]),
    "red": GroundingContext(properties=["COLOR", "RED"]),
    "fast": GroundingContext(properties=["SPEED", "FAST"]),
    "happy": GroundingContext(properties=["EMOTION", "HAPPY"]),

    # --- PREPOSITIONS (spatial grounding) --- 4 words
    "on": GroundingContext(spatial=["ON", "ABOVE"]),
    "in": GroundingContext(spatial=["IN", "INSIDE"]),
    "under": GroundingContext(spatial=["UNDER", "BELOW"]),
    "near": GroundingContext(spatial=["NEAR", "PROXIMITY"]),

    # --- PRONOUNS (social grounding) --- 4 words
    "he": GroundingContext(social=["PERSON", "MALE"]),
    "she": GroundingContext(social=["PERSON", "FEMALE"]),
    "it": GroundingContext(social=["THING"]),
    "they": GroundingContext(social=["GROUP"]),

    # --- ADVERBS (temporal grounding) --- 3 words
    "quickly": GroundingContext(temporal=["QUICK", "MANNER"]),
    "slowly": GroundingContext(temporal=["SLOW", "MANNER"]),
    "yesterday": GroundingContext(temporal=["PAST", "TIME"]),

    # --- DETERMINERS (no grounding) --- 2 words
    "the": GroundingContext(),
    "a": GroundingContext(),

    # --- CONJUNCTIONS (no grounding) --- 1 word
    "and": GroundingContext(),
}

assert len(VOCABULARY) == 37, f"Expected 37 words, got {len(VOCABULARY)}"
