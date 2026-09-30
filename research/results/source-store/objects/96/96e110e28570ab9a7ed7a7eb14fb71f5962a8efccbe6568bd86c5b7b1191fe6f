"""Classification evidence keeps its measurement source through conversion."""
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal, Mapping

from .areas import CORE_TO_CATEGORY


@dataclass(frozen=True)
class ClassificationEvidence:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-classification-evidence

    Neural scores use core-area keys; distributional scores use category keys.
    Values are source-specific strengths, not interchangeable probabilities.
    The legacy tuple is a compatibility view that loses source information.
    """

    category: str
    source: Literal["neural", "distributional", "none"]
    scores: Mapping[str, float]
    cue_mode: Literal["combined", "phon_only", "grounding_only"] | None = None
    cues: tuple[str, ...] = ()

    def __post_init__(self):
        if self.cue_mode not in (None, "combined", "phon_only", "grounding_only"):
            raise ValueError("Unknown classification cue_mode")
        if type(self.cues) not in (tuple, list) or any(not isinstance(c, str) or not c for c in self.cues):
            raise ValueError("Classification cues must be nonempty stimulus names")
        cues = tuple(self.cues)
        if len(set(cues)) != len(cues) or (cues and self.cue_mode is None):
            raise ValueError("Classification cues require a mode and distinct names")
        object.__setattr__(self, "cues", cues)
        domains = {"neural": CORE_TO_CATEGORY,
                   "distributional": set(CORE_TO_CATEGORY.values()), "none": set()}
        if self.source not in domains:
            raise ValueError(f"Unknown classification source: {self.source}")
        if self.category not in {*CORE_TO_CATEGORY.values(), "UNKNOWN"}:
            raise ValueError(f"Unknown classification category: {self.category}")
        scores = dict(self.scores)
        if set(scores) - set(domains[self.source]):
            raise ValueError(f"Scores do not belong to the {self.source} domain")
        if any(not isfinite(value) or value < 0 for value in scores.values()):
            raise ValueError("Classification scores must be finite and nonnegative")
        if self.source == "none" and self.category != "UNKNOWN":
            raise ValueError("Absent evidence cannot supply a category")
        object.__setattr__(self, "scores", MappingProxyType(scores))

    def category_scores(self) -> dict[str, float]:
        """Explicitly convert index domains without relabeling the source."""
        if self.source != "neural":
            return dict(self.scores)
        return {CORE_TO_CATEGORY[area]: score for area, score in self.scores.items()}

    def as_legacy_tuple(self) -> tuple[str, dict[str, float]]:
        return self.category, dict(self.scores)
