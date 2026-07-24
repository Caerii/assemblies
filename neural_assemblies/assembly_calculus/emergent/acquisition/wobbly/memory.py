"""Wobbly episodic memory and ERROR routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from ...core.areas import CONTEXT, ERROR
from ...evaluation.erp import ErpProbeResult

if TYPE_CHECKING:
    from ...parser import EmergentParser

WobblyProbe = ErpProbeResult

STRUCTURAL_SIGNATURES = frozenset({
    "structural_wobble",
    "lexical_and_structural",
    "phrase_instability",
})


@dataclass
class WobblyEpisode:
    """One wobbly position in a heard sentence."""
    sentence: Tuple[str, ...]
    probe: WobblyProbe
    hypotheses: Tuple[Tuple[str, str], ...] = ()
    resolved_category: Optional[str] = None
    resolved_combined: float = 1.0
    resolved_stability: float = 0.0


@dataclass
class WobblyMemory:
    """Episodic store of wobbly parses from raw exposure."""
    episodes: List[WobblyEpisode] = field(default_factory=list)
    max_episodes: int = 500

    def add(self, episode: WobblyEpisode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]

    def unique_targets(self) -> List[Tuple[str, str]]:
        seen: Set[Tuple[str, str]] = set()
        out: List[Tuple[str, str]] = []
        for ep in self.episodes:
            for word, cat in ep.hypotheses:
                key = (word, cat)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def wobbly_words(self) -> Set[str]:
        return {ep.probe.word for ep in self.episodes}


def activate_error(parser: "EmergentParser", surprise: float) -> bool:
    """Route combined surprise into ERROR (parse-failure control area)."""
    brain = parser.brain
    if ERROR not in brain.areas:
        return False
    with brain.frozen():
        brain.inhibit_areas([ERROR])
        rounds = max(1, min(parser.inference_rounds, int(surprise * 6)))
        if CONTEXT in brain.areas:
            brain.project({}, {CONTEXT: [ERROR], ERROR: [ERROR]})
            if rounds > 1:
                brain.project_rounds(
                    target=ERROR,
                    areas_by_stim={},
                    dst_areas_by_src_area={ERROR: [ERROR], CONTEXT: [ERROR]},
                    rounds=rounds - 1,
                )
        return True
