"""Multi-turn dialogue working memory for EmergentParser sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..structured_io import InstructionFrame


@dataclass
class DialogueState:
    """Working memory: turn history, entities, and discourse context."""

    history: List[dict] = field(default_factory=list)
    recent_entities: Dict[str, str] = field(default_factory=dict)
    last_agent: Optional[str] = None
    last_patient: Optional[str] = None
    last_action: Optional[str] = None
    turn_count: int = 0

    def record_turn(
        self,
        speaker: str,
        words: List[str],
        frame: Optional[InstructionFrame] = None,
        reply: Optional[str] = None,
    ) -> None:
        self.turn_count += 1
        entry = {
            "speaker": speaker,
            "words": list(words),
            "frame": frame,
            "reply": reply,
        }
        self.history.append(entry)
        if frame is not None:
            self._update_entities(frame)

    def _update_entities(self, frame: InstructionFrame) -> None:
        if frame.agent and frame.agent not in ("you", "i"):
            self.last_agent = frame.agent
            self.recent_entities["agent"] = frame.agent
        if frame.patient:
            self.last_patient = frame.patient
            self.recent_entities["patient"] = frame.patient
        if frame.action:
            self.last_action = frame.action
            self.recent_entities["action"] = frame.action
        for word, role in frame.roles.items():
            if role == "AGENT" and word not in ("who", "what", "you", "i"):
                self.recent_entities.setdefault("agent", word)
            if role == "PATIENT":
                self.recent_entities.setdefault("patient", word)

    def resolve_words(self, words: List[str]) -> List[str]:
        """Replace pronouns with recent entities when known."""
        resolved = []
        for w in words:
            lw = w.lower()
            if lw == "it" and self.last_patient:
                resolved.append(self.last_patient)
            elif lw == "they" and self.recent_entities.get("agent"):
                resolved.append(self.recent_entities["agent"])
            elif lw == "he" and self.last_agent:
                resolved.append(self.last_agent)
            elif lw == "she" and self.last_agent:
                resolved.append(self.last_agent)
            else:
                resolved.append(w)
        return resolved

    def recent_context_words(self, n_turns: int = 2) -> List[str]:
        """Flatten words from the last *n* user/system turns."""
        out: List[str] = []
        for entry in self.history[-n_turns * 2 :]:
            out.extend(entry.get("words", []))
        return out

    def clear(self) -> None:
        self.history.clear()
        self.recent_entities.clear()
        self.last_agent = None
        self.last_patient = None
        self.last_action = None
        self.turn_count = 0
