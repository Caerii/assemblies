"""
Dialogue State Management
=========================

Tracks conversation history and context.
This is the "working memory" for interactive NEMO.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Turn:
    """A single turn in the dialogue."""
    speaker: str  # "user" or "system"
    text: str
    words: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # What was learned from this turn
    learned_patterns: List[str] = field(default_factory=list)
    learned_words: List[str] = field(default_factory=list)
    
    # Classification of the turn
    turn_type: Optional[str] = None  # "statement", "question", "command", etc.
    
    def __str__(self):
        return f"{self.speaker}: {self.text}"


@dataclass 
class DialogueState:
    """
    Tracks the state of an ongoing dialogue.
    
    This is NEMO's "working memory" - what it remembers
    about the current conversation.
    """
    
    # Conversation history
    history: List[Turn] = field(default_factory=list)
    
    # Current topic (emergent from recent turns)
    current_topic: Optional[str] = None
    
    # Words encountered but not yet known
    unknown_words: List[str] = field(default_factory=list)
    
    # Recent entities mentioned (for reference resolution)
    recent_entities: Dict[str, Any] = field(default_factory=dict)
    
    # Dialogue statistics
    total_turns: int = 0
    user_turns: int = 0
    system_turns: int = 0
    
    def add_turn(self, speaker: str, text: str, words: List[str], 
                 turn_type: Optional[str] = None) -> Turn:
        """Add a new turn to the dialogue."""
        turn = Turn(
            speaker=speaker,
            text=text,
            words=words,
            turn_type=turn_type
        )
        self.history.append(turn)
        self.total_turns += 1
        
        if speaker == "user":
            self.user_turns += 1
        else:
            self.system_turns += 1
        
        return turn
    
    def get_recent_turns(self, n: int = 5) -> List[Turn]:
        """Get the n most recent turns."""
        return self.history[-n:] if self.history else []
    
    def get_recent_words(self, n_turns: int = 3) -> List[str]:
        """Get words from recent turns (for context)."""
        words = []
        for turn in self.get_recent_turns(n_turns):
            words.extend(turn.words)
        return words
    
    def mark_word_unknown(self, word: str):
        """Mark a word as unknown (for later learning)."""
        if word not in self.unknown_words:
            self.unknown_words.append(word)
    
    def mark_word_learned(self, word: str):
        """Mark a word as now known."""
        if word in self.unknown_words:
            self.unknown_words.remove(word)
    
    def update_recent_entities(self, entities: Dict[str, Any]):
        """Update recently mentioned entities."""
        self.recent_entities.update(entities)
    
    def clear(self):
        """Clear dialogue state (start new conversation)."""
        self.history = []
        self.current_topic = None
        self.unknown_words = []
        self.recent_entities = {}
        self.total_turns = 0
        self.user_turns = 0
        self.system_turns = 0

