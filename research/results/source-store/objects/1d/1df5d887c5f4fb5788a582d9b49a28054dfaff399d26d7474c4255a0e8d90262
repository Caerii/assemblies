"""Blocks-world instruction training sentences (closed vocabulary)."""

from __future__ import annotations

from typing import List

from ..core.grounding import GroundingContext, VOCABULARY
from .data import GroundedSentence


def blocks_vocabulary() -> dict:
    """Extra words for move commands (merged into parser at train time)."""
    extra = {
        "move": GroundingContext(motor=["MOVE", "ACTION"]),
        "put": GroundingContext(motor=["PUT", "ACTION"]),
        "place": GroundingContext(motor=["PLACE", "ACTION"]),
        "to": GroundingContext(spatial=["TO", "GOAL"]),
        "blk_a": GroundingContext(visual=["BLOCK", "A"]),
        "blk_b": GroundingContext(visual=["BLOCK", "B"]),
        "blk_c": GroundingContext(visual=["BLOCK", "C"]),
    }
    return {**VOCABULARY, **extra}


def create_blocks_instruction_sentences() -> List[GroundedSentence]:
    """Imperative move commands with block + goal roles."""
    data = []

    patterns = [
        (["move", "blk_a", "to", "table"], [None, "patient", None, "goal"]),
        (["move", "blk_b", "to", "table"], [None, "patient", None, "goal"]),
        (["move", "blk_a", "to", "blk_b"], [None, "patient", None, "goal"]),
        (["put", "blk_a", "on", "blk_b"], [None, "patient", None, "goal"]),
        (["place", "blk_c", "on", "table"], [None, "patient", None, "goal"]),
        (["move", "blk_b", "to", "blk_c"], [None, "patient", None, "goal"]),
    ]
    for words, roles in patterns:
        vocab = blocks_vocabulary()
        data.append(GroundedSentence(
            words=words,
            contexts=[vocab[w] for w in words],
            roles=roles,
            mood="imperative",
        ))

    return data


def blocks_compliance_test_cases() -> List[dict]:
    """Pinned language → expected tool args for evaluation."""
    return [
        {
            "words": ["move", "a", "to", "b"],
            "expected_tool": "move_block",
            "expected_args": {"block": "blk_a", "destination": "blk_b"},
            "expected_action": ("A", "B"),
        },
        {
            "words": ["put", "a", "on", "table"],
            "expected_tool": "move_block",
            "expected_args": {"block": "blk_a", "destination": "table"},
            "expected_action": ("A", "table"),
        },
        {
            "words": ["move", "blk_b", "to", "blk_c"],
            "expected_tool": "move_block",
            "expected_args": {"block": "blk_b", "destination": "blk_c"},
            "expected_action": ("B", "C"),
        },
    ]
