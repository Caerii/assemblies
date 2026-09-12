"""
TACL 2021 rule-based parser (Mitropolsky, Collins, Papadimitriou).

Wraps the maintained ``neural_assemblies.language`` implementation.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union

from neural_assemblies.language import (
    EnglishParserBrain,
    RussianParserBrain,
    ReadoutMethod,
    parse as _parse,
)
from neural_assemblies.language.grammar_rules import AreaRule, FiberRule


class RuleParser:
    """Fiber-gated dependency parser from the TACL 2021 paper."""

    def __init__(
        self,
        language: str = "English",
        p: float = 0.1,
        non_LEX_n: int = 1000,
        lex_k: int = 20,
        project_rounds: int = 30,
        engine: str = "auto",
    ):
        self.language = language
        self.p = p
        self.lex_k = lex_k
        self.non_LEX_n = non_LEX_n
        self.project_rounds = project_rounds
        self.engine = engine

    def parse(
        self,
        sentence: str,
        readout: ReadoutMethod = ReadoutMethod.FIBER_READOUT,
    ) -> Union[List[Tuple[str, str, str]], dict[str, Any], None]:
        return _parse(
            sentence=sentence,
            language=self.language,
            p=self.p,
            LEX_k=self.lex_k,
            non_LEX_n=self.non_LEX_n,
            project_rounds=self.project_rounds,
            verbose=False,
            readout_method=readout,
            engine=self.engine,
        )


def parse_sentence(
    sentence: str,
    language: str = "English",
    non_LEX_n: int = 1000,
    **kwargs,
) -> Union[List[Tuple[str, str, str]], dict[str, Any], None]:
    """Parse a sentence with the TACL rule-based parser."""
    return RuleParser(language=language, non_LEX_n=non_LEX_n, **kwargs).parse(sentence)


__all__ = [
    "RuleParser",
    "parse_sentence",
    "EnglishParserBrain",
    "RussianParserBrain",
    "ReadoutMethod",
    "AreaRule",
    "FiberRule",
]
