"""TACL 2021 parser role-assignment smoke (TACL21-E05 proxy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set


@dataclass
class TaclParserF1Result:
    sentence: str
    roles_found: List[str]
    expected_roles: List[str]
    role_recall: float
    parameters: dict


def _role_recall(found: Set[str], expected: Set[str]) -> float:
    if not expected:
        return 1.0
    return len(found & expected) / len(expected)


def run_tacl_parser_f1_smoke(
    *,
    language: str = "english",
) -> TaclParserF1Result:
    from neural_assemblies.programs.rule_parser import parse_sentence

    if language == "english":
        sentence = "cats chase mice"
        expected = {"SUBJ", "VERB", "OBJ"}
        lang = "English"
    else:
        sentence = "мальчик дал девочке мяч"
        expected = {"NOM", "ACC", "DAT"}
        lang = "Russian"

    result = parse_sentence(sentence, language=lang)
    if result is None:
        raise RuntimeError(f"parser returned no role assignment for {sentence!r}")
    roles = {role for _, _, role in result}
    recall = _role_recall(roles, expected)
    return TaclParserF1Result(
        sentence=sentence,
        roles_found=sorted(roles),
        expected_roles=sorted(expected),
        role_recall=recall,
        parameters={"language": language},
    )
