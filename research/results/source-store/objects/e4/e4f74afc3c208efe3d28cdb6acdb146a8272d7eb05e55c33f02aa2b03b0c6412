"""
TACL 2021 parser sentence suite — English and Russian parity tests.
"""

import pytest

from neural_assemblies.programs import RuleParser, parse_sentence
from neural_assemblies.language.grammar_rules import LEXEME_DICT, RUSSIAN_LEXEME_DICT


ENGLISH_SENTENCES = [
    "cats chase mice",
    "dogs chase cats",
    "big cats chase mice",
    "cats love mice",
    "dogs bite cats",
]

# Center-embedded relative clause: outer "cats chase mice ... love mice"
# with inner "dogs chase cats" between "that" and "," markers.
CENTER_EMBEDDING_SENTENCES = [
    "cats chase mice that dogs chase cats , love mice",
    "dogs chase cats that cats bite mice , love dogs",
]

STRUCTURAL_TOKENS = frozenset({"that", ","})

RUSSIAN_SENTENCES = [
    "kot vidit sobaku",
    "sobaka vidit kota",
    "kot lyubit sobaku",
    "sobaka lyubit kota",
    "kot dayet kotu sobaku",
    "мальчик дал девочке мяч",
]


def _words_in_lexicon(sentence: str, lexicon: dict) -> bool:
    return all(
        w in lexicon or w in STRUCTURAL_TOKENS
        for w in sentence.split()
    )


def _content_words(sentence: str) -> list:
    return [w for w in sentence.split() if w not in STRUCTURAL_TOKENS]


@pytest.mark.parametrize("sentence", ENGLISH_SENTENCES)
def test_english_parser_sentences(sentence):
    assert _words_in_lexicon(sentence, LEXEME_DICT)
    result = RuleParser(
        language="English", p=0.1, lex_k=20, non_LEX_n=5000,
    ).parse(sentence)
    assert isinstance(result, list)
    assert len(result) >= 1
    for head, dep, role in result:
        assert isinstance(head, str)
        assert isinstance(dep, str)
        assert isinstance(role, str)


@pytest.mark.parametrize("sentence", RUSSIAN_SENTENCES)
def test_russian_parser_sentences(sentence):
    assert _words_in_lexicon(sentence, RUSSIAN_LEXEME_DICT)
    result = parse_sentence(
        sentence, language="Russian", p=0.1, lex_k=10, non_LEX_n=3000,
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    for _, _, role in result:
        assert isinstance(role, str)


def test_russian_ditransitive_roles():
    """TACL 2021 Russian: ditransitive activates NOM, ACC, and DAT roles."""
    result = parse_sentence(
        "kot dayet kotu sobaku", language="Russian", p=0.1, lex_k=10, non_LEX_n=3000,
    )
    roles = {role for _, _, role in result}
    words = {h for h, d, _ in result} | {d for h, d, _ in result}
    assert "NOM" in roles
    assert "ACC" in roles
    assert "DAT" in roles
    assert {"kot", "kotu", "sobaku", "dayet"} & words


def test_russian_free_word_order():
    """TACL 2021: case marking supports non-SVO order (roles present)."""
    svo = parse_sentence(
        "kot vidit sobaku", language="Russian", p=0.1, lex_k=10, non_LEX_n=3000,
    )
    osv = parse_sentence(
        "sobaku vidit kot", language="Russian", p=0.1, lex_k=10, non_LEX_n=3000,
    )
    assert len(svo) >= 1
    assert len(osv) >= 1
    assert {r for _, _, r in svo} & {"NOM", "ACC", "VERB"}
    assert {r for _, _, r in osv} & {"NOM", "ACC", "VERB"}


def test_russian_dative_initial_order_roles():
    """Free word order: DAT-initial ditransitive still activates case roles."""
    result = parse_sentence(
        "sobakie dayet kot sobaku", language="Russian", p=0.1, lex_k=10, non_LEX_n=3000,
    )
    roles = {role for _, _, role in result}
    assert roles & {"NOM", "ACC", "DAT"}


def test_english_cats_chase_mice_roles():
    result = RuleParser(
        language="English", p=0.1, lex_k=20, non_LEX_n=5000,
    ).parse("cats chase mice")
    roles = {r for _, _, r in result}
    assert "SUBJ" in roles or "OBJ" in roles or "VERB" in roles
    words = {h for h, d, _ in result} | {d for h, d, _ in result}
    assert "cats" in words
    assert "chase" in words
    assert "mice" in words
    assert "<NON-WORD>" not in words


@pytest.mark.parametrize("sentence", CENTER_EMBEDDING_SENTENCES)
def test_english_center_embedding_sentences(sentence):
    assert all(w in LEXEME_DICT for w in _content_words(sentence))
    result = RuleParser(
        language="English", p=0.1, lex_k=20, non_LEX_n=5000,
    ).parse(sentence)
    assert isinstance(result, list)
    assert len(result) >= 5
    roles = {role for _, _, role in result}
    assert "DEP-VERB" in roles
    words = {h for h, d, _ in result if h} | {d for h, d, _ in result if d}
    assert "love" in words or "chase" in words or "bite" in words
    dep_verbs = {d for _, d, r in result if r == "DEP-VERB"}
    assert dep_verbs & {"chase", "bite"}
    if "dogs" in _content_words(sentence):
        assert "dogs" in words
    for head, dep, role in result:
        assert isinstance(head, str)
        assert isinstance(dep, str)
        assert isinstance(role, str)
