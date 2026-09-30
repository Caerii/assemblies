"""CHAT reader (#150): adult utterances out, honesty counters visible.

The fixture is a hand-built transcript exercising every cleaning rule the
reader claims: continuation lines, retracing scopes, events, pauses,
parenthesized completions, unintelligible tokens, clitics in %mor, a
DELIBERATELY misaligned %mor (the true-negative the alignment rule needs
-- a guard whose failing case has never been built has unmeasured power),
and a child utterance that must not leak into child-directed input.
"""
from __future__ import annotations

from neural_assemblies.assembly_calculus.emergent.curriculum.childes import (
    frequency_spectrum,
    mor_number_teacher,
    read_cha,
)

FIXTURE = """@UTF8
@Begin
@Participants:\tCHI Adam Target_Child , MOT Mother
*MOT:\tthe dogs are running .
%mor:\tdet:art|the n|dog-PL cop|be&PRES part|run-PRESP .
*CHI:\tdoggie !
%mor:\tn|doggie .
*MOT:\t<the the> [/] the dog (.) chases xxx .
%mor:\tdet:art|the n|dog v|chase-3S .
*MOT:\tyou're a big boy now &=laughs
\tand (be)cause I said so .
%mor:\tpro:per|you~cop|be&PRES det:art|a adj|big n|boy adv|now
\tconj|and conj|because pro:sub|I v|say&PAST adv|so .
*MOT:\tsee the cats ?
%mor:\tco|see det:art|the .
@End
"""


def test_adult_filter_and_cleaning():
    utts, stats = read_cha(FIXTURE)
    assert all(u.speaker == "MOT" for u in utts)
    assert stats.by_speaker["CHI"] == 1
    texts = [" ".join(u.words) for u in utts]
    # Retracing scope removed, pause removed, xxx token (not utterance)
    # dropped, completion expanded.
    assert "the dog chases" in texts
    assert stats.unintelligible_tokens == 1


def test_mor_alignment_carries_or_refuses():
    utts, stats = read_cha(FIXTURE)
    by_text = {" ".join(u.words): u for u in utts}
    # Aligned: 4 words, 4 mor tokens.
    dogs = by_text["the dogs are running"]
    assert dogs.mor is not None and len(dogs.mor) == 4
    # you're -> one main token, clitic ~ in one mor token: still aligned.
    boy = by_text["you're a big boy now and because i said so"]
    assert boy.mor is not None and boy.mor[0].startswith("pro:per|you~")
    # The deliberately short %mor (3 tokens vs "see the cats" = 3 words...
    # main has 3 words, mor has 2 after terminator drop) must REFUSE.
    cats = by_text["see the cats"]
    assert cats.mor is None
    assert stats.mor_misaligned == 1


def test_number_teacher_reads_only_aligned_nouns():
    utts, _ = read_cha(FIXTURE)
    by_text = {" ".join(u.words): u for u in utts}
    t = mor_number_teacher(by_text["the dogs are running"])
    assert t == {"dogs": "PL"}
    t2 = mor_number_teacher(by_text["the dog chases"])
    assert t2 == {"dog": "SG"}
    # No aligned mor -> no teacher, never a guess from the surface form.
    assert mor_number_teacher(by_text["see the cats"]) == {}


def test_frequency_spectrum_counts_tokens():
    utts, _ = read_cha(FIXTURE)
    freq = frequency_spectrum(utts)
    assert freq["the"] >= 3
    assert freq["dogs"] == 1 and freq["dog"] == 1


def test_childesdb_jsonl_route():
    """The auth-wall deviation route (#150): childes-db rows -> the same
    Utterance stream, with null-mor counted where misalignment was."""
    from neural_assemblies.assembly_calculus.emergent.curriculum.childes \
        import read_childesdb_jsonl

    jsonl = "\n".join([
        '{"speaker": "Mother", "words": ["the", "dogs", "run"],'
        ' "mor": ["det:art|the", "n|dog-PL", "v|run"]}',
        '{"speaker": "Mother", "words": ["hm"], "mor": null}',
    ])
    utts, stats = read_childesdb_jsonl(jsonl)
    assert len(utts) == 2
    assert mor_number_teacher(utts[0]) == {"dogs": "PL"}
    assert utts[1].mor is None and stats.mor_misaligned == 1
