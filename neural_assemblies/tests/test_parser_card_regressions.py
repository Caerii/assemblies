"""Regressions for card E (the emergent parser), SEMANTIC_CARDS.md.

Each test names the item it discharges. Sizes follow test_reconstruction_readout
(n=2000, k=30, the default 30-sentence corpus, numpy_sparse pinned): three
trainings, about ten seconds on CPU. Every bound below was observed by running
the test; the observed values are stated in the docstrings.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    CORE_TO_CATEGORY, GROUNDING_TO_CORE, MUTUAL_INHIBITION_GROUPS, THEMATIC_AREAS,
)
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.parser_mixins._shared import (
    _ROLE_BINDING_ROUNDS,
)
from neural_assemblies.assembly_calculus.ops import _snap, activate_assembly

N, K, SEED = 2000, 30, 42
SENTENCES = ("the dog chases the cat", "the cat was chases by the dog",
             "the boy finds the ball")
PARSE_KEYS = ("categories", "roles", "phrases", "tense", "mood", "polarity")


def _parser() -> EmergentParser:
    return EmergentParser(n=N, k=K, seed=SEED, engine="numpy_sparse", fast_training=True)


def _spy_brain(mp, brain, targets, fired):
    """Record every Brain.project target set and every inhibition that would fire."""
    project, apply = brain.project, brain._apply_mutual_inhibition

    def spy_project(areas_by_stim=None, dst_areas_by_src_area=None,
                    external_inputs=None, projections=None, **kwargs):
        seen = set()
        for routing in (areas_by_stim, dst_areas_by_src_area, projections):
            for dests in (routing or {}).values():
                seen.update(dests)
        targets.append(frozenset(seen))
        return project(areas_by_stim, dst_areas_by_src_area, external_inputs,
                       projections, **kwargs)

    def spy_apply(scores):
        for group in brain._mutual_inhibition_groups:
            if sum(name in scores for name in group) > 1:
                fired.append(sorted(scores))
        return apply(scores)

    mp.setattr(brain, "project", spy_project)
    mp.setattr(brain, "_apply_mutual_inhibition", spy_apply)


def _co_targeting(targets, groups):
    return [t for t in targets for g in groups if len(t & set(g)) > 1]


def _retraverse(parser, word, role):
    """roles.py `_traverse`, replicated outside its closure: same protocol, frozen."""
    brain = parser.brain
    core = parser._word_core_area(word)
    with brain.read_only():
        brain.clear_activity([role])
        brain.areas[role].unfix_assembly()
        activate_assembly(brain, parser.core_lexicons[core][word])
        brain.areas[core].fix_assembly()
        try:
            brain.project({}, {core: [role]})
            for _ in range(_ROLE_BINDING_ROUNDS - 1):
                brain.project({}, {core: [role], role: [role]})
        finally:
            brain.areas[core].unfix_assembly()
        return tuple(int(x) for x in _snap(brain, role).winners)


@pytest.fixture(scope="module")
def trained():
    """(parser, project target log, inhibition firings) -- spies live across train."""
    with pytest.MonkeyPatch.context() as mp:
        parser, targets, fired = _parser(), [], []
        _spy_brain(mp, parser.brain, targets, fired)
        parser.train()
        yield parser, targets, fired


@pytest.fixture(scope="module")
def beta_zero():
    """The card's control: beta 0 on every core->role fiber before train."""
    parser = _parser()
    parser.role_bind_gain = 0.0
    parser.train()
    return parser


@pytest.fixture(scope="module")
def switches_off():
    parser = _parser()
    parser.train(include_word_order=False, include_morphology=False)
    return parser


def test_e2_every_corpus_category_is_a_dictionary_entry(trained):
    """E2: compile_corpus writes all 36 corpus categories before any projection,
    and after train each equals CORE_TO_CATEGORY of the lexicon holding the word."""
    corpus = create_training_sentences()
    words = {w for s in corpus for w in s.words}
    fresh, targets = _parser(), []
    with pytest.MonkeyPatch.context() as mp:
        _spy_brain(mp, fresh.brain, targets, [])
        fresh.compile_corpus(corpus)
    assert not targets and words <= set(fresh._category_cache) and len(words) == 36
    parser = trained[0]
    for w in words:
        holders = [c for c, lex in parser.core_lexicons.items() if w in lex]
        assert holders and parser._category_cache[w] == CORE_TO_CATEGORY[holders[0]], w


def test_e6_parse_classifies_by_lookup_and_probes_only_an_unlexiconed_form(trained, monkeypatch):
    """E6: a trained-corpus sentence reaches classify_word_evidence 0 times and
    issues 0 probes. An unregistered word ('zorp') reaches it twice (once per
    classification pass; UNKNOWN is never cached) and still issues 0 probes: no
    stimulus, no cue. Only a form with a stimulus and no lexicon entry (public
    route: add_phon_stimulus) is probed, once per non-empty core lexicon (7),
    labelled by chance overlap with a core category, then cached (second parse: 0)."""
    parser = trained[0]
    evidence, probes = [], []
    orig_evidence, orig_rounds = parser.classify_word_evidence, parser.brain.project_rounds
    monkeypatch.setattr(parser, "classify_word_evidence",
                        lambda *a, **k: (evidence.append(a), orig_evidence(*a, **k))[1])
    monkeypatch.setattr(parser.brain, "project_rounds",
                        lambda *a, **k: (probes.append(a[0]), orig_rounds(*a, **k))[1])
    parser.parse(SENTENCES[0].split())
    assert (len(evidence), len(probes)) == (0, 0)
    result = parser.parse("the zorp chases the cat".split())
    assert result["categories"]["zorp"] == "UNKNOWN"
    assert (len(evidence), len(probes)) == (2, 0)
    evidence.clear()
    parser.add_phon_stimulus("wug")
    result = parser.parse("the wug chases the cat".split())
    nonempty = {c for c, lex in parser.core_lexicons.items() if lex}
    assert len(evidence) == 1 and set(probes) == nonempty and len(probes) == 7
    assert result["categories"]["wug"] in CORE_TO_CATEGORY.values()
    assert not any("wug" in lex for lex in parser.core_lexicons.values())
    probes.clear()
    parser.parse("the wug chases the cat".split())
    assert not probes and parser._category_cache["wug"] == result["categories"]["wug"]


def test_e3_e7_no_projection_co_targets_a_mutual_inhibition_group(trained):
    """E3/E7: both declared groups are registered, and of the 2871 projections
    train issues and the 14 each parse issues, none co-targets two members, so
    Brain._apply_mutual_inhibition never silences an area. The 'now wired'
    claim in core.py is corrected to say this."""
    parser, targets, fired = trained
    registered = parser.brain._mutual_inhibition_groups
    assert all(list(g) in registered for g in MUTUAL_INHIBITION_GROUPS)
    assert len(targets) > 1000 and not _co_targeting(targets, registered) and not fired
    for text in SENTENCES:
        targets.clear()
        parser.parse(text.split())
        assert len(targets) == 14 and not _co_targeting(targets, registered), text
    assert not fired


def test_e8_word_order_and_morphology_switches_do_not_reach_parse(trained, switches_off):
    """E8: the switches are live in train (SEQ w 1940 vs 0, MOOD 162 vs 0) and
    change nothing parse returns: all six keys, winners and gaps identical."""
    parser = trained[0]
    for area in ("SEQ", "MOOD"):
        assert parser.brain.areas[area].w > 0 and switches_off.brain.areas[area].w == 0
    for text in SENTENCES:
        on, off = parser.parse(text.split()), switches_off.parse(text.split())
        assert {k: on[k] for k in PARSE_KEYS} == {k: off[k] for k in PARSE_KEYS}, text
        assert on["role_diagnostics"]["winners"] == off["role_diagnostics"]["winners"]
        assert on["role_diagnostics"]["gaps"] == off["role_diagnostics"]["gaps"]


def test_e9_occupant_reproduces_its_own_winners_and_gap_is_the_runner_up(trained):
    """E9: top is exactly 1.0 on every gap (6 of 6), so gap == 1 - runner-up;
    an independent re-traversal by the recorded protocol reproduces the
    recorded winners exactly."""
    parser = trained[0]
    seen = 0
    for text in SENTENCES:
        _roles, diag = parser.parse_roles_by_reconstruction(text.split())
        for role, occupant, top, runner, gap in diag["gaps"]:
            assert top == 1.0 and gap == pytest.approx(1.0 - runner), (text, role)
            assert _retraverse(parser, occupant, role) == diag["winners"][role]
            seen += 1
    assert seen == 6


def test_e9_control_beta_zero_core_to_role_does_not_move_the_gap(trained, beta_zero):
    """E9 control, NOT moved: with beta 0 on all 49 core->role fibers the labels
    are identical and the mean gap is 0.978 against 0.983 trained, above the
    0.5 bar test_reconstruction_readout uses. Learning did change the images
    (winner overlap between arms 0.43-0.67 per role), not their separation:
    the gap measures fixed image separation, not learned role binding."""
    parser = trained[0]
    for core in set(GROUNDING_TO_CORE.values()):
        for role in THEMATIC_AREAS:
            assert beta_zero.brain.plasticity_rate(core, role) == 0.0
            assert parser.brain.plasticity_rate(core, role) > 0.0
    gaps, overlaps = [], []
    for text in SENTENCES:
        r1, d1 = parser.parse_roles_by_reconstruction(text.split())
        r0, d0 = beta_zero.parse_roles_by_reconstruction(text.split())
        assert r0 == r1, text
        gaps += [g[4] for g in d0["gaps"]]
        for role, winners in d1["winners"].items():
            common = set(winners) & set(d0["winners"][role])
            overlaps.append(len(common) / len(winners))
    assert len(gaps) == 6 and sum(gaps) / len(gaps) > 0.5, gaps
    assert max(overlaps) < 0.9, overlaps


def test_e10_unknown_word_reads_unknown_and_consumes_no_slot(trained):
    """E10, the reachable half: an unregistered word classifies UNKNOWN, takes
    no slot, and leaves no diagnostic; the next noun takes AGENT."""
    parser = trained[0]
    roles, diag = parser.parse_roles_by_reconstruction("the zorp chases the cat".split())
    assert roles == {"the": None, "zorp": None, "chases": "ACTION", "cat": "AGENT"}
    assert [g[0] for g in diag["gaps"]] == ["ROLE_AGENT"] and not diag["unavailable_areas"]


@pytest.mark.xfail(strict=True, reason="no public construction")
def test_e10_noun_without_snapshot_or_stimulus_consumes_a_slot(trained):
    """E10, the card's case: a NOUN with no core snapshot and no phon stimulus
    returns False from _traverse and still consumes a slot. Every public route
    that gives a word a category also registers its stimulus (constructor
    vocabulary, corpus registration, ingest_raw_sentence, add_phon_stimulus),
    so the state exists only by writing stim_map or _category_cache directly.
    The public attempt below (an unregistered subject noun) must displace the
    object to PATIENT; it does not, and strict xfail reports if that changes."""
    parser = trained[0]
    roles, _diag = parser.parse_roles_by_reconstruction("the zorp chases the cat".split())
    assert roles["cat"] == "PATIENT"
