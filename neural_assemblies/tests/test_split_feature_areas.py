"""Per-value feature areas (E15, #144): the split must exist, route, and
leave the default byte-path alone.

WHY THIS ARCHITECTURE EXISTS. E14 (#143) measured the one-area feature
design's scaling blocker: TWO label values sharing ONE k-WTA area merge as
total label projections grow (shared SG∩PL image columns 2.6 -> 17.8 of 30
at 200 frames, through fully VARIED sentences -- count, not replay). The
split puts each value in its own area inside a mutual-inhibition group --
the paper's ROLE-triple device (Mitropolsky & Papadimitriou 2025: "firing
only in the area that receives the greatest total synaptic input") -- so
the label images cannot share a neuron.

WHAT IS PINNED HERE, deliberately mechanical: area creation + MI group
registration (idempotent), training routes each detected value to its own
area, recall answers through the MI competition, the flag can be flipped
POST-CONSTRUCTION (the checkpoint-reuse path every experiment depends on),
and a default parser creates none of it. Accuracy claims live in the
pre-registered experiment, not here.
"""
from __future__ import annotations

import pytest

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    NUMBER, TENSE, FEATURE_VALUE_LABELS, feature_value_area,
)

SENTS = [
    ["the", "dog", "chases", "the", "cat"],
    ["the", "dogs", "chase", "the", "cat"],
    ["the", "cat", "sees", "the", "dog"],
    ["the", "cats", "see", "the", "dog"],
]


def _register_plurals(p):
    # Inflected surfaces enter stim_map sharing the lemma's grounding --
    # the register_surface_forms contract, done by hand for this tiny corpus.
    for form, lemma in (("dogs", "dog"), ("cats", "cat")):
        if form not in p.stim_map:
            p.register_word(form)
            ctx = p.word_grounding.get(lemma)
            if ctx is not None:
                p.word_grounding[form] = ctx


@pytest.fixture(scope="module")
def split_parser():
    p = EmergentParser(n=600, k=20, seed=42, fast_training=True,
                       split_feature_areas=True)
    _register_plurals(p)
    p.train_number(SENTS)
    p.train_tense(SENTS)
    return p


def test_value_areas_created_with_mi_group(split_parser):
    brain = split_parser.brain
    for feature in (NUMBER, TENSE):
        names = [feature_value_area(feature, lab)
                 for lab in FEATURE_VALUE_LABELS[feature]]
        for name in names:
            assert name in brain.areas, f"{name} missing"
        assert any(set(g) == set(names)
                   for g in brain._mutual_inhibition_groups), (
            f"MI group for {feature} not registered")


def test_training_routed_to_value_areas(split_parser):
    """The detected values' areas were actually driven; the shared area
    never fired -- the split ROUTES, it does not duplicate.

    (Stimulus connectomes are pre-wired to every area at add_stimulus --
    the stim->area init consistency work -- so fiber PRESENCE proves
    nothing; ever-fired is the honest routing probe.)"""
    for area in (feature_value_area(NUMBER, "SG"),
                 feature_value_area(NUMBER, "PL")):
        assert split_parser.brain.areas[area].get_num_ever_fired() > 0, (
            f"{area} never fired despite trained SG and PL tokens")
    assert split_parser.brain.areas[NUMBER].get_num_ever_fired() == 0, (
        "the SHARED area fired under the split -- routing leaked")


def test_recall_answers_through_mi(split_parser):
    got_pl, diag_pl = split_parser.recall_number("dogs")
    got_sg, diag_sg = split_parser.recall_number("dog")
    assert diag_pl["readout"] == "mi_split"
    # Both drives measured, a decision made, at most one MI survivor.
    assert set(diag_pl["scores"]) == {"SG", "PL"}
    assert got_pl in ("SG", "PL", None)
    assert len(diag_pl.get("mi_survivors", [])) <= 1
    # Tiny-brain accuracy is not pinned (that is the experiment's job),
    # but the readout must not be degenerate: the two probes must not
    # both refuse.
    assert not (got_pl is None and got_sg is None)


def test_flag_flips_post_construction():
    """The checkpoint-reuse path: default-built parser, flag flipped after
    construction, training creates the areas lazily."""
    p = EmergentParser(n=600, k=20, seed=43, fast_training=True)
    assert feature_value_area(NUMBER, "SG") not in p.brain.areas
    p.split_feature_areas = True
    _register_plurals(p)
    p.train_number(SENTS)
    assert feature_value_area(NUMBER, "SG") in p.brain.areas
    assert feature_value_area(NUMBER, "PL") in p.brain.areas


def test_default_parser_untouched():
    """No flag: no value areas, no extra MI groups, shared-area readout."""
    p = EmergentParser(n=600, k=20, seed=44, fast_training=True)
    p.train_number(SENTS)
    for lab in FEATURE_VALUE_LABELS[NUMBER]:
        assert feature_value_area(NUMBER, lab) not in p.brain.areas
    _got, diag = p.recall_number("dogs")
    assert diag.get("readout") != "mi_split"
