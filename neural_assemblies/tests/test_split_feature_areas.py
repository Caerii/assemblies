"""Per-value feature areas (E15, #144): the split must exist, route, and
keep the legacy shared-area path reachable.

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
and the LEGACY shared-area path stays reachable via
split_feature_areas=False. Accuracy claims live in the pre-registered
experiment, not here.

DEFAULT: True since the #149 adoption -- decided by an n=10 paired gate
at the default corpus (tense delta -0.039 +/- 0.058 NS, SG 0.920, PL
0.415 vs 0.085 shared), not by E15's scale numbers alone. The
default-construction fixture below is itself the pin on that.
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
    # Default construction on purpose: the #149 adoption means a plain
    # parser IS the split parser, and this fixture pins that.
    p = EmergentParser(n=600, k=20, seed=42, fast_training=True)
    assert p.split_feature_areas is True, "adopted default regressed"
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
    """The checkpoint-reuse path: legacy-built parser, flag flipped after
    construction, training creates the areas lazily."""
    p = EmergentParser(n=600, k=20, seed=43, fast_training=True,
                       split_feature_areas=False)
    assert feature_value_area(NUMBER, "SG") not in p.brain.areas
    p.split_feature_areas = True
    _register_plurals(p)
    p.train_number(SENTS)
    assert feature_value_area(NUMBER, "SG") in p.brain.areas
    assert feature_value_area(NUMBER, "PL") in p.brain.areas


def test_legacy_shared_path_reachable():
    """split_feature_areas=False: no value areas, no extra MI groups,
    shared-area readout -- byte-identical pre-E15 behavior, kept for
    literature-parity reproductions (the norm_init substrate-vs-reference
    pattern)."""
    p = EmergentParser(n=600, k=20, seed=44, fast_training=True,
                       split_feature_areas=False)
    p.train_number(SENTS)
    for lab in FEATURE_VALUE_LABELS[NUMBER]:
        assert feature_value_area(NUMBER, lab) not in p.brain.areas
    _got, diag = p.recall_number("dogs")
    assert diag.get("readout") != "mi_split"


def test_competition_dynamics_modes(split_parser):
    """E16 (#145): latched and settled modes answer through the same API.

    Accuracy claims live in the experiment; pinned here is that each mode
    runs, reports itself in diag, and the latched mode cannot CHANGE the
    one-shot decision (MI silences the loser at step 1, so its recurrence
    is gone -- the registered holds-not-changes prediction, at unit scale).
    """
    base_mode = getattr(split_parser, "mi_readout_mode", "oneshot")
    try:
        answers = {}
        for mode, t in (("oneshot", 1), ("latched", 5), ("settled", 5)):
            split_parser.mi_readout_mode = mode
            split_parser.mi_latch_rounds = t
            got, diag = split_parser.recall_number("dogs")
            assert diag["mode"] == mode and diag["latch_rounds"] == t
            assert diag["readout"] == "mi_split"
            answers[mode] = got
        assert answers["latched"] == answers["oneshot"], (
            "the latch changed the decision -- it must only hold it")
    finally:
        split_parser.mi_readout_mode = base_mode
        split_parser.mi_latch_rounds = 1


def test_morph_readout_switch(split_parser):
    """#149: the returned answer follows `morph_readout`; both answers ride
    in diag either way. The readouts CROSS by budget (E15: MI wins at 50
    frames, overlap at >=200), so this is a measured parameter with a
    per-regime best -- not two interchangeable ways to do one thing."""
    base = getattr(split_parser, "morph_readout", "mi")
    try:
        split_parser.morph_readout = "mi"
        got_mi, diag_mi = split_parser.recall_number("dogs")
        assert "mi_answer" in diag_mi and "overlap_answer" in diag_mi
        assert got_mi == diag_mi["mi_answer"]

        split_parser.morph_readout = "overlap"
        got_ov, diag_ov = split_parser.recall_number("dogs")
        expected = (diag_ov["overlap_answer"]
                    if diag_ov["overlap_answer"] is not None
                    else diag_ov["mi_answer"])  # tie falls back to MI
        assert got_ov == expected
    finally:
        split_parser.morph_readout = base


def test_morph_flush_every_rate():
    """E19 (#148): morph_flush_every=K triggers interim deferred flushes.

    Pinned mechanically: K=2 on a 4-episode corpus fires interim flushes
    (counted by wrapping the engine's flush), and K=0 fires NONE beyond
    the phase-end one. The DEFAULT is 40 since the #149 adoption (E19b:
    right wall unconditional, K=40 the measured-best cell) -- inert
    without deferred scaling, which is why the default byte-path stays
    identical. The schedule's accuracy claims live in the pre-registered
    experiment.
    """
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        FEATURE_VALUE_LABELS,
    )

    assert EmergentParser(n=600, k=20, seed=46,
                          fast_training=True).morph_flush_every == 40, (
        "adopted flush-schedule default regressed")

    def build(flush_every):
        p = EmergentParser(n=600, k=20, seed=45, fast_training=True,
                           split_feature_areas=True)
        _register_plurals(p)
        eng = p.brain._engine
        vals = frozenset(feature_value_area(NUMBER, lab)
                         for lab in FEATURE_VALUE_LABELS[NUMBER])
        eng.synaptic_scaling = vals
        p.brain._synaptic_scaling = vals
        eng.synaptic_scaling_deferred = True
        p.morph_flush_every = flush_every
        calls = []
        orig = eng.flush_synaptic_scaling
        eng.flush_synaptic_scaling = lambda: calls.append(1) or orig()
        p.train_number(SENTS)
        return len(calls)

    assert build(0) == 1, "K=0 must flush exactly once, at phase end"
    assert build(2) >= 2, "K=2 over 4+ episodes must fire interim flushes"
