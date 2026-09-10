"""A category query may read learned evidence but cannot train its instrument."""
import copy

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import CORE_AREAS, NOUN_CORE, VERB_CORE


@pytest.fixture
def parser():
    p = EmergentParser(n=1000, k=20, seed=13, rounds=3,
                       engine="numpy_sparse", fast_training=True)
    p.train_lexicon(skip_known=False)
    # Give the destructive-reset control a nonempty area fiber to detect.
    p.brain.project({}, {NOUN_CORE: [VERB_CORE]})
    return p


def fibers(brain):
    return {(kind, src, dst): np.asarray(conn.weights).copy()
            for kind, mapping in (("stim", brain._engine._stim_conns),
                                  ("area", brain._engine._area_conns))
            for src, targets in mapping.items() for dst, conn in targets.items()}


def assert_brain_unchanged(brain, before):
    current, previous = fibers(brain), fibers(before)
    assert current.keys() == previous.keys()
    for key in current:
        np.testing.assert_array_equal(current[key], previous[key], err_msg=str(key))
    for name, area in brain.areas.items():
        original = before.areas[name]
        np.testing.assert_array_equal(area.winners, original.winners)
        assert area.fixed_assembly == original.fixed_assembly
        assert brain._engine.is_fixed(name) == before._engine.is_fixed(name)
        assert brain._engine.get_num_ever_fired(name) == before._engine.get_num_ever_fired(name)
        assert brain._engine.get_neuron_id_mapping(name) == before._engine.get_neuron_id_mapping(name)
    assert brain._engine._rng.bit_generator.state == before._engine._rng.bit_generator.state


@pytest.mark.parametrize("cached", [False, True])
def test_category_query_preserves_the_brain_and_later_learning(parser, cached):
    parser._category_cache.clear()
    before = copy.deepcopy(parser.brain)
    classify = parser.classify_word_cached if cached else parser.classify_word
    category, _ = classify("dog")
    assert category == "NOUN"
    assert_brain_unchanged(parser.brain, before)
    cue = {parser.stim_map["cat"]: [NOUN_CORE]}
    parser.brain.project(cue, {})
    before.project(cue, {})
    assert_brain_unchanged(parser.brain, before)


def test_inhibited_populations_cannot_report_residual_recognition(parser):
    assert parser.classify_word("dog")[0] == "NOUN"
    for name in CORE_AREAS:
        parser.brain.inhibit_area(name)
    category, scores = parser.classify_word("dog")
    assert category == "UNKNOWN"
    assert not any(scores.values())


def test_classifier_does_not_inherit_the_legacy_recurrence_flag(parser):
    before = parser.classify_word("dog")
    parser.brain.recurrent_projection = True
    assert parser.classify_word("dog") == before


def test_exception_restores_activity_and_clamps_without_learning(parser, monkeypatch):
    import neural_assemblies.assembly_calculus.emergent.parser_mixins.classify as module
    parser.brain.areas[NOUN_CORE].fix_assembly()
    parser.brain._engine.fix_assembly(NOUN_CORE)
    before = copy.deepcopy(parser.brain)
    def fail(*args, **kwargs):
        raise RuntimeError("constructed readout failure")
    monkeypatch.setattr(module, "readout_all", fail)
    with pytest.raises(RuntimeError, match="constructed"):
        parser.classify_word("dog")
    assert_brain_unchanged(parser.brain, before)



def test_query_order_does_not_change_scores(parser):
    dog = parser.classify_word("dog")
    cat = parser.classify_word("cat")
    assert parser.classify_word("cat") == cat
    assert parser.classify_word("dog") == dog


def test_unregistered_grounding_does_not_create_stimuli(parser):
    from neural_assemblies.assembly_calculus.emergent.core.grounding import GroundingContext
    before = copy.deepcopy(parser.brain)
    category, scores = parser.classify_word("unseen", GroundingContext(visual=["unregistered"]))
    assert category == "UNKNOWN" and not any(scores.values())
    assert parser.brain.stimuli.keys() == before.stimuli.keys()
    assert_brain_unchanged(parser.brain, before)


def test_stale_lexicon_cannot_initialize_a_cold_population(parser):
    # Construct the inconsistent metadata that the cold-probe guard must stop.
    parser.brain._engine._areas[NOUN_CORE].w = 0
    before = copy.deepcopy(parser.brain)
    with pytest.raises(ValueError, match="materializ"):
        parser.classify_word("dog")
    assert_brain_unchanged(parser.brain, before)



@pytest.mark.parametrize('mode', ['combined', 'phon_only', 'grounding_only'])
def test_explicit_cue_modes_retain_actual_schedule_and_readonly_state(parser, monkeypatch, mode):
    ctx = parser.word_grounding['dog']
    expected = [] if mode == 'grounding_only' else [parser.stim_map['dog']]
    if mode != 'phon_only':
        expected += parser._grounding_stim_names(ctx)
    observed = []
    project = parser.brain.project_rounds
    def traced(target, stimuli, areas, rounds):
        observed.append(tuple(stimuli))
        return project(target, stimuli, areas, rounds)
    monkeypatch.setattr(parser.brain, 'project_rounds', traced)
    before = copy.deepcopy(parser.brain)
    evidence = parser.classify_word_evidence('dog', ctx, cue_mode=mode)
    assert evidence.cue_mode == mode
    assert evidence.cues == tuple(expected)
    assert observed and all(cues == tuple(expected) for cues in observed)
    assert_brain_unchanged(parser.brain, before)


def test_invalid_cue_mode_fails_before_neural_observation(parser, monkeypatch):
    def fail():
        pytest.fail('invalid query reached neural observation')
    monkeypatch.setattr(parser.brain, 'read_only', fail)
    with pytest.raises(ValueError, match='cue_mode'):
        parser.classify_word_evidence('dog', cue_mode='groundng')



def test_alternate_grounding_cache_path_matches_inference_without_neural_mutation(parser):
    from neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference import classify_word_bootstrapped
    context = parser.word_grounding['sees']
    expected = classify_word_bootstrapped(parser, 'dog', context)
    before = copy.deepcopy(parser.brain)
    cached = dict(parser._category_cache)
    actual = parser.classify_word_cached('dog', grounding=context)
    assert actual == expected and actual[1]
    assert parser._category_cache == cached
    assert_brain_unchanged(parser.brain, before)
