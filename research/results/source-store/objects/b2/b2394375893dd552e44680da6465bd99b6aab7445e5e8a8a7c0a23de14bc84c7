"""Shared measurement battery for the curriculum experiments.

The ablation and developmental studies intentionally differ in training
schedule, but their seven-phenomenon readout is the same protocol. Keeping
one implementation prevents a silent change in one study from changing the
meaning of the other.
"""

from typing import Any, Dict
import numpy as np

from research.experiments.lib.brain_setup import activate_word
from research.experiments.lib.measurement import measure_n400, measure_p600
from research.experiments.base import measure_overlap

def measure_battery(
    brain,
    lexicon,
    vocab,
    cfg,
) -> Dict[str, Any]:
    """Measure all 7 phenomena. Returns raw paired values for effect size."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    locs = vocab.words_for_category("LOCATION")
    ni = cfg.n_test_items

    results = {}

    # 1. Assembly stability: activate word twice, measure self-overlap
    stability_scores = []
    for noun in nouns[:ni]:
        activate_word(brain, noun, "NOUN_CORE", 3)
        a1 = np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
        activate_word(brain, noun, "NOUN_CORE", 3)
        a2 = np.array(brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
        stability_scores.append(measure_overlap(a1, a2))
    results["assembly_stability"] = float(np.mean(stability_scores))

    # 2. Forward prediction N400 at object position
    n400_gram, n400_cv = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]

        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)

        n400_gram.append(measure_n400(predicted, lexicon[gram_obj]))
        n400_cv.append(measure_n400(predicted, lexicon[cv_obj]))

    results["prediction_n400_gram"] = n400_gram
    results["prediction_n400_cv"] = n400_cv

    # 3. Binding P600 double dissociation at object position
    p600_gram, p600_cv = [], []
    for i in range(ni):
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]
        p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    results["binding_p600_gram"] = p600_gram
    results["binding_p600_cv"] = p600_cv

    # 4. PP binding P600
    pp_p600_gram, pp_p600_cv = [], []
    for i in range(ni):
        gram_pp = locs[i % len(locs)]
        cv_pp = verbs[(i + 2) % len(verbs)]
        pp_p600_gram.append(measure_p600(
            brain, gram_pp, "NOUN_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))
        pp_p600_cv.append(measure_p600(
            brain, cv_pp, "VERB_CORE", "ROLE_PP_OBJ", cfg.n_settling_rounds))

    results["pp_p600_gram"] = pp_p600_gram
    results["pp_p600_cv"] = pp_p600_cv

    # 5. SRC dual binding P600
    src_dual = []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        src_dual.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))
    results["src_dual_binding"] = src_dual

    # 6. Garden-path N400 (GP vs unambiguous at 2nd verb)
    gp_n400, unamb_n400 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # Unambiguous (with "that")
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_u = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        unamb_n400.append(measure_n400(pred_u, lexicon[main_verb]))

        # Garden-path (no "that")
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_g = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        gp_n400.append(measure_n400(pred_g, lexicon[main_verb]))

    results["garden_path_gp"] = gp_n400
    results["garden_path_unamb"] = unamb_n400

    # 7. SRC/ORC asymmetry (dual-binding P600)
    src_p600, orc_p600 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        src_p600.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))
        orc_p600.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_PATIENT", cfg.n_settling_rounds))

    results["src_orc_src"] = src_p600
    results["src_orc_orc"] = orc_p600

    return results


