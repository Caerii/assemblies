"""The paper's actual task: generate a sentence from a scene, in the word order
that was trained (Mitropolsky & Papadimitriou 2025, Sec 2.4-2.5).

NEMO does not do next-token prediction from a word prefix. It is given a SCENE
(assemblies in the three ROLE areas) plus a MOOD and generates the sentence with
the correct CONSTITUENT ORDER; success is defined on a scene "sampled randomly
and withheld during training". The decision is 3-way (which ROLE area fires
next) and lives in ``SYN[i] -> ROLE[i+1]`` synapses.

THE CONTROL THESE TESTS ENCODE: an untrained parser already emits SVO by
default, so "generates SVO" proves nothing. What must hold is that generation
FOLLOWS THE TRAINED ORDER -- so we train on permuted corpora and require each to
reproduce its own order on a withheld scene.
"""
import pytest

from research.experiments.word_order_generation import ORDERS, run_order

pytestmark = pytest.mark.slow  # full parser training per order (~1 min each)


@pytest.mark.parametrize("name", sorted(ORDERS))
def test_generation_follows_trained_word_order(name):
    produced = run_order(name, ORDERS[name], verbose=False)
    assert produced.startswith(name), (
        f"trained {name} but generated '{produced}' -- generation is not "
        "following the trained constituent order (a default ordering, or the "
        "SYN[i] -> ROLE[i+1] pathway, has regressed)")


def test_untrained_parser_is_not_evidence():
    """Pin the reason the permutation control is necessary: with no order
    training the parser still emits a default order, so a bare 'SVO' result
    must never be read as evidence of learning."""
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT,
    )
    parser = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
    parser.train(create_training_sentences())
    parser.prepare_scene({
        ROLE_AGENT: "boy", ROLE_ACTION: "finds", ROLE_PATIENT: "ball",
    })
    out = parser.generate_from_roles(max_len=6)
    assert out, "untrained generation returned nothing"
