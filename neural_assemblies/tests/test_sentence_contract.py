import pytest

from neural_assemblies.assembly_calculus.emergent.core.grounding import GroundingContext
from neural_assemblies.assembly_calculus.emergent.core.sentence import GroundedSentence


def test_grounded_sentence_rejects_misaligned_contexts():
    with pytest.raises(ValueError, match="contexts"):
        GroundedSentence(words=["dog"], contexts=[])


def test_grounded_sentence_rejects_misaligned_roles():
    with pytest.raises(ValueError, match="roles"):
        GroundedSentence(
            words=["dog"], contexts=[GroundingContext()], roles=[]
        )


def test_grounded_sentence_normalizes_unannotated_roles():
    sentence = GroundedSentence(words=["dog"], contexts=[GroundingContext()])
    assert sentence.roles == [None]


def test_grounded_utterance_rejects_misaligned_pos_tags():
    from neural_assemblies.lexicon.curriculum.grounded_training import (
        GroundedUtterance, SpeechAct,
    )
    with pytest.raises(ValueError, match="pos_tags"):
        GroundedUtterance(
            words=["dog"], pos_tags=[], speech_act=SpeechAct.NAMING,
            context=GroundingContext(),
        )
