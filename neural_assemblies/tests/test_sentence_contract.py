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
