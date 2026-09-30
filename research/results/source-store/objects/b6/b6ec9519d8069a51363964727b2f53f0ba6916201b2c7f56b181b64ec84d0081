"""Regression tests for interactive grounding schema boundaries."""

from types import SimpleNamespace

from neural_assemblies.nemo.language.emergent.interactive.grounding import (
    GroundingInference,
)


def test_adjective_and_adverb_inference_populates_properties() -> None:
    learner = SimpleNamespace(
        word_count={"bright": 1, "quickly": 1},
        get_emergent_category=lambda word: (
            ("ADJECTIVE", {}) if word == "bright" else ("ADVERB", {})
        ),
    )
    inference = GroundingInference(learner)

    adjective = inference.infer_grounding("bright", ["a", "bright", "dog"], 1)
    adverb = inference.infer_grounding("quickly", ["runs", "quickly"], 1)

    assert adjective.properties == ["bright"]
    assert adverb.properties == ["quickly"]
    assert not hasattr(adjective, "property")
