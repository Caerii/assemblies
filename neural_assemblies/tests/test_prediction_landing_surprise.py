"""prediction_landing_surprise: the landing-channel N400 must be live.

#28. The energy N400 is recorded as insensitive to expectancy. The landing
adapter shares the SAME settling (`_settle_context_into_prediction` -- one
implementation) and reads the competition's outcome. Power both directions:

  * NEGATIVE -- undefined escapes fire on absent preconditions (no prefix;
    word without a prediction entry) instead of returning 1.0, the value
    that reads as maximum surprise.
  * POSITIVE (liveness, not science) -- on a curriculum-trained parser the
    adapter returns DEFINED values that are NOT one constant across words:
    a dead readout returns the same number for every probe, which is the
    saturation signature this unit exists to measure. The test pins
    liveness only; WHICH word wins is the registered experiment's question,
    not a test's.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    prediction_landing_surprise,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)


def _parser(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    return EmergentParser(n=600, k=20, seed=seed, fast_training=True)


def test_no_prefix_is_undefined():
    p = _parser()
    m = prediction_landing_surprise(p, (), "cat")
    assert not m.defined
    assert "no prefix" in m.why


@pytest.mark.slow
def test_landing_surprise_is_live_not_constant():
    p = _parser()
    t = CurriculumTrainer(p)
    for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES"):
        t.train_stage(stage)

    prefix = ("the", "dog")
    words = [w for w in ("ball", "cat", "cookie", "milk", "book")
             if w in p.stim_map][:4]
    assert len(words) >= 3, "core vocabulary changed under the test"
    vals = {}
    for w in words:
        m = prediction_landing_surprise(p, prefix, w)
        assert m.defined, f"{w!r}: {m.why}"
        vals[w] = float(m)
    assert len(set(round(v, 6) for v in vals.values())) > 1, (
        f"landing surprise is ONE constant across words ({vals}) -- the "
        f"dead-readout signature this adapter exists to escape")
