"""role_bind_gain: one store, every writer -- and it must reach the fiber.

#52 recipe propagation. The gain is a PROPERTY whose setter writes the
engine's per-fiber beta store rather than a bracket at one call site,
because role_lexicons has FIVE writers and the curriculum path does NOT
run `train_roles` -- a bracket there would be a dormant selector on the
path every standing exam trains through (the near-miss that motivated
this design; see the property's docstring).

Power: the liveness test trains through `CurriculumTrainer` -- the SAME
path as sentence_conditioned_readout -- and asserts trained mass on the
core->role fiber DIFFERS between gain arms. On a build where the gain
fails to reach that path, it fails.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    GROUNDING_TO_CORE,
    NOUN_CORE,
    ROLE_AGENT,
    THEMATIC_AREAS,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)
from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (
    build_vocabulary_preset,
)


def _parser(seed=7, **kw):
    random.seed(seed)
    np.random.seed(seed)
    return EmergentParser(n=600, k=20, seed=seed, fast_training=True, **kw)


class TestBetaStore:

    def test_setter_writes_every_core_role_pair(self):
        p = _parser()
        eng = p.brain._engine
        bases = {(r, c): eng.get_beta(r, c)
                 for c in set(GROUNDING_TO_CORE.values())
                 for r in THEMATIC_AREAS}
        p.role_bind_gain = 4.0
        for (r, c), b in bases.items():
            assert eng.get_beta(r, c) == pytest.approx(4.0 * b)

    def test_setter_is_reversible(self):
        p = _parser()
        eng = p.brain._engine
        base = eng.get_beta(ROLE_AGENT, NOUN_CORE)
        p.role_bind_gain = 4.0
        p.role_bind_gain = 1.0
        assert eng.get_beta(ROLE_AGENT, NOUN_CORE) == pytest.approx(base)
        assert p.role_bind_gain == 1.0


def _fiber_mass(p, role_area):
    """Summed weight of the max-mass core->role block (the morph gain
    test's idiom: the trained fiber is whichever core actually drove)."""
    eng = p.brain._engine
    best = 0.0
    for core in set(GROUNDING_TO_CORE.values()):
        conn = getattr(eng, "_area_conns", {}).get(core, {}).get(role_area)
        w = getattr(conn, "weights", None) if conn is not None else None
        if w is not None and getattr(w, "ndim", 0) == 2 and w.size:
            best = max(best, float(np.asarray(w).sum()))
    return best


@pytest.mark.slow
def test_gain_reaches_the_curriculum_binding_path():
    """Same seed, gain 1 vs 4, trained via CurriculumTrainer (the exam's
    path, which does NOT call train_roles): the core->role fiber's trained
    mass must differ. Byte-identical mass = the gain is a dormant selector
    on the production path and the registered 2x2 would measure a no-op."""
    masses = {}
    for gain in (1.0, 4.0):
        p = _parser(vocabulary=build_vocabulary_preset("core"))
        p.role_bind_gain = gain
        t = CurriculumTrainer(p)
        for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
                      "SENTENCES"):
            t.train_stage(stage)
        masses[gain] = _fiber_mass(p, ROLE_AGENT)
    assert masses[4.0] > 0
    assert masses[4.0] != pytest.approx(masses[1.0]), (
        f"gain 4 left the ROLE_AGENT fiber byte-identical to gain 1 "
        f"({masses[1.0]}) -- the gain does not reach the curriculum "
        f"binding path")
