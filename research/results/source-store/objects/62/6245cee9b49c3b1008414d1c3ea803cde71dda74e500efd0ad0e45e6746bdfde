"""role_binding_deficit: the WHERE-channel P600 readout must see binding.

#121. The drive channel provably cannot see word-level binding history
(pathway-only AUC 0.515; test_drive_and_binding_are_orthogonal.py). This
adapter reads the binding channel via `ops.read_binding`. Power is verified
in BOTH directions (one-canonical-way rule 6):

  * POSITIVE -- a word bound into the role area via `ops.bind` reads a
    SMALLER deficit than a matched word never bound there.
  * NEGATIVE -- the undefined escapes fire when the preconditions are
    genuinely absent (no stored core; empty role lexicon), instead of
    returning 0.0 or 1.0, the two values that read as findings.
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
    NOUN_CORE,
    ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (
    role_binding_deficit,
)
from neural_assemblies.assembly_calculus.ops import bind


def _parser(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    return EmergentParser(n=600, k=20, seed=seed, fast_training=True)


class TestUndefinedEscapes:

    def test_missing_core_assembly_is_undefined(self):
        p = _parser()
        m = role_binding_deficit(p, NOUN_CORE, ROLE_PATIENT, "never_seen")
        assert not m.defined
        assert "no stored assembly" in m.why

    def test_empty_role_lexicon_is_undefined(self):
        p = _parser()
        p.train_lexicon(words={"cat"})
        assert p.core_lexicons.get(NOUN_CORE, {}).get("cat") is not None
        m = role_binding_deficit(p, NOUN_CORE, ROLE_PATIENT, "cat")
        assert not m.defined
        assert "no stored bindings" in m.why


@pytest.mark.slow
def test_bound_word_reads_smaller_deficit_than_unbound():
    """Two nouns, same core area, same training depth; ONE is bound into
    ROLE_PATIENT through the production protocol (`ops.bind`, the one
    implementation). The bound word must read a smaller deficit. Equality
    would mean the readout cannot see binding -- the exact property the
    drive channel fails."""
    p = _parser()
    p.train_lexicon(words={"cat", "ball", "dog"})
    lex = p.core_lexicons[NOUN_CORE]
    # Bind cat and dog into ROLE_PATIENT; ball never.
    for w in ("cat", "dog"):
        for _ in range(3):
            stored = bind(p.brain, NOUN_CORE, ROLE_PATIENT, lex[w])
        p.role_lexicons.setdefault(ROLE_PATIENT, {})[w] = stored

    bound = role_binding_deficit(p, NOUN_CORE, ROLE_PATIENT, "cat")
    unbound = role_binding_deficit(p, NOUN_CORE, ROLE_PATIENT, "ball")
    assert bound.defined and unbound.defined
    assert bound.detail["landed_on"] == "cat"
    assert float(bound) < float(unbound), (
        f"bound deficit {float(bound):.4f} not below unbound "
        f"{float(unbound):.4f} -- the WHERE channel is not seeing binding")
