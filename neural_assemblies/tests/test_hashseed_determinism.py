"""Determinism across PROCESSES, not just within one.

Every determinism test in this suite before this file rebuilt a `Brain` several
times inside ONE interpreter and compared the results. That cannot detect the
largest reproducibility bug this repo has had: Python randomizes `hash()` of
str/bytes per process (PEP 456), so anything seeding an RNG from a string hash
-- or iterating a set of strings while allocating neurons -- is perfectly stable
within a run and different in the next one.

Measured before the fix, varying ONLY `PYTHONHASHSEED` on
`research/experiments/lesion_aphasia.py`: hash seeds 6/8/42 reported
irreversible accuracy 0.50 where 0/1/2/9/13 reported 1.00, and hash seed 9
crashed. `Brain(seed=42)` was trained to different weights in every process.

So these tests re-exec the interpreter with explicit, differing
`PYTHONHASHSEED` values and compare digests. Keep them subprocess-based: an
in-process version passes vacuously.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _run(snippet: str, hashseed: str) -> str:
    """Run `snippet` in a fresh interpreter under a given PYTHONHASHSEED."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
    )
    assert proc.returncode == 0, (
        f"PYTHONHASHSEED={hashseed} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout.strip().splitlines()
    assert out, f"no output under PYTHONHASHSEED={hashseed}: {proc.stderr[-2000:]}"
    return out[-1].strip()


# Hash seeds 6/8/42 are not arbitrary: those are the values that produced the
# WRONG answer before the fix, so they are the ones with power to catch it.
HASH_SEEDS = ("0", "6", "8", "42")


_CORE = """
import hashlib
import numpy as np
from neural_assemblies.core.brain import Brain
b = Brain(p=0.05, seed=42)
b.add_stimulus("S", 50)
for a in ("A", "B", "C"):
    b.add_area(a, n=1000, k=50, beta=0.05)
for _ in range(15):
    b.project({"S": ["A"]}, {"A": ["B"], "B": ["C"], "C": ["A"]})
m = hashlib.md5()
for n in ("A", "B", "C"):
    m.update(np.asarray(b.areas[n].winners).tobytes())
    m.update(str(b.areas[n].w).encode())
print(m.hexdigest())
"""


def test_core_brain_is_identical_across_hash_seeds():
    """A seeded Brain must train to the same weights in every process."""
    digests = {hs: _run(_CORE, hs) for hs in HASH_SEEDS}
    assert len(set(digests.values())) == 1, (
        "Brain(seed=42) is not reproducible across processes; digests per "
        f"PYTHONHASHSEED: {digests}. Something is seeding an RNG from hash() "
        "of a str, or iterating a set of str while allocating neurons. See "
        "numpy_engine._sparse.stable_seed."
    )


_PARSER = """
import hashlib
import numpy as np
from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
p = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
p.train(create_training_sentences())
e = p.brain._engine
m = hashlib.md5()
for n, a in sorted(e._areas.items()):
    m.update(f"{n}:{a.w}".encode())
    m.update(np.asarray(a.winners).tobytes())
for s, tm in sorted(e._stim_conns.items()):
    for t, c in sorted(tm.items()):
        m.update(f"{s}->{t}".encode())
        m.update(np.ascontiguousarray(c.weights).tobytes())
print(m.hexdigest())
"""


@pytest.mark.slow
def test_emergent_parser_is_identical_across_hash_seeds():
    """The trained parser -- weights included -- must not depend on hash order.

    This is the one that failed: divergence entered in `train_lexicon`, where
    `_expand_stim_vectors` iterated a set of stimulus NAMES while drawing from
    the shared seeded RNG, so each stimulus got a different slice of the stream
    depending on hash order. Same names, same shapes, different weights.
    """
    digests = {hs: _run(_PARSER, hs) for hs in HASH_SEEDS}
    assert len(set(digests.values())) == 1, (
        "EmergentParser training is not reproducible across processes; "
        f"digests per PYTHONHASHSEED: {digests}"
    )


_CURRICULUM = """
import hashlib
import os
os.environ["ASSEMBLIES_BACKBONE_CACHE"] = "0"
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"
import numpy as np
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    train_parser_to_depth,
)
p = train_parser_to_depth("VOCABULARY_SPURT", n=3000, k=30, seed=42)
b = p.brain
m = hashlib.md5()
for name in sorted(b.areas):
    eng = b._engine_for(b.areas[name])
    m.update(f"{name}:{eng.materialized_count(name)}".encode())
    m.update(np.asarray(eng.get_winners(name)).tobytes())
print(m.hexdigest())
"""


@pytest.mark.slow
def test_curriculum_training_is_identical_across_hash_seeds():
    """The CURRICULUM path, which the test above never reached.

    THE COVERAGE HOLE THIS CLOSES. `test_emergent_parser_is_identical_across_
    hash_seeds` exercises `p.train(create_training_sentences())` and passed
    throughout. But every parser in the suite and in `research/` is built by
    `train_parser_to_depth`, which runs the CURRICULUM -- a different and much
    longer path. That path had two live set-iteration sites and the guard could
    not see either, so it stayed green for months while training was
    irreproducible (#80).

    MEASURED before the fix, `train_parser_to_depth("SENTENCES", seed=42)` in
    three fresh processes: 16564 / 16572 / 16638 materialized neurons, with
    Cohen's d on the ERP contrast moving 1.452 -> 1.291. Pinning
    PYTHONHASHSEED=0 gave 16491 every time, which is what localised it.

    THE TWO SITES, both `list()`/iteration over a SET OF STRINGS feeding a call
    that PROJECTS -- i.e. that recruits neurons, so a different order is a
    different brain:

      * `acquisition/pos_inference.infer_holdout_categories` -- `for word in
        targets` where targets is a set comprehension, calling
        `classify_word_bootstrapped`;
      * `acquisition/pos_inference.sentences_from_transition_paths` -- same
        pattern, and its output order becomes the training corpus order;
      * `parser_mixins/prediction.train_next_token` -- `list(bridge_vocab)`
        where `bridge_vocab` is a set INTERSECTION.

    Uses VOCABULARY_SPURT rather than SENTENCES to stay affordable; that stage
    reproduced the divergence on its own (7035 vs 7022) and covers both sites.

    N IS 3000, THE REAL CONFIGURATION, and that is load-bearing. The first
    version shrank it to 1000 for speed, which silently cost the test its power.
    A guard that cannot fail is the exact defect this file exists to catch, and
    it was very nearly reintroduced inside the fix for it.

    VERIFIED DISCRIMINATING, not assumed: with both `sorted()` calls in
    `pos_inference` reverted this test FAILS; with them restored it passes.
    Worth noting what that experiment also showed -- reverting ONLY
    `infer_holdout_categories` was not enough to reproduce the divergence at
    this stage, so `sentences_from_transition_paths`, which fixes the order of
    the training CORPUS, is the dominant of the two. Both are fixed; only the
    pair has been shown necessary.
    """
    digests = {hs: _run(_CURRICULUM, hs) for hs in HASH_SEEDS}
    assert len(set(digests.values())) == 1, (
        "curriculum training is not reproducible across processes; "
        f"digests per PYTHONHASHSEED: {digests}"
    )
