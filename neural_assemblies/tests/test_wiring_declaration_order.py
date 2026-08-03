"""Dense wiring must depend on (seed, source, target) -- not on declaration order.

`Connectome._initialize_weights` draws
``self._gen.binomial(1, p, size=(source_size, target_size))``: ONE sequential
draw per connectome, from one shared generator. `ExplicitEngine.add_area`
constructs stimulus fibers before recurrent ones, so declaring one extra
stimulus inserts a draw and rewires every connectome built after it.

This is the same defect class as the lazy-connectome seeds, and it was already
fixed once -- for the SPARSE path, which materializes with a content-addressed
``stable_seed(row, col)``. The dense path never got that fix, and the dense
path is the entire explicit engine.

Why it matters beyond reproducibility: an ablation that removes a stimulus or
an area does not only remove that component, it rewires the substrate
downstream of it, so the ablated arm differs from its control by more than the
thing under test.

The last test here is the invariant we WANT and it is xfail -- keying dense
init on content rewires every explicit-engine golden in the repo and needs its
own re-recording pass. The first two pin the behaviour that is correct today,
so a future content-addressed init cannot regress them.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain

P = 0.05
N = 500
K = 50
BETA = 0.05
SEED = 42


def _wiring(add_unused_stimulus: bool, seed: int = SEED):
    """Build an explicit area; return (s1->A, A->A) weight matrices."""
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse",
              norm_init=False)
    b.add_stimulus("s1", K)
    if add_unused_stimulus:
        b.add_stimulus("s2", K)      # declared, never projected
    b.add_area("A", N, K, BETA, explicit=True)
    eng = b._engine_for(b.areas["A"])
    return (np.asarray(eng._stim_conns["s1"]["A"].weights).copy(),
            np.asarray(eng._area_conns["A"]["A"].weights).copy())


def test_identical_declarations_are_reproducible():
    """The seed does what it claims for a FIXED declaration order.

    This is the part that already works, and it is what the previous round of
    this fix bought (passing the Brain's generator instead of reading the
    global stream). Keep it pinned: it is the floor any future change to the
    init path has to stay above.
    """
    _, a = _wiring(True)
    _, b = _wiring(True)
    assert np.array_equal(a, b), (
        "two identical builds disagree -- the seed is not controlling wiring "
        "at all, which is a bigger problem than declaration order")


def test_fibers_drawn_before_the_insertion_point_are_unaffected():
    """s1->A is constructed BEFORE the extra stimulus's fiber, so it survives.

    Asserted to localize the defect: the stream is not globally scrambled, it
    is shifted from the insertion point onward. That is what makes the failure
    below so easy to miss -- the fiber you are looking at is often fine.
    """
    s1_without, _ = _wiring(False)
    s1_with, _ = _wiring(True)
    assert np.array_equal(s1_without, s1_with)


def test_unused_stimulus_does_not_rewire_the_recurrent_fiber():
    """Declaring a stimulus you never fire must not change the connectome.

    WAS XFAIL, NOW PASSES -- door 5 of [[content-addressed-synapse-init]] is
    closed (d253b33). The old reason named the fix exactly: "keying dense init
    on (source, target) content the way the sparse path already does". That is
    what `Connectome(pair_seed=...)` plus `NumpyExplicitEngine._fiber_seed` do,
    so the predicted golden re-recording pass was not needed after all -- the
    fiber a golden was recorded on now gets the SAME wiring it had, because
    wiring is a function of which fiber it is rather than of draw order.

    Measured when this was a defect: 23752 of 250000 A->A synapses differed,
    and the downstream effect was not subtle -- explicit `project` persistence
    read 0.9048 without the unused stimulus and 0.7460 with it, at identical
    parameters.
    """
    _, aa_without = _wiring(False)
    _, aa_with = _wiring(True)
    differing = int((aa_without != aa_with).sum())
    assert differing == 0, (
        f"{differing} of {aa_without.size} A->A synapses were rewired by a "
        f"stimulus that is never projected")
