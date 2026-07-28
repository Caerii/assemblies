"""Positive control for research/experiments/lucky_seed_audit.py.

An audit that cannot fail is worthless, and this one reported "0 of 29 tests
are seed-dependent" on the conformance suite the first time it ran. That is
either good news or a broken harness, and nothing in the output distinguishes
them -- the same failure mode the audit exists to catch, one level up.

So the audit is run against these two first. Measured over the same 12 seeds:

    10/12  test_definitely_seed_dependent
    12/12  test_definitely_stable

which is the expected shape (the first is ~50/50 by construction and landed
10/12 on this particular seed list) and confirms that ASSEMBLIES_AUDIT_SEED
really does reach the module and change behaviour.

These pass in ordinary test runs: the default seed 42 gives 0.774 > 0.5. They
are fixtures for the audit, not claims about the library, and they are named
zz_ so they sort last and are obviously not conformance tests.
"""

import os

import numpy as np

SEED = int(os.environ.get("ASSEMBLIES_AUDIT_SEED", "42"))


def test_definitely_seed_dependent():
    """Passes for roughly half of all seeds, by construction."""
    assert np.random.default_rng(SEED).random() > 0.5


def test_definitely_stable():
    """Passes for every seed, by construction."""
    assert np.random.default_rng(SEED).random() >= 0.0
