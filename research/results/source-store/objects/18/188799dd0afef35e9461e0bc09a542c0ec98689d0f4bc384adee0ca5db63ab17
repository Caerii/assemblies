"""`AttentionArea` against mdabagia/nemo's own implementation, bit for bit.

This is the last unported construction in that reference
(research/literature/CONFORMANCE.md), and the only one carrying REVERSIBLE
plasticity. Two of its properties differ from everything else here, so the port
is checked against the reference directly rather than described:

  a SET rule, `w = (1 + p) * (w > 0)`, which saturates instead of compounding
  a ONE-STEP undo, `w -= change`, which restores the baseline exactly

The reference is imported from `.reference/mdabagia-nemo` when present and the
tests skip otherwise, following `test_cross_repo_parity.py`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REF = _REPO_ROOT / ".reference" / "mdabagia-nemo"

N, CAP, DENSITY, PLASTICITY = 64, 8, 0.4, 0.25


def _reference_module():
    """Load the reference `brain` module without polluting `sys.path`."""
    path = _REF / "brain.py"
    if not path.exists():
        pytest.skip(f"reference clone absent: {path}")
    spec = importlib.util.spec_from_file_location("_ref_brain", path)
    if spec is None or spec.loader is None:
        pytest.skip("reference brain.py is not loadable")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ref_brain"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"reference brain.py did not import: {exc}")
    return mod


def _paired(seed=42):
    """One reference area and one port sharing an identical connectome."""
    from neural_assemblies.assembly_calculus.attention_area import AttentionArea
    ref_mod = _reference_module()
    ref = ref_mod.AttentionArea([N], N, CAP, DENSITY, PLASTICITY)
    ref.reset()
    ours = AttentionArea(N, CAP, DENSITY, PLASTICITY,
                         rng=np.random.default_rng(seed))
    # the two draw their own connectomes; copy the reference's across so the
    # comparison is of the RULE, not of two random matrices
    ours.recurrent_weights = np.array(ref.recurrent_weights, dtype=np.float64)
    ours.recurrent_change = np.zeros_like(ours.recurrent_weights)
    return ref, ours


def test_the_set_rule_matches_the_reference_exactly():
    ref, ours = _paired()
    pre = np.arange(0, 12)
    post = np.arange(20, 33)
    ref.activations = pre
    ref.update(post)
    ours.fire(pre)
    ours.update(post)
    assert np.array_equal(ours.recurrent_weights, ref.recurrent_weights)
    assert np.array_equal(ours.recurrent_change, ref.recurrent_change)


def test_the_undo_matches_the_reference_exactly():
    ref, ours = _paired()
    baseline = np.array(ref.recurrent_weights, copy=True)
    pre, post = np.arange(0, 9), np.arange(30, 41)
    ref.activations = pre
    ref.update(post)
    ref.decay_weights()
    ours.fire(pre)
    ours.update(post)
    ours.decay_weights()
    assert np.array_equal(ours.recurrent_weights, ref.recurrent_weights)
    assert np.allclose(ours.recurrent_weights, baseline), (
        "a bind then a release restores the baseline exactly")


def test_potentiation_SATURATES_rather_than_compounding():
    """The property that separates this rule from every other one here."""
    _, ours = _paired()
    pre, post = np.arange(0, 6), np.arange(10, 17)
    ours.fire(pre)
    ours.update(post)
    once = np.array(ours.recurrent_weights, copy=True)
    ours.decay_weights()
    ours.update(post)
    assert np.array_equal(ours.recurrent_weights, once), (
        "a second bind of the same pair must change nothing: the rule assigns "
        "(1 + p) rather than scaling by it")
    live = once[np.ix_(pre, post)]
    assert set(np.unique(live)).issubset({0.0, 1.0 + PLASTICITY}), (
        "weights are binary under the set rule, so no clip is needed")


def test_a_second_bind_without_release_is_REFUSED():
    """Where this port deliberately departs from the reference.

    `change` is assigned, not accumulated, so a second update makes the undo
    over-subtract: from baseline 1 it would leave 1 - p**2. The reference does
    not guard this; refusing is the honest behaviour and the divergence is
    recorded in the module docstring.
    """
    _, ours = _paired()
    ours.fire(np.arange(0, 5))
    ours.update(np.arange(9, 15))
    with pytest.raises(RuntimeError, match="already holds an unreleased bind"):
        ours.update(np.arange(9, 15))


def test_the_reference_would_corrupt_its_baseline_there():
    """The divergence above is justified only if the hazard is real."""
    ref, _ = _paired()
    baseline = np.array(ref.recurrent_weights, copy=True)
    pre, post = np.arange(0, 5), np.arange(9, 15)
    ref.activations = pre
    ref.update(post)
    ref.update(post)          # the reference permits this
    ref.decay_weights()
    live = np.ix_(pre, post)
    was_on = baseline[live] > 0
    assert not np.allclose(ref.recurrent_weights[live][was_on],
                           baseline[live][was_on]), (
        "two binds then one release does NOT restore the reference")
    expected = 1.0 - PLASTICITY ** 2
    assert np.allclose(ref.recurrent_weights[live][was_on], expected), (
        f"it leaves {expected}, which is the arithmetic the guard prevents")


def test_the_recurrent_read_matches_the_reference():
    ref, ours = _paired()
    pre = np.arange(3, 14)
    ref.activations = pre
    ours.fire(pre)
    assert np.allclose(ours.recurrent_input(),
                       ref.recurrent_weights[ref.activations].sum(axis=0))


def test_attend_is_a_bind_read_release_cycle():
    from neural_assemblies.assembly_calculus.attention_area import attend
    _, ours = _paired()
    baseline = np.array(ours.recurrent_weights, copy=True)
    picked = attend(ours, np.arange(0, 7), np.arange(21, 29))
    assert picked.shape == (CAP,)
    assert np.allclose(ours.recurrent_weights, baseline), (
        "attend must leave the area exactly as it found it")
