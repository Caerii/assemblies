"""Retracted DIRECT prototype retained as an explicit refusal boundary.

The former overlap readouts were insensitive to the learned CAUSE-to-BIND
fiber: wiping that fiber, swapping cues, and feed-forward controls did not move
the claimed metrics. Returning those numbers would therefore present substrate
overlap as learned directional or interventional evidence.
"""

from __future__ import annotations

from typing import NoReturn

from neural_assemblies.exceptions import RetractedProtocol

_RETRACTION = (
    "The DIRECT overlap prototype is retracted: its readout is insensitive to "
    "the learned CAUSE->BIND fiber. Implement a synaptic-asymmetry readout and "
    "a wipe negative control under a new protocol ID."
)


def _reject() -> NoReturn:
    raise RetractedProtocol(_RETRACTION)


def direct_bind(
    brain,
    cause: str,
    effect: str,
    bind: str,
    cause_stim: str | None = None,
    effect_stim: str | None = None,
    rounds: int = 10,
) -> NoReturn:
    """Reject the retired directional-binding training protocol."""
    _reject()


def measure_directional_asymmetry(
    brain,
    cause: str,
    effect: str,
    bind: str,
    probe_rounds: int = 5,
) -> NoReturn:
    """Reject the retired fiber-insensitive overlap readout."""
    _reject()


def intervention_bind_overlap(
    brain,
    fixed_area: str,
    cue_area: str,
    bind: str,
    probe_rounds: int = 5,
) -> NoReturn:
    """Reject the retired intervention-labelled overlap readout."""
    _reject()


def validate_direct_do_calculus(
    brain,
    cause: str,
    effect: str,
    bind: str,
    probe_rounds: int = 5,
    min_forward: float = 0.1,
) -> NoReturn:
    """Reject the retired validation protocol before it returns a score."""
    _reject()
