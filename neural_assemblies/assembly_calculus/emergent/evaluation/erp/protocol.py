"""The ERP measurement PROTOCOL as an explicit value, not process-global state.

WHY. `ERP_EXPECTED_SLOT` and `ERP_AFFERENT_ENERGY` are read INSIDE
`measure_live_integration` / `phrase_stability` -- three call levels below any
caller who chose them. Consequences, all of which happened:

  * The same function call means different things depending on process
    environment, so a result cannot be reproduced from the call site alone.
  * Experiments A/B by MUTATING `os.environ` with save/restore around each arm.
    That is global state as an argument-passing mechanism: it cannot be nested,
    it leaks on exception, and it makes "which arm produced this number" a
    property of when you looked rather than of the value.
  * A flag flipped as a DEFAULT behaves differently from the same flag set in
    the environment, because the default also applies inside cache-fill
    calibration where no caller is present. Chasing that difference cost hours.

WHAT THIS IS NOT. It is not a second way to configure the ERP path. `ErpProtocol`
is the value; `from_environment()` is the ONE adapter that reads the legacy
variables, called once at the entry point. Everything below takes the value.
Adding another `os.environ.get` inside the measurement path re-opens the door.

MIGRATION STATE (deliberately partial, and stated rather than implied):
`from_environment()` exists and is tested, and the flag semantics now live in
exactly one place. The call chain from `measure_live_integration` down to
`phrase_stability` is NOT yet threaded -- that touches the live ERP metric,
which is under active investigation (#108) and whose magnitudes must not move as
a side effect of a plumbing change. Two adoptions were rolled back today for
exactly that coupling. Thread it behind a measurement, not with one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "on", "yes")


@dataclass(frozen=True)
class ErpProtocol:
    """Every choice that changes what an ERP number MEANS.

    Frozen because a protocol that can be mutated mid-study is the environment
    variable again with extra steps. Use `replace()` (or `.with_(...)`) to
    derive an arm, which makes the difference between arms a visible expression
    rather than a temporal one.

    Attributes:
        expected_slot: dispatch the probed area on the slot the PARSE PREDICTS
            rather than the observed word's category. The only way to
            area-match the grammatical/violation contrast -- and it INVERTS on
            freshly-trained parsers, so it is off by default (#108).
        afferent_energy: measure drive INTO an area instead of self-recurrent
            energy. Rejected on measurement (AUC 0.000, zero variance across
            four seeds) but kept as a correct implementation for once the arms
            are area-matched.
        debug: print per-probe diagnostics.
    """

    expected_slot: bool = False
    afferent_energy: bool = False
    debug: bool = False

    @classmethod
    def from_environment(cls, env=None) -> "ErpProtocol":
        """THE one adapter for the legacy variables. Call once, at the top.

        Kept so existing scripts and CI invocations keep working unchanged. New
        code should construct an `ErpProtocol` directly and pass it, because a
        value that arrives as an argument can be logged with the result -- which
        is what makes a number attributable to an arm.
        """
        env = os.environ if env is None else env
        return cls(
            expected_slot=_truthy(env.get("ERP_EXPECTED_SLOT")),
            afferent_energy=_truthy(env.get("ERP_AFFERENT_ENERGY")),
            debug=_truthy(env.get("ERP_DEBUG")),
        )

    def with_(self, **changes) -> "ErpProtocol":
        """Derive an arm. `base.with_(expected_slot=True)` IS the manipulation."""
        return replace(self, **changes)

    def describe(self) -> str:
        """One line naming every non-default choice, for the result artefact.

        "default" rather than an empty string on purpose: a study that records
        nothing and a study that recorded "no deviations" must not look alike.
        """
        on = [f for f in ("expected_slot", "afferent_energy", "debug")
              if getattr(self, f)]
        return "+".join(on) if on else "default"


#: The shipped protocol. Both measurement flags are OFF: `expected_slot` inverts
#: on freshly-trained parsers and `afferent_energy` was rejected on measurement.
DEFAULT_PROTOCOL = ErpProtocol()
