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
            area-match the grammatical/violation contrast, and ON BY DEFAULT
            since 2026-08-06 (#108).

            It was off, on the claim that it "inverts on freshly-trained
            parsers". That claim is REFUTED. Re-measured through the harness
            with `ASSEMBLIES_BACKBONE_CACHE=0`, 10 seeds, `disk_hits=0
            trained_fresh=10`: p600_auc_of_raw 0.9056+/-0.0268 ->
            0.7167+/-0.0805, above chance on every seed. It does not invert --
            it SHRINKS, which is what removing a confound is supposed to do.
            The 0.906 was the confound; 0.717 is the effect.
        afferent_energy: measure drive INTO an area instead of self-recurrent
            energy. Rejected on measurement (AUC 0.000, zero variance across
            four seeds) but kept as a correct implementation for once the arms
            are area-matched.
        debug: print per-probe diagnostics.
    """

    expected_slot: bool = True
    afferent_energy: bool = False
    debug: bool = False

    @classmethod
    def from_environment(cls, env=None) -> "ErpProtocol":
        """THE one adapter for the legacy variables. Call once, at the top.

        Kept so existing scripts and CI invocations keep working unchanged. New
        code should construct an `ErpProtocol` directly and pass it, because a
        value that arrives as an argument can be logged with the result -- which
        is what makes a number attributable to an arm.

        `ERP_EXPECTED_SLOT` is now an OVERRIDE of a True default rather than an
        opt-in, so it must distinguish ABSENT from PRESENT. Absent takes the
        default; PRESENT is parsed for truthiness, so `ERP_EXPECTED_SLOT=0`
        restores the old observed-category dispatch for reproducing
        pre-adoption numbers.

        The line is drawn at absent-vs-present, NOT at absent-or-empty. An
        empty value is falsey like "0" and "off", which keeps the whole falsey
        set behaving identically -- `test_falsey_spellings_do_not_enable`
        exists because an earlier version of this flag made two spellings of
        one intent behave differently, and a True default is exactly the
        condition that invites that mistake back.
        """
        env = os.environ if env is None else env
        raw_slot = env.get("ERP_EXPECTED_SLOT")
        return cls(
            expected_slot=(cls.expected_slot if raw_slot is None
                           else _truthy(raw_slot)),
            afferent_energy=_truthy(env.get("ERP_AFFERENT_ENERGY")),
            debug=_truthy(env.get("ERP_DEBUG")),
        )

    def with_(self, **changes) -> "ErpProtocol":
        """Derive an arm. `base.with_(expected_slot=True)` IS the manipulation."""
        return replace(self, **changes)

    def describe(self) -> str:
        """One line naming every ACTIVE choice, for the result artefact.

        ACTIVE, not "non-default", and the difference started mattering on
        2026-08-06 when `expected_slot` became the default. A description
        written as a delta from the default is only readable if you also know
        WHICH default was shipped that week -- so an artefact from before the
        flip and one from after would both say "default" while describing
        opposite dispatches. Name what was on.

        Never an empty string: a study that records nothing and a study that
        recorded "no flags" must not look alike.
        """
        on = [f for f in ("expected_slot", "afferent_energy", "debug")
              if getattr(self, f)]
        return "+".join(on) if on else "no-flags"


#: The shipped protocol. `expected_slot` is ON -- it area-matches the
#: grammatical/violation contrast, which nothing else can do, and the cold
#: 10-seed study says it shrinks the effect rather than inverting it.
#: `afferent_energy` stays OFF: rejected on measurement (AUC 0.000, zero seed
#: variance), and that rejection was itself an artefact of the arms probing
#: different areas -- it is worth RE-measuring now that they do not.
DEFAULT_PROTOCOL = ErpProtocol()
