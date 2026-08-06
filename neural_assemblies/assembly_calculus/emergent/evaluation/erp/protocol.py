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

MIGRATION STATE: THREADED (#115, 2026-08-06). `erp/adapters.py` -- the
measurement layer -- now contains no `import os` at all, and a comment where the
import was says why. `from_environment()` is called at exactly two boundaries,
`measure_live_integration` and `measure_lexical_surprise`, plus the two drivers
above them (`run_incremental_erp_probes`, `collect_frame_samples`) so a study
can resolve it ONCE per parse and pass the value down.

What that buys, concretely:

  * `phrase_stability` takes a REQUIRED `protocol` and has no default. It runs
    once per phrase area per probe, so the old `os.environ` read made the
    environment a per-area input to a measurement three levels below anyone who
    chose it. A default of `from_environment()` would have looked threaded
    while leaving the door open.
  * `_expected_slot_enabled()` is GONE. A zero-argument function returning an
    environment variable is the shape this module exists to eliminate.
  * `_ERP_DEBUG` is gone too, and it was the worst of the three: a MODULE-LEVEL
    read, frozen at import, so `ERP_DEBUG=1` set by any test or study after the
    module loaded did nothing whatsoever, silently.
  * One parse can no longer measure its first word under one protocol and its
    last under another.

The earlier note here said to thread it "behind a measurement, not with one",
because #108 was live and its magnitudes must not move as a side effect. #108
closed (f79c4f5) and the afferent-energy question closed with it (0fcedb4), so
that condition is met: this change resolves the same values from the same
environment and is intended to be numerically inert.
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
            energy. NOT ADOPTED, re-measured 2026-08-06 now that the arms ARE
            area-matched: AUC 0.7167 -> 0.7500, delta +0.0333+/-0.0627 with the
            CI spanning zero, while the span nearly doubles. It spreads the
            conditions further apart without ordering them better. (The
            original "AUC 0.000, zero variance" rejection is void -- taken while
            the arms probed different areas.)
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
#: `afferent_energy` stays OFF -- RE-measured cold on area-matched arms
#: (2026-08-06) and its AUC delta's CI spans zero, so it does not discriminate
#: better; see research/notes/afferent_energy_is_not_adopted.md.
DEFAULT_PROTOCOL = ErpProtocol()
