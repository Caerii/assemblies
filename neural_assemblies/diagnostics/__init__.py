"""Runtime diagnostics for assemblies: is this measurement trustworthy?

WHY THIS MODULE EXISTS
----------------------
On 2026-07-28/29 a research session produced eight experiments, six of which
refuted the one before them. Not one of the six was refuted by new theory --
each was refuted by a CONTROL that cost almost nothing to add and had simply
not been there. The failures fell into four repeatable shapes:

  1. COLLAPSE UPSTREAM. An area holding many items merges them into one, and
     everything downstream then reads exactly chance. Six hours were spent on
     "composition fails at depth" before a spread column showed the PARENTS had
     become a single assembly before composition ever ran.
  2. DEAD PROBE. Reading under `read_only()` from an area that was never
     materialised returns the same degenerate winners for every input, so
     accuracy is EXACTLY chance and margin is EXACTLY 1.00. Twice mistaken for
     a negative result.
  3. WRONG INDEX SPACE. `area.winners` holds compact engine indices;
     `ops._snap` returns stable neuron IDs. Comparing across them is silently
     at chance.
  4. STABILITY MISTAKEN FOR DISCRIMINABILITY. After collapse every item still
     re-cues to high overlap with what was stored, because it returns THE
     collapsed assembly. A probe reading only self-overlap reports success at
     0.7763 while rank-1 identity is 0.0143.

Every one of those is mechanically detectable. This module detects them. The
organising idea is that a diagnostic should answer "can I believe this number?",
not merely "what is this number?" -- so the functions here return verdicts and
reasons, not just floats.

WHAT THIS IS NOT
----------------
Not a metrics library. `research/experiments/_substrate.py` covers reading,
similarity and probing for experiments. This is for interrogating a LIVE brain,
including mid-training, and is safe to call from production code.

LAYOUT
------
One module per question; this package re-exports every name, so
`from neural_assemblies.diagnostics import X` is unchanged:

    verdict      Verdict: a diagnostic's answer, with its reasons
    readout      the one sanctioned readout (read_assembly, assembly_overlap)
    health       area_health: collapse, dead probes, stability vs identity
    drive        drive_breakdown: where a target's drive comes from
    sweeps       whole-brain sweeps: recurrence_audit, collapse_scan
    fibers       fiber_census, pricing_exposure: what each fiber holds and costs
    regime       regime_audit, require_regime: is this the regime the laws need
    report       format_report
    arbitration  arbitrate: ask an engine that does not sample
    ensembles    ensembles over seeds, paired deltas, arm comparison
    ranks        rank statistics without scipy
    load         load matching: when an A/B on the sampler is not an A/B
    gain         gain stability: when a sweep measures the boundary
    separation   score the ORDERING, not the scale
    probes       verify_probe: a measurement that cannot vary is not one
"""
from __future__ import annotations

from .verdict import (Verdict)  # noqa: F401
from .readout import (read_assembly, assembly_overlap, _spread)  # noqa: F401
from .health import (_NO_CUES, AreaHealth, area_health)  # noqa: F401
from .drive import (DriveBreakdown, drive_breakdown)  # noqa: F401
from .sweeps import (recurrence_audit, collapse_scan)  # noqa: F401
from .fibers import (FiberState, _safe, _fiber_shape, fiber_census, PricingExposure, pricing_exposure)  # noqa: F401
from .regime import (Regime, regime_audit, require_regime)  # noqa: F401
from .report import (format_report)  # noqa: F401
from .arbitration import (ARBITER_ARMS, Arm, _ARM_SPECS, arm_spec, Arbitration, arbitrate, arbitrate_prebuilt)  # noqa: F401
from .ensembles import (Ensemble, ensemble, _validate_ensemble_keys, _t_interval, ensemble_from_values, compare_arms, paired_delta)  # noqa: F401
from .ranks import (rankdata, spearman, partial_spearman)  # noqa: F401
from .load import (LoadGap, area_load, load_audit)  # noqa: F401
from .gain import (GainStability, gain_stability)  # noqa: F401
from .separation import (Separation, separation)  # noqa: F401
from .probes import (ProbeCheck, verify_probe)  # noqa: F401

__all__ = [
    "Verdict", "AreaHealth", "DriveBreakdown", "FiberState",
    "PricingExposure",
    "read_assembly", "assembly_overlap",
    "area_health", "drive_breakdown", "recurrence_audit", "collapse_scan",
    "fiber_census", "pricing_exposure",
    "format_report",
    "Arbitration", "arbitrate", "arbitrate_prebuilt", "ARBITER_ARMS",
    "Arm", "arm_spec",
    "Ensemble", "ensemble", "compare_arms", "paired_delta",
    "LoadGap", "area_load", "load_audit",
    "GainStability", "gain_stability",
    "Separation", "separation",
    "ProbeCheck", "verify_probe",
]
