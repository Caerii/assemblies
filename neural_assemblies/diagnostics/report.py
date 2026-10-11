"""A plain-text report of the diagnostics for a brain.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from typing import (List, Mapping)


from neural_assemblies.core.measurement import Measured

from .drive import DriveBreakdown
from .fibers import FiberState, PricingExposure
from .health import AreaHealth
from .regime import Regime
from .verdict import Verdict


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_report(items) -> str:
    """Render health objects, breakdowns or verdicts as aligned text."""
    lines: List[str] = []
    seq = items.values() if isinstance(items, Mapping) else items
    for it in (seq if isinstance(seq, (list, tuple, type({}.values()))) else [seq]):
        if isinstance(it, AreaHealth):
            # `_num` prints "n/a" for an undefined quantity rather than a
            # number. It must never silently substitute one: the whole reason
            # these are `Measured` is that a missing spread used to print as
            # NaN and a missing margin as nothing at all.
            def _num(m: Measured, spec: str, suffix: str = "") -> str:
                return f"{m:{spec}}{suffix}" if m.defined else "n/a"

            lines.append(
                f"  {it.area:<16} items {it.n_items:<5} spread "
                f"{_num(it.spread, '.4f')} (floor {_num(it.floor, '.4f')})"
                # Printed next to spread ON PURPOSE: these two disagree exactly
                # when partial collapse is happening, and seeing the pair is
                # what makes that visible at a glance.
                + f"  distinct {_num(it.distinct_frac, '.3f')}"
                + (f"  acc {_num(it.accuracy, '.4f')}"
                   f"  margin {_num(it.margin, '.2f', 'x')}"
                   if it.accuracy.defined or it.margin.defined else "")
                + (f"  [{it.unbounded_margins} unbounded]"
                   if it.unbounded_margins else "")
                + (f"  [{it.unmatched_reads} matched nothing]"
                   if it.unmatched_reads else ""))
            lines += [f"      {v}" for v in it.verdicts if not v.ok]
        elif isinstance(it, Regime):
            lines.append("  " + str(it))
        elif isinstance(it, DriveBreakdown):
            lines.append(f"  drive into {it.target}:")
            for s, v in sorted(it.per_source.items(),
                               key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
                lines.append(f"      {s:<16}{v:>10.4f}  {it.share(s):>7.1%}")
            lines += [f"      {v}" for v in it.verdicts if not v.ok]
        elif isinstance(it, FiberState):
            tag = ("DEAD*" if it.silently_ignored else
                   "dead " if it.dead else
                   "hot  " if it.potentiated else "ok   ")
            lines.append(
                f"  [{tag}] {it.src:>14} -> {it.dst:<14} "
                f"{it.rows:>6}x{it.cols:<6} nnz {it.nnz:>8}  "
                f"dst_w {it.dst_w:>6}"
                + (f"  p99/med {it.p99_over_median:.1f}x"
                   if it.p99_over_median == it.p99_over_median else "")
                + ("   <-- delivers ZERO drive into a live area"
                   if it.silently_ignored else ""))
        elif isinstance(it, PricingExposure):
            lines.append(f"  [{'EXPOSED' if it.exposed else 'clean  '}] {it}")
        elif isinstance(it, Verdict):
            if not it.ok:
                lines.append(f"  {it}")
    return "\n".join(lines) if lines else "  (nothing flagged)"
