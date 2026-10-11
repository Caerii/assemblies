"""A diagnostic's answer: a Verdict says whether a number can be believed, and why.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

from dataclasses import dataclass




# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

@dataclass
class Verdict:
    """A judgement plus the reason for it. The reason is the point.

    A bare boolean sends the reader back to the code to find out what was
    checked; every failure this module exists to catch was originally missed by
    someone reading a number without its provenance.
    """

    ok: bool
    label: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        mark = "OK  " if self.ok else "WARN"
        return f"[{mark}] {self.label}{(': ' + self.detail) if self.detail else ''}"
