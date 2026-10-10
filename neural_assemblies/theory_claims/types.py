"""The theory register's types: a claim's status, its evidence references, its sensitivity
checks, the claim itself, and the citation syntax.

Moved from neural_assemblies/theory.py unchanged."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

CITATION = re.compile(r"\[\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\]\]")
EVIDENCE_ROLES = frozenset({
    "artifact", "registration", "producer", "analysis", "comparison", "log",
})


class Status:
    """How much weight a claim can carry, and what would invalidate it."""

    #: Stated and proved in the cited source. Transfers only inside its
    #: preconditions -- check them before relying on it.
    PROVED = "PROVED"

    #: Established empirically in this repository, with the experiment named.
    #: Transfers to the regime it was measured in; re-measure outside it.
    MEASURED = "MEASURED"

    #: We rely on this BEYOND where it is proved or measured. Every use is a
    #: standing risk and should say what would falsify it.
    EXTENSION = "EXTENSION"


@dataclass(frozen=True)
class EvidenceRef:
    """One resolvable repository file and its role in supporting a result."""

    path: str
    role: str
    limitation: str = ""


@dataclass(frozen=True)
class SensitivityCheck:
    """A retained treatment/control comparison that must still move.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-result-sensitivity

    Paths use RFC 6901 tokens without the optional leading slash and ``*`` to
    expand a JSON list. The resulting vectors are compared pairwise, so a
    missing seed, reordered arm, or dead probe is a register-validation failure
    rather than prose debt.
    """

    artifact: str
    sample_path: str
    treatment_path: str
    control_path: str
    relation: str
    minimum_effect: float
    mechanism: str


@dataclass(frozen=True)
class Result:
    """One citable claim, with the conditions under which it holds."""

    id: str
    status: str
    claim: str
    source: str
    preconditions: Sequence[str] = field(default_factory=tuple)
    evidence: Sequence[str] = field(default_factory=tuple)
    evidence_refs: Sequence[EvidenceRef] = field(default_factory=tuple)
    provenance_gap: str = ""
    sensitivity_checks: Sequence[SensitivityCheck] = field(default_factory=tuple)
    sensitivity_gap: str = ""
    implemented_by: Sequence[str] = field(default_factory=tuple)
    caveat: str = ""
    engine: str = ""  # measurement substrate; empty for non-empirical entries

    def __str__(self) -> str:
        head = f"[{self.status}] {self.id}: {self.claim}"
        bits = [f"    source: {self.source}"]
        if self.engine:
            bits.append(f"    engine: {self.engine}")
        if self.preconditions:
            bits.append("    requires: " + "; ".join(self.preconditions))
        if self.evidence:
            bits.append("    evidence: " + "; ".join(self.evidence))
        if self.evidence_refs:
            bits.append("    evidence files: " + "; ".join(ref.path for ref in self.evidence_refs))
        if self.provenance_gap:
            bits.append("    PROVENANCE GAP: " + self.provenance_gap)
        if self.sensitivity_checks:
            bits.append("    sensitivity: " + "; ".join(
                f"{check.mechanism} ({check.artifact})"
                for check in self.sensitivity_checks))
        if self.sensitivity_gap:
            bits.append("    SENSITIVITY GAP: " + self.sensitivity_gap)
        if self.implemented_by:
            bits.append("    used by: " + "; ".join(self.implemented_by))
        if self.caveat:
            bits.append(f"    CAVEAT: {self.caveat}")
        return head + "\n" + "\n".join(bits)
