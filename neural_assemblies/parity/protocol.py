"""Literature reproduction protocol types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict


class Backend(str, Enum):
    """Execution backend for a protocol."""

    BRAIN = "brain"
    NEMO_NUMPY = "nemo_numpy"
    LEGACY = "legacy"
    MIXED = "mixed"
    CROSS_LANG = "cross_lang"


@dataclass(frozen=True)
class Protocol:
    """Pinned reproduction specification."""

    protocol_id: str
    paper_id: str
    claim_ids: tuple[str, ...]
    golden_path: str
    backend: Backend = Backend.BRAIN
    config_file: str | None = None
    regime: str | None = None
    repro_command: str | None = None
    slow: bool = False
    description: str = ""

    @property
    def primary_claim_id(self) -> str:
        return self.claim_ids[0] if self.claim_ids else self.protocol_id


@dataclass
class ProtocolResult:
    """Outcome of running or verifying a protocol."""

    protocol_id: str
    claim_id: str
    passed: bool
    backend: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    golden_metrics: Dict[str, Any] = field(default_factory=dict)
    diffs: Dict[str, Any] = field(default_factory=dict)
    repro_command: str | None = None
    exit_code: int | None = None
    duration_s: float = 0.0
    git_sha: str | None = None
    message: str = ""

    def to_manifest(self) -> dict:
        return asdict(self)
