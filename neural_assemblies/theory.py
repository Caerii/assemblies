"""The indexed register of results this codebase stands on.

WHY THIS EXISTS
---------------
Most of what this system does is an instance of something proved in the
literature, a measurement we made ourselves, or an EXTRAPOLATION beyond both.
Those three are not interchangeable, and until they are written down they are
indistinguishable in a docstring: "the theory requires kp >= 3 ln n" reads the
same whether it is a theorem, our own sweep, or a guess.

So every claim a docstring leans on gets an ID here, with its PRECONDITIONS and
its STATUS. Code cites it by ID in double brackets. Three consequences:

* an extension is obvious at the point of use, because its status says so;
* preconditions travel with the claim, so "we are inside the theorem" is
  checkable rather than assumed -- this is what `diagnostics.regime_audit`
  automates for the one precondition we kept violating;
* citations cannot rot: `unresolved_citations` scans the package and
  `neural_assemblies/tests/test_theory_citations.py` fails on a dangling one.

CITATION SYNTAX. Result IDs are UPPERCASE-DASHED, written in double
brackets at the point of use. The
lowercase-kebab ``[[silent-no-op-dead-fibers]]`` links already used throughout
the codebase point at operator memory, not at results, and the checker
deliberately ignores them.

STATUS IS NOT QUALITY. A `MEASURED` result can be better evidence for our
purposes than a `PROVED` one whose preconditions we cannot meet. The point of
the label is to say what would have to be true for the claim to transfer, not
to rank the claims.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Sequence

#: Matches an UPPERCASE-DASHED result citation in double brackets. Lowercase
#: kebab links are operator memory, not results, and never match.
from .theory_claims.types import (CITATION, EVIDENCE_ROLES, EvidenceRef,  # noqa: F401  re-exported
                                   Result, SensitivityCheck, Status)
from .theory_claims import ALL as _CLAIMS

#: every claim, in the register's order (the programme modules of theory_claims/)
_RESULTS: List[Result] = list(_CLAIMS)

RESULTS: Dict[str, Result] = {r.id: r for r in _RESULTS}


def cite(result_id: str) -> Result:
    """Look up a result, raising if the ID is unknown."""
    try:
        return RESULTS[result_id]
    except KeyError:
        raise KeyError(
            f"unknown result {result_id!r}. Add it to neural_assemblies/theory.py "
            f"rather than citing an ID that does not resolve. Known: "
            f"{', '.join(sorted(RESULTS))}"
        ) from None


def extensions() -> List[Result]:
    """Everything the codebase relies on beyond what is proved or measured."""
    return [r for r in _RESULTS if r.status == Status.EXTENSION]


def _source_files(root: str) -> List[str]:
    out = []
    for base, _dirs, names in os.walk(root):
        if any(part in base for part in (".git", "__pycache__", "reference")):
            continue
        out += [os.path.join(base, n) for n in names
                if n.endswith((".py", ".md"))]
    return out


def unresolved_citations(root: str) -> Dict[str, List[str]]:
    """Map each dangling result citation to the files citing it.

    Lowercase-kebab links are operator memory and are ignored by `CITATION`.
    """
    missing: Dict[str, List[str]] = {}
    for path in _source_files(root):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:                                      # noqa: PERF203
            continue
        for rid in set(CITATION.findall(text)):
            if rid not in RESULTS:
                missing.setdefault(rid, []).append(path)
    return missing


def _json_values(document, path: str) -> list:
    """Resolve RFC 6901 tokens with ``*`` list expansion into scalar values.

    A token that is a decimal index without leading zeros selects one element
    of a list, as RFC 6901 specifies; ``*`` selects every element. Anything
    else must name a key of a mapping.
    """
    nodes = [document]
    for raw_token in path.split('/'):
        if not raw_token:
            raise ValueError("empty sensitivity path component")
        if re.search(r"~(?![01])", raw_token):
            raise ValueError("invalid RFC 6901 escape in sensitivity path")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        expanded = []
        for node in nodes:
            if (isinstance(node, list) and re.fullmatch(r"0|[1-9][0-9]*", token)
                    and int(token) < len(node)):
                expanded.append(node[int(token)])
            elif token == '*':
                if not isinstance(node, list):
                    raise ValueError("sensitivity wildcard requires a list")
                expanded.extend(node)
            elif isinstance(node, dict) and token in node:
                expanded.append(node[token])
            else:
                raise ValueError(f"missing sensitivity path component {token!r}")
        nodes = expanded
    if nodes and all(isinstance(node, list) for node in nodes):
        return [value for node in nodes for value in node]
    if any(isinstance(node, (dict, list)) for node in nodes):
        raise ValueError("sensitivity path must resolve only to scalar values")
    return nodes


def _sensitivity_errors(result: Result, check: SensitivityCheck, root: str) -> list[str]:
    prefix = f"{result.id}: sensitivity {check.mechanism or '<unnamed>'}"
    if not check.mechanism.strip():
        return [f"{prefix} must name the mechanism under test"]
    if not any(ref.role == "artifact" and ref.path == check.artifact
               for ref in result.evidence_refs):
        return [f"{prefix} artifact is not a typed artifact evidence edge"]
    allowed = {"all-greater", "all-less", "all-different"}
    if check.relation not in allowed:
        return [f"{prefix} has unsupported relation {check.relation!r}"]
    if (type(check.minimum_effect) not in (int, float)
            or not math.isfinite(check.minimum_effect)
            or check.minimum_effect <= 0):
        return [f"{prefix} has invalid minimum effect"]
    normalized = os.path.normpath(check.artifact).replace("\\", "/")
    target = os.path.abspath(os.path.join(root, check.artifact))
    if (not check.artifact or normalized != check.artifact
            or os.path.commonpath((root, target)) != root):
        return [f"{prefix} has unsafe artifact path {check.artifact!r}"]
    try:
        with open(target, encoding="utf-8") as handle:
            document = json.load(handle)
        samples = _json_values(document, check.sample_path)
        treatment = _json_values(document, check.treatment_path)
        control = _json_values(document, check.control_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"{prefix} cannot resolve retained values: {exc}"]
    if (not samples or len(samples) != len(set(map(repr, samples)))):
        return [f"{prefix} sample identities are empty or duplicated"]
    if len(samples) != len(treatment) or len(treatment) != len(control):
        return [f"{prefix} sample/treatment/control vectors have unequal lengths"]
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in [*treatment, *control]):
        return [f"{prefix} values must be finite numbers"]
    if check.relation == "all-greater":
        effects = [left - right for left, right in zip(treatment, control, strict=True)]
    elif check.relation == "all-less":
        effects = [right - left for left, right in zip(treatment, control, strict=True)]
    else:
        effects = [abs(left - right) for left, right in zip(treatment, control, strict=True)]
    if any(effect < check.minimum_effect for effect in effects):
        return [
            f"{prefix} does not move by {check.minimum_effect:g} under "
            f"{check.relation}; minimum retained effect is {min(effects):g}",
        ]
    return []


def evidence_reference_errors(root: str) -> List[str]:
    """Validate local evidence edges and retained mechanism sensitivity.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-result-sensitivity

    This checks whether the evidence still exists and whether each declared
    treatment/control contrast still clears its frozen minimum effect. It does
    not decide whether the scientific claim follows from that contrast.
    """
    root = os.path.abspath(root)
    errors = []
    for result in _RESULTS:
        if result.status == Status.MEASURED and not (result.evidence_refs or result.provenance_gap):
            errors.append(f"{result.id}: measured result has no typed evidence or provenance gap")
        if (result.status == Status.MEASURED
                and not any(ref.role == "artifact" for ref in result.evidence_refs)
                and not result.provenance_gap):
            errors.append(f"{result.id}: no result artifact and no provenance gap")
        if (result.status == Status.MEASURED
                and not result.sensitivity_checks and not result.sensitivity_gap):
            errors.append(f"{result.id}: no retained sensitivity check or explicit gap")
        for ref in result.evidence_refs:
            normalized = os.path.normpath(ref.path).replace("\\", "/")
            target = os.path.abspath(os.path.join(root, ref.path))
            if (not ref.path or normalized != ref.path or os.path.commonpath((root, target)) != root):
                errors.append(f"{result.id}: unsafe evidence path {ref.path!r}")
            elif not os.path.isfile(target):
                errors.append(f"{result.id}: dangling evidence path {ref.path}")
            if ref.role not in EVIDENCE_ROLES:
                errors.append(f"{result.id}: unsupported evidence role {ref.role!r}")
        for check in result.sensitivity_checks:
            errors.extend(_sensitivity_errors(result, check, root))
    return errors


def format_index(results: Sequence[Result] = ()) -> str:
    """Render the register, extensions last so they are what you read last."""
    order = {Status.PROVED: 0, Status.MEASURED: 1, Status.EXTENSION: 2}
    items = list(results) or _RESULTS
    return "\n\n".join(str(r) for r in
                       sorted(items, key=lambda r: (order[r.status], r.id)))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _split_findings(text: str) -> List[str]:
    """A caveat written as '(1) ... (2) ...' becomes one item per number;
    any other caveat is one item."""
    parts = re.split(r"\s*\(\d+\)\s+", text.strip())
    parts = [x.strip() for x in parts if x.strip()]
    return parts if len(parts) > 1 else [text.strip()]


def render_markdown() -> str:
    """The register as Markdown: one section per entry, in file order."""
    out = ["# Register of results", "",
           "Rendered from `neural_assemblies/theory.py` by "
           "`python -m neural_assemblies.theory --render`; do not edit by hand. "
           "Each entry is cited elsewhere by its ID in double brackets. Statuses: PROVED (in the "
           "cited source, inside its preconditions), MEASURED (in this repository, "
           "in the regime named), EXTENSION (relied on beyond either).", ""]
    by_status: Dict[str, List[Result]] = {}
    for r in _RESULTS:
        by_status.setdefault(r.status, []).append(r)
    out.append("| ID | Status | Engine / substrate | Claim |")
    out.append("|----|--------|--------------------|-------|")
    for r in _RESULTS:
        first = r.claim.split(". ")[0].rstrip(".") + "."
        out.append(f"| [`{r.id}`](#{r.id.lower()}) | {r.status} | {r.engine or 'Not an empirical entry'} | {first} |")
    out.append("")
    for r in _RESULTS:
        out.append(f"## {r.id}")
        out.append("")
        out.append(f"**Status.** {r.status}. **Source.** {r.source}")
        out.append("")
        if r.engine:
            out.append(f"**Engine / substrate.** {r.engine}")
            out.append("")
        out.append(f"**Claim.** {r.claim}")
        out.append("")
        if r.preconditions:
            out.append("**Requires.**")
            out.extend(f"- {x}" for x in r.preconditions)
            out.append("")
        if r.evidence:
            out.append("**Evidence.**")
            out.extend(f"- {x}" for x in r.evidence)
            out.append("")
        if r.evidence_refs:
            out.append("**Evidence files.**")
            for ref in r.evidence_refs:
                suffix = f" â€” {ref.limitation}" if ref.limitation else ""
                out.append(f"- [{ref.path}](../{ref.path}) ({ref.role}){suffix}")
            out.append("")
        if r.provenance_gap:
            out.append(f"**Provenance gap.** {r.provenance_gap}")
            out.append("")
        if r.sensitivity_checks:
            out.append("**Mechanism sensitivity.**")
            for check in r.sensitivity_checks:
                out.append(
                    f"- {check.mechanism}: `{check.treatment_path}` "
                    f"{check.relation} `{check.control_path}` by at least "
                    f"{check.minimum_effect:g}, retained in "
                    f"[{check.artifact}](../{check.artifact}) and paired by "
                    f"`{check.sample_path}`."
                )
            out.append("")
        if r.sensitivity_gap:
            out.append(f"**Sensitivity gap.** {r.sensitivity_gap}")
            out.append("")
        if r.implemented_by:
            out.append("**Used by.** " + "; ".join(f"`{x}`" for x in r.implemented_by))
            out.append("")
        if r.caveat:
            items = _split_findings(r.caveat)
            if len(items) > 1:
                out.append("**Findings and caveats.**")
                out.extend(f"{i}. {x}" for i, x in enumerate(items, 1))
            else:
                out.append(f"**Caveat.** {items[0]}")
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


if __name__ == "__main__":
    import sys as _sys
    if "--render" in _sys.argv:
        _sys.stdout.write(render_markdown())
    else:
        print(format_index())
