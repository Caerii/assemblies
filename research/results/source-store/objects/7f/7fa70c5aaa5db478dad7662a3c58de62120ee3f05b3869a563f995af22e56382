#!/usr/bin/env python3
"""Validate research/literature/index.json structure.

Also checks that the bibliography and the CODE agree, which is the part that
matters. Docstrings across this repo cite claims as ``[PNAS20]``, ``[HOFF26]``
and so on, and until those tags were added to the index they resolved to
nothing: a citation was a bare string, and a typo or an invented tag read
exactly like a real reference. The checks below close that -- every citation-
shaped tag used in code must name a real entry, and every entry must carry a
unique tag.

Run directly, or via tests/test_literature_index.py so it cannot rot unnoticed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED_ENTRY_KEYS = {
    "id",
    "tier",
    "year",
    "title",
    "authors",
    "venue",
    "bibkey",
    "cite_tag",
    "urls",
    "mechanisms",
    "implementation_status",
    "package_modules",
    "gaps",
}

VALID_STATUSES = {"implemented", "partial", "legacy", "research", "not_started"}
VALID_TIERS = {"foundations", "nemo_language", "extensions"}

# A cite_tag is uppercase alphanumerics: PNAS20, HOFF26, NEMOREF.
CITE_TAG_RE = re.compile(r"^[A-Z][A-Z0-9]{2,11}$")

# What counts as a citation when scanning source. Deliberately NARROW -- letters
# then a two-digit year -- because this codebase is full of bracketed tags that
# are not citations: [AREA], [HIGH], [MOOD], [SEMANTIC], [CONTEXT], [PREDICTION],
# [P600], [N400], [SUBJ], [VERB]. Requiring >=3 letters AND exactly 2 trailing
# digits matches [PNAS20]/[HOFF26] and none of those (verified: the only matches
# in the tree are the two real citation tags). All-letter reference tags like
# [NEMOREF] cannot be distinguished this way, so they are matched against the
# declared set instead of by shape.
CITATION_RE = re.compile(r"\[([A-Z]{3,10}[0-9]{2})\]")

# Trees scanned for citations. research/ is included because the experiment
# scripts cite papers too, and a stale tag there misleads just as effectively.
SCAN_DIRS = ("neural_assemblies", "research")


def scan_citations(
    root: Path,
    declared: set[str],
    scan_dirs: tuple[str, ...] = SCAN_DIRS,
) -> tuple[set[str], dict[str, list[str]]]:
    """Find citation tags used in source under *root*.

    Returns ``(used, unknown)`` -- the declared tags found, and a mapping from
    each unrecognised citation tag to the files citing it.

    Separate from :func:`main` so it can be pointed at a temporary tree, which
    is how the test gives itself a negative control: a checker that silently
    approves everything and a checker that works are indistinguishable when both
    are only ever run against a clean repo.
    """
    used: set[str] = set()
    unknown: dict[str, list[str]] = {}
    for name in scan_dirs:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for tag in set(CITATION_RE.findall(text)):
                used.add(tag)
                if tag not in declared:
                    unknown.setdefault(tag, []).append(
                        path.relative_to(root).as_posix())
            # Second pass for DECLARED tags the shape regex cannot see -- the
            # all-letter reference tags such as [NEMOREF]. Without this they are
            # reported as never cited while in fact they are, which makes the
            # "catalogued but never cited" list wrong in the direction that
            # hides work already done.
            for tag in declared:
                if tag not in used and f"[{tag}]" in text:
                    used.add(tag)
    # Only declared tags are "used"; unrecognised ones come back via `unknown`.
    return used & declared, unknown


def main() -> int:
    index_path = Path(__file__).with_name("index.json")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for i, entry in enumerate(data.get("entries", [])):
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            errors.append(f"entries[{i}] ({entry.get('id', '?')}): missing {sorted(missing)}")
        if entry.get("implementation_status") not in VALID_STATUSES:
            errors.append(
                f"entries[{i}] ({entry.get('id')}): bad status {entry.get('implementation_status')!r}"
            )
        if entry.get("tier") not in VALID_TIERS:
            errors.append(f"entries[{i}] ({entry.get('id')}): bad tier {entry.get('tier')!r}")

    roadmap = data.get("implementation_roadmap_priority", [])
    ids = {e["id"] for e in data.get("entries", [])}
    for pid in roadmap:
        if pid not in ids:
            errors.append(f"roadmap references unknown id: {pid}")

    root = index_path.parents[2]
    entries = data.get("entries", [])
    refs = data.get("reference_implementations", [])

    # --- cite_tag: well-formed and unique across papers AND reference impls ---
    tag_owner: dict[str, str] = {}
    for entry in entries + refs:
        tag = entry.get("cite_tag")
        who = entry.get("id", "?")
        if tag is None:
            errors.append(f"{who}: no cite_tag")
            continue
        if not CITE_TAG_RE.match(tag):
            errors.append(f"{who}: malformed cite_tag {tag!r}")
        if tag in tag_owner:
            errors.append(
                f"duplicate cite_tag {tag!r}: {tag_owner[tag]} and {who}. "
                f"Tags are how code refers to a paper, so they must be unique."
            )
        else:
            tag_owner[tag] = who

    # --- declared file paths actually exist ---------------------------------
    for entry in entries:
        who = entry["id"]
        pdf = entry.get("urls", {}).get("local_pdf")
        if pdf and not (root / pdf).is_file():
            errors.append(f"{who}: local_pdf missing on disk: {pdf}")
        for module in entry.get("package_modules", []):
            if not (root / module).exists():
                errors.append(
                    f"{who}: package_modules path does not exist: {module}. "
                    f"Either the file moved or the parity claim is stale."
                )
    for ref in refs:
        path = ref.get("local_path")
        if path and not (root / path).exists():
            # Not an error: reference checkouts are optional (sync_reference_repos
            # fetches them) and CI may run without them.
            print(f"note: reference checkout absent: {path} ({ref['id']})")

    # --- every citation used in code resolves to an entry -------------------
    used, unknown = scan_citations(root, set(tag_owner), SCAN_DIRS)
    for tag, where in sorted(unknown.items()):
        errors.append(
            f"code cites [{tag}] but no index entry declares that cite_tag "
            f"(in {', '.join(where[:3])}"
            f"{f' +{len(where) - 3} more' if len(where) > 3 else ''})"
        )

    if errors:
        print("Literature index validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    n_pdf = sum(1 for e in entries if e.get("urls", {}).get("local_pdf"))
    print(f"Literature index OK ({len(entries)} papers, {len(refs)} reference "
          f"implementations, {n_pdf} local PDFs)")
    print(f"  citations resolved: {', '.join(sorted(used)) or 'none'}")
    uncited = sorted(t for t in tag_owner if t not in used)
    if uncited:
        # Informational, not an error: a paper can be catalogued before anything
        # cites it. Printed so the gap between "we know about it" and "we build
        # on it" stays visible.
        print(f"  catalogued but never cited in code: {', '.join(uncited)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
