"""Applying an amendment to the documents, the same way every time.

Registering or recording an amendment of the refraction-memory programme touched seven places by
hand -- the pre-registration, its scorecard, the theory register, the citation allowlist, the
generated register.md, the notebook, the manuscript -- and every slip on the way was one of the
same few: a file's line endings flipped (the repository mixes CRLF and LF per file), a backslash
eaten by a shell, a row inserted at the wrong anchor. This module does the mechanical part:

    append_section(path, text)            append a section, in the file's own line endings
    add_scorecard_rows(path, rows)        rows after the scorecard table's last row
    theory_result(...)                    a theory.Result block, as code, from its fields
    add_claim(path, block, before_id)     that block before an existing claim
    allow_citation(path, claim_id, after_id, comment)
                                          the claim in test_theory_citations' active list
    render_register(root)                 docs/register.md regenerated from theory.py
    Prereg(root)                          the pre-registration as per-amendment sources:
                                          register / record / scorecard, then rebuild
    anchor(heading)                       the Markdown anchor a heading gets

Each refuses an ambiguous anchor rather than guessing. What is not mechanical -- the prose of a
registration and the reading of a result -- comes from research.experiments.memory_lib.Registration.
"""
from __future__ import annotations

import re
from pathlib import Path


def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    return text, nl


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _in(text, nl):
    """LF-authored text in the file's line endings"""
    return text.replace("\r\n", "\n").replace("\n", nl)


def append_section(path, text):
    """append ``text`` (LF-authored) after one blank line, in the file's line endings"""
    body, nl = _read(path)
    _write(path, body.rstrip("\r\n") + nl + nl + _in(text.strip("\n"), nl) + nl)


def add_scorecard_rows(path, rows, header="| Bar | Registered | Verdict | Deciding number |"):
    """insert ``rows`` (LF-authored table rows) after the last row of the table under ``header``"""
    body, nl = _read(path)
    lines = body.split(nl)
    starts = [i for i, line in enumerate(lines) if line.strip() == header]
    if len(starts) != 1:
        raise ValueError(f"expected one scorecard header, found {len(starts)}")
    end = starts[0] + 1
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    new = [r for r in rows.strip("\n").split("\n") if r]
    if not all(r.startswith("|") and r.endswith("|") for r in new):
        raise ValueError("scorecard rows must be table rows")
    lines[end:end] = new
    _write(path, nl.join(lines))


def anchor(heading):
    """the GitHub anchor of a Markdown heading: '### Amendment 55 result (2026-10-10)' ->
    'amendment-55-result-2026-10-10'"""
    text = heading.lstrip("#").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _lines(s, width=88, lead='              '):
    """a Python string expression wrapped over lines (adjacent literals)"""
    words, out, cur = s.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur + " ")
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    out.append(cur)
    return ("\n" + lead).join(_q(x) for x in out)


def theory_result(*, id, engine, claim, source, preconditions, evidence, artifact, artifact_role,
                  checks, caveat=None):
    """a theory.Result(...) block as code (LF), MEASURED, one artifact (``artifact_role`` the
    EvidenceRef's description of it), sensitivity checks given as
    dicts {treatment_path, control_path, relation, minimum_effect, mechanism, sample_path?}"""
    out = [
        "    Result(",
        f"        id={_q(id)},",
        f"        engine={_q(engine)},",
        "        status=Status.MEASURED,",
        f"        claim={_lines(claim, lead='              ')},",
        f"        source={_q(source)},",
        "        preconditions=(" + ",\n                       ".join(_q(p) for p in preconditions) + "),",
        f"        evidence=({_q(evidence)},),",
        "        evidence_refs=(",
        f"            EvidenceRef({_q(artifact)}, \"artifact\",",
        f"                        {_q(artifact_role)}),",
        "        ),",
    ]
    sc = []
    for c in checks:
        sc.append("SensitivityCheck(\n"
                  f"            artifact={_q(artifact)},\n"
                  f"            sample_path={_q(c.get('sample_path', 'run/seeds'))},\n"
                  f"            treatment_path={_q(c['treatment_path'])},\n"
                  f"            control_path={_q(c['control_path'])},\n"
                  f"            relation={_q(c['relation'])}, minimum_effect={c['minimum_effect']!r},\n"
                  f"            mechanism={_q(c['mechanism'])},\n"
                  "        )")
    out.append("        sensitivity_checks=(" + ", ".join(sc) + "),")
    if caveat:
        out.append(f"        caveat={_lines(caveat, lead='               ')},")
    out.append("    ),")
    return "\n".join(out) + "\n"


def add_claim(path, block, before_id):
    """insert a Result block (LF-authored) before the claim ``before_id`` -- in the file ``path``,
    or, given the claims package (neural_assemblies/theory_claims/), in the module that holds it"""
    path = Path(path)
    if path.is_dir():
        holders = [p for p in sorted(path.glob("*.py")) if f'id="{before_id}",' in p.read_text(encoding="utf-8")]
        if len(holders) != 1:
            raise ValueError(f"expected one module holding {before_id!r}, found {len(holders)}")
        path = holders[0]
    body, nl = _read(path)
    anchor_text = "    Result(" + nl + f'        id="{before_id}",'
    if body.count(anchor_text) != 1:
        raise ValueError(f"expected one claim {before_id!r}, found {body.count(anchor_text)}")
    _write(path, body.replace(anchor_text, _in(block, nl) + anchor_text))


def allow_citation(path, claim_id, after_id, comment):
    """list ``claim_id`` after ``after_id`` in the active sensitivity-check list"""
    body, nl = _read(path)
    a = f'            "{after_id}",' + nl
    if body.count(a) != 1:
        raise ValueError(f"expected one {after_id!r} in the list, found {body.count(a)}")
    _write(path, body.replace(a, a + f"            # {comment}" + nl + f'            "{claim_id}",' + nl))


class Prereg:
    """The refraction-memory pre-registration as SOURCES: research/notes/memory/prereg/ holds one
    file per section group (A00 the preface, A<N> an amendment with its result and adoption notes,
    scorecard the scorecard), concatenated in the order ORDER lists into
    PREREG_refraction_memory.md -- the file every run's record points at and the theory register
    cites by anchor, so it stays whole, rebuilt after every edit (and checked fresh by a test)."""

    def __init__(self, root="."):
        self.dir = Path(root) / "research" / "notes" / "memory" / "prereg"
        self.file = Path(root) / "research" / "notes" / "memory" / "PREREG_refraction_memory.md"

    def order(self):
        return [n for n in (self.dir / "ORDER").read_text(encoding="utf-8").split("\n") if n]

    def text(self):
        """the whole document, concatenated from its sources (bytes as stored)"""
        parts = []
        for name in self.order():
            with open(self.dir / name, encoding="utf-8", newline="") as fh:
                parts.append(fh.read())
        return "".join(parts)

    def build(self):
        _write(self.file, self.text())

    def _source(self, amendment):
        return self.dir / f"A{int(amendment):02d}.md"

    def register(self, amendment, text):
        """a new amendment's registration (LF-authored) as its own source, last in ORDER"""
        names = self.order()
        new = self._source(amendment).name
        if new in names:
            raise ValueError(f"{new} is already registered")
        last = self.dir / names[-1]
        body, nl = _read(last)
        _write(last, body.rstrip("\r\n") + nl + nl)
        _write(self.dir / new, _in(text.strip("\n"), nl) + nl)
        (self.dir / "ORDER").write_text("\n".join(names + [new]) + "\n", encoding="utf-8", newline="\n")
        self.build()

    def record(self, amendment, text):
        """a result section (LF-authored) appended to its amendment's source"""
        names = self.order()
        src = self._source(amendment)
        if src.name not in names:
            raise ValueError(f"Amendment {amendment} is not registered")
        body, nl = _read(src)
        tail = nl if src.name != names[-1] else ""           # keep the blank line before the next
        _write(src, body.rstrip("\r\n") + nl + nl + _in(text.strip("\n"), nl) + nl + tail)
        self.build()

    def scorecard(self, rows):
        add_scorecard_rows(self.dir / "scorecard.md", rows)
        self.build()

    def index(self):
        """README.md for the sources: each file, its title, when it was registered and recorded"""
        rows = ["# The refraction-memory pre-registration, by amendment", "",
                "`PREREG_refraction_memory.md` is BUILT from these files, in the order `ORDER` lists "
                "(`research.amend.Prereg`): edit a source, then rebuild; never edit the built file. "
                "Generated by `Prereg.index()`.", "",
                "| file | section | registered | result |", "|---|---|---|---|"]
        for name in self.order():
            with open(self.dir / name, encoding="utf-8", newline="") as fh:
                lines = fh.read().replace("\r\n", "\n").split("\n")
            head = next((ln for ln in lines if ln.startswith("## Amendment ") or ln == "## Scorecard"),
                        next((ln for ln in lines if ln.startswith("# ")), lines[0]))
            m = re.match(r"## Amendment (\d+) \(([^)]*)\):?\s*(.*)", head)
            title = f"{m.group(1)}: {m.group(3)}" if m else head.lstrip("# ")
            date = re.search(r"\((\d{4}-\d\d-\d\d)", head)
            result = next((re.search(r"\((\d{4}-\d\d-\d\d)\)", ln) for ln in lines
                           if ln.startswith("### Amendment") and " result " in ln), None)
            rows.append(f"| [{name}]({name}) | {title} | {date.group(1) if date else ''} | "
                        f"{result.group(1) if result else ''} |")
        return "\n".join(rows) + "\n"

    def write_index(self):
        (self.dir / "README.md").write_text(self.index(), encoding="utf-8", newline="\n")


def render_register(root="."):
    """docs/register.md from neural_assemblies.theory (LF, UTF-8, as its freshness test expects)"""
    from neural_assemblies import theory
    with open(Path(root) / "docs" / "register.md", "w", newline="\n", encoding="utf-8") as fh:
        fh.write(theory.render_markdown())
