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
    """insert a Result block (LF-authored) before the claim ``before_id`` in theory.py"""
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


def render_register(root="."):
    """docs/register.md from neural_assemblies.theory (LF, UTF-8, as its freshness test expects)"""
    from neural_assemblies import theory
    with open(Path(root) / "docs" / "register.md", "w", newline="\n", encoding="utf-8") as fh:
        fh.write(theory.render_markdown())
