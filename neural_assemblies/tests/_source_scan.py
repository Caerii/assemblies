"""One way to scan source for a pattern: CODE ONLY, never prose.

WHY THIS MODULE EXISTS. Two ratchets in this suite freeze per-file counts of a
risky pattern. Both originally scanned with a line regex, and both therefore
counted the pattern inside DOCSTRINGS AND COMMENTS. That is not a cosmetic
inaccuracy -- it is a guard that punishes documenting the very hazard it tracks,
and it makes a frozen baseline part code and part commentary, so an edit that
deletes a real occurrence and adds a comment about it nets to zero.

  * `test_index_space_ratchet` counted a docstring warning that `.w` is the
    wrong divisor as a NEW ambiguous access. Fixed there with `tokenize`.
  * `test_methodology_ratchet` was NOT fixed, and later flagged
    `research/experiments/erp_rng_leak_check.py` for a `Brain(seed=)` that
    appears only in a sentence explaining why global RNG state matters.

Same defect, two doors, one fixed. That is the pattern this refactor exists to
stop, so the fix lives in ONE place both ratchets call.

`blank_prose` is the primary entry point BECAUSE IT PRESERVES LINE STRUCTURE.
The alternative -- emitting a token stream -- would have forced both ratchets to
rewrite their per-line logic (`_BRAIN.search(line) and "engine=" not in line`
reads a whole call site from one line), and rewriting a guard while fixing it is
how guards lose their baselines. Blanking in place keeps every existing line
scan working, minus the prose.
"""
from __future__ import annotations

import io
import re
import tokenize


def blank_prose(text: str) -> str:
    """*text* with string and comment CONTENT replaced by spaces.

    Line count, line lengths and column offsets are all preserved, so any
    line-based scan keeps working and reported line numbers stay true. A
    multi-line docstring becomes the same number of blank lines.

    Returns *text* UNCHANGED when it will not tokenize. That over-counts (prose
    included) rather than under-counting: some research scripts do not parse,
    and a guard that silently reads zero on an unparseable file has a hole
    exactly where sloppy code lives. Over-counting fails loud and is
    recoverable; under-counting is invisible.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return text

    lines = text.splitlines(keepends=True)
    out = [list(line) for line in lines]
    for tok in toks:
        if tok.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        for row in range(r1, r2 + 1):
            idx = row - 1
            if idx < 0 or idx >= len(out):
                continue
            line = out[idx]
            start = c1 if row == r1 else 0
            end = c2 if row == r2 else len(line)
            for col in range(start, min(end, len(line))):
                if line[col] != "\n":
                    line[col] = " "
    return "".join("".join(line) for line in out)


def code_lines(text: str) -> list:
    """`blank_prose` split into lines -- the common shape for a line scan."""
    return blank_prose(text).splitlines()


def count_attribute_reads(text: str, attr: str) -> int:
    """Count ``.attr`` ATTRIBUTE READS in code -- an OP '.' then NAME *attr*.

    Kept as a token walk rather than a regex over `blank_prose` because for
    attribute access the token pair IS the definition: it cannot be fooled by a
    same-named local, a keyword argument, or a partial word.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pat = re.compile(rf"\.{re.escape(attr)}\b")
        return sum(1 for line in text.splitlines() if pat.search(line))

    n = 0
    prev_op_dot = False
    for tok in toks:
        if tok.type == tokenize.OP and tok.string == ".":
            prev_op_dot = True
            continue
        if prev_op_dot and tok.type == tokenize.NAME and tok.string == attr:
            n += 1
        prev_op_dot = False
    return n
