"""Where did each golden's numbers come from -- the paper, or us?

A golden recorded by running our own code is a REGRESSION test: it pins that
today's output matches last month's. That is worth having, but it is not
reproduction, and the two are easy to confuse because they live in the same
directory, are consumed by the same runner, and both go green.

Reproduction requires a number that exists independently of this repo. This
script tiers every artifact in ``parity/golden/`` by that criterion, and
reports how many paper PDFs are on disk and therefore minable for tier-A
targets.

    A  the number is from the paper (table, figure, or quoted text)
    B  recorded by running the AUTHORS' reference code, ported here
    C  recorded by running OUR implementation -- regression only

Run:  python research/literature/audit_golden_provenance.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

GOLDEN_DIR = "research/literature/parity/golden"
PAPER_DIR = "research/literature/papers"

# A source string is tier A only if it points OUTSIDE this repo at a document.
_TIER_A = re.compile(r"extracted from pdf|arxiv|table\s*\d|figure\s*\d|fig\.\s*\d|"
                     r"paper\s+(?:reports|states|table|figure)", re.I)
# Tier B: the authors' own released code, which we ported.
_TIER_B = re.compile(r"learning-with-assemblies|\.ipynb|mdabagia|nemo-demo|"
                     r"reference/nemo_numpy", re.I)


def tier_of(source: str, notes: str) -> str:
    blob = f"{source} {notes}"
    if _TIER_A.search(blob):
        return "A"
    if _TIER_B.search(blob):
        return "B"
    return "C"


def scan_goldens(golden_dir: str) -> List[Tuple[str, str, str]]:
    out = []
    for path in sorted(glob.glob(os.path.join(golden_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        name = os.path.basename(path)
        if not isinstance(data, dict):
            out.append((name, "C", "(no provenance -- not an object)"))
            continue
        src = str(data.get("source", ""))
        notes = str(data.get("notes", ""))
        out.append((name, tier_of(src, notes), src or "(no source field)"))
    return out


def mine_pdf(path: str, max_len: int = 320) -> List[str]:
    """Sentences carrying both a digit and an assembly-calculus keyword."""
    try:
        import fitz
    except ImportError:
        return []
    kw = re.compile(r"(overlap|converg|assembl|pattern complet|associat|merge|"
                    r"recall|accuracy|precision|F1|percent|%)", re.I)
    doc = fitz.open(path)
    text = " ".join(p.get_text() for p in doc).replace("\n", " ")
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in sents
            if kw.search(s) and re.search(r"\d", s) and len(s) < max_len]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine", action="store_true",
                    help="also mine paper PDFs for candidate tier-A numbers")
    ap.add_argument("--show", type=int, default=0,
                    help="print N mined sentences per paper")
    args = ap.parse_args()

    rows = scan_goldens(GOLDEN_DIR)
    by: Dict[str, List[Tuple[str, str]]] = {"A": [], "B": [], "C": []}
    for name, tier, src in rows:
        by[tier].append((name, src))

    label = {"A": "from the PAPER", "B": "from the AUTHORS' code (our port)",
             "C": "from OUR implementation -- regression only"}
    for tier in "ABC":
        print(f"\n=== TIER {tier}: {label[tier]} -- {len(by[tier])} ===")
        for name, src in by[tier]:
            print(f"   {name:<40} {src[:74]}")

    total = len(rows)
    print(f"\nTOTAL {total} goldens -> "
          f"A={len(by['A'])}  B={len(by['B'])}  C={len(by['C'])}")
    if total:
        print(f"share independent of this repo (A): {len(by['A'])/total:.0%}")

    pdfs = sorted(glob.glob(os.path.join(PAPER_DIR, "*.pdf")))
    print(f"\npaper PDFs on disk: {len(pdfs)} (tier-A targets are minable from these)")
    if not args.mine:
        print("re-run with --mine to size the extractable claims")
        return

    grand = 0
    for p in pdfs:
        hits = mine_pdf(p)
        grand += len(hits)
        print(f"  {os.path.basename(p):<36} {len(hits):>4} candidate quantitative claims")
        for s in hits[:args.show]:
            print(f"        * {s[:150]}")
    print(f"\ntotal candidate claims across PDFs: {grand}")


if __name__ == "__main__":
    main()
