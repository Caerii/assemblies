#!/usr/bin/env python3
"""Extract figure/table text snippets from downloaded literature PDFs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "research" / "literature" / "papers"
OUT = ROOT / "research" / "literature" / "FIGURES.md"


PAPER_META = {
    "hoff2026_epwta.pdf": {
        "id": "hoff2026epwta",
        "focus": ["Table 1", "Table 2", "Figure 2", "Figure 3", "recovery", "E%-WTA", "ω_inh", "beta"],
    },
    "kopadi2026_direct.pdf": {
        "id": "kopadi2026causal",
        "focus": ["Figure 9", "Figure 10", "Figure 11", "Alzheimer", "Precision", "do-calculus", "adaptive"],
    },
    "dabagia2024_coinflipping.pdf": {
        "id": "dabagia2024coinflipping",
        "focus": ["softmax", "Markov", "Figure", "probability", "empirical"],
    },
    "mitropolsky2025_acquisition.pdf": {
        "id": "mitropolsky2025simulated",
        "focus": ["curriculum", "Figure", "accuracy", "POS", "role", "vocabulary"],
    },
    "dabagia2025_sequences.pdf": {
        "id": "dabagia2025sequences",
        "focus": ["Figure", "recall", "Turing", "LRI", "sequence"],
    },
    "dabagia2022_colt.pdf": {
        "id": "dabagia2022classify",
        "focus": ["MNIST", "Figure", "accuracy", "classif"],
    },
    "ting2026_speech.pdf": {
        "id": "ting2026speech",
        "focus": ["TIMIT", "47.5", "F1", "Figure", "MFCC", "mel"],
    },
    "mitropolsky2021_tacl.pdf": {
        "id": "mitropolsky2021parser",
        "focus": ["Figure", "Russian", "English", "fiber", "parser", "accuracy"],
    },
    "papadimitriou2019_itcs.pdf": {
        "id": "papadimitriou2019random",
        "focus": ["Figure", "convergence", "associate", "merge", "O(log"],
    },
}


def extract_snippets(pdf_path: Path, keywords: list[str], max_pages: int = 20) -> list[dict]:
    doc = fitz.open(pdf_path)
    hits: list[dict] = []
    for i in range(min(len(doc), max_pages)):
        text = doc.load_page(i).get_text()
        lower = text.lower()
        if any(k.lower() in lower for k in keywords):
            # grab paragraphs containing keywords
            for para in re.split(r"\n\s*\n", text):
                if any(k.lower() in para.lower() for k in keywords):
                    snippet = " ".join(para.split())
                    if len(snippet) > 40:
                        hits.append({"page": i + 1, "text": snippet[:800]})
    doc.close()
    return hits[:25]


def main() -> None:
    lines = [
        "# Literature figure and table notes",
        "",
        "Auto-extracted from PDFs in `papers/`. Use for golden protocol design.",
        "",
    ]
    index: dict[str, list] = {}

    for pdf_name, meta in PAPER_META.items():
        pdf = PAPERS / pdf_name
        if not pdf.is_file():
            lines.append(f"## {meta['id']} — **missing** `{pdf_name}`")
            lines.append("")
            continue
        snippets = extract_snippets(pdf, meta["focus"])
        index[meta["id"]] = snippets
        lines.append(f"## {meta['id']} (`{pdf_name}`)")
        lines.append("")
        if not snippets:
            lines.append("_No keyword hits in first 20 pages; see full PDF._")
        for s in snippets:
            lines.append(f"- **p.{s['page']}:** {s['text']}")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = ROOT / "research" / "literature" / "figure_snippets.json"
    json_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({sum(len(v) for v in index.values())} snippets)")


if __name__ == "__main__":
    main()
